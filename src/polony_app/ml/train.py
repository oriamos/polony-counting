"""Fine-tune Spotiflow from a selected base model on a snapshot's train / val crops -> registry candidate.

Explicit only: nothing in the app calls this automatically. The result is registered with status ``candidate`` and is
never used by the app until someone promotes it (``ml promote``) after a locked-test comparison.

How fine-tuning starts from existing weights (verified against spotiflow 0.6.5): the base is loaded with
``Spotiflow.from_pretrained(name, inference_mode=False)`` / ``from_folder(path, inference_mode=False)`` (this is exactly
what the ``spotiflow-train --finetune-from`` CLI does) and ``model.fit(...)`` then continues from those weights. Training
runs in a staging directory so a failed run leaves nothing in the registry.
"""
from __future__ import annotations

import csv
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .common import (DEFAULT_NORMALISER, LO_HI_POLICY, MLError, check_downscale, check_normaliser, hash_obj, models_dir,
                     prepare_crop, sha256_file, utc_now)
from .dataset import list_snapshots, load_crop, load_manifest, snapshot_path, verify_snapshot
from .manifest import CropRecord, SnapshotManifest
from .registry import Registry
from .splits import load_lock, lock_fingerprint

Say = Callable[[str], None]


@dataclass
class TrainOptions:
    base: str = "general"
    epochs: int = 30
    batch_size: int = 4
    lr: float = 1e-4
    crop_size: int = 256
    seed: int = 0
    device: str = "cpu"                 # 'auto' = CUDA/MPS when present
    num_train_samples: Optional[int] = None
    pos_weight: float = 10.0
    smart_crop: bool = True
    augment: bool = True
    optimize_threshold: bool = True
    early_stopping_patience: int = 0
    num_workers: int = 0
    notes: str = ""
    # safeguards against "fine-tuning" on a handful of fields of one or two slides (unstable, and not evidence of anything)
    min_train_groups: int = 4           # independent groups (specimen/slide level of the split) in TRAIN
    min_train_crops: int = 20
    allow_small: bool = False           # --allow-small: train anyway (recorded as a warning in the registry entry)
    divergence_factor: float = 1.25     # last-epoch val loss > factor x best  =>  'diverged' (reported; best weights kept)


@dataclass
class TrainPlan:
    version: str
    manifest: SnapshotManifest
    train: list[CropRecord]
    val: list[CropRecord]
    base: dict
    checks: list[str]
    warnings: list[str]
    lock_sha256: Optional[str]
    config: dict = None            # {downscale, normaliser} the snapshot was built for = what the model is trained at


def _locked_sources(data_dir, lock: dict) -> tuple[set[str], set[str]]:
    """(file sha256 set, group set) of the locked test images."""
    lm = load_manifest(data_dir, lock["snapshot"])
    shas = {c.source_sha256 for c in lm.crops if c.split == "test"}
    for s in lm.sources:
        if s.split == "test":
            shas |= set(s.tiff_sha256.values())
    return shas, set(lock.get("groups", []))


def plan_training(data_dir: str | Path, version: str, opts: TrainOptions) -> TrainPlan:
    from .spotiflow_io import describe_base
    data_dir = Path(data_dir)
    checks: list[str] = []
    warns: list[str] = []
    m = load_manifest(data_dir, version)
    bad = verify_snapshot(data_dir, version)
    if bad:
        raise MLError(f"snapshot {version} failed its integrity check ({'; '.join(bad[:3])}); refusing to train on it")
    checks.append(f"snapshot {version} is intact ({len(m.files)} files verified)")
    if opts.epochs < 1 or opts.batch_size < 1 or opts.lr <= 0 or opts.crop_size < 16:
        raise MLError("invalid hyper-parameters: need epochs >= 1, batch_size >= 1, lr > 0, crop_size >= 16")
    train = [c for c in m.crops if c.split == "train"]
    val = [c for c in m.crops if c.split == "val"]
    if not train:
        raise MLError(f"snapshot {version} has no valid TRAIN crops (see its warnings: too few independent groups or "
                      "reviewed regions). Nothing to train on.")
    if not val:
        raise MLError(f"snapshot {version} has no valid VALIDATION crops; a validation set from other slides is required "
                      "(validation crops of the training slides would leak). Nothing was trained.")
    if sum(c.n_positive for c in train) == 0:
        raise MLError("the train crops contain no human-confirmed positive points; nothing to learn from")
    if sum(c.n_positive for c in val) == 0:
        warns.append("the validation crops contain no positives: validation loss / threshold optimisation are weak")
    cfgs = {(c.downscale, c.normaliser) for c in train + val}
    if len(cfgs) != 1:
        raise MLError(f"the train/val crops of snapshot {version} do not share one downscale / normaliser ({sorted(cfgs)}); "
                      "rebuild the snapshot")
    ds_, norm_ = next(iter(cfgs))
    if not norm_ or any(c.norm_lo is None or c.norm_hi is None for c in train + val if norm_ != "none"):
        raise MLError(f"snapshot {version} was built before page-level normalisation was recorded (its crops carry no page lo/hi): "
                      "training on it would normalise per crop, which the app does not do at inference. Rebuild the snapshot.")
    config = {"downscale": check_downscale(ds_), "normaliser": check_normaliser(norm_)}
    checks.append(f"trained at downscale {config['downscale']} with normaliser '{config['normaliser']}' "
                  "(page-level lo/hi from the manifest); the app must run the model with the same settings")
    n_groups = len({c.group_id for c in train})
    small = []
    if n_groups < opts.min_train_groups:
        small.append(f"only {n_groups} independent training group(s) (< {opts.min_train_groups})")
    if len(train) < opts.min_train_crops:
        small.append(f"only {len(train)} training crop(s) (< {opts.min_train_crops})")
    if small:
        msg = ("training data too small to say anything about generalisation: " + "; ".join(small) +
               ". A leave-specimen-out check on 2-3 slides showed unstable training and no gain over the pretrained model.")
        if not opts.allow_small:
            raise MLError(msg + " Refused; add more reviewed slides, or pass --allow-small to train anyway (recorded).")
        warns.append(msg + " (--allow-small: training anyway)")
    checks.append(f"{len(train)} train crop(s) / {sum(c.n_positive for c in train)} positives; "
                  f"{len(val)} val crop(s) / {sum(c.n_positive for c in val)} positives")
    lock = load_lock(data_dir)
    lock_sha = None
    if lock:
        lock_sha = lock.get("lock_sha256") or lock_fingerprint(lock)
        shas, groups = _locked_sources(data_dir, lock)
        leaked = [c.crop_id for c in train + val if c.group_id in groups or c.source_sha256 in shas or c.frozen_from]
        if leaked:
            raise MLError("REFUSED: training/validation data contains locked test material "
                          f"({len(leaked)} crop(s), e.g. {leaked[:3]}). The locked test set may never be trained on.")
        checks.append(f"no train/val crop belongs to a locked test group or file (lock {lock_sha[:12]})")
    else:
        warns.append("no locked test set exists: this candidate can be trained but cannot be promoted until one is locked "
                     "(ml split --snapshot <version> --lock-test)")
    # snapshot-internal leakage
    tr_g = {c.group_id for c in train}
    va_g = {c.group_id for c in val}
    if tr_g & va_g:
        raise MLError(f"train and validation share group(s) {sorted(tr_g & va_g)}; refusing (leakage)")
    checks.append("train and validation groups are disjoint")
    base = describe_base(opts.base, data_dir)
    checks.append(f"base model resolves to {base['kind']} '{base['name']}'")
    return TrainPlan(version, m, train, val, base, checks, warns + list(m.warnings), lock_sha, config)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        try:
            import pytorch_lightning as pl
            pl.seed_everything(seed, workers=True)
        except Exception:
            pass
    except Exception:
        pass


def _load_xy(data_dir, version, crops, config):
    """Network inputs: native crop -> page-level normalisation -> block-mean downscale; labels mapped with the exact
    pixel-centre rule x_s = (x - (d-1)/2) / d. Returns (images, spots, n_labels_dropped_outside_the_downscaled_frame)."""
    imgs, spots, dropped = [], [], 0
    for c in crops:
        im, lab = load_crop(data_dir, version, c)
        x, lab_s, n, _ = prepare_crop(im, lab, lo=c.norm_lo, hi=c.norm_hi, downscale=config["downscale"],
                                      normaliser=config["normaliser"], origin_xy=c.box_xyxy[:2])
        imgs.append(x)
        spots.append(lab_s)
        dropped += n
    return imgs, spots, dropped


def _val_curve(logdir: Path) -> list:
    """[(epoch, val_loss)] from the CSV logger (one entry per validation evaluation)."""
    out = []
    for f in sorted(logdir.rglob("metrics.csv")):
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                v = row.get("val_loss")
                if v not in (None, ""):
                    e = row.get("epoch")
                    out.append((int(float(e)) if e not in (None, "") else len(out), float(v)))
    return out


def _val_loss_from_logs(logdir: Path, divergence_factor: float = 1.25) -> dict:
    """Best-epoch AND last-epoch validation loss. ``diverged`` = the last epoch is clearly worse than the best one (or not
    finite). The registered weights are the BEST epoch's (best.pt); last-epoch numbers are reported next to them."""
    curve = _val_curve(logdir)
    last_train = None
    for f in sorted(logdir.rglob("metrics.csv")):
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                t = row.get("train_loss") or row.get("train_loss_epoch")
                if t not in (None, ""):
                    last_train = float(t)
    if not curve:
        return {"best_val_loss": None, "best_epoch": None, "last_val_loss": None, "last_epoch": None,
                "last_train_loss": last_train, "n_val_evaluations": 0, "diverged": None}
    finite = [(e, v) for e, v in curve if np.isfinite(v)]
    be, bv = min(finite, key=lambda t: t[1]) if finite else (None, None)
    le, lv = curve[-1]
    div = (not np.isfinite(lv)) or (bv is not None and lv > divergence_factor * bv)
    return {"best_val_loss": bv, "best_epoch": be, "last_val_loss": lv, "last_epoch": le, "last_train_loss": last_train,
            "n_val_evaluations": len(curve), "diverged": bool(div), "val_loss_curve": [[e, v] for e, v in curve]}


def train_candidate(data_dir: str | Path, version: str, opts: TrainOptions, *, dry_run: bool = False, say: Say = print,
                    fit_fn: Optional[Callable] = None) -> dict:
    """Returns the registry entry (or, for ``dry_run``, the validated plan summary). ``fit_fn(model, train, val, save_dir,
    train_config, device, logger)`` replaces ``model.fit`` in tests."""
    data_dir = Path(data_dir)
    plan = plan_training(data_dir, version, opts)
    for c in plan.checks:
        say(f"  ok: {c}")
    for w in plan.warnings:
        say(f"  WARNING: {w}")
    cfg = {"crop_size": opts.crop_size, "batch_size": opts.batch_size, "lr": float(opts.lr), "num_epochs": opts.epochs,
           "num_train_samples": opts.num_train_samples, "pos_weight": float(opts.pos_weight), "smart_crop": opts.smart_crop,
           "early_stopping_patience": opts.early_stopping_patience, "optimize_threshold": opts.optimize_threshold,
           "finetuned_from": plan.base["name"], "downscale": plan.config["downscale"],
           "normaliser": plan.config["normaliser"]}
    from .spotiflow_io import hardware_info, load_model, pick_device
    device = pick_device(opts.device)
    if dry_run:
        say(f"[dry-run] would fine-tune '{plan.base['name']}' for {opts.epochs} epoch(s) on {device} with {cfg}; "
            "nothing was trained and nothing was written")
        return {"dry_run": True, "snapshot": version, "base": plan.base, "device": device, "train_config": cfg,
                "inference_config": plan.config,
                "n_train": len(plan.train), "n_val": len(plan.val), "checks": plan.checks, "warnings": plan.warnings}

    stage = models_dir(data_dir) / f".staging-{uuid.uuid4().hex[:10]}"
    stage.mkdir(parents=True, exist_ok=True)
    try:
        _seed_all(opts.seed)
        say(f"loading images ({len(plan.train)} train, {len(plan.val)} val) ...")
        tri, trs, d1 = _load_xy(data_dir, version, plan.train, plan.config)
        vai, vas, d2 = _load_xy(data_dir, version, plan.val, plan.config)
        if d1 + d2:
            plan.warnings.append(f"{d1 + d2} label(s) fell outside the downscaled image (remainder rows/columns trimmed) and were dropped")
            say(f"  WARNING: {plan.warnings[-1]}")
        say(f"  training input: downscale {plan.config['downscale']}, normaliser {plan.config['normaliser']} (page-level lo/hi)")
        say(f"loading base model '{plan.base['name']}' (trainable) on {device} ...")
        model = load_model(opts.base, data_dir, inference_mode=False, device=device)
        save_dir = stage / "weights"
        logger = None
        if fit_fn is None:
            from pytorch_lightning.loggers import CSVLogger
            logger = CSVLogger(save_dir=str(stage / "logs"), name="fit")
        t0 = time.time()
        say(f"training {opts.epochs} epoch(s) (seed {opts.seed}) ...")
        if fit_fn is not None:
            fit_fn(model, (tri, trs), (vai, vas), save_dir, cfg, device, logger)
        else:
            fit_cfg = {k: v for k, v in cfg.items() if k not in ("downscale", "normaliser")}    # Spotiflow's own config keys only
            model.fit(tri, trs, vai, vas, augment_train=opts.augment, save_dir=save_dir, train_config=fit_cfg, device=device,
                      logger=logger, num_workers=opts.num_workers, deterministic=True,
                      normalizer=None)      # inputs are ALREADY normalised with the page lo/hi: no per-image re-normalisation
        dur = time.time() - t0
        logs = _val_loss_from_logs(stage / "logs", opts.divergence_factor)
        if (save_dir / "best.pt").is_file():
            wfile = "best.pt"
        elif (save_dir / "last.pt").is_file() and not logs["diverged"]:
            wfile = "last.pt"
            plan.warnings.append("no best.pt was written: the LAST epoch's weights are registered (validation loss did not diverge)")
        elif (save_dir / "last.pt").is_file():
            raise MLError("training produced no best.pt and the validation loss diverged "
                          f"(best {logs['best_val_loss']} at epoch {logs['best_epoch']}, last {logs['last_val_loss']}); "
                          "refusing to register the last epoch's weights")
        else:
            raise MLError(f"training produced no weights in {save_dir}")
        if logs["diverged"]:
            plan.warnings.append(f"validation loss DIVERGED: best {logs['best_val_loss']:.4g} (epoch {logs['best_epoch']}), last "
                                 f"{logs['last_val_loss']:.4g} (epoch {logs['last_epoch']}); the registered weights are the best "
                                 "epoch's. Treat this candidate with suspicion.")
            say(f"  WARNING: {plan.warnings[-1]}")
        wsha = sha256_file(save_dir / wfile)
        mid = f"ft-{version.split('-')[-1][:8]}-{wsha[:8]}"
        files = {p.relative_to(save_dir).as_posix(): sha256_file(p) for p in sorted(save_dir.rglob("*")) if p.is_file()}
        final = models_dir(data_dir) / mid
        if final.exists():
            raise MLError(f"model artifact {mid} already exists; identical weights were registered before")
        shutil.move(str(save_dir), str(final))
        pages = {}
        for src in plan.manifest.sources:
            if src.split in ("train", "val") and src.page_normalization:
                pages[src.analysis_id] = src.page_normalization
        entry = {
            "id": mid, "kind": "finetuned", "base": plan.base, "dataset_version": version, "status": "candidate",
            "split_membership": {"train": [c.crop_id for c in plan.train], "val": [c.crop_id for c in plan.val],
                                 "train_groups": sorted({c.group_id for c in plan.train}),
                                 "val_groups": sorted({c.group_id for c in plan.val}),
                                 "test_excluded": len([c for c in plan.manifest.crops if c.split == "test"]),
                                 "lock_sha256": plan.lock_sha256},
            "inference_config": plan.config, "lo_hi_policy": LO_HI_POLICY, "page_normalization": pages,
            "hyperparameters": dict(cfg, augment=opts.augment, device=device, num_workers=opts.num_workers,
                                    allow_small=opts.allow_small),
            "seed": opts.seed, "duration_s": round(dur, 2), "hardware": hardware_info(device),
            "metrics": {"training": dict(logs, weights_epoch=("best" if wfile == "best.pt" else "last")), "n_train_positives": sum(c.n_positive for c in plan.train),
                        "n_val_positives": sum(c.n_positive for c in plan.val)},
            "weights_file": wfile, "weights_sha256": wsha, "artifact": mid, "artifact_files": files,
            "artifact_sha256": hash_obj(files), "created_at": utc_now(), "notes": opts.notes,
            "warnings": plan.warnings}
        Registry(data_dir).add(entry)
        say(f"registered candidate {mid} (weights sha256 {wsha[:16]}..., {dur:.1f}s). It is NOT active: evaluate it on the "
            f"locked test set and promote it explicitly.")
        return entry
    finally:
        shutil.rmtree(stage, ignore_errors=True)
