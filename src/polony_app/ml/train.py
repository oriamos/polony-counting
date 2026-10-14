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
import hashlib
import os
import random
import shutil
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tifffile

from .common import (DEFAULT_NORMALISER, LO_HI_POLICY, MLError, check_downscale, check_normaliser, grid_offset, hash_obj,
                     labels_in_frame, models_dir, prepare_crop, sha256_file, to_downscaled, utc_now)
from .dataset import list_snapshots, load_crop, load_manifest, read_labels_csv, snapshot_path, verify_snapshot
from .manifest import CropRecord, SnapshotManifest
from .registry import Registry
from .splits import load_lock, lock_fingerprint, lock_identity_conflicts

Say = Callable[[str], None]


@dataclass
class TrainOptions:
    base: str = "general"
    epochs: int = 30
    batch_size: int = 4
    lr: float = 1e-4
    crop_size: int = 256
    seed: int = 0
    device: str = "cpu"                 # auto | cpu | cuda | rocm (= alias of cuda) | mps ; see spotiflow_io.pick_device
    num_train_samples: Optional[int] = None
    pos_weight: float = 10.0
    smart_crop: bool = True
    augment: bool = True
    optimize_threshold: bool = True
    early_stopping_patience: int = 0
    num_workers: int = 0                # DataLoader workers of Spotiflow.fit (train AND validation loaders)
    load_mode: str = "auto"             # auto | eager | lazy : how the prepared (normalised, downscaled) crops are held
    allow_large_memory: bool = False    # eager load although the prepared data exceeds the available RAM
    lazy_cache_mb: int = 256            # lazy mode: LRU cache of prepared crops, per dataset (train / val) and per process
    max_val_crops: Optional[int] = None  # deterministic validation subsample (stable hash order); None = all val crops
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
    val_total: int = 0             # number of valid val crops in the snapshot (``val`` may be a --max-val-crops subset)
    val_excluded: list = None      # crop ids left out of ``val`` by --max-val-crops


def _locked_sources(data_dir, lock: dict) -> tuple[set[str], set[str]]:
    """(file sha256 set, group set) of the locked test images."""
    lm = load_manifest(data_dir, lock["snapshot"])
    shas = {c.source_sha256 for c in lm.crops if c.split == "test"}
    for s in lm.sources:
        if s.split == "test":
            shas |= set(s.tiff_sha256.values())
    return shas, set(lock.get("groups", []))


LOAD_MODES = ("auto", "eager", "lazy")


def _check_examination_provenance(m: SnapshotManifest) -> str:
    """Bridge imports (``options.legacy_import`` or a non-empty ``import_summary`` or any crop with provenance) must prove that every crop was examined by the researcher.
    ``legacy_bridge.verify_examination_provenance`` re-checks the recorded provenance of every train / val / test crop; on top,
    ANY crop that carries provenance (or any crop of an import) must say ``researcher_examined == true``."""
    from . import legacy_bridge as lb
    is_import = lb.is_legacy_import(m)
    problems: list[str] = []
    if is_import:
        problems += lb.verify_examination_provenance(m)
    for c in m.crops:
        prov = c.provenance or {}
        if (is_import or prov) and prov.get("researcher_examined") is not True:
            msg = f"crop {c.crop_id} ({c.split}): provenance.researcher_examined is not true"
            if msg not in problems:
                problems.append(msg)
    if problems:
        raise MLError(f"REFUSED: {len(problems)} crop(s) of snapshot {m.version} are not proven researcher-examined; training on "
                      f"them would teach the model from unexamined cells. First problems: {'; '.join(problems[:5])}")
    return ("every crop of this bridge import carries researcher_examined provenance" if is_import
            else "no bridge-import provenance to verify (snapshot built from reviewed analyses)")


VALID_SPLITS = ("train", "val", "test")


def _check_manifest_sanity(m: SnapshotManifest) -> None:
    """Structural gate on a (possibly hand-edited and re-sealed) manifest: split names, unique crop ids, finite page bounds,
    consistent geometry. ``verify`` only proves the manifest matches its own hash, not that its content makes sense."""
    import math
    problems: list[str] = []
    seen: dict[str, str] = {}
    for c in m.crops:
        if c.split not in VALID_SPLITS:
            problems.append(f"crop {c.crop_id}: unknown split {c.split!r} (must be one of {', '.join(VALID_SPLITS)})")
        if c.crop_id in seen:
            problems.append(f"crop id {c.crop_id!r} appears more than once (splits {seen[c.crop_id]} and {c.split}): each crop "
                            "must be unique, or it would be trained on twice / sit in train and validation")
        seen.setdefault(c.crop_id, c.split)
        if c.n_positive < 0:
            problems.append(f"crop {c.crop_id}: negative n_positive {c.n_positive}")
        sh, bx = c.shape_hw, c.box_xyxy
        ok_shape = (len(sh) == 2 and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in sh))
        if not ok_shape:
            problems.append(f"crop {c.crop_id}: shape_hw {sh} is not two positive integers (empty crop)")
        elif len(bx) != 4 or (bx[3] - bx[1], bx[2] - bx[0]) != (sh[0], sh[1]):
            problems.append(f"crop {c.crop_id}: shape_hw {sh} is inconsistent with box_xyxy {bx} (expected {[bx[3] - bx[1], bx[2] - bx[0]] if len(bx) == 4 else '4 values'})")
        if c.split in ("train", "val") and c.normaliser != "none":
            lo, hi = c.norm_lo, c.norm_hi
            if lo is not None and hi is not None and not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                problems.append(f"crop {c.crop_id}: page lo/hi ({lo}, {hi}) are not finite with lo < hi")
    if problems:
        raise MLError(f"REFUSED: the manifest of snapshot {m.version} is inconsistent ({len(problems)} problem(s)); nothing was "
                      f"trained. First problems: {'; '.join(problems[:5])}")


def subsample_val(val: list, n: int) -> tuple[list, list]:
    """Deterministic validation subsample: the first ``n`` crops in sha256(crop_id) order (independent of seed, platform and
    manifest order). Returns (kept in that order, excluded crop ids)."""
    ranked = sorted(val, key=lambda c: hashlib.sha256(c.crop_id.encode("utf-8")).hexdigest())
    return ranked[:n], sorted(c.crop_id for c in ranked[n:])


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
    if opts.num_workers < 0 or (opts.max_val_crops is not None and opts.max_val_crops < 1) or \
            (opts.num_train_samples is not None and opts.num_train_samples < 1):
        raise MLError("invalid options: need num_workers >= 0, max_val_crops >= 1, num_train_samples >= 1")
    if opts.load_mode not in LOAD_MODES:
        raise MLError(f"unknown --load-mode '{opts.load_mode}' (choose one of {', '.join(LOAD_MODES)})")
    checks.append(_check_examination_provenance(m))
    _check_manifest_sanity(m)
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
    val_total, val_excluded = len(val), []
    if opts.max_val_crops is not None and opts.max_val_crops < len(val):
        groups_all = len({c.group_id for c in val})
        val, val_excluded = subsample_val(val, opts.max_val_crops)
        warns.append(f"validation uses only {len(val)} of {val_total} val crops (--max-val-crops, stable sha256 order of the crop "
                     "id; the excluded ids are recorded). Best-checkpoint selection, early stopping and threshold optimisation "
                     f"see only this subset ({len({c.group_id for c in val})} of {groups_all} val group(s)).")
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
        ident_bad = lock_identity_conflicts(m, lock)
        if ident_bad:
            raise MLError("REFUSED: training/validation data shares a specimen or slide identity with the locked test set "
                          f"(leakage; nothing was trained): {'; '.join(ident_bad[:3])}")
        checks.append(f"no train/val crop belongs to a locked test group, file or identity (lock {lock_sha[:12]})")
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
    return TrainPlan(version, m, train, val, base, checks, warns + list(m.warnings), lock_sha, config, val_total, val_excluded)


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
    """EAGER loading. Network inputs: native crop -> page-level normalisation -> block-mean downscale; labels mapped with the
    exact pixel-centre rule x_s = (x - (d-1)/2) / d. Returns (images, spots, n_labels_dropped_outside_the_downscaled_frame)."""
    imgs, spots, dropped = [], [], 0
    for c in crops:
        im, lab = load_crop(data_dir, version, c)
        x, lab_s, n, _ = prepare_crop(im, lab, lo=c.norm_lo, hi=c.norm_hi, downscale=config["downscale"],
                                      normaliser=config["normaliser"], origin_xy=c.box_xyxy[:2])
        imgs.append(x)
        spots.append(lab_s)
        dropped += n
    return imgs, spots, dropped


def prepared_hw(crop: CropRecord, downscale: int) -> tuple[int, int]:
    """Shape (h, w) of the network input of ``crop`` after the grid trim and the block mean, from the manifest alone."""
    oy, ox = grid_offset(crop.box_xyxy[:2], downscale)
    h, w = crop.shape_hw
    return (max(0, h - oy) // downscale, max(0, w - ox) // downscale)


class LazyCropImages(Sequence):
    """Prepared network inputs read from disk on demand (``--load-mode lazy``).

    ``seq[i]`` = exactly what :func:`_load_xy` puts at position i (it calls the same :func:`prepare_crop`), kept in a small LRU
    cache bounded in bytes (read-only arrays: a consumer that tried to modify one in place would raise, not corrupt the cache).
    ITERATION yields zero-memory stand-ins of the right shape (``np.broadcast_to``): Spotiflow's start-up passes only iterate to
    look at ``.shape`` / ``.ndim`` (input validation, minimum crop size, 'smaller than the crop size' scan), so they cost no I/O;
    pixels only ever come from indexing, which is what the DataLoader does. ``seq[i] = padded`` (Spotiflow pads images
    smaller than the crop size by assignment) is kept as an override. Picklable (the cache is dropped), so DataLoader workers
    work with any start method; every process has its own cache."""

    def __init__(self, root: str, items: list, shapes: list, downscale: int, normaliser: str, labels: list,
                 cache_bytes: int = 256 * 1024 ** 2):
        self._root, self._items, self._shapes = str(root), list(items), [tuple(s) for s in shapes]
        self._d, self._norm, self._labels = int(downscale), normaliser, labels   # labels: native (y, x) arrays
        self._cache_bytes = int(cache_bytes)
        self._cache: OrderedDict = OrderedDict()
        self._cache_used = 0
        self._override: dict = {}
        self.loads = 0          # disk reads in THIS process
        self.hits = 0

    def __len__(self) -> int:
        return len(self._items)

    def __getstate__(self):
        st = dict(self.__dict__)
        st["_cache"], st["_cache_used"], st["loads"], st["hits"] = OrderedDict(), 0, 0, 0
        return st

    def _load(self, i: int) -> np.ndarray:
        rel, lo, hi, origin = self._items[i]
        img = tifffile.imread(os.path.join(self._root, rel))
        x, _, _, _ = prepare_crop(img, self._labels[i], lo=lo, hi=hi, downscale=self._d, normaliser=self._norm, origin_xy=origin)
        if tuple(x.shape) != self._shapes[i]:
            raise MLError(f"{rel}: the prepared crop is {tuple(x.shape)} but the manifest says {self._shapes[i]} "
                          "(the snapshot does not match its manifest)")
        x.flags.writeable = False
        return x

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        i = int(i)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        if i in self._override:
            return self._override[i]
        if i in self._cache:
            self._cache.move_to_end(i)
            self.hits += 1
            return self._cache[i]
        x = self._load(i)
        self.loads += 1
        if x.nbytes <= self._cache_bytes:
            self._cache[i] = x
            self._cache_used += x.nbytes
            while self._cache_used > self._cache_bytes:
                _, old = self._cache.popitem(last=False)
                self._cache_used -= old.nbytes
        return x

    def __setitem__(self, i, value):
        self._override[int(i) % len(self)] = value

    def __iter__(self):
        for i in range(len(self)):
            if i in self._override:
                yield self._override[i]
            else:
                yield np.broadcast_to(np.float32(0), self._shapes[i])


def load_lazy(data_dir, version, crops, config, cache_mb: int = 256):
    """(LazyCropImages, spots, n_labels_dropped). Labels are small, so they are read and mapped up front (this is what gives the
    dropped-label count, with no pixels read); the labels equal those of :func:`_load_xy` bit for bit (tested)."""
    d, root = config["downscale"], snapshot_path(data_dir, version)
    items, shapes, native_labels, spots, dropped = [], [], [], [], 0
    for c in crops:
        lab = read_labels_csv(root / c.path_csv)
        oy, ox = grid_offset(c.box_xyxy[:2], d)
        shape = prepared_hw(c, d)
        lab_s, n = labels_in_frame(to_downscaled(np.asarray(lab, float).reshape(-1, 2) - [oy, ox], d), shape)
        items.append((c.path_tif, c.norm_lo, c.norm_hi, tuple(c.box_xyxy[:2])))
        shapes.append(shape)
        native_labels.append(lab)
        spots.append(lab_s.astype("float32"))
        dropped += n
    return LazyCropImages(root, items, shapes, d, config["normaliser"], native_labels, cache_mb * 1024 ** 2), spots, dropped


# ------------------------------------------------------------------------------------------------ scale of a run
def scale_summary(plan: TrainPlan, opts: TrainOptions, data_dir) -> dict:
    """Actual numbers of the run (from the snapshot manifest) for ``--dry-run`` and for the memory guard."""
    from .estimates import available_memory_bytes, choose_load_mode, estimate_training
    m, d = plan.manifest, plan.config["downscale"]

    def trimmed(c):
        oy, ox = grid_offset(c.box_xyxy[:2], d)
        return (c.shape_hw[0] - oy, c.shape_hw[1] - ox)

    est = estimate_training(len(plan.train), len(plan.val), opts.batch_size, opts.epochs, opts.num_train_samples, d,
                            train_shapes_hw=[trimmed(c) for c in plan.train], val_shapes_hw=[trimmed(c) for c in plan.val],
                            optimize_threshold=opts.optimize_threshold)
    avail = available_memory_bytes()
    mode, why = choose_load_mode(est["prepared_bytes"], avail, opts.load_mode, opts.allow_large_memory)
    native = 0
    root = snapshot_path(data_dir, plan.version)
    for dirpath, _, names in os.walk(root):
        for n in names:
            try:
                native += os.path.getsize(os.path.join(dirpath, n))
            except OSError:
                pass
    by_split = {k: sum(1 for c in m.crops if c.split == k) for k in ("train", "val", "test")}
    return {"estimate": est, "available_ram_bytes": avail, "load_mode": mode, "load_mode_reason": why,
            "native_snapshot_bytes": native, "sources": len(m.sources), "sources_with_crops": sum(1 for s in m.sources if s.n_crops),
            "eligible_crops": len(m.crops), "crops_by_split": by_split, "val_used": len(plan.val), "val_total": plan.val_total,
            "eager_exceeds_ram": bool(avail and est["prepared_bytes"] > avail)}


def format_scale(sc: dict, plan: TrainPlan, opts: TrainOptions, device: str, version: str) -> list[str]:
    from .estimates import format_bytes
    e, bs = sc["estimate"], sc["crops_by_split"]
    L = [f"SCALE OF THIS RUN (actual numbers from snapshot {version}):",
         f"  sources (source images): {sc['sources']} in the manifest, {sc['sources_with_crops']} with crops",
         f"  eligible crops: {sc['eligible_crops']}  (train {bs['train']}, val {bs['val']}, test {bs['test']} - locked/held out, never trained on)",
         f"  used here: {e['n_train_crops']} train crop(s), {e['n_val_crops']} val crop(s)"
         + (f"  (--max-val-crops {opts.max_val_crops}: {sc['val_used']} of {sc['val_total']})" if sc['val_used'] != sc['val_total'] else ""),
         f"  epochs: {e['epochs']}   batch size: {e['batch_size']}   downscale: {e['downscale']}   device: {device}   workers: {opts.num_workers}",
         f"  num_train_samples: {e['num_train_samples'] if e['num_train_samples'] else 'not set'}  -> {e['samples_per_epoch']} sample(s) drawn per epoch"
         + ("" if e['num_train_samples'] else " (one pass over the train crops)"),
         f"  steps per epoch: {e['steps_per_epoch']}   estimated optimizer steps: {e['optimizer_steps']}",
         f"  validation: {e['validation_batches']} batch(es) per pass (Spotiflow validates with batch size 1) x {e['validation_passes']} pass(es) "
         f"= {e['validation_batches_total']} batches, + {e['sanity_check_batches']} sanity-check batch(es) before epoch 1"
         + (f", + {e['final_threshold_batches']} for the final threshold optimisation" if e['final_threshold_batches'] else ""),
         f"  prepared in-memory size (float32 network inputs): {format_bytes(e['prepared_bytes'])} "
         f"(train {format_bytes(e['prepared_train_bytes'])}, val {format_bytes(e['prepared_val_bytes'])}); available RAM: {format_bytes(sc['available_ram_bytes'])}",
         f"  load mode: {sc['load_mode']} ({sc['load_mode_reason']})",
         f"  native snapshot size on disk: {format_bytes(sc['native_snapshot_bytes'])}"]
    if sc["load_mode"] == "eager" and sc["eager_exceeds_ram"]:
        L.append("  WARNING: the prepared data would not fit in the available RAM in eager mode; use --load-mode lazy "
                 "(or --allow-large-memory to try anyway)")
    return L


def format_recommendations(version: str, opts: TrainOptions, sc: dict) -> list[str]:
    dev = f" --device {opts.device}" if opts.device != "cpu" else ""
    base = f"python -m polony_app.ml train --snapshot {version} --base {opts.base}{dev} --batch-size {opts.batch_size}"
    if opts.num_workers:
        base += f" --num-workers {opts.num_workers}"
    e = sc["estimate"]
    return ["RECOMMENDATIONS",
            f"  1. Benchmark first (does not need to finish a long run): {base} --epochs 1   -> note the seconds per epoch, then "
            f"estimate the total as seconds/epoch x epochs ({e['steps_per_epoch']} steps per epoch here).",
            f"  2. Then train about 10 epochs with early stopping: {base} --epochs 10 --early-stopping-patience 3   "
            "(the BEST validation checkpoint is the one registered, not the last).",
            "  3. --num-train-samples N bounds an epoch: each epoch draws N crops (with replacement) instead of one pass over all "
            f"{e['n_train_crops']} train crops, so an epoch costs ceil(N/batch_size) steps whatever the snapshot size; validation still "
            "runs over all validation crops once per epoch" +
            (" (--max-val-crops N limits that, recorded)." if opts.max_val_crops is None else "."),
            "  4. Nothing is promoted automatically: after training run eval-model for the candidate and for 'general', "
            "compare, and promote only if the locked-test evidence supports it."]


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
    device = pick_device(opts.device, say=say)          # MLError for an explicitly requested GPU that is not available
    say(f"  device: requested '{opts.device}' -> resolved '{device}'"
        + (" ('rocm' is an alias of 'cuda': PyTorch ROCm builds use the cuda API)" if opts.device == "rocm" else ""))
    sc = scale_summary(plan, opts, data_dir)
    cfg.update(num_workers=opts.num_workers, max_val_crops=opts.max_val_crops)
    if dry_run:
        say(f"[dry-run] would fine-tune '{plan.base['name']}' for {opts.epochs} epoch(s) on {device} with {cfg}; "
            "nothing was trained and nothing was written")
        for line in format_scale(sc, plan, opts, device, version):
            say(line)
        say("Training will not start because --dry-run was supplied.")
        for line in format_recommendations(version, opts, sc):
            say(line)
        return {"dry_run": True, "snapshot": version, "base": plan.base, "device": device, "train_config": cfg,
                "inference_config": plan.config, "scale": sc, "recommendations": format_recommendations(version, opts, sc),
                "n_train": len(plan.train), "n_val": len(plan.val), "checks": plan.checks, "warnings": plan.warnings}
    load_mode = sc["load_mode"]
    if load_mode == "eager" and sc["eager_exceeds_ram"] and not opts.allow_large_memory:
        from .estimates import format_bytes
        raise MLError(f"the prepared training data ({format_bytes(sc['estimate']['prepared_bytes'])}) exceeds the available RAM "
                      f"({format_bytes(sc['available_ram_bytes'])}) in eager mode. Use --load-mode lazy (crops are read from the "
                      "snapshot on demand with a small cache) or pass --allow-large-memory to try anyway. Nothing was trained.")
    say(f"  load mode: {load_mode} ({sc['load_mode_reason']}); prepared size {sc['estimate']['prepared_bytes'] / 1024 ** 2:.0f} MiB")

    stage = models_dir(data_dir) / f".staging-{uuid.uuid4().hex[:10]}"
    stage.mkdir(parents=True, exist_ok=True)
    try:
        _seed_all(opts.seed)
        say(f"loading images ({len(plan.train)} train, {len(plan.val)} val; {load_mode}) ...")
        if load_mode == "lazy":
            tri, trs, d1 = load_lazy(data_dir, version, plan.train, plan.config, opts.lazy_cache_mb)
            vai, vas, d2 = load_lazy(data_dir, version, plan.val, plan.config, opts.lazy_cache_mb)
        else:
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
        if opts.num_workers:
            _prepare_worker_sharing(say)
        t0 = time.time()
        say(f"training {opts.epochs} epoch(s) (seed {opts.seed}) ...")
        if fit_fn is not None:
            fit_fn(model, (tri, trs), (vai, vas), save_dir, cfg, device, logger)
        else:
            fit_cfg = {k: v for k, v in cfg.items() if k not in ("downscale", "normaliser", "num_workers", "max_val_crops")}
            extra = {"defer_normalization": True} if load_mode == "lazy" else {}   # else SpotsDataset would materialise every crop
            model.fit(tri, trs, vai, vas, augment_train=opts.augment, save_dir=save_dir, train_config=fit_cfg, device=device,
                      logger=logger, num_workers=opts.num_workers, deterministic=True,
                      normalizer=None,      # inputs are ALREADY normalised with the page lo/hi: no per-image re-normalisation
                      **extra)
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
                                 "val_excluded_by_max_val_crops": list(plan.val_excluded or []),
                                 "test_excluded": len([c for c in plan.manifest.crops if c.split == "test"]),
                                 "lock_sha256": plan.lock_sha256},
            "inference_config": plan.config, "lo_hi_policy": LO_HI_POLICY, "page_normalization": pages,
            "hyperparameters": dict(cfg, augment=opts.augment, device=device, device_requested=opts.device,
                                    num_workers=opts.num_workers, allow_small=opts.allow_small, load_mode=load_mode,
                                    load_mode_requested=opts.load_mode, lazy_cache_mb=opts.lazy_cache_mb,
                                    max_val_crops=opts.max_val_crops, n_val_total=plan.val_total,
                                    allow_large_memory=opts.allow_large_memory),
            "seed": opts.seed, "duration_s": round(dur, 2), "hardware": dict(hardware_info(device), requested_device=opts.device, resolved_device=device),
            "metrics": {"training": dict(logs, weights_epoch=("best" if wfile == "best.pt" else "last")), "n_train_positives": sum(c.n_positive for c in plan.train),
                        "n_val_positives": sum(c.n_positive for c in plan.val)},
            "weights_file": wfile, "weights_sha256": wsha, "artifact": mid, "artifact_files": files,
            "artifact_sha256": hash_obj(files), "created_at": utc_now(), "notes": opts.notes,
            "warnings": plan.warnings}
        Registry(data_dir).add(entry)
        say(f"registered candidate {mid} (weights sha256 {wsha[:16]}..., {dur:.1f}s; "
            f"registered weights: {'BEST validation checkpoint' if wfile == 'best.pt' else 'LAST epoch'}). It is NOT active: "
            "evaluate it on the locked test set and promote it explicitly.")
        dev = f" --device {opts.device}" if opts.device != "cpu" else ""
        say("next (nothing is promoted automatically):")
        say(f"  python -m polony_app.ml eval-model --model {mid}{dev}")
        say(f"  python -m polony_app.ml eval-model --model general{dev}      # the baseline, same locked test crops")
        say(f"  python -m polony_app.ml compare --models {mid} general")
        say(f"  python -m polony_app.ml promote {mid} --reason \"...\"        # only if the comparison supports it")
        return entry
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _prepare_worker_sharing(say) -> None:
    """DataLoader workers hand every batch over as shared-memory file descriptors; with many small tensors the default soft
    limit (often 1024) is exhausted at the epoch end ("Too many open files ... Communication with the workers is no longer
    possible"). Raise the soft limit to the hard limit (max 65536) and pass tensors by file name instead of by descriptor."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = 65536 if hard == resource.RLIM_INFINITY else min(hard, 65536)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            say(f"  raised the open-files limit {soft} -> {want} for the DataLoader workers")
    except (ImportError, ValueError, OSError):
        pass
    try:
        import torch.multiprocessing as tmp
        tmp.set_sharing_strategy("file_system")
    except Exception:   # noqa: BLE001 - best effort; training still works with the default strategy
        pass
