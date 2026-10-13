"""``python -m polony_app.ml <command>`` - explicit, step-by-step; nothing trains or promotes by itself.

Commands: build, import-legacy, split, verify, train, gpu-smoke, eval-model, evaluate, compare, promote, rollback, list, show.
(``dataset build`` is accepted as an alias of ``build``.)"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from .common import MLError, datasets_dir, pretty_json, utc_now


def _default_data_dir() -> str:
    env = os.environ.get("POLONY_DATA_DIR")
    if env:
        return env
    try:
        from polony_app.launcher import data_dir
        return str(data_dir())
    except Exception:
        return "./data"


def _store(data_dir: str):
    from polony_app.config import load_settings
    from polony_app.store import open_store
    return open_store(data_dir, load_settings(persist_machine_id=False))


def _fractions(s: Optional[str]) -> Optional[dict]:
    if not s:
        return None
    try:
        a, b, c = (float(x) for x in s.split(","))
    except ValueError:
        raise MLError("--fractions must be 'train,val,test', e.g. 0.7,0.15,0.15")
    return {"train": a, "val": b, "test": c}


def _p(*a):
    print(*a, flush=True)


# Command groups that live in their own modules (each exposes ``COMMANDS``, ``register(subparsers)`` and
# ``run(args, data_dir) -> int``): ``cli_legacy`` = import-legacy, ``cli_devices`` = gpu-smoke.
_EXTENSIONS = ("cli_legacy", "cli_devices")


def _extension_modules() -> list:
    import importlib
    mods = []
    for name in _EXTENSIONS:
        try:
            mods.append(importlib.import_module(f"{__package__}.{name}"))
        except ModuleNotFoundError as exc:      # the module does not exist (yet): its commands are simply absent
            if exc.name != f"{__package__}.{name}":
                raise
    return mods


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m polony_app.ml", description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=None, help="app data directory (default: POLONY_DATA_DIR or the app default)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build an immutable dataset snapshot from reviewed analyses")
    b.add_argument("--analyses", nargs="+", required=True)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--fractions", help="train,val,test (default 0.7,0.15,0.15)")
    b.add_argument("--level", default="auto", choices=["auto", "specimen", "slide", "scan_session", "source_image"])
    b.add_argument("--allow-unconfirmed-identity", action="store_true")
    b.add_argument("--identity-map", help="JSON {analysis_id: {specimen_id, slide_id, assay}} - confirmed identities")
    b.add_argument("--allow-bbox", action="store_true", help="cut the bounding box of non-rectangular regions (warned)")
    b.add_argument("--strict-reviewed", action="store_true",
                   help="only objects the reviewer touched are positives (an approved region alone is not enough)")
    b.add_argument("--require-finalized", action="store_true")
    b.add_argument("--roles", default="red,green", help="channel roles to cut (only from a confirmed mapping)")
    b.add_argument("--min-crop-px", type=int, default=32)
    b.add_argument("--no-raw", action="store_true", help="do not list raw engine proposals in points.csv")
    b.add_argument("--downscale", type=int, default=1, choices=[1, 2, 4, 8],
                   help="the model input is downscaled by this factor (block mean) at training / evaluation time; crops are still "
                        "exported at native resolution. Must equal the downscale the app will run the model with.")
    b.add_argument("--normaliser", default="valid_unsaturated", choices=["valid_unsaturated", "none"],
                   help="valid_unsaturated: page-level percentiles 1/99.8 of the valid, unsaturated pixels (as at inference)")
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--created-at", help=argparse.SUPPRESS)

    s = sub.add_parser("split", help="show the group split of a snapshot; --lock-test freezes its test groups")
    s.add_argument("--snapshot", required=True)
    s.add_argument("--lock-test", action="store_true")
    s.add_argument("--replace-lock", action="store_true")
    s.add_argument("--reason", default="")
    s.add_argument("--allow-unconfirmed-identity", action="store_true")

    v = sub.add_parser("verify", help="re-hash a snapshot")
    v.add_argument("snapshot")

    t = sub.add_parser("train", help="fine-tune Spotiflow on a snapshot -> registry candidate")
    t.add_argument("--snapshot", required=True)
    t.add_argument("--base", default="general", help="official pretrained name, registry id, or model folder")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--crop-size", type=int, default=256)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"])
    t.add_argument("--num-train-samples", type=int)
    t.add_argument("--pos-weight", type=float, default=10.0)
    t.add_argument("--no-augment", action="store_true")
    t.add_argument("--no-optimize-threshold", action="store_true")
    t.add_argument("--early-stopping-patience", type=int, default=0)
    t.add_argument("--notes", default="")
    t.add_argument("--allow-small", action="store_true",
                   help="train even with fewer than 4 independent training groups / fewer than --min-train-crops crops")
    t.add_argument("--min-train-groups", type=int, default=4)
    t.add_argument("--min-train-crops", type=int, default=20)
    t.add_argument("--dry-run", action="store_true", help="validate everything, train nothing, write nothing")

    e = sub.add_parser("eval-model", help="score a model on the LOCKED test crops and record it in the registry")
    e.add_argument("--model", required=True, help="registry id, official pretrained name, or model folder")
    e.add_argument("--snapshot", help="default: the snapshot the test lock refers to")
    e.add_argument("--tolerance", type=float, default=10.0, help="headline matching tolerance in px (5, 10 and 15 are always reported)")
    e.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"])
    e.add_argument("--prob-thresh", type=float)
    e.add_argument("--downscale", type=int, choices=[1, 2, 4, 8],
                   help="official pretrained models only (default 1): a fine-tuned model is always run at the downscale it was "
                        "trained at; a different value is refused")
    e.add_argument("--normaliser", choices=["valid_unsaturated", "none"],
                   help="official pretrained models only (default valid_unsaturated); fine-tuned: the one it was trained with")

    ev = sub.add_parser("evaluate", help="compare detection configurations against a reference")
    ev.add_argument("--predictions", nargs="+", required=True, help="analysis ids (optionally id:runid[,runid])")
    ev.add_argument("--reference", choices=["analysis", "roi"], default="analysis")
    ev.add_argument("--tolerance", type=float, default=10.0, help="headline matching tolerance in px (5, 10 and 15 are always reported)")
    ev.add_argument("--numbering", choices=["index", "ordinal"], default="index")
    ev.add_argument("--configs", nargs="+")
    ev.add_argument("--out", help="Markdown report path (a .json with the same stem is written next to it)")

    c = sub.add_parser("compare", help="side-by-side locked-test results of models")
    c.add_argument("--models", nargs="+", required=True)

    pr = sub.add_parser("promote", help="make a model the active one (explicit)")
    pr.add_argument("model")
    pr.add_argument("--force", action="store_true")
    pr.add_argument("--reason", default="")
    pr.add_argument("--app-downscale", type=int, choices=[1, 2, 4, 8],
                    help="the app's current inference downscale (default: the app's default Spotiflow setting)")
    pr.add_argument("--app-normaliser", choices=["auto", "none", "valid_unsaturated"],
                    help="the app's current Spotiflow normaliser (default: the app's default setting)")
    pr.add_argument("--acknowledge-config", action="store_true",
                    help="accept that the candidate was trained/evaluated at a different downscale/normaliser than the app currently uses")

    sub.add_parser("rollback", help="restore the previously active model")

    ls = sub.add_parser("list", help="list datasets or models")
    ls.add_argument("what", nargs="?", default="models", choices=["models", "datasets"])

    sh = sub.add_parser("show", help="print the manifest of a snapshot")
    sh.add_argument("snapshot")
    for mod in _extension_modules():
        mod.register(sub)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"build", "split", "verify", "train", "eval-model", "evaluate", "compare", "promote", "rollback", "list", "show"}
    for mod in _extension_modules():
        cmds |= set(mod.COMMANDS)
    for i, tok in enumerate(argv[:-1]):
        if tok == "dataset" and argv[i + 1] in cmds:    # `dataset build ...` == `build ...`
            argv = argv[:i] + argv[i + 1:]
            break
    ap = build_parser()
    a = ap.parse_args(argv)
    data_dir = a.data_dir or _default_data_dir()
    try:
        return _run(a, data_dir)
    except MLError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr, flush=True)
        return 2


def _run(a, data_dir: str) -> int:
    from . import dataset as ds
    from .registry import Registry

    for mod in _extension_modules():
        if a.cmd in mod.COMMANDS:
            return int(mod.run(a, data_dir))

    if a.cmd == "build":
        opts = ds.BuildOptions(
            seed=a.seed, fractions=_fractions(a.fractions), level=a.level,
            allow_unconfirmed_identity=a.allow_unconfirmed_identity, allow_bbox=a.allow_bbox,
            strict_reviewed=a.strict_reviewed, roles=tuple(r for r in a.roles.split(",") if r),
            identity_map=(json.loads(Path(a.identity_map).read_text(encoding="utf-8")) if a.identity_map else None),
            require_finalized=a.require_finalized, min_crop_px=a.min_crop_px, created_at=a.created_at,
            include_raw=not a.no_raw, downscale=a.downscale, normaliser=a.normaliser)
        res = ds.build_snapshot(_store(data_dir), data_dir, a.analyses, opts, say=_p, dry_run=a.dry_run)
        _p(("[dry-run] " if res.dry_run else "") + f"dataset version: {res.version}")
        return 0

    if a.cmd == "split":
        rep = ds.split_report(data_dir, a.snapshot)
        _p(f"snapshot {rep['version']}  seed={rep['seed']} level={rep['level']} fractions={rep['fractions']}")
        for g, d in rep["groups"].items():
            _p(f"  {d['split']:<5} {g}  analyses={','.join(d['analyses']) or '(carried over)'}  crops={d['crops']}  positives={d['positives']}")
        for w in rep["warnings"]:
            _p(f"  WARNING: {w}")
        if a.lock_test:
            lock = ds.lock_test_set(data_dir, a.snapshot, replace=a.replace_lock, reason=a.reason,
                                    allow_unconfirmed_identity=a.allow_unconfirmed_identity)
            _p(f"LOCKED test set of snapshot {a.snapshot}: {len(lock['groups'])} group(s); lock {lock['lock_sha256'][:16]}. "
               "Later snapshots carry these test crops over unchanged and may never train on them.")
        return 0

    if a.cmd == "verify":
        bad = ds.verify_snapshot(data_dir, a.snapshot)
        _p("intact" if not bad else "PROBLEMS: " + "; ".join(bad))
        return 0 if not bad else 3

    if a.cmd == "show":
        _p(pretty_json(ds.load_manifest(data_dir, a.snapshot).model_dump(mode="json")))
        return 0

    if a.cmd == "train":
        from .train import TrainOptions, train_candidate
        o = TrainOptions(base=a.base, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, crop_size=a.crop_size, seed=a.seed,
                         device=a.device, num_train_samples=a.num_train_samples, pos_weight=a.pos_weight,
                         augment=not a.no_augment, optimize_threshold=not a.no_optimize_threshold,
                         early_stopping_patience=a.early_stopping_patience, notes=a.notes, allow_small=a.allow_small,
                         min_train_groups=a.min_train_groups, min_train_crops=a.min_train_crops)
        _p(f"training request: snapshot {a.snapshot}, base {a.base}" + (" (dry run)" if a.dry_run else ""))
        out = train_candidate(data_dir, a.snapshot, o, dry_run=a.dry_run, say=_p)
        if not a.dry_run:
            _p(f"model id: {out['id']}")
        return 0

    if a.cmd == "eval-model":
        return _eval_model(a, data_dir)

    if a.cmd == "evaluate":
        from .evaluate import CONFIGS, evaluate_analyses, report_markdown
        reg_ids = {m["id"] for m in Registry(data_dir).list() if m["kind"] == "finetuned"}
        rep = evaluate_analyses(_store(data_dir), a.predictions, reference=a.reference, tolerance_px=a.tolerance,
                                configs=tuple(a.configs or CONFIGS), numbering=a.numbering, registry_ids=reg_ids)
        md = report_markdown(rep)
        if a.out:
            out = Path(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(md, encoding="utf-8")
            out.with_suffix(".json").write_text(pretty_json(rep), encoding="utf-8")
            _p(f"wrote {out} and {out.with_suffix('.json')}")
        else:
            _p(md)
        return 0

    if a.cmd == "compare":
        return _compare(a, data_dir)

    if a.cmd == "promote":
        from .splits import load_lock
        lock = load_lock(data_dir)
        from .registry import make_config
        dflt = _app_defaults()
        app_cfg = {"downscale": a.app_downscale or dflt["downscale"], "normaliser": a.app_normaliser or dflt["normaliser"]}
        e = Registry(data_dir).promote(a.model, lock_sha256=(lock or {}).get("lock_sha256"), force=a.force, reason=a.reason,
                                       app_config=app_cfg, acknowledge_config=a.acknowledge_config)
        prom = e["promotion"]
        _p(f"promoted {a.model} (previous active: {e['previous_active'] or 'app default'})"
           + (f"  FORCED, overridden: {prom['overridden_checks']}" if prom["forced"] else ""))
        for k, v in (prom["comparison"].get("delta") or {}).items():
            _p(f"  delta {k}: {v}")
        return 0

    if a.cmd == "rollback":
        r = Registry(data_dir).rollback()
        _p(f"rolled back {r['rolled_back']}; active is now {r['restored'] or 'the app default (no promoted model)'}")
        return 0

    if a.cmd == "list":
        if a.what == "datasets":
            for v in ds.list_snapshots(data_dir):
                m = ds.load_manifest(data_dir, v)
                _p(f"{v}  {m.created_at}  crops={len(m.crops)}  {m.label_counts.get('by_split')}")
            if not ds.list_snapshots(data_dir):
                _p(f"no dataset snapshots under {datasets_dir(data_dir)}")
        else:
            reg = Registry(data_dir)
            rows = reg.list()
            for e in rows:
                from .registry import config_of, describe_config
                _p(f"{'*' if e['is_active'] else ' '} {e['id']:<28} {e['kind']:<9} {e['status']:<9} dataset={e.get('dataset_version')} "
                   f"evaluations={len(e.get('evaluations', []))} config=[{describe_config(config_of(e))}]")
            if not rows:
                _p("registry is empty (the app uses the default pretrained model)")
        return 0
    return 1


def _app_defaults() -> dict:
    """The app's default Spotiflow inference settings (what an analysis uses unless the researcher changes them)."""
    try:
        from polony_app.schemas import SpotiflowParams
        sp = SpotiflowParams()
        return {"downscale": int(sp.downscale), "normaliser": str(sp.normalizer)}
    except Exception:  # pragma: no cover
        return {"downscale": 1, "normaliser": "auto"}


def _eval_model(a, data_dir: str) -> int:
    from .common import DEFAULT_NORMALISER
    from .evaluate import evaluate_on_snapshot, make_spotiflow_predictor
    from .registry import Registry, config_of, describe_config, make_config, pretrained_id
    from .spotiflow_io import describe_base
    from .splits import load_lock
    lock = load_lock(data_dir)
    if not lock:
        raise MLError("there is no locked test set (ml split --snapshot <v> --lock-test); a model cannot be evaluated on "
                      "'the test set' before one is frozen")
    version = a.snapshot or lock["snapshot"]
    reg = Registry(data_dir)
    base = describe_base(a.model, data_dir)
    mid = base["registry_id"] or None
    if mid is None:
        if base["kind"] == "pretrained":
            reg.ensure_pretrained(base["name"])
            mid = pretrained_id(base["name"])
        else:
            raise MLError("only registry ids and official pretrained names can be evaluated into the registry")
    own = config_of(reg.get(mid)) if base["kind"] == "registry" else None
    want_d, want_n = getattr(a, "downscale", None), getattr(a, "normaliser", None)
    if own:                                     # fine-tuned: ALWAYS its own recorded configuration, never native
        if (want_d is not None and want_d != own["downscale"]) or (want_n is not None and want_n != own["normaliser"]):
            raise MLError(f"'{mid}' was trained at {describe_config(own)}; evaluating it at a different downscale / normaliser "
                          "would not measure the model (use the recorded configuration)")
        cfg = own
        if cfg["normaliser"] not in ("valid_unsaturated", "none"):
            raise MLError(f"'{mid}' was trained with the legacy per-crop normalisation ({describe_config(own)}) that the "
                          "locked test crops (page-level lo/hi) cannot reproduce; retrain it with a rebuilt snapshot")
    else:
        cfg = make_config(want_d or 1, want_n or DEFAULT_NORMALISER)
    _p(f"evaluating '{a.model}' at {describe_config(cfg)} on the locked test crops of snapshot {version} "
       f"(tolerances 5, 10, 15 px; headline {a.tolerance:g}) ...")
    pred = make_spotiflow_predictor(a.model, data_dir, device=a.device, prob_thresh=a.prob_thresh, downscale=cfg["downscale"])
    res = evaluate_on_snapshot(data_dir, version, pred, tolerance_px=a.tolerance, downscale=cfg["downscale"],
                               normaliser=cfg["normaliser"])
    reg.add_evaluation(mid, {"at": utc_now(), "lock_sha256": lock["lock_sha256"], "snapshot": version, "config": cfg,
                             "params": {"tolerance_px": a.tolerance, "prob_thresh": a.prob_thresh, **cfg}, **res})
    m = res["metrics"]
    _p(f"{mid} ({describe_config(cfg)}): precision vs annotator's marks={m['precision']} recall={m['recall']} f1={m['f1']} "
       f"mean_match_distance_px={m['mean_match_distance_px']} "
       f"[{res['label']}{': ' + '; '.join(res['preliminary_reasons']) if res['preliminary_reasons'] else ''}]")
    for k, v in m["by_tolerance"].items():
        _p(f"  @{k} px: P={v['precision']} R={v['recall']} F1={v['f1']}")
    return 0


def _compare(a, data_dir: str) -> int:
    from .registry import Registry, pretrained_id
    from .splits import load_lock
    lock = load_lock(data_dir)
    if not lock:
        raise MLError("there is no locked test set to compare on")
    reg = Registry(data_dir)
    rows = []
    for m in a.models:
        mid = m if m in {e["id"] for e in reg.list()} else pretrained_id(m)
        ev = reg.evaluation_on(mid, lock["lock_sha256"])
        if ev is None:
            raise MLError(f"'{mid}' has no evaluation on the current locked test set (run: ml eval-model --model {m})")
        rows.append((mid, ev))
    cols = ["precision", "recall", "f1", "mean_match_distance_px", "fp_per_field", "fn_per_field", "count_mae"]
    _p("Precision is vs this annotator's marks (unmarked polonies count as false positives).")
    _p("| model | configuration | " + " | ".join(cols) + " | F1 @5 px | F1 @10 px | F1 @15 px | label |")
    _p("|" + "|".join(["---"] * (len(cols) + 6)) + "|")
    from .registry import describe_config
    for mid, ev in rows:
        bt = ev["metrics"].get("by_tolerance", {})
        f1s = ["-" if (bt.get(k) or {}).get("f1") is None else f"{bt[k]['f1']:.4f}" for k in ("5", "10", "15")]
        _p(f"| {mid} | {describe_config(ev.get('config'))} | "
           + " | ".join("-" if ev['metrics'].get(c) is None else f"{ev['metrics'][c]:.4f}" for c in cols)
           + " | " + " | ".join(f1s) + f" | {ev['label']} |")
    return 0
