"""Command line: ``python -m polony_app.legacy_import {inspect,import,split,export-spotiflow}``.

``import`` is a dry run unless ``--commit`` is given: the full conversion and the report are
produced, nothing is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .exports import export_spotiflow
from .importer import ImportConfig, build_dataset
from .schema import read_dataset, write_dataset
from .splits import DEFAULT_FRACTIONS, IdentityNotConfirmed, LEVELS, SplitAssignment, SplitError, assign_splits


def _kv(items: Sequence[str], what: str) -> dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"error: {what} expects KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _config(a: argparse.Namespace) -> ImportConfig:
    cm = {}
    for k, v in _kv(a.class_map or [], "--class-map").items():
        if not k.isdigit() or int(k) < 1:
            raise SystemExit(f"error: --class-map keys are 1-based ImageJ plane positions, got {k!r}")
        cm[int(k)] = v
    return ImportConfig(
        class_map=cm, reserve=_kv(a.reserve or [], "--reserve"),
        valid_region_names=tuple(a.valid_name or ()), excluded_region_names=tuple(a.excluded_name or ()),
        duplicate_tol_px=a.duplicate_tol, check_signal=not a.no_signal_check,
        reserve_legacy_test_dirs=not a.no_reserve_test_dirs, created_at=a.created_at)


def _add_import_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", type=Path, help="directory (searched recursively) or a single TIFF")
    p.add_argument("--class-map", action="append", metavar="POS=LABEL",
                   help="EXPLICIT class label for an ImageJ plane position (1-based); default: no classes assigned")
    p.add_argument("--reserve", action="append", metavar="GLOB_OR_SHA=SPLIT",
                   help="pin matching files (relative-path glob or sha256 prefix) to a split, e.g. '*/holdout/*=test'")
    p.add_argument("--valid-name", action="append", help="ROI name that marks a VALID region")
    p.add_argument("--excluded-name", action="append", help="ROI name that marks an EXCLUDED region")
    p.add_argument("--duplicate-tol", type=float, default=1.0, help="px; points closer than this on one plane are duplicates")
    p.add_argument("--no-signal-check", action="store_true")
    p.add_argument("--no-reserve-test-dirs", action="store_true",
                   help="do not auto-reserve files under a 'test' directory to the test split")
    p.add_argument("--created-at", help="fixed timestamp for the manifest (reproducible output)")
    p.add_argument("--json", action="store_true", help="print the report as JSON")


def _print_report(report, as_json: bool) -> None:
    if as_json:
        d = {k: (v if not hasattr(v, "model_dump") else v.model_dump(mode="json")) for k, v in report.__dict__.items()}
        d["warnings"] = [w.model_dump(mode="json") for w in report.warnings]
        print(json.dumps(d, indent=2, sort_keys=True, default=str))
    else:
        print(report.render_text())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m polony_app.legacy_import", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="read-only import report (never writes)")
    _add_import_args(p)

    p = sub.add_parser("import", help="convert historical files; DRY RUN unless --commit")
    _add_import_args(p)
    p.add_argument("--out", type=Path, help="dataset directory to create (required with --commit)")
    p.add_argument("--commit", action="store_true", help="actually write the dataset")
    p.add_argument("--allow-errors", action="store_true", help="commit even if some files failed")

    p = sub.add_parser("split", help="leakage-safe deterministic split of a committed dataset")
    p.add_argument("dataset", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fractions", default=",".join(f"{k}={v}" for k, v in DEFAULT_FRACTIONS.items()))
    p.add_argument("--level", default="auto", choices=("auto",) + LEVELS)
    p.add_argument("--allow-unconfirmed-identity", action="store_true")
    p.add_argument("--confirmed-slides", type=Path,
                   help='JSON file {"confirmed_slides": ["AM_A", ...], "confirmed_by": "..."} recording that the '
                        "researcher confirmed the specimen/slide identity of these slides")
    p.add_argument("--name", default="default", help="writes <dataset>/splits/<name>.json")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("export-spotiflow", help="y,x CSVs + counted-field polygons for fine-tuning")
    p.add_argument("dataset", type=Path)
    p.add_argument("--splits-file", type=Path, required=True)
    p.add_argument("--pages", required=True, help="comma-separated 0-based planes to export, e.g. 0")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--write-planes", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd in ("inspect", "import"):
        ds, report = build_dataset(a.source, _config(a))
        _print_report(report, a.json)
        if a.cmd == "inspect" or not a.commit:
            print(f"\nDRY RUN: nothing was written (dataset_id would be {ds.manifest.dataset_id}). "
                  "Re-run `import ... --out DIR --commit` to write it." if a.cmd == "import" else "")
            return 1 if report.n_errors else 0
        if a.out is None:
            raise SystemExit("error: --commit requires --out DIR")
        if report.n_errors and not a.allow_errors:
            print(f"\nrefusing to commit: {report.n_errors} error(s); fix them or pass --allow-errors", file=sys.stderr)
            return 2
        out = write_dataset(ds, a.out)
        print(f"\nCOMMITTED dataset {ds.manifest.dataset_id} -> {out}")
        return 0
    if a.cmd == "split":
        ds = read_dataset(a.dataset)
        fr = {k: float(v) for k, v in _kv(a.fractions.split(","), "--fractions").items()}
        confirmed = None
        if a.confirmed_slides:
            confirmed = set(json.loads(a.confirmed_slides.read_text())["confirmed_slides"])
        try:
            asg = assign_splits(ds, seed=a.seed, fractions=fr, level=a.level,
                                allow_unconfirmed=a.allow_unconfirmed_identity, confirmed_slides=confirmed)
        except IdentityNotConfirmed as exc:
            print(f"NEEDS CONFIRMATION: {exc}", file=sys.stderr)
            return 3
        except SplitError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for im in ds.images:
            print(f"{asg.image_to_split[im.image_id]:6s} {im.file_name}  [{asg.image_to_group[im.image_id]}]")
        for w in asg.warnings:
            print(f"WARNING: {w}")
        if not a.dry_run:
            d = a.dataset / "splits"
            d.mkdir(exist_ok=True)
            (d / f"{a.name}.json").write_text(asg.dumps())
            print(f"written {d / (a.name + '.json')}")
        return 0
    if a.cmd == "export-spotiflow":
        ds = read_dataset(a.dataset)
        raw = json.loads(a.splits_file.read_text())
        asg = SplitAssignment(seed=raw["seed"], level=raw["level"], fractions=raw["fractions"],
                              image_to_split=raw["image_to_split"], group_to_split=raw["group_to_split"],
                              image_to_group=raw["image_to_group"], warnings=raw["warnings"],
                              unconfirmed_identity=raw["unconfirmed_identity"])
        m = export_spotiflow(ds, a.out, asg, pages=[int(x) for x in a.pages.split(",")],
                             splits=tuple(a.splits.split(",")), source_root=a.source_root, write_planes=a.write_planes)
        print(f"exported {len(m['entries'])} (image, plane) entries to {a.out}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
