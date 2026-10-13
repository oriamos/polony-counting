"""``python -m polony_app.ml preflight`` - read-only audit of the historical TIFF directory before ``import-legacy``.

    python -m polony_app.ml --data-dir DATA preflight --input DIR [--channel-map C] [--identity-map I] [--examined-fields F]
                                                      [--sample 40|all] [--json] [--report-dir OUT]

Never writes into ``--input`` and never writes a snapshot. Exit codes: 0 = no FAIL (READY / READY WITH WARNINGS), 1 = at least
one FAIL (NOT READY), 2 = refused (``REFUSED: ...`` on stderr: bad arguments, ``--input`` missing, ``--report-dir`` inside
``--input``). See :mod:`polony_app.ml.preflight` for the checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from ..fields import examined as ex
from .common import MLError

COMMANDS = ("preflight",)


def _fractions(s: Optional[str]) -> Optional[dict]:
    if not s:
        return None
    try:
        a, b, c = (float(x) for x in s.split(","))
    except ValueError:
        raise MLError("--fractions must be 'train,val,test', e.g. 0.7,0.15,0.15")
    return {"train": a, "val": b, "test": c}


def register(sub) -> None:
    p = sub.add_parser("preflight", help="READ-ONLY audit of the historical TIFF directory before import-legacy (environment, input "
                                         "sanity, mapping files, independent cross-check of the bridge, policy summary, VERDICT)",
                       epilog="exit codes: 0 no FAIL; 1 at least one FAIL (NOT READY); 2 refused (bad arguments)")
    p.add_argument("--input", help="directory (or one file) with the historical TIFFs; searched recursively")
    p.add_argument("--channel-map")
    p.add_argument("--identity-map")
    p.add_argument("--examined-fields")
    p.add_argument("--mode", default=ex.MODE_MARKED_ONLY, choices=list(ex.MODES))
    p.add_argument("--lab-csv", action="append", default=[])
    p.add_argument("--lab-series", action="append", default=[])
    p.add_argument("--acknowledge-no-counters", action="store_true")
    p.add_argument("--allow-unconfirmed-identity", action="store_true")
    p.add_argument("--acknowledge-field-equals-counter-plus-one", action="store_true")
    p.add_argument("--acknowledge-lab-csv-field-numbering", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fractions", help="train,val,test (default 0.7,0.15,0.15)")
    p.add_argument("--level", default="auto", choices=["auto", "specimen", "slide", "scan_session", "source_image"])
    p.add_argument("--valid-mask", default="chamber", choices=["chamber", "grid-cells", "eligible-fields"],
                   help="valid mask for the page lo/hi, as in import-legacy (default chamber: the app's automatic chamber)")
    p.add_argument("--downscale", type=int, default=1, choices=[1, 2, 4, 8])
    p.add_argument("--normaliser", default="valid_unsaturated", choices=["valid_unsaturated", "none"])
    p.add_argument("--batch-size", type=int, default=4, help="only used for the training estimate")
    p.add_argument("--epochs", type=int, default=30, help="only used for the training estimate")
    p.add_argument("--workers", type=int, default=0, help="header-scan threads (default: automatic, at most 8)")
    p.add_argument("--sample", default="40", help="files re-derived independently in check D: a number K (deterministic sample, plus "
                                                  "every odd file) or 'all' (default 40)")
    p.add_argument("--max-examples", type=int, default=10, help="offending files listed per finding (default 10)")
    p.add_argument("--no-pixels", action="store_true", help="skip the local-intensity-maximum sanity check (reads page 0 of sampled files)")
    p.add_argument("--json", action="store_true", help="print one machine-readable JSON document on stdout")
    p.add_argument("--report-dir", help="also write preflight-report.json and preflight-report.txt here (never inside --input)")
    p.add_argument("--no-progress", action="store_true", help="no progress lines on stderr")


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run(args, data_dir: str) -> int:
    from . import preflight as pf
    if not args.input:
        raise MLError("--input DIR is required")
    root = Path(args.input)
    if not root.exists():
        raise MLError(f"--input {root} does not exist")
    rd = Path(args.report_dir) if args.report_dir else None
    if rd is not None and pf._is_within(rd, root if root.is_dir() else root.parent):
        raise MLError(f"--report-dir {rd} is inside --input: preflight never writes into the historical data")
    if args.max_examples < 1:
        raise MLError("--max-examples must be >= 1")
    p = pf.PreflightParams(
        input=root, data_dir=Path(data_dir), channel_map=Path(args.channel_map) if args.channel_map else None,
        identity_map=Path(args.identity_map) if args.identity_map else None,
        examined_fields=Path(args.examined_fields) if args.examined_fields else None, lab_csv=list(args.lab_csv),
        lab_series=list(args.lab_series), mode=args.mode, acknowledge_no_counters=args.acknowledge_no_counters,
        allow_unconfirmed_identity=args.allow_unconfirmed_identity,
        acknowledge_field_equals_counter_plus_one=args.acknowledge_field_equals_counter_plus_one,
        acknowledge_lab_csv_field_numbering=args.acknowledge_lab_csv_field_numbering, seed=args.seed,
        fractions=_fractions(args.fractions), level=args.level, valid_mask=args.valid_mask, downscale=args.downscale, normaliser=args.normaliser,
        batch_size=args.batch_size, epochs=args.epochs, workers=args.workers, sample=str(args.sample),
        max_examples=args.max_examples, report_dir=rd, pixels=not args.no_pixels,
        progress=None if args.no_progress else _err)
    doc = pf.run_preflight(p)
    text = pf.format_text(doc)
    if rd is not None:
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "preflight-report.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (rd / "preflight-report.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(doc, indent=2, sort_keys=True), flush=True)
    else:
        for line in text:
            print(line, flush=True)
    return int(doc["exit_code"])
