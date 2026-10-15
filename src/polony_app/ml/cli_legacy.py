"""``python -m polony_app.ml import-legacy`` - bring historical ImageJ-annotated TIFFs into the fine-tuning lifecycle.

    # 1. write review templates (no pixel data is read)
    python -m polony_app.ml import-legacy --input DIR --write-templates OUT
    # 2. dry run: writes NOTHING, reports what a real import would do and what blocks it
    python -m polony_app.ml import-legacy --input DIR --channel-map channels.json --identity-map identities.json --dry-run
    # 3. real import -> immutable snapshot, then: ml verify / ml split --lock-test / ml train / ml eval-model / ml compare

Only (file, field, page) tuples the researcher examined are used (see :mod:`polony_app.ml.legacy_bridge`). Exit codes: 0 ok,
2 refused (``REFUSED: ...`` on stderr; in --dry-run only malformed mapping / manifest files refuse - identity and mapping
gaps are printed as BLOCKERS and exit 0).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ..fields import examined as ex
from .common import MLError, datasets_dir
from . import dataset as ds

COMMANDS = ("import-legacy",)


def _fractions(s: Optional[str]) -> Optional[dict]:
    if not s:
        return None
    try:
        a, b, c = (float(x) for x in s.split(","))
    except ValueError:
        raise MLError("--fractions must be 'train,val,test', e.g. 0.7,0.15,0.15")
    return {"train": a, "val": b, "test": c}


def register(sub) -> None:
    p = sub.add_parser("import-legacy", help="import historical ImageJ-annotated TIFFs as a dataset snapshot (researcher-examined "
                                              "fields only)")
    p.add_argument("--input", help="directory (or one file) with the historical TIFFs; searched recursively, archives are never opened")
    p.add_argument("--mode", default=ex.MODE_MARKED_ONLY, choices=list(ex.MODES),
                   help="marked-only (default): only field/pages with >=1 embedded ImageJ mark. examined-fields-manifest: only tuples "
                        "listed in --examined-fields (verified zeros allowed). There is no mode that treats unmarked cells as negatives.")
    p.add_argument("--examined-fields", help="examined-fields manifest (JSON or CSV, see polony_app.fields.examined)")
    p.add_argument("--channel-map", help="JSON 'polony-channel-map': what each page shows (never inferred); required for a real import")
    p.add_argument("--identity-map", help="JSON/CSV: file -> specimen_id, slide_id, scan_session, assay, replicate, confirmed")
    p.add_argument("--acknowledge-no-counters", action="store_true",
                   help="accept files whose marks carry no ImageJ counter (fields = geometric grid cells; recorded as an assumption)")
    p.add_argument("--allow-unconfirmed-identity", action="store_true",
                   help="split on identities inferred from file names (recorded; leakage between replicates cannot be excluded)")
    p.add_argument("--acknowledge-field-equals-counter-plus-one", action="store_true",
                   help="read a manifest 'field N' as ImageJ counter N-1 (otherwise such entries are unresolved)")
    p.add_argument("--lab-csv", action="append", default=[], help="the lab's per-slide counts CSV (blank = not counted, 0 = counted "
                   "zero); repeat per series. Needs --lab-series and --acknowledge-lab-csv-field-numbering")
    p.add_argument("--lab-series", action="append", default=[], help="SERIES=PAGE[,SERIES=PAGE] e.g. T7B=0,T7A=1 (never inferred)")
    p.add_argument("--acknowledge-lab-csv-field-numbering", action="store_true",
                   help="confirm that field N of the lab CSV is ImageJ counter N-1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fractions", help="train,val,test (default 0.7,0.15,0.15)")
    p.add_argument("--level", default="auto", choices=["auto", "specimen", "slide", "scan_session", "source_image"])
    p.add_argument("--downscale", type=int, default=1, choices=[1, 2, 4, 8])
    p.add_argument("--normaliser", default="valid_unsaturated", choices=["valid_unsaturated", "none"])
    p.add_argument("--valid-mask", default="chamber", choices=list(lb_valid_masks()),
                   help="mask the page-level lo/hi (normalisation) are computed over. chamber (default): the app's automatic "
                        "chamber (rim 40 px) = the inference mask; when a file has no detectable chamber the union of ALL grid cells "
                        "is used and recorded. grid-cells: always the union of all grid cells. eligible-fields: the examined "
                        "fields only (NOT the inference mask: kept for comparison)")
    p.add_argument("--batch-size", type=int, default=4, help="only used for the training estimate")
    p.add_argument("--epochs", type=int, default=30, help="only used for the training estimate")
    p.add_argument("--workers", type=int, default=1, help="crop-writing threads (the snapshot is identical for any value)")
    p.add_argument("--dry-run", action="store_true", help="scan and report only; nothing is written")
    p.add_argument("--no-resume", action="store_true", help="do not reuse a staging directory of an interrupted run")
    p.add_argument("--created-at", help=argparse.SUPPRESS)
    p.add_argument("--write-templates", metavar="DIR", help="write identities / channels / examined-fields templates and exit")


def lb_valid_masks() -> tuple:
    from . import legacy_bridge as lb
    return lb.VALID_MASKS


def _p(*a) -> None:
    print(*a, flush=True)


def run(args, data_dir: str) -> int:
    from . import legacy_bridge as lb
    if args.write_templates:
        paths = lb.write_templates(args.write_templates, args.input)
        for q in paths:
            _p(f"wrote {q}")
        _p("Review and fill them in (nothing in them is confirmed). Then re-run with --channel-map / --identity-map "
           "[--examined-fields] and --dry-run first.")
        return 0
    if not args.input:
        raise MLError("--input DIR is required (or use --write-templates DIR)")
    opts = lb.BridgeOptions(
        mode=args.mode, acknowledge_no_counters=args.acknowledge_no_counters,
        allow_unconfirmed_identity=args.allow_unconfirmed_identity,
        acknowledge_field_equals_counter_plus_one=args.acknowledge_field_equals_counter_plus_one,
        acknowledge_lab_csv_field_numbering=args.acknowledge_lab_csv_field_numbering, seed=args.seed,
        fractions=_fractions(args.fractions), level=args.level, downscale=args.downscale, normaliser=args.normaliser,
        batch_size=args.batch_size, epochs=args.epochs, workers=args.workers, created_at=args.created_at,
        valid_mask=args.valid_mask)
    out = lb.import_legacy(data_dir, args.input, opts, channel_map=args.channel_map, identity_map=args.identity_map,
                           examined_fields=args.examined_fields, lab_csv=[Path(p) for p in args.lab_csv],
                           lab_series=args.lab_series, dry_run=args.dry_run, resume=not args.no_resume, say=_p)
    if args.dry_run:
        for line in format_report(out, data_dir):
            _p(line)
        return 0
    if out.already_imported:
        _p(f"already imported as {out.already_imported}")
        return 0
    _p(f"import summary: {out.plan.summary['fields']['eligible_examined_fields']} examined (field, page) tuple(s) used, "
       f"{out.plan.summary['fields']['excluded_unproven']} excluded because examination could not be proven")
    ver = out.result.version
    _p("next:")
    _p(f"  python -m polony_app.ml verify {ver}")
    _p(f"  python -m polony_app.ml split --snapshot {ver}            # inspect the group split")
    _p(f"  python -m polony_app.ml split --snapshot {ver} --lock-test  # freeze the test groups (permanent)")
    _p(f"  python -m polony_app.ml train --snapshot {ver} --dry-run   # then without --dry-run; eval-model / compare / promote follow")
    return 0


def _n(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items())) or "none"


def format_report(out, data_dir) -> list[str]:
    s = out.plan.summary
    L: list[str] = []
    ad = L.append
    ad("")
    ad("=" * 78)
    ad("IMPORT-LEGACY DRY RUN - nothing was written (no staging directory, no snapshot)")
    ad("=" * 78)
    ad(f"mode: {s['mode']}   downscale: {s['options']['downscale']}   normaliser: {s['options']['normaliser']}   "
       f"valid mask for page lo/hi: {s['options']['valid_mask']}"
       + ("" if s['options']['valid_mask'] != "eligible-fields" else "  (NOT the inference mask)"))
    if s["options"]["valid_mask"] == "chamber":
        ad("  (chamber detection reads the pixels: it runs at import time; files without a chamber fall back to the union of all "
           "grid cells and are recorded per source in import_summary.valid_mask)")
    for sk in s["tiffs"].get("skipped_links", []):
        ad(f"  SKIPPED {sk['path']}: {sk['reason']}")
    for k, v in s["input_files"].items():
        if v:
            for f in (v if isinstance(v, list) else [v]):
                ad(f"input file {k}: {f['name']}  sha256 {f['sha256'][:16]}...")
    t = s["tiffs"]
    ad("")
    ad(f"SOURCES: discovered {t['discovered']}, unique {t['unique']}, accepted {t['accepted']}, rejected {t['rejected']}, "
       f"deduplicated (byte-identical alias paths) {t['deduplicated']}")
    if t["rejected_by_reason"]:
        ad("  rejected by reason: " + _n(t["rejected_by_reason"]))
        for rel, codes in list(t["rejected_files"].items())[:15]:
            ad(f"    {rel}: {', '.join(codes)}")
        if len(t["rejected_files"]) > 15:
            ad(f"    ... {len(t['rejected_files']) - 15} more (see import_summary.tiffs.rejected_files in the manifest)")
    for k, al in list(t["deduplicated_paths"].items())[:10]:
        ad(f"  kept {k}; identical copies: {', '.join(al)}")
    so = s["sources"]
    ad(f"  accepted sources contributing crops: {so['with_crops']}; accepted but without any examined field/page: {so['without_crops']}")
    cmap = s["channel_mappings"]
    ad("")
    ad("PAGES AND CHANNEL MAPPING" + ("" if cmap["mapped"] else " - NO --channel-map GIVEN (roles unknown, no crop can be planned)"))
    if cmap["map"]:
        ad(f"  map {cmap['map']['name']} confirmed by {cmap['map']['confirmed_by']} ({cmap['map']['confirmed_on']}); origins: {_n(cmap['origin_counts'])}")
    for r in cmap["by_page_role"]:
        ad(f"  page {r['page']} (ImageJ position {r['page'] + 1}) -> role {r['role']!r}, train={r['train']}: {r['files']} file(s)")
    ilc = s.get("info_label_check")
    if ilc and (ilc["usual_hint_by_page0_role"] or ilc["absent"]):
        ad(f"  Info label hint of page 0 (information only, never decides a role): usual {ilc['usual_hint_by_page0_role']}; "
           f"present in {ilc['present']} file(s), absent in {len(ilc['absent'])}, different in {len(ilc['different'])}")
        for rel in (ilc["different"] + ilc["absent"])[:15]:
            ad(f"    deviates: {rel} ({'absent' if rel in ilc['absent'] else 'different'}) - add a rule/file override if its pages differ")
    f = s["fields"]
    ad("")
    ad(f"FIELD/PAGE TUPLES ({f['tuples_total']} in accepted files): examined and eligible {f['eligible_examined_fields']} "
       f"(positive_marked {f['positive_marked_fields']}, verified_zero {f['verified_zero_fields']})")
    ad(f"  EXCLUDED because the researcher's examination could not be proven: {f['excluded_unproven']} "
       f"(unknown {f['unknown_fields_excluded']}, conflict {f['conflict_fields_excluded']}); ambiguous zero-vs-unreviewed: "
       f"{f['ambiguous_zero_vs_unreviewed']}")
    for reason, n in list(f["reasons"].items())[:8]:
        ad(f"    {n} x {reason}")
    for pg, d in f["by_page"].items():
        ad(f"  page {pg}: {_n(d)}")
    c = s["crops"]
    ad(f"  crops planned {c['created']}; examined tuples without a crop {c['skipped']} {_n(c['skipped_reasons']) if c['skipped'] else ''}")
    ad("")
    ad("POSITIVES PER PAGE/CHANNEL (inside examined fields only)")
    for r in s["positives_per_page"]:
        ad(f"  page {r['page']} / {r['role'] or 'role unmapped'}: {r['positives']} positive(s) in {r['crops']} {'crop(s)' if r['role'] else 'examined field(s)'}")
    a = s["annotations"]
    ad(f"ANNOTATIONS: {a['points_total']} ImageJ point(s); {a['labels_exported']} exported as labels; {a['dropped_total']} NOT labels "
       f"(listed as 'dropped' in points.csv): {_n(a['dropped_by_reason'])}")
    if a["straddling_labels_in_examined_cells"]:
        ad(f"  {a['straddling_labels_in_examined_cells']} label(s) in examined cells were clicked with another cell's counter (kept; recorded)")
    sp = s["splits"]
    ad("")
    if sp:
        ad(f"GROUPS AND SPLITS (seed {sp['seed']}, level {sp['level']}): {sp['groups']} independent group(s)")
        for k, v in sp["by_split"].items():
            ad(f"  {k:<5} {v['crops']} crop(s), {v['positives']} positive(s), {v['groups']} group(s), {v['sources']} source(s)")
        if sp["locked_sources"]:
            ad(f"  {sp['locked_sources']} source(s) belong to the LOCKED test set and are not exported again")
    else:
        ad("GROUPS AND SPLITS: not available (no exportable crop)")
    e = s["estimates"]
    ad(f"ESTIMATES: native snapshot size ~ {e['native_disk']}")
    tr = e["training"]
    if tr:
        ad(f"  prepared in-memory training size (float32, downscale {tr['downscale']}): ~ {ds.fmt_bytes(tr['prepared_bytes'])}")
        ad(f"  batch size {tr['batch_size']}: {tr['steps_per_epoch']} batch(es) per epoch, {tr['optimizer_steps']} optimiser step(s) in "
           f"{tr['epochs']} epoch(s); validation {tr['validation_batches']} batch(es) per pass")
    free = None
    try:
        from .dataset import _estimate_free_bytes
        free = _estimate_free_bytes(datasets_dir(data_dir))
        ad(f"  free space under {datasets_dir(data_dir)}: {ds.fmt_bytes(free)}")
        if e["native_disk_bytes"] * 1.1 > free:
            ad("  BLOCKER: not enough free disk space for the snapshot")
    except Exception:  # pragma: no cover
        pass
    idn = s["identity"]
    ad("")
    ad(f"IDENTITY: {idn['confirmed']} confirmed, {idn['unconfirmed']} unconfirmed among sources with crops ({_n(idn['sources'])})")
    unc = [p for p in out.plan.plans if p.ext.cands and not (p.scan.identity or {}).get("confirmed")]
    if unc:
        ad("  file | inferred specimen | inferred slide | source")
        for p in unc[:40]:
            i = p.scan.identity
            ad(f"  {p.scan.rel} | {i.get('specimen_id')} | {i.get('slide_id')} | {i.get('source')}")
        if len(unc) > 40:
            ad(f"  ... {len(unc) - 40} more")
    m = idn["map"]
    for k in ("ambiguous_basename", "missing_on_disk", "conflicting_rows"):
        if m.get(k):
            ad(f"  identity map: {k}: {', '.join(m[k][:10])}")
    if idn["map"]["rows"] and m.get("unlisted_files"):
        ad(f"  identity map: {len(m['unlisted_files'])} file(s) on disk are not in the map")
    if s["unresolved_manifest_entries"]:
        ad(f"UNRESOLVED MANIFEST ENTRIES: {len(s['unresolved_manifest_entries'])} (not used)")
        for u in s["unresolved_manifest_entries"][:10]:
            ad(f"    {u}")
    ad("")
    if out.plan.blockers:
        ad("BLOCKERS - a real import would be REFUSED:")
        for b in out.plan.blockers:
            ad(f"  - {b}")
    else:
        ad("no blockers: a real import would proceed.")
    ad("")
    ad("next commands:")
    ad("  python -m polony_app.ml import-legacy --input DIR --write-templates OUT      # templates to review")
    ad("  python -m polony_app.ml import-legacy --input DIR --channel-map C --identity-map I [--examined-fields F]   # real import")
    ad("  python -m polony_app.ml verify <snapshot>; split --snapshot <snapshot> --lock-test; train --snapshot <snapshot> --dry-run")
    return L
