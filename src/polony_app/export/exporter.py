"""Exports: objects.csv, regions.csv, slide.csv, provenance.json, bundle.zip."""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from importlib import metadata
from pathlib import Path

import polony_app
from polony_app.review.counts import compute_counts, counted_objects
from polony_app.review.engine import (
    action_log_summary,
    counting_area_px2,
    mapping_confirmed,
    compute_stale,
    now,
    region_area_px2,
)
from polony_app.review.state import AnalysisState
from polony_app.schemas import SCHEMA_VERSION, Analysis, RegionKind

KINDS = ("objects", "regions", "slide", "provenance", "bundle")
_PKGS = ["numpy", "scipy", "big-fish", "tifffile", "pydantic", "fastapi", "Pillow", "scikit-image", "roifile"]


def status_label(a: Analysis) -> str:
    return "FINAL" if a.finalized_at is not None else "DRAFT"


def safe_stem(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return (s or "analysis")[:60]


_EXT = {"objects": "csv", "regions": "csv", "slide": "csv", "history": "csv",
        "provenance": "json", "actions": "jsonl", "bundle": "zip"}


def export_filename(a: Analysis, kind: str) -> str:
    """THE download naming rule (used by the API): <name>_<kind>_<DRAFT|FINAL>.<ext>."""
    return f"{safe_stem(a.name)}_{kind}_{status_label(a)}.{_EXT[kind]}"


def member_name(a: Analysis, kind: str) -> str:
    """Name of a CSV inside the bundle: <kind>_<DRAFT|FINAL>.csv (same status rule, no stem)."""
    return f"{kind}_{status_label(a)}.{_EXT[kind]}"


# the detection pipeline reports import names; the exporter uses distribution names
_CANON = {"bigfish": "big-fish", "skimage": "scikit-image", "PIL": "Pillow", "pil": "Pillow",
          "polony-app": "polony_app"}


def canonical_versions(v: dict[str, str] | None) -> dict[str, str]:
    """Package versions under ONE set of key names (distribution names)."""
    return {_CANON.get(k, k): str(x) for k, x in (v or {}).items()}


def package_versions() -> dict[str, str]:
    """Versions installed NOW (at export time)."""
    out = {"polony_app": polony_app.__version__}
    for p in _PKGS:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def read_upload_manifest(originals_dir: str | Path | None) -> dict[str, str] | None:
    """{stored_name: original_upload_name} from <originals_dir>/upload_manifest.json
    (written by the API at upload). None when absent/unreadable."""
    if originals_dir is None:
        return None
    try:
        d = json.loads((Path(originals_dir) / "upload_manifest.json").read_text())
        return {str(k): str(v) for k, v in d.items()} if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _csv(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(["" if v is None else v for v in r])
    return buf.getvalue()


def _b(v: bool) -> str:
    return "true" if v else "false"


# ------------------------------------------------------------------ objects
def objects_csv(a: Analysis) -> str:
    status = status_label(a)
    header = ["id", "x", "y", "class", "origin", "region_id", "counted"]
    for ch in ("red", "green"):
        header += [f"{ch}_detected", f"{ch}_x", f"{ch}_y", f"{ch}_peak", f"{ch}_local_background",
                   f"{ch}_snr", f"{ch}_saturated"]
    header += ["association_distance_px", "review_reasons", "ambiguous_with", "reviewed", "status"]
    counted_ids = {o.id for o in counted_objects(a)}
    rows = []
    for o in a.objects:
        if o.deleted:
            continue
        row = [o.id, round(o.x, 3), round(o.y, 3), o.cls.value, o.origin.value, o.region_id,
               _b(o.id in counted_ids)]
        for ev in (o.red, o.green):
            row += [_b(ev.detected), ev.x, ev.y, ev.peak_intensity, ev.local_background, ev.snr,
                    _b(ev.saturated)]
        row += [o.association_distance_px, "|".join(r.value for r in o.review_reasons),
                "|".join(o.ambiguous_with), _b(o.reviewed), status]
        rows.append(row)
    return _csv(header, rows)


# ------------------------------------------------------------------ regions
def regions_csv(a: Analysis) -> str:
    counts = {c.scope: c for c in compute_counts(a)["regions"]}
    status = status_label(a)
    header = [
        "region_id", "name", "kind", "area_px2",
        "red_only", "green_only", "yellow", "uncertain",
        "red_positive_incl_yellow", "green_positive_incl_yellow", "yellow_overlap",
        "total_objects_exclusive", "reporting_mode", "inspected", "approval_status", "approved_at", "status",
    ]
    rows = []
    for r in a.regions:
        area = region_area_px2(a, r)
        c = counts.get(r.id)
        if r.kind == RegionKind.counting and c is not None:
            rows.append([
                r.id, r.name, r.kind.value, area, c.red_only, c.green_only, c.yellow, c.uncertain,
                c.red_positive, c.green_positive, c.yellow, c.total_objects_exclusive,
                a.reporting_mode.value, _b(r.inspected),
                "approved" if r.approved else "not_approved", r.approved_at, status,
            ])
        else:
            rows.append([r.id, r.name, r.kind.value, area, None, None, None, None, None, None, None,
                         None, a.reporting_mode.value, None, "n/a", None, status])
    return _csv(header, rows)


# ------------------------------------------------------------------ slide
_CONC_KEYS = ["red_only", "green_only", "yellow", "total_objects_exclusive", "red_positive", "green_positive"]


def concentration(a: Analysis, counts) -> tuple[dict[str, float | None], str, list[str]]:
    """Return (estimates, status text, missing inputs). Never invents inputs.

    PROPOSED / UNCONFIRMED formula (area-scaled, count per ml of the sample):

        conc_per_ml = (n_counted / counting_area_px2) * sample_total_area_px2
                      / (sample_volume_ul / 1000) * concentration_factor

    i.e. the counted density (objects per px^2 of the net counting area) is scaled to the
    whole sample area and divided by the sample volume. ALL of the following are required,
    otherwise every estimate cell is empty and the status lists what is missing:
    concentration_formula_confirmed, sample_volume_ul, concentration_factor,
    counting_area_px2 (union of counting regions net of exclusions and overlaps),
    sample_total_area_px2 (same pixel units), confirmed channel mapping. No confidence
    interval is computed; raw counts are always exported next to the estimates."""
    s = a.slide
    area = counting_area_px2(a)
    missing = []
    if not s.concentration_formula_confirmed:
        missing.append("concentration_formula_confirmed")
    if not s.sample_volume_ul or s.sample_volume_ul <= 0:
        missing.append("sample_volume_ul")
    if not s.concentration_factor or s.concentration_factor <= 0:
        missing.append("concentration_factor")
    if not area or area <= 0:
        missing.append("counting_area_px2")
    if not s.sample_total_area_px2 or s.sample_total_area_px2 <= 0:
        missing.append("sample_total_area_px2")
    if not mapping_confirmed(a):
        missing.append("confirmed_channel_mapping")
    if missing:
        if missing[0] == "concentration_formula_confirmed":
            msg = "inputs incomplete: formula/geometry not confirmed by researcher"
            if len(missing) > 1:
                msg += "; also missing: " + ", ".join(missing[1:])
        else:
            msg = "inputs incomplete: " + ", ".join(missing)
        return {k: None for k in _CONC_KEYS}, msg, missing
    vol_ml = s.sample_volume_ul / 1000.0
    scale = s.sample_total_area_px2 / area / vol_ml * s.concentration_factor
    est = {k: getattr(counts, k) * scale for k in _CONC_KEYS}
    msg = "computed (PROPOSED area-scaled formula, UNCONFIRMED; no confidence interval)"
    if area > s.sample_total_area_px2:
        msg += "; WARNING counting_area_px2 exceeds sample_total_area_px2 (check units)"
    if counts.uncertain:
        msg += f"; {counts.uncertain} unresolved uncertain object(s) excluded"
    return est, msg, []


CONC_FORMULA = (
    "PROPOSED (UNCONFIRMED): (n_counted / counting_area_px2) * sample_total_area_px2 / (sample_volume_ul / 1000) * "
    "concentration_factor; assumes the counted area is representative of the whole sample area (no chamber geometry, "
    "no edge/loss model, no confidence interval)"
)


def slide_csv(a: Analysis) -> str:
    c = compute_counts(a)["slide"]
    regs = [r for r in a.regions if r.kind == RegionKind.counting]
    est, cstat, _ = concentration(a, c)
    pix = a.slide.pixel_size_um or (a.image.pixel_size_um if a.image else None)
    header = [
        "slide_id", "analysis_id", "reporting_mode", "red_only", "green_only", "yellow", "uncertain",
        "red_positive_incl_yellow", "green_positive_incl_yellow", "yellow_overlap",
        "total_objects_exclusive", "n_counting_regions", "n_regions_approved", "counting_area_px2",
        "pixel_size_um", "sample_volume_ul", "concentration_factor", "sample_total_area_px2",
        "est_conc_per_ml_red_only", "est_conc_per_ml_green_only", "est_conc_per_ml_yellow",
        "est_conc_per_ml_total_exclusive", "est_conc_per_ml_red_positive", "est_conc_per_ml_green_positive",
        "concentration_status", "concentration_formula", "channel_mapping_confirmed", "stale_detection",
        "finalized_at", "status",
    ]
    row = [
        a.slide.slide_id, a.id, a.reporting_mode.value, c.red_only, c.green_only, c.yellow, c.uncertain,
        c.red_positive, c.green_positive, c.yellow, c.total_objects_exclusive, len(regs),
        sum(1 for r in regs if r.approved), counting_area_px2(a), pix, a.slide.sample_volume_ul,
        a.slide.concentration_factor, a.slide.sample_total_area_px2,
        est["red_only"], est["green_only"], est["yellow"], est["total_objects_exclusive"],
        est["red_positive"], est["green_positive"], cstat,
        CONC_FORMULA if cstat.startswith("computed") else "",
        _b(mapping_confirmed(a)), _b(a.provenance.stale_detection), a.finalized_at, status_label(a),
    ]
    return _csv(header, [row])


# ------------------------------------------------------------------ provenance
def provenance_dict(state: AnalysisState, originals_dir: str | Path | None = None) -> dict:
    a = state.analysis
    img = a.image
    image = None
    manifest = read_upload_manifest(originals_dir)
    if img is not None:
        image = {
            "original_upload_filenames": manifest,  # stored_name -> name as uploaded (null: no manifest)
            "tiff_info_text": img.tiff_info_text,
            "kind": img.kind.value, "height": img.height, "width": img.width, "dtype": img.dtype,
            "bit_depth": img.bit_depth, "sha256": img.sha256, "pixel_size_um": img.pixel_size_um,
            "composite_only": img.composite_only, "warnings": img.warnings,
            "embedded_rois": [r.model_dump(mode="json") for r in img.embedded_rois],
            "channel_mapping": [
                {"index": c.index, "source_file": c.source_file,
                 "original_upload_filename": (manifest or {}).get(c.source_file), "page": c.page,
                 "label_hint": c.label_hint, "role": c.role.value if c.role else None,
                 "role_confirmed": c.role_confirmed, "min": c.min, "max": c.max,
                 "saturated_fraction": c.saturated_fraction}
                for c in img.channels
            ],
        }
    prov = a.provenance.model_dump(mode="json")
    counts = compute_counts(a)
    return {
        "status": status_label(a),
        "exported_at": now(),
        "schema_version": SCHEMA_VERSION,
        "app_version": polony_app.__version__,
        "analysis_id": a.id,
        "analysis_name": a.name,
        "finalized_at": a.finalized_at,
        "reporting_mode": a.reporting_mode.value,
        "provenance": prov,
        "detection_parameters": a.detection_request.model_dump(mode="json"),
        "detection_parameters_used": prov.get("detection_request"),
        "detection_inputs_recorded": state.detection_sig,
        "detection_is_stale": compute_stale(a, state.detection_sig, a.provenance.stale_detection),
        # same key names in both: distribution names (see canonical_versions)
        "package_versions_at_detection": canonical_versions(a.provenance.package_versions),
        "package_versions_at_export": package_versions(),
        "image": image,
        "alignment_applied_to_green_px": state.alignment,
        "regions": [r.model_dump(mode="json") for r in a.regions],
        "slide": a.slide.model_dump(mode="json"),
        "counts": {"regions": [c.model_dump(mode="json") for c in counts["regions"]],
                   "slide": counts["slide"].model_dump(mode="json")},
        "action_log_summary": action_log_summary(state),
        "events": list(state.events),  # complete: mapping/alignment/settings/undo/redo/reopen, never truncated
        "files_note": {
            "actions.jsonl": "full review-action log (seq, ts, type, payload); the undo/redo "
                             "'inverse' restore snapshots are omitted (large, redundant with the "
                             "payloads + objects_history.csv)",
            "objects_history.csv": "per-object history trail, including deleted objects",
        },
        "coordinate_convention": "pixel coordinates, x=column, y=row, pixel (0,0) centre = (0.0, 0.0)",
    }


def provenance_json(state: AnalysisState, originals_dir: str | Path | None = None) -> str:
    return json.dumps(provenance_dict(state, originals_dir), indent=2, sort_keys=False, default=str)


def actions_jsonl(state: AnalysisState) -> str:
    """Full ReviewAction log, one JSON object per line. Inverse (undo/redo) snapshots omitted."""
    lines = []
    status = status_label(state.analysis)
    for act in state.log:
        lines.append(json.dumps(
            {"seq": act.seq, "ts": act.ts, "type": act.type.value, "payload": act.payload,
             "inverse_omitted": True, "status": status},
            sort_keys=False, default=str))
    return "\n".join(lines) + ("\n" if lines else "")


def objects_history_csv(a: Analysis) -> str:
    """Every history entry of every object (deleted ones included, flagged)."""
    rows = []
    status = status_label(a)
    for o in a.objects:
        for i, h in enumerate(o.history):
            ts, _, msg = h.partition(" ")
            if "T" not in ts:  # entry without a leading timestamp
                ts, msg = "", h
            rows.append([o.id, i + 1, ts, msg, _b(o.deleted), o.origin.value, status])
    return _csv(["object_id", "step", "timestamp", "event", "object_deleted_now", "origin", "status"], rows)


# ------------------------------------------------------------------ bundle
def readme_text(a: Analysis) -> str:
    final = a.finalized_at is not None
    banner = (
        f"FINAL - finalized at {a.finalized_at}. All counting regions were approved by a user."
        if final
        else "DRAFT - NOT FINAL. This analysis has not been finalized; counts may still change."
    )
    return f"""{banner}

Polony counting export for analysis "{a.name}" (id {a.id})
Generated {now()} by polony_app {polony_app.__version__}.

FILES
  objects_<S>.csv one row per non-deleted object, with per-channel evidence
  regions_<S>.csv per-region counts, areas, approval status
  slide_<S>.csv   slide totals and (only if inputs are complete) concentration
                  (<S> is DRAFT or FINAL, the same status rule as the downloaded file names
                  <name>_<kind>_<DRAFT|FINAL>.<ext>)
  provenance.json parameters, image hash/shape/bit depth, TIFF info text, original upload
                  file names, channel mapping, regions, package versions at detection and at
                  export, recorded detection inputs, reporting mode, schema version,
                  action-log summary and the complete event list
  actions.jsonl   complete review-action log (seq, ts, type, payload). The undo/redo restore
                  snapshots ("inverse") are omitted (marked inverse_omitted)
  objects_history.csv  history trail of every object, deleted ones included

CONVENTIONS
  * Coordinates are image pixels; x = column, y = row; pixel (0,0) centre is (0.0, 0.0).
  * Classes: red = red-only, green = green-only, yellow = dual-positive (a red and a green
    spot associated one-to-one: each detected spot is used at most once in the one-to-one
    association; objects may still be duplicates or splits that need review), uncertain = unresolved.
  * `uncertain` objects are NEVER included in any colour total; they are reported separately.
  * Both counting conventions are always given with unambiguous names:
      exclusive : red_only, green_only, yellow
      inclusive : red_positive_incl_yellow = red_only + yellow,
                  green_positive_incl_yellow = green_only + yellow, yellow_overlap = yellow
    The analysis' selected reporting mode is "{a.reporting_mode.value}".
  * Soft-deleted objects are not exported. Objects outside every counting region (or inside
    an exclusion) have counted=false and are not in any count.
  * Channel colours come only from the user-confirmed channel mapping (see provenance.json).

CONCENTRATION  (PROPOSED formula - UNCONFIRMED, not validated)
  Estimates are filled in ONLY when ALL of these are provided: concentration_formula_confirmed
  (default false), sample_volume_ul, concentration_factor, a counting area (union of the
  counting regions net of exclusions and of overlaps between regions), sample_total_area_px2
  (the total area, in the same pixel units, that the sample volume spreads over) and a
  confirmed channel mapping. Otherwise the estimate columns are empty and concentration_status
  reads "inputs incomplete: <what is missing>". Raw counts are always present.
  Proposed formula (an assumption that the counted area is representative of the whole sample
  area; no chamber geometry, loss or edge model):
      conc_per_ml = (n_counted / counting_area_px2) * sample_total_area_px2
                    / (sample_volume_ul / 1000) * concentration_factor
  The same count in a 4x larger counting area therefore gives 1/4 of the concentration.
  Please compare with the legacy CSV convention
  abundance = mean(fields) * (field/grid) / volume_ul * 1000.

UNCERTAINTY
  No confidence interval is provided (the estimates are point values of an unconfirmed
  formula). Only Poisson counting uncertainty could be given and it is not provided yet; no
  interval here accounts for missed detections, false detections, debris or area/volume errors.

STATUS
  Every CSV row, every actions.jsonl line and provenance.json carry status={status_label(a)};
  the file names (downloads and the three CSVs in this bundle) carry _DRAFT or _FINAL.
"""


def bundle_zip(state: AnalysisState, originals_dir: str | Path | None = None) -> bytes:
    a = state.analysis
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", readme_text(a))
        z.writestr(member_name(a, "objects"), objects_csv(a))
        z.writestr(member_name(a, "regions"), regions_csv(a))
        z.writestr(member_name(a, "slide"), slide_csv(a))
        z.writestr("provenance.json", provenance_json(state, originals_dir))
        z.writestr("actions.jsonl", actions_jsonl(state))
        z.writestr("objects_history.csv", objects_history_csv(a))
    return buf.getvalue()
