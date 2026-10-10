"""Exports: objects.csv, regions.csv, slide.csv, provenance.json, bundle.zip."""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from importlib import metadata

import polony_app
from polony_app.review.counts import compute_counts, counted_objects
from polony_app.review.engine import (
    action_log_summary,
    counting_area_px2,
    mapping_confirmed,
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


def export_filename(a: Analysis, kind: str) -> str:
    ext = {"objects": "csv", "regions": "csv", "slide": "csv", "provenance": "json", "bundle": "zip"}[kind]
    suffix = "" if a.finalized_at is not None else "_DRAFT"
    return f"{safe_stem(a.name)}_{kind}{suffix}.{ext}"


def package_versions() -> dict[str, str]:
    out = {"polony_app": polony_app.__version__}
    for p in _PKGS:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


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
def concentration(a: Analysis, counts) -> tuple[dict[str, float | None], str, list[str]]:
    """Return (estimates, status text, missing inputs). Never invents inputs."""
    s = a.slide
    area = counting_area_px2(a)
    missing = []
    if not s.concentration_formula_confirmed:
        keys0 = ["red_only", "green_only", "yellow", "total_objects_exclusive", "red_positive", "green_positive"]
        return ({k: None for k in keys0},
                "inputs incomplete: formula/geometry not confirmed by researcher", ["concentration_formula_confirmed"])
    if not s.sample_volume_ul or s.sample_volume_ul <= 0:
        missing.append("sample_volume_ul")
    if not s.concentration_factor or s.concentration_factor <= 0:
        missing.append("concentration_factor")
    if not area or area <= 0:
        missing.append("counting_area_px2")
    if not mapping_confirmed(a):
        missing.append("confirmed_channel_mapping")
    keys = ["red_only", "green_only", "yellow", "total_objects_exclusive", "red_positive", "green_positive"]
    if missing:
        return {k: None for k in keys}, "inputs incomplete: " + ", ".join(missing), missing
    vol_ml = s.sample_volume_ul / 1000.0
    est = {k: getattr(counts, k) / vol_ml * s.concentration_factor for k in keys}
    msg = "computed"
    if counts.uncertain:
        msg += f"; {counts.uncertain} unresolved uncertain object(s) excluded"
    return est, msg, []


CONC_FORMULA = (
    "PROPOSED (UNCONFIRMED assumption): count / (sample_volume_ul / 1000) * concentration_factor, i.e. the counting region(s) are "
    "taken to represent the whole sample volume (assumption; no chamber geometry is modelled)"
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
        "pixel_size_um", "sample_volume_ul", "concentration_factor",
        "est_conc_per_ml_red_only", "est_conc_per_ml_green_only", "est_conc_per_ml_yellow",
        "est_conc_per_ml_total_exclusive", "est_conc_per_ml_red_positive", "est_conc_per_ml_green_positive",
        "concentration_status", "concentration_formula", "channel_mapping_confirmed", "stale_detection",
        "finalized_at", "status",
    ]
    row = [
        a.slide.slide_id, a.id, a.reporting_mode.value, c.red_only, c.green_only, c.yellow, c.uncertain,
        c.red_positive, c.green_positive, c.yellow, c.total_objects_exclusive, len(regs),
        sum(1 for r in regs if r.approved), counting_area_px2(a), pix, a.slide.sample_volume_ul,
        a.slide.concentration_factor,
        est["red_only"], est["green_only"], est["yellow"], est["total_objects_exclusive"],
        est["red_positive"], est["green_positive"], cstat,
        CONC_FORMULA if not cstat.startswith("inputs incomplete") else "",
        _b(mapping_confirmed(a)), _b(a.provenance.stale_detection), a.finalized_at, status_label(a),
    ]
    return _csv(header, [row])


# ------------------------------------------------------------------ provenance
def provenance_dict(state: AnalysisState) -> dict:
    a = state.analysis
    img = a.image
    image = None
    if img is not None:
        image = {
            "kind": img.kind.value, "height": img.height, "width": img.width, "dtype": img.dtype,
            "bit_depth": img.bit_depth, "sha256": img.sha256, "pixel_size_um": img.pixel_size_um,
            "composite_only": img.composite_only, "warnings": img.warnings,
            "embedded_rois": [r.model_dump(mode="json") for r in img.embedded_rois],
            "channel_mapping": [
                {"index": c.index, "source_file": c.source_file, "page": c.page,
                 "label_hint": c.label_hint, "role": c.role.value if c.role else None,
                 "role_confirmed": c.role_confirmed, "min": c.min, "max": c.max,
                 "saturated_fraction": c.saturated_fraction}
                for c in img.channels
            ],
        }
    prov = a.provenance.model_dump(mode="json")
    versions = dict(package_versions())
    versions.update(a.provenance.package_versions or {})
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
        "package_versions": versions,
        "image": image,
        "alignment_applied_to_green_px": state.alignment,
        "regions": [r.model_dump(mode="json") for r in a.regions],
        "slide": a.slide.model_dump(mode="json"),
        "counts": {"regions": [c.model_dump(mode="json") for c in counts["regions"]],
                   "slide": counts["slide"].model_dump(mode="json")},
        "action_log_summary": action_log_summary(state),
        "coordinate_convention": "pixel coordinates, x=column, y=row, pixel (0,0) centre = (0.0, 0.0)",
    }


def provenance_json(state: AnalysisState) -> str:
    return json.dumps(provenance_dict(state), indent=2, sort_keys=False, default=str)


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
  objects*.csv    one row per non-deleted object, with per-channel evidence
  regions*.csv    per-region counts, areas, approval status
  slide*.csv      slide totals and (only if inputs are complete) concentration
  provenance.json parameters, image hash/shape/bit depth, channel mapping, regions,
                  package versions, reporting mode, schema version, action-log summary

CONVENTIONS
  * Coordinates are image pixels; x = column, y = row; pixel (0,0) centre is (0.0, 0.0).
  * Classes: red = red-only, green = green-only, yellow = dual-positive (matched red+green
    evidence, each physical object counted once), uncertain = unresolved.
  * `uncertain` objects are NEVER included in any colour total; they are reported separately.
  * Both counting conventions are always given with unambiguous names:
      exclusive : red_only, green_only, yellow
      inclusive : red_positive_incl_yellow = red_only + yellow,
                  green_positive_incl_yellow = green_only + yellow, yellow_overlap = yellow
    The analysis' selected reporting mode is "{a.reporting_mode.value}".
  * Soft-deleted objects are not exported. Objects outside every counting region (or inside
    an exclusion) have counted=false and are not in any count.
  * Channel colours come only from the user-confirmed channel mapping (see provenance.json).

CONCENTRATION
  Estimates are filled in ONLY when sample_volume_ul, concentration_factor and a counting
  area are provided and the channel mapping is confirmed; otherwise the columns are empty and
  concentration_status reads "inputs incomplete: <missing>". Raw counts are always present.
  Concentration output is gated by SlideMeta.concentration_formula_confirmed (default false): until
  the researcher confirms the formula and geometry the columns are empty with status
  "inputs incomplete: formula/geometry not confirmed by researcher".
  Proposed formula (UNCONFIRMED assumption: the counted region represents the whole sample
  volume): count / (sample_volume_ul / 1000) * concentration_factor.
  Please compare with the legacy CSV convention
  abundance = mean(fields) * (field/grid) / volume_ul * 1000.

UNCERTAINTY
  No confidence interval is provided. Only Poisson counting uncertainty could be given and it
  is not provided yet; no interval here accounts for missed detections or false detections.

STATUS
  Every CSV row carries status={status_label(a)}; file names carry _DRAFT until finalized.
"""


def bundle_zip(state: AnalysisState) -> bytes:
    a = state.analysis
    suffix = "" if a.finalized_at is not None else "_DRAFT"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", readme_text(a))
        z.writestr(f"objects{suffix}.csv", objects_csv(a))
        z.writestr(f"regions{suffix}.csv", regions_csv(a))
        z.writestr(f"slide{suffix}.csv", slide_csv(a))
        z.writestr("provenance.json", provenance_json(state))
    return buf.getvalue()
