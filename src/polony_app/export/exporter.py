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
from polony_app.export import lab
from polony_app.review.consensus import QC_NOTE, qc_status
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
# per-channel two-engine columns (empty for Big-FISH-only runs). Score wording: Big-FISH = LoG response in
# noise units; Spotiflow = raw "model score". Neither is a probability or a confidence percentage.
ENGINE_COLS = ["consensus_status", "bigfish_x", "bigfish_y", "bigfish_loG_noise_units",
               "spotiflow_x", "spotiflow_y", "spotiflow_model_score", "engine_match_distance_px"]


def raw_detections_csv(detections) -> str:
    """Every raw engine detection (valid and excluded), exactly as the engines produced it."""
    header = ["id", "run_id", "engine", "page", "channel_role", "x", "y", "z", "radius_px",
              "model_score", "loG_noise_units", "score_kind", "valid", "region_id", "excluded_reason"]
    rows = []
    for d in detections:
        eng = d.engine.value
        rows.append([d.id, d.run_id, eng, d.page, d.channel_role, d.x, d.y, d.z, d.radius_px,
                     d.score if eng == "spotiflow" else None, d.score if eng == "bigfish" else None,
                     d.score_kind, _b(d.valid), d.region_id, d.excluded_reason])
    return _csv(header, rows)


def engine_runs_json(runs) -> str:
    return json.dumps([r.model_dump(mode="json") for r in runs], indent=2, default=str)


def objects_csv(a: Analysis) -> str:
    status = status_label(a)
    header = ["id", "x", "y", "class", "origin", "region_id", "counted"]
    for ch in ("red", "green"):
        header += [f"{ch}_detected", f"{ch}_x", f"{ch}_y", f"{ch}_peak", f"{ch}_local_background",
                   f"{ch}_snr", f"{ch}_saturated"]
        header += [f"{ch}_{c}" for c in ENGINE_COLS]
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
            row += [ev.consensus_status.value if ev.consensus_status is not None else None,
                    ev.bigfish_x, ev.bigfish_y, ev.bigfish_score,
                    ev.spotiflow_x, ev.spotiflow_y, ev.spotiflow_score, ev.match_distance_px]
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
        "total_objects_exclusive", "red_positive_excl_debris", "green_positive_excl_debris",
        "reporting_mode", "inspected", "approval_status", "approved_at", "status",
    ]
    rows = []
    for r in a.regions:
        area = region_area_px2(a, r)
        c = counts.get(r.id)
        if r.kind == RegionKind.counting and c is not None:
            rows.append([
                r.id, r.name, r.kind.value, area, c.red_only, c.green_only, c.yellow, c.uncertain,
                c.red_positive, c.green_positive, c.yellow, c.total_objects_exclusive,
                c.red_positive_excl_debris, c.green_positive_excl_debris,
                a.reporting_mode.value, _b(r.inspected),
                "approved" if r.approved else "not_approved", r.approved_at, status,
            ])
        else:
            rows.append([r.id, r.name, r.kind.value, area, None, None, None, None, None, None, None,
                         None, None, None, a.reporting_mode.value, None, "n/a", None, status])
    return _csv(header, rows)


# ------------------------------------------------------------------ slide
CONC_FORMULA = (
    "LAB FORMULA (as in the researcher's T7 sheets): abundance_per_ml = mean(field counts) * "
    "(field_area / grid_area) / sample_volume_ul * 1000, where the mean is over the lab fields "
    "(counting regions with a field_index; blank = no region, 0 = region with zero objects) and the count "
    "type of each series is the one chosen by the user (lab_series). The concentration factor is NOT "
    "part of this formula (it is exported as a column; the Counts-style workbook multiplies by it); "
    "no confidence interval"
)


def concentration(a: Analysis) -> tuple[dict[str, dict], str, list[str]]:
    """Lab abundance per configured series. Returns ({series: {...}}, status text, missing inputs).

    Never invents inputs. ALL of these are required, otherwise every estimate cell is empty and
    the status lists what is missing: concentration_formula_confirmed (= field geometry and
    series mapping confirmed by the researcher), sample_volume_ul, field_area, grid_area,
    at least one lab_series, at least one counting region with a valid field_index (1..24, no
    duplicates), confirmed channel mapping. `concentration_factor`, `sample_total_area_px2` and
    `counting_area_px2` are not used by the lab formula."""
    s = a.slide
    missing = []
    if not s.concentration_formula_confirmed:
        missing.append("concentration_formula_confirmed")
    if not s.sample_volume_ul or s.sample_volume_ul <= 0:
        missing.append("sample_volume_ul")
    if not s.field_area or s.field_area <= 0:
        missing.append("field_area")
    if not s.grid_area or s.grid_area <= 0:
        missing.append("grid_area")
    if not s.lab_series:
        missing.append("lab_series")
    est: dict[str, dict] = {}
    warns: list[str] = []
    n_field_regions = 0
    for ser in s.lab_series:
        fields, table, w, probs = lab.field_counts(a, ser.count_type)
        n_field_regions = max(n_field_regions, len(table))
        if probs:
            missing.append(f"valid_field_index ({'; '.join(probs)})")
        warns.extend(w)
        est[ser.name] = {"count_type": ser.count_type.value, "fields": fields, "table": table,
                         "n_fields": sum(1 for v in fields if v is not None), "abundance_per_ml": None}
    if s.lab_series and not any(e["n_fields"] for e in est.values()) and not any(
            m.startswith("valid_field_index") for m in missing):
        missing.append("field_regions")
    if not mapping_confirmed(a):
        missing.append("confirmed_channel_mapping")
    if missing:
        if missing[0] == "concentration_formula_confirmed":
            msg = "inputs incomplete: formula/geometry not confirmed by researcher"
            if len(missing) > 1:
                msg += "; also missing: " + ", ".join(missing[1:])
        else:
            msg = "inputs incomplete: " + ", ".join(missing)
        return est, msg, missing
    for e in est.values():
        vals = [v for v in e["fields"] if v is not None]
        if vals:
            e["abundance_per_ml"] = (sum(vals) / len(vals)) * (s.field_area / s.grid_area) / s.sample_volume_ul * 1000
    msg = "computed (lab formula; no confidence interval)"
    if warns:
        msg += f"; {len(warns)} warning(s): " + " | ".join(warns)
    return est, msg, []


def slide_csv(a: Analysis) -> str:
    c = compute_counts(a)["slide"]
    regs = [r for r in a.regions if r.kind == RegionKind.counting]
    est, cstat, missing = concentration(a)
    ok = not missing
    pix = a.slide.pixel_size_um or (a.image.pixel_size_um if a.image else None)
    header = [
        "slide_id", "analysis_id", "reporting_mode", "red_only", "green_only", "yellow", "uncertain",
        "red_positive_incl_yellow", "green_positive_incl_yellow", "yellow_overlap",
        "total_objects_exclusive", "red_positive_excl_debris", "green_positive_excl_debris",
        "n_counting_regions", "n_regions_approved", "counting_area_px2",
        "pixel_size_um", "sample_volume_ul", "concentration_factor", "field_area", "grid_area",
        "lab_series", "n_field_regions",
    ]
    series_cols = []
    for ser in a.slide.lab_series:
        series_cols += [f"n_fields_{ser.name}", f"abundance_per_ml_{ser.name}"]
    header += series_cols + [
        "concentration_status", "concentration_formula", "channel_mapping_confirmed", "stale_detection",
        "finalized_at", "status",
    ]
    row = [
        a.slide.slide_id, a.id, a.reporting_mode.value, c.red_only, c.green_only, c.yellow, c.uncertain,
        c.red_positive, c.green_positive, c.yellow, c.total_objects_exclusive,
        c.red_positive_excl_debris, c.green_positive_excl_debris, len(regs),
        sum(1 for r in regs if r.approved), counting_area_px2(a), pix, a.slide.sample_volume_ul,
        a.slide.concentration_factor, a.slide.field_area, a.slide.grid_area,
        "|".join(f"{x.name}={x.count_type.value}" for x in a.slide.lab_series),
        sum(1 for r in regs if r.field_index is not None),
    ]
    for ser in a.slide.lab_series:
        e = est.get(ser.name, {})
        row += [e.get("n_fields"), e.get("abundance_per_ml") if ok else None]
    row += [cstat, CONC_FORMULA if ok else "", _b(mapping_confirmed(a)), _b(a.provenance.stale_detection),
            a.finalized_at, status_label(a)]
    return _csv(header, [row])


# ------------------------------------------------------------------ provenance
SCORE_NOTE = ("Big-FISH score = LoG response in noise units; Spotiflow score = raw model score. Neither is a "
              "probability or a confidence percentage. Agreement between the engines is not ground truth.")


def provenance_dict(state: AnalysisState, originals_dir: str | Path | None = None, engine_runs=None) -> dict:
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
        "lab": lab_provenance(a),
        "counts": {"regions": [c.model_dump(mode="json") for c in counts["regions"]],
                   "slide": counts["slide"].model_dump(mode="json")},
        "engine_runs": [r.model_dump(mode="json") for r in (engine_runs or [])],  # model name/version/checksum, params, preprocessing, device, errors
        "engine_scores_note": SCORE_NOTE,
        # agreement QC: confirmed/rejected among the SAMPLED agreements + exact 95 % interval (null until reviewed)
        "agreement_qc": qc_status(state),
        "agreement_review_note": QC_NOTE,
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


def lab_provenance(a: Analysis) -> dict:
    """Lab-export inputs: the series -> count type mapping the user chose, the field -> region
    table per series and the abundance (or why it is withheld)."""
    est, status, missing = concentration(a)
    return {
        "formula": CONC_FORMULA,
        "concentration_status": status,
        "missing_inputs": missing,
        "lab_series": [x.model_dump(mode="json") for x in a.slide.lab_series],
        "series": {
            name: {
                "count_type": e["count_type"],
                "n_fields": e["n_fields"],
                "abundance_per_ml": e["abundance_per_ml"] if not missing else None,
                "fields": [{"field_index": t.field_index, "region_id": t.region_id, "region_name": t.region_name,
                            "count": t.count, "counted_area_px2": t.area_px2,
                            "counted_area_fraction": t.area_fraction, "note": t.note or None}
                           for t in e["table"]],
            }
            for name, e in est.items()
        },
        "field_regions_without_series": [r.id for r in a.regions
                                         if r.kind == RegionKind.counting and r.field_index is not None
                                         and not a.slide.lab_series],
    }


def provenance_json(state: AnalysisState, originals_dir: str | Path | None = None, engine_runs=None) -> str:
    return json.dumps(provenance_dict(state, originals_dir, engine_runs), indent=2, sort_keys=False, default=str)


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
  slide_<S>.csv   slide totals and (only if inputs are complete) the lab-formula abundance
                  (<S> is DRAFT or FINAL, the same status rule as the downloaded file names
                  <name>_<kind>_<DRAFT|FINAL>.<ext>)
  provenance.json parameters, image hash/shape/bit depth, TIFF info text, original upload
                  file names, channel mapping, regions, package versions at detection and at
                  export, recorded detection inputs, reporting mode, schema version,
                  action-log summary and the complete event list
  actions.jsonl   complete review-action log (seq, ts, type, payload). The undo/redo restore
                  snapshots ("inverse") are omitted (marked inverse_omitted)
  objects_history.csv  history trail of every object, deleted ones included
  raw_detections.csv  (only when engine runs are stored) every raw detection of every engine, valid or
                  excluded, in full-resolution pixel coordinates
  engine_runs.json (only when engine runs are stored) one record per engine run: engine/model
                  name, version, weights checksum, parameters, preprocessing, device, status, errors

TWO-ENGINE DETECTION
  Big-FISH (classical) and Spotiflow (learned) can run independently on the same page; their
  detections are matched one-to-one within a pixel tolerance. objects_<S>.csv has, per channel,
  <red|green>_consensus_status (agreement | bigfish_only | spotiflow_only | human_added) and each
  engine's own coordinates and score. AGREEMENT IS NOT TRUTH: two engines can make the same
  mistake, and an engine that missed a spot is not evidence that the spot is absent. Nothing is
  removed because one engine did not see it; disagreements are reviewed by a person first.
  AGREEMENT OBJECTS ARE NOT REVIEWED UNLESS SAMPLED. The optional agreement QC sample is a fixed number
  of agreement objects per slide (default 40), drawn deterministically from (seed, object id); objects
  carrying the review reason qc_sample are the sample. provenance.json (agreement_qc) reports, for the
  sampled agreements, confirmed vs rejected and the exact Clopper-Pearson 95 % interval of the
  agreement error rate. That interval bounds the error rate of the UNSAMPLED agreements only under
  random sampling (one seed per slide) and only for the agreements present when the sample was drawn;
  it says nothing about spots both engines missed. Agreement is not ground truth.
  A MODEL SCORE IS NOT A PROBABILITY (Spotiflow: raw model score; Big-FISH: LoG response in noise
  units): neither is calibrated and neither is a confidence percentage. human_rejected (a candidate a
  user deleted) is derived from the action log and not stored on objects.

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
  * red_positive_excl_debris / green_positive_excl_debris: convenience count; objects flagged debris_like remain counted elsewhere
    (= the positive count minus non-deleted positive objects whose review reasons include debris_like; nothing is removed).
  * Soft-deleted objects are not exported. Objects outside every counting region (or inside
    an exclusion) have counted=false and are not in any count.
  * Channel colours come only from the user-confirmed channel mapping (see provenance.json).

CONCENTRATION  (LAB FORMULA)
  The estimate columns of slide_<S>.csv follow the researcher's own T7 sheets:
      abundance_per_ml = mean(field counts) * (field_area / grid_area) / sample_volume_ul * 1000
  mean(field counts) is over the lab fields = counting regions that carry a field_index (1..24);
  a field without a region is blank and not in the mean, a region with zero objects is 0 and is.
  Each series in `lab_series` (e.g. T7A, T7B) uses the count type the USER chose for it
  (red_only / green_only / yellow / red_positive / green_positive); it is never inferred.
  The concentration factor is NOT part of this formula (as in the T7 sheets); it is exported as
  a column, and the Counts-style lab workbook (polonies/ml, column AK) multiplies by it.
  Estimates are filled in ONLY when concentration_formula_confirmed (= field geometry and series
  mapping confirmed; default false), sample_volume_ul, field_area, grid_area, lab_series, at least
  one field region and a confirmed channel mapping are present; otherwise the estimate columns
  are empty and concentration_status reads "inputs incomplete: <what is missing>".
  The field -> region table and the series mapping used are in provenance.json ("lab").
  A field region partly covered by an exclusion is counted over the remaining area only (flagged
  in concentration_status / the lab export Notes sheet); sample_total_area_px2 and
  counting_area_px2 are not used by this formula.

UNCERTAINTY
  No confidence interval is provided (the estimates are point values of an unconfirmed
  formula). Only Poisson counting uncertainty could be given and it is not provided yet; no
  interval here accounts for missed detections, false detections, debris or area/volume errors.

STATUS
  Every CSV row, every actions.jsonl line and provenance.json carry status={status_label(a)};
  the file names (downloads and the three CSVs in this bundle) carry _DRAFT or _FINAL.
"""


def bundle_zip(state: AnalysisState, originals_dir: str | Path | None = None,
               engine_runs=None, raw_detections=None) -> bytes:
    a = state.analysis
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", readme_text(a))
        z.writestr(member_name(a, "objects"), objects_csv(a))
        z.writestr(member_name(a, "regions"), regions_csv(a))
        z.writestr(member_name(a, "slide"), slide_csv(a))
        z.writestr("provenance.json", provenance_json(state, originals_dir, engine_runs))
        z.writestr("actions.jsonl", actions_jsonl(state))
        z.writestr("objects_history.csv", objects_history_csv(a))
        if engine_runs or raw_detections:
            z.writestr("raw_detections.csv", raw_detections_csv(raw_detections or []))
            z.writestr("engine_runs.json", engine_runs_json(engine_runs or []))
    return buf.getvalue()
