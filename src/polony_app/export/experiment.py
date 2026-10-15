"""One CSV + one provenance JSON for a whole experiment.

CSV layout (decided default, plan 3.5/3.6): the researcher's own ``experiment_counts.csv`` columns from
``slide_save`` (``Slide name, What counted, NofPol, NofGrids, Field1..Field24``; one row per slide per count
series) followed by ``Experiment``, ``Status`` (finalized / approved / draft) and ``Engine``. Same field semantics:
a counted field without polonies is 0, a field that is not counted is blank. Only slides with a detection result
and at least one numbered grid field appear; the others are listed in the provenance as skipped.
Switching to another layout later is a change in this module only.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import polony_app
from polony_app.review.engine import now
from polony_app.review.state import ReviewError
from polony_app.schemas import SCHEMA_VERSION, Analysis, Experiment, ObjectClass, RegionKind

from . import slide_save as ss

HEADER = ss.EXPERIMENT_HEADER + ["Experiment", "Status", "Engine"]
DEFAULT_SERIES = ("red", "green", "yellow")   # one row each: red-only, green-only, dual-positive


def parse_series(values: Optional[Iterable[str]]) -> list[tuple[str, tuple[ObjectClass, ...]]]:
    """``["red", "green,yellow"]`` -> [(label, classes)]; each value is one count series (classes summed)."""
    out: list[tuple[str, tuple[ObjectClass, ...]]] = []
    for v in (list(values) if values else list(DEFAULT_SERIES)):
        classes = ss.parse_classes([c.strip() for c in str(v).split(",") if c.strip()])
        label = ss.default_label(classes)
        if label in (x[0] for x in out):
            raise ReviewError("invalid_selection", f"count series '{label}' is given twice", 422)
        out.append((label, classes))
    return out


def slide_status(a: Analysis) -> str:
    """finalized / approved (every counting region approved, not finalized) / draft."""
    if a.finalized_at is not None:
        return "finalized"
    regs = [r for r in a.regions if r.kind == RegionKind.counting]
    return "approved" if regs and all(r.approved for r in regs) else "draft"


def slide_engine(a: Analysis) -> str:
    return a.provenance.engine if a.provenance.detection_ran_at is not None else ""


def has_counts(a: Analysis) -> bool:
    return a.provenance.detection_ran_at is not None


def build_csv(exp: Experiment, analyses: Sequence[Analysis],
              series: list[tuple[str, tuple[ObjectClass, ...]]]) -> tuple[str, list[dict], list[dict]]:
    """(CSV text, included slides, skipped slides with the reason)."""
    rows: list[list] = []
    included, skipped = [], []
    for a in analyses:
        if not has_counts(a):
            skipped.append({"analysis_id": a.id, "name": a.name, "reason": "no_detection"})
            continue
        try:
            per = [(lab, ss.count_fields(a, cls)) for lab, cls in series]
        except ReviewError as e:   # no numbered grid field / duplicate / out of range
            skipped.append({"analysis_id": a.id, "name": a.name, "reason": e.code, "message": e.message})
            continue
        status, eng, name = slide_status(a), slide_engine(a), ss.slide_name(a)
        for lab, counts in per:
            rows.append(ss.experiment_row(name, lab, counts) + [exp.name, status, eng])
        included.append({"analysis_id": a.id, "name": a.name, "slide_name": name, "status": status})
    return ss._csv(HEADER, rows), included, skipped


def _grid(a: Analysis):
    g = getattr(a.slide, "grid", None)   # stage 1 GridSpec; absent on older installs
    if g is None:
        return None
    return g.model_dump(mode="json") if hasattr(g, "model_dump") else g


def provenance(exp: Experiment, slides: Sequence[tuple[Analysis, list]], batch: Optional[dict],
               series: list[tuple[str, tuple[ObjectClass, ...]]], included: list[dict], skipped: list[dict]) -> dict:
    """``slides``: (analysis, engine runs) in experiment order; ``batch``: the GET /run view of the last batch."""
    per_batch = {s["analysis_id"]: s for s in (batch or {}).get("slides", [])}
    bjob = (batch or {}).get("batch") or None
    out_slides = []
    for a, runs in slides:
        b = per_batch.get(a.id)
        out_slides.append({
            "analysis_id": a.id, "name": a.name, "slide_name": ss.slide_name(a),
            "order_in_experiment": a.order_in_experiment, "status": slide_status(a), "finalized_at": a.finalized_at,
            "image_sha256": dict(a.image.sha256) if a.image else {},
            "grid": _grid(a),
            "counted_fields": sorted(r.field_index for r in a.regions
                                     if r.kind == RegionKind.counting and r.field_index is not None),
            "skipped_fields": sorted(s.field_index for s in a.slide.skipped_fields),
            "engine": slide_engine(a), "detection_ran_at": a.provenance.detection_ran_at,
            "detection_request": (a.provenance.detection_request.model_dump(mode="json")
                                  if a.provenance.detection_request else None),
            "engine_runs": [{"run_id": r.run_id, "engine": r.engine.value, "page": r.page,
                             "channel_role": r.channel_role, "model_name": r.model_name,
                             "model_version": r.model_version, "model_checksum": r.model_checksum,
                             "params": r.params, "device": r.device, "status": r.status, "error": r.error,
                             "started_at": r.started_at} for r in runs],
            "batch": ({"batch_job_id": bjob["id"] if bjob else None, "job_id": b.get("job_id"),
                       "status": b.get("status"), "error": b.get("error")} if b else None),
        })
    return {
        "exported_at": now(), "app_version": polony_app.__version__, "schema_version": SCHEMA_VERSION,
        "experiment": {"id": exp.id, "name": exp.name, "notes": exp.notes, "created_at": exp.created_at,
                       "updated_at": exp.updated_at, "default_grid": exp.default_grid,
                       "default_detection": exp.default_detection.model_dump(mode="json")},
        "csv": {"header": HEADER, "series": [{"what_counted": lab, "classes": [c.value for c in cls]}
                                             for lab, cls in series],
                "included_slides": included, "skipped_slides": skipped},
        "last_batch": ({"job_id": bjob["id"], "status": bjob["status"], "message": bjob["message"],
                        "error": bjob["error"], "started_at": bjob["started_at"], "finished_at": bjob["finished_at"],
                        "summary": batch.get("summary")} if bjob else None),
        "slides": out_slides,
    }
