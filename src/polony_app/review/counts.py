"""Counting and review-queue logic (pure)."""
from __future__ import annotations

from polony_app.schemas import (
    Analysis,
    Counts,
    DetectedObject,
    ObjectClass,
    RegionKind,
    ReviewReason,
)

# lower number = reviewed first
REASON_PRIORITY = {
    ReviewReason.ambiguous_match: 0,
    ReviewReason.saturated: 1,
    ReviewReason.weak_signal: 2,
    ReviewReason.near_threshold: 2,
    ReviewReason.debris_like: 3,   # below saturated/weak, above dense_cluster: every flagged object must be queued
    ReviewReason.dense_cluster: 4,
    ReviewReason.suspicious_shape: 5,
    ReviewReason.near_edge: 6,
}
# Every ReviewReason must have a priority: an object whose ONLY reason is missing here would
# never reach the review queue (this was the case for debris_like).
UNCERTAIN_PRIORITY = 0


def _tally(objs, scope: str, analysis: Analysis) -> Counts:
    r = g = y = u = 0
    for o in objs:
        if o.deleted:
            continue
        if o.cls == ObjectClass.red:
            r += 1
        elif o.cls == ObjectClass.green:
            g += 1
        elif o.cls == ObjectClass.yellow:
            y += 1
        else:
            u += 1
    return Counts(
        scope=scope,
        red_only=r,
        green_only=g,
        yellow=y,
        uncertain=u,
        red_positive=r + y,
        green_positive=g + y,
        total_objects_exclusive=r + g + y,
        reporting_mode=analysis.reporting_mode,
        is_final=analysis.finalized_at is not None,
    )


def counted_objects(analysis: Analysis) -> list[DetectedObject]:
    """Non-deleted objects that count towards slide totals: those inside a
    counting region (region_id set). With no counting regions defined, every
    non-deleted object counts."""
    has_regions = any(r.kind == RegionKind.counting for r in analysis.regions)
    out = []
    for o in analysis.objects:
        if o.deleted:
            continue
        if has_regions and o.region_id is None:
            continue
        out.append(o)
    return out


def compute_counts(analysis: Analysis) -> dict:
    regions = [
        _tally([o for o in analysis.objects if o.region_id == r.id], r.id, analysis)
        for r in analysis.regions
        if r.kind == RegionKind.counting
    ]
    slide = _tally(counted_objects(analysis), "slide", analysis)
    return {"regions": regions, "slide": slide}


def object_priority(o: DetectedObject) -> int | None:
    pr = [REASON_PRIORITY[r] for r in o.review_reasons if r in REASON_PRIORITY]
    if o.cls == ObjectClass.uncertain:
        pr.append(UNCERTAIN_PRIORITY)
    return min(pr) if pr else None


def review_queue(analysis: Analysis) -> list[dict]:
    items = []
    for o in counted_objects(analysis):
        if o.reviewed and o.cls != ObjectClass.uncertain:
            continue
        p = object_priority(o)
        if p is None:
            continue
        reasons = [r.value for r in o.review_reasons]
        if o.cls == ObjectClass.uncertain and "uncertain" not in reasons:
            reasons = ["uncertain"] + reasons
        items.append(
            {"object_id": o.id, "reasons": reasons, "priority": p, "x": o.x, "y": o.y}
        )
    items.sort(key=lambda d: (d["priority"], d["y"], d["x"], d["object_id"]))
    return items
