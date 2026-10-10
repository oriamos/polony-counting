"""Two-engine consensus review helpers (pure; no I/O).

Principles (docs/app/TWO_ENGINE_CONTRACT.md, docs/app/REVIEW_POLICY.md):
  * agreement is NOT truth; nothing is ever auto-rejected or auto-deleted because one engine missed it;
  * `human_added` is stored on the channel evidence of a manually added object (always true by definition);
  * `human_rejected` is DERIVED here (a candidate whose object the user soft-deleted with the `delete`
    action = tombstone) and never stored on the object;
  * the QC sample is a deterministic function of (seed, object id): no RNG state.
"""
from __future__ import annotations

import hashlib

from polony_app.schemas import (
    Analysis,
    ActionType,
    ConsensusStatus,
    ConsensusSummary,
    DetectedObject,
    ObjectClass,
    ReviewReason,
)

from .counts import counted_objects
from .state import AnalysisState, ReviewError

_STATUS_FIELDS = ("agreement", "bigfish_only", "spotiflow_only", "human_added")


def _channel_pages(a: Analysis) -> list[tuple[str, int | None]]:
    out = []
    for role in ("red", "green"):
        page = None
        for c in (a.image.channels if a.image else []):
            if c.role is not None and c.role.value == role and c.role_confirmed:
                page = c.index
        out.append((role, page))
    return out


def _is_candidate(ev) -> bool:
    return bool(ev.detected) or ev.consensus_status is not None


def consensus_summary(state: AnalysisState) -> dict:
    """{pages:[ConsensusSummary per page/channel], total: ConsensusSummary}.

    Per channel candidate (an object's red/green evidence that an engine detected or a human added):
      status counts  agreement | bigfish_only | spotiflow_only | human_added  - live (non-deleted) candidates;
      human_rejected - candidates whose object the user deleted (tombstone; derived);
      confirmed      - live and reviewed;  rejected - user-deleted;
      unresolved     - live and (unreviewed or class uncertain).
    Objects deleted by merge/split (not a user decision) are not counted. Channels are never merged
    silently: `total` is only the sum of the per-channel rows."""
    a = state.analysis
    user_deleted = set(state.user_deleted)
    rows = {role: ConsensusSummary(page=page, channel_role=role) for role, page in _channel_pages(a)}
    for o in a.objects:
        rejected = o.deleted and o.id in user_deleted
        if o.deleted and not rejected:
            continue
        for role, ev in (("red", o.red), ("green", o.green)):
            if not _is_candidate(ev):
                continue
            r = rows[role]
            if rejected:
                r.human_rejected += 1
                r.rejected += 1
                continue
            if ev.consensus_status is not None and ev.consensus_status.value in _STATUS_FIELDS:
                setattr(r, ev.consensus_status.value, getattr(r, ev.consensus_status.value) + 1)
            if o.reviewed and o.cls != ObjectClass.uncertain:
                r.confirmed += 1
            else:
                r.unresolved += 1
    total = ConsensusSummary()
    for r in rows.values():
        for f in (*_STATUS_FIELDS, "human_rejected", "confirmed", "rejected", "unresolved"):
            setattr(total, f, getattr(total, f) + getattr(r, f))
    return {"pages": [r.model_dump(mode="json") for r in rows.values()], "total": total.model_dump(mode="json")}


def qc_score(seed: int | str, object_id: str) -> float:
    """Deterministic value in [0,1) from (seed, object id)."""
    h = hashlib.sha256(f"{seed}:{object_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def qc_eligible(o: DetectedObject) -> bool:
    """Non-deleted, unreviewed, and EVERY detected channel evidence is an engine agreement."""
    if o.deleted or o.reviewed:
        return False
    evs = [ev for ev in (o.red, o.green) if ev.detected]
    return bool(evs) and all(ev.consensus_status == ConsensusStatus.agreement for ev in evs)


def mark_qc_sample(state: AnalysisState, fraction: float, seed: int | str) -> dict:
    """Add ReviewReason.qc_sample to a deterministic sample of eligible agreement objects.
    Idempotent (already marked objects are kept, never duplicated); one undoable action, logged as
    ActionType.region_edit with payload.event == 'qc_sample'. Only ADDS a review reason."""
    from . import engine as eng

    if not (isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and 0.0 <= fraction <= 1.0):
        raise ReviewError("invalid_fraction", "fraction must be a number in 0..1", 422)
    eng.guard_editable(state)
    a = state.analysis
    eligible = [o for o in counted_objects(a) if qc_eligible(o)]
    selected = [o for o in eligible if qc_score(seed, o.id) < fraction]
    tx = eng.Tx(state)
    newly = []
    for o in selected:
        if ReviewReason.qc_sample in o.review_reasons:
            continue
        tx.touch_obj(o)
        o.review_reasons = [*o.review_reasons, ReviewReason.qc_sample]
        newly.append(o.id)
    payload = {"event": "qc_sample", "fraction": fraction, "seed": seed, "eligible": len(eligible),
               "selected": len(selected), "newly_marked": len(newly), "object_ids": sorted(o.id for o in selected)}
    if newly:
        tx.commit(ActionType.region_edit, payload)
    return {"marked": len(selected), "newly_marked": len(newly), "eligible": len(eligible),
            "fraction": fraction, "seed": seed}
