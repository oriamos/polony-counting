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
    return {"pages": [r.model_dump(mode="json") for r in rows.values()], "total": total.model_dump(mode="json"),
            "qc": qc_status(state)}


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


QC_DEFAULT_N = 40                 # fixed per-slide sample size (NOT a fraction of the agreements)
QC_PLANNING_P = 0.153             # pooled agreement false-positive rate measured on the 7 research slides
QC_MARGIN = 0.05
QC_NOTE = ("agreement is not ground truth. Agreement objects are NOT reviewed unless sampled; the interval bounds the "
           "error rate of the UNSAMPLED agreements only if the sample is random (it is a deterministic function of "
           "seed and object id, taken once per slide with a single seed) and only for the agreements present when it "
           "was drawn.")


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval for k errors in n reviewed objects."""
    from scipy.stats import beta

    if n <= 0:
        return 0.0, 1.0
    a = (1.0 - conf) / 2.0
    lo = 0.0 if k <= 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def n_for_margin(population: int, margin: float = QC_MARGIN, p: float = QC_PLANNING_P, z: float = 1.96) -> int:
    """Sample size to estimate an error rate p within +/- margin (normal approximation, 95 %) in a finite
    population N: n0 = z^2 p (1-p) / margin^2,  n = ceil(N n0 / (N + n0 - 1)).
    With p = 0.153 and margin 0.05: N = 24, 49, 67, 203, 296  ->  22, 40, 51, 101, 120. p = 0.5 is the
    conservative planning value (largest n)."""
    import math

    if population <= 0:
        return 0
    n0 = z * z * p * (1.0 - p) / (margin * margin)
    return min(int(population), int(math.ceil(population * n0 / (population + n0 - 1.0) - 1e-9)))


def _agreement_population(a: Analysis) -> int:
    """Objects whose every detected channel evidence is an engine agreement (user-deleted ones included)."""
    n = 0
    for o in a.objects:
        evs = [ev for ev in (o.red, o.green) if ev.detected]
        if evs and all(ev.consensus_status == ConsensusStatus.agreement for ev in evs):
            n += 1
    return n


def qc_status(state: AnalysisState) -> dict:
    """Result of the agreement QC sample so far: confirmed vs rejected among the SAMPLED agreements, the exact
    Clopper-Pearson 95 % interval of the agreement error rate, and the finite-population sample sizes for +/-0.05."""
    a = state.analysis
    user_deleted = set(state.user_deleted)
    sampled = confirmed = rejected = 0
    for o in a.objects:
        if ReviewReason.qc_sample not in o.review_reasons:
            continue
        gone = o.deleted and o.id in user_deleted
        if o.deleted and not gone:
            continue                       # merged/split away: not a user decision
        sampled += 1
        if gone:
            rejected += 1
        elif o.reviewed:
            confirmed += 1
    reviewed = confirmed + rejected
    pop = _agreement_population(a)
    seed = None
    for act in reversed(state.log):
        pl = getattr(act, "payload", None) or {}
        if pl.get("event") == "qc_sample":
            seed = pl.get("seed")
            break
    out = {"sample_size": sampled, "reviewed": reviewed, "pending": sampled - reviewed, "confirmed": confirmed,
           "rejected": rejected, "population_agreements": pop, "seed": seed,
           "error_rate": (rejected / reviewed) if reviewed else None, "ci95": None,
           "interval_method": "Clopper-Pearson exact, 95 % two-sided",
           "n_for_margin_0_05": n_for_margin(pop), "n_for_margin_0_05_conservative": n_for_margin(pop, p=0.5),
           "planning_error_rate": QC_PLANNING_P, "note": QC_NOTE}
    if reviewed:
        lo, hi = clopper_pearson(rejected, reviewed)
        out["ci95"] = [lo, hi]
    return out


def mark_qc_sample(state: AnalysisState, fraction: float | None = None, seed: int | str = 0,
                   n: int | None = None) -> dict:
    """Add ReviewReason.qc_sample to a FIXED-SIZE deterministic sample of eligible agreement objects:
    the n eligible objects with the smallest qc_score(seed, id) (a random sample without replacement). n defaults to
    QC_DEFAULT_N (40), clamped to [1, N_eligible]. ``fraction`` is kept for compatibility and maps to
    max(QC_DEFAULT_N, ceil(fraction * N_eligible)); an explicit ``n`` wins. A fraction cannot bound the error: 5 % of
    24-296 agreements is 1-15 objects.
    Idempotent (already marked objects are kept, never duplicated); one undoable action, logged as
    ActionType.region_edit with payload.event == 'qc_sample'. Only ADDS a review reason."""
    import math
    from . import engine as eng

    if fraction is not None and not (isinstance(fraction, (int, float)) and not isinstance(fraction, bool)
                                     and 0.0 <= fraction <= 1.0):
        raise ReviewError("invalid_fraction", "fraction must be a number in 0..1", 422)
    if n is not None and not (isinstance(n, int) and not isinstance(n, bool) and n >= 0):
        raise ReviewError("invalid_n", "n must be a non-negative integer", 422)
    eng.guard_editable(state)
    a = state.analysis
    eligible = sorted((o for o in counted_objects(a) if qc_eligible(o)), key=lambda o: (qc_score(seed, o.id), o.id))
    ne = len(eligible)
    if n is not None:
        want = n
    elif fraction is not None:
        want = max(QC_DEFAULT_N, int(math.ceil(fraction * ne)))
    else:
        want = QC_DEFAULT_N
    k = 0 if ne == 0 else min(max(want, 1), ne)
    selected = eligible[:k]
    tx = eng.Tx(state)
    newly = []
    for o in selected:
        if ReviewReason.qc_sample in o.review_reasons:
            continue
        tx.touch_obj(o)
        o.review_reasons = [*o.review_reasons, ReviewReason.qc_sample]
        newly.append(o.id)
    payload = {"event": "qc_sample", "n": k, "n_requested": want, "fraction": fraction, "seed": seed,
               "eligible": ne, "selected": len(selected), "newly_marked": len(newly),
               "object_ids": sorted(o.id for o in selected)}
    if newly:
        tx.commit(ActionType.region_edit, payload)
    pop = _agreement_population(a)
    return {"marked": len(selected), "newly_marked": len(newly), "eligible": ne, "n": k, "n_requested": want,
            "fraction": fraction, "seed": seed, "population_agreements": pop,
            "n_for_margin_0_05": n_for_margin(ne), "n_for_margin_0_05_conservative": n_for_margin(ne, p=0.5),
            "qc": qc_status(state), "note": QC_NOTE}
