"""Point-matching evaluation and comparison of detection configurations.

Predictions and references are compared per field and per TIFF page (never pooled across pages silently):
one-to-one Hungarian matching inside a pixel tolerance, then precision / recall / F1, mean matching distance and
field-count statistics. Results are labelled ``preliminary`` automatically when the evidence is thin.
Nothing here decides what is true: the reference is either the final human-reviewed points or the researcher's
reference marks embedded in the TIFF, and both are shown as such.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from polony_app.detection.masks import points_in_polygon
from polony_app.review.engine import mapping_confirmed
from polony_app.schemas import ConsensusStatus, ObjectClass, Origin, RegionKind

from .common import (DEFAULT_NORMALISER, MLError, check_downscale, check_normaliser, from_downscaled,
                     hash_obj, prepare_crop, utc_now)
from .dataset import (POSITIVE, ROLES, classify_objects, fully_reviewed_regions, load_crop, reviewers_from,
                      unmarked_proposals, UNMARKED_RADIUS_PX)

DEFAULT_TOLERANCE_PX = 10.0              # click scatter between two annotators is ~5 px: 5 px measures the scatter, not detection
TOLERANCES = (5.0, 10.0, 15.0)           # ALWAYS reported side by side
MIN_FIELDS = 30
MIN_EVENTS_PER_PAGE = 30
IDLE_CAP_S = 120.0
LOA_Z = 1.96
CONFIGS = ("bigfish_default", "bigfish_sensitive", "spotiflow_pretrained", "spotiflow_finetuned",
           "agreement_subset", "union_before_review", "final_human_reviewed")
PRECISION_LABEL = "precision vs this annotator's marks"
POOLED_R_LABEL = "between-slide (not accuracy)"

DEFINITIONS = {
    "tp / fp / fn": "matched predictions / predictions without a reference partner / reference points nobody predicted "
                    "(one-to-one matching within the tolerance, minimum total distance; matched ONCE per page over the union "
                    "of its evaluated fields, so a pair that straddles a field border is not lost; the pair is credited to the "
                    "field of the reference point)",
    "tolerance": "matching radius in pixels. 5, 10 and 15 px are always reported side by side: 5 px mostly measures the "
                 "click scatter of the annotator, not whether the spot was detected",
    "precision": "tp / (tp + fp) = precision vs THIS annotator's marks: a real polony nobody marked counts as a false "
                 "positive, so it is a lower bound on the true precision",
    "recall": "tp / (tp + fn): of the reference points, the share that was predicted",
    "f1": "harmonic mean of precision and recall",
    "mean_match_distance_px": "mean distance between matched prediction and reference (pixels)",
    "count_mae": "mean over fields of |predicted count - reference count|",
    "count_bias / bias": "mean over fields of (predicted - reference); positive = over-counting",
    "slope / intercept": "ordinary least-squares line predicted = intercept + slope x reference over fields (an offset is NOT hidden "
                         "by forcing the line through the origin)",
    "loa_lower / loa_upper": "Bland-Altman limits of agreement: bias +/- 1.96 x SD of (predicted - reference), computed per slide",
    "pearson_r": "Pearson r of per-field counts WITHIN a single slide (None when the row mixes slides)",
    "pearson_r_pooled": f"Pearson r over fields of several slides: {POOLED_R_LABEL}. Slides differ in density, so it is high even "
                        "when the counts of fields of one slide are unrelated to the reference; report the per-slide r instead",
    "pearson_r_median_per_slide": "median over slides of the within-slide Pearson r",
    "fp_per_field / fn_per_field": "average false positives / false negatives per field",
    "score": "model score = Spotiflow heatmap value at the peak. It is NOT a calibrated probability (measured: ECE about "
             "0.27-0.30, AUC about 0.55-0.64 against reviewer decisions); never read it as 'probability of being a polony'",
}


# ====================================================================== core
def match_points(pred: np.ndarray, ref: np.ndarray, tol: float = DEFAULT_TOLERANCE_PX):
    """One-to-one matching. ``pred`` (N,2), ``ref`` (M,2) in the same (x,y) frame.
    Returns (pairs [(i, j, distance)], unmatched_pred_idx, unmatched_ref_idx). Only pairs with distance <= tol are
    allowed; among all such matchings the one with the most pairs wins, then the smallest total distance."""
    pred = np.asarray(pred, float).reshape(-1, 2)
    ref = np.asarray(ref, float).reshape(-1, 2)
    n, m = len(pred), len(ref)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    d = np.sqrt(((pred[:, None, :] - ref[None, :, :]) ** 2).sum(-1))
    big = 1e6
    cost = np.where(d <= tol, d, big)
    ri, ci = linear_sum_assignment(cost)
    pairs = [(int(i), int(j), float(d[i, j])) for i, j in zip(ri, ci) if d[i, j] <= tol]
    mp = {p[0] for p in pairs}
    mr = {p[1] for p in pairs}
    return pairs, [i for i in range(n) if i not in mp], [j for j in range(m) if j not in mr]


def _div(a: float, b: float) -> Optional[float]:
    return a / b if b else None


def prf(tp: int, fp: int, fn: int) -> dict:
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = None if p is None or r is None else (0.0 if p + r == 0 else 2 * p * r / (p + r))
    return {"precision": p, "recall": r, "f1": f1}


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if xa.std() == 0 or ya.std() == 0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def tol_key(t: float) -> str:
    return f"{float(t):g}"


def tolerances_for(tol: float) -> list[float]:
    return sorted({float(tol), *TOLERANCES})


def annotate_matches(items: list[dict], tols: Iterable[float]) -> None:
    """Match each *unit* (items sharing ``it['unit']``; default: the item alone) ONCE over the union of its fields, per
    tolerance, and store the per-item result in ``it['_match'][tol_key]``. A pair whose prediction lies in one field and
    whose reference lies in the neighbouring one is therefore kept (credited to the reference's field); an unmatched
    prediction is an fp of its own field, an unmatched reference an fn of its own field."""
    units: dict = defaultdict(list)
    for k, it in enumerate(items):
        units[it.get("unit", ("_item", k))].append(it)
    for its in units.values():
        pred, ref, powner, rowner = [], [], [], []
        for k, it in enumerate(its):
            for p in it["pred"]:
                pred.append(p)
                powner.append(k)
            for r in it["ref"]:
                ref.append(r)
                rowner.append(k)
        for t in tols:
            pairs, up, ur = match_points(np.array(pred, float), np.array(ref, float), t)
            res = [{"tp": 0, "fp": 0, "fn": 0, "dists": []} for _ in its]
            for i, j, dist in pairs:
                res[rowner[j]]["tp"] += 1
                res[rowner[j]]["dists"].append(dist)
            for i in up:
                res[powner[i]]["fp"] += 1
            for j in ur:
                res[rowner[j]]["fn"] += 1
            for it, r in zip(its, res):
                it.setdefault("_match", {})[tol_key(t)] = r


def _ensure_matches(items: list[dict], tols: Sequence[float]) -> None:
    need = [t for t in tols if any(tol_key(t) not in it.get("_match", {}) for it in items)]
    if need:
        annotate_matches(items, need)


def count_agreement(ref: Sequence[float], pred: Sequence[float]) -> dict:
    """Agreement of per-field counts: OLS slope AND intercept, mean bias, Bland-Altman limits, Pearson r."""
    r, p = np.asarray(ref, float), np.asarray(pred, float)
    n = len(r)
    out = {"n_fields": n, "bias": None, "sd_diff": None, "loa_lower": None, "loa_upper": None, "slope": None,
           "intercept": None, "pearson_r": None, "count_mae": None}
    if n == 0:
        return out
    diff = p - r
    out.update({"bias": float(diff.mean()), "count_mae": float(np.abs(diff).mean())})
    if n >= 2:
        sd = float(diff.std(ddof=1))
        out.update({"sd_diff": sd, "loa_lower": float(diff.mean() - LOA_Z * sd), "loa_upper": float(diff.mean() + LOA_Z * sd)})
        sxx = float(((r - r.mean()) ** 2).sum())
        if sxx > 0:
            slope = float(((r - r.mean()) * (p - p.mean())).sum() / sxx)
            out.update({"slope": slope, "intercept": float(p.mean() - slope * r.mean())})
        out["pearson_r"] = pearson(r, p)
    return out


def metrics_of(items: Iterable[dict], tol: float = DEFAULT_TOLERANCE_PX) -> dict:
    """Aggregate metrics over field items ({'pred': [(x,y)], 'ref': [(x,y)], 'slide': ..}). Each item = one field on one
    page. Headline numbers are at ``tol``; ``by_tolerance`` always carries 5 / 10 / 15 px."""
    items = list(items)
    tols = tolerances_for(tol)
    _ensure_matches(items, tols)

    def agg(t: float) -> dict:
        ms = [it["_match"][tol_key(t)] for it in items]
        tp, fp, fn = (sum(m[k] for m in ms) for k in ("tp", "fp", "fn"))
        dists = [d for m in ms for d in m["dists"]]
        return {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn),
                "mean_match_distance_px": (float(np.mean(dists)) if dists else None)}

    head = agg(tol)
    cp = [len(it["pred"]) for it in items]
    cr = [len(it["ref"]) for it in items]
    n = len(items)
    out = {"n_fields": n, "n_reference": int(sum(cr)), "n_predicted": int(sum(cp)), "tolerance_px": float(tol), **head,
           "by_tolerance": {tol_key(t): agg(t) for t in tols}}
    slides: dict = defaultdict(lambda: ([], []))
    for it, a_, b_ in zip(items, cr, cp):
        slides[str(it.get("slide", "?"))][0].append(a_)
        slides[str(it.get("slide", "?"))][1].append(b_)
    per_slide_r = {k: pearson(v[0], v[1]) for k, v in sorted(slides.items())}
    ca = count_agreement(cr, cp)
    pooled = ca["pearson_r"]
    rs = [v for v in per_slide_r.values() if v is not None]
    out.update({"count_mae": ca["count_mae"], "count_bias": ca["bias"], "bias": ca["bias"], "sd_diff": ca["sd_diff"],
                "loa_lower": ca["loa_lower"], "loa_upper": ca["loa_upper"], "slope": ca["slope"], "intercept": ca["intercept"],
                "n_slides": len(slides), "pearson_r_pooled": pooled,
                "pearson_r": (pooled if len(slides) == 1 else None),
                "pearson_r_label": ("within-slide" if len(slides) <= 1 else POOLED_R_LABEL),
                "pearson_r_per_slide": per_slide_r,
                "pearson_r_median_per_slide": (float(np.median(rs)) if rs else None),
                "fp_per_field": (head["fp"] / n if n else None), "fn_per_field": (head["fn"] / n if n else None)})
    return out


def preliminary_reasons(n_fields: int, events_on_page: Optional[int], n_annotators: int) -> list[str]:
    why = []
    if n_fields < MIN_FIELDS:
        why.append(f"only {n_fields} field(s) (< {MIN_FIELDS})")
    if events_on_page is not None and events_on_page < MIN_EVENTS_PER_PAGE:
        why.append(f"only {events_on_page} reference event(s) on this page (< {MIN_EVENTS_PER_PAGE})")
    if n_annotators <= 1:
        why.append("a single annotator / review session (reviewer bias cannot be assessed)")
    return why


# ====================================================================== grouping into slices
def _slice(items: list[dict], key, tol: float, n_annotators: int, page_events: dict) -> dict:
    """``key`` = item field name or a callable item -> label. Matching was done on the whole configuration beforehand
    (``annotate_matches``), never re-done on the slice."""
    kf = key if callable(key) else (lambda it: it.get(key))
    groups: dict = defaultdict(list)
    for it in items:
        groups[str(kf(it))].append(it)
    out = {}
    for k, its in sorted(groups.items()):
        m = metrics_of(its, tol)
        pages = {it["page"] for it in its}
        ev = min((page_events.get(p, 0) for p in pages), default=0) if pages else 0
        why = preliminary_reasons(m["n_fields"], ev, n_annotators)
        out[k] = dict(m, label="preliminary" if why else "ok", preliminary_reasons=why)
    return out


def _slide_field_key(it: dict) -> str:
    return f"{it.get('slide')} / field {it.get('field')}"


def _slide_page_key(it: dict) -> str:
    return f"{it.get('slide')} / page {it.get('page')}"


def count_agreement_by_slide(items: list[dict]) -> dict:
    """Per (slide, page): OLS slope + intercept, bias, Bland-Altman limits, Pearson r of the field counts; plus the median
    per-slide r and the pooled r, which is labelled between-slide (not accuracy)."""
    groups: dict = defaultdict(list)
    for it in items:
        groups[_slide_page_key(it)].append(it)
    rows = {}
    for k, its in sorted(groups.items()):
        rows[k] = count_agreement([len(i["ref"]) for i in its], [len(i["pred"]) for i in its])
        rows[k]["slide"], rows[k]["page"] = its[0].get("slide"), its[0].get("page")
    rs = [r["pearson_r"] for r in rows.values() if r["pearson_r"] is not None]
    allc = count_agreement([len(i["ref"]) for i in items], [len(i["pred"]) for i in items])
    n_slides = len({str(i.get("slide")) for i in items})
    return {"per_slide": rows, "median_per_slide_r": (float(np.median(rs)) if rs else None),
            "pooled_r": allc["pearson_r"], "pooled_r_label": (POOLED_R_LABEL if n_slides > 1 else "single slide"),
            "n_slides": n_slides}


def summarise_config(items: list[dict], tol: float, n_annotators: int) -> dict:
    """Per page (never merged across pages), plus a pooled row over the pages that carry enough events only."""
    _ensure_matches(items, tolerances_for(tol))
    page_events: dict = defaultdict(int)
    for it in items:
        page_events[it["page"]] += len(it["ref"])
    per_page = _slice(items, "page", tol, n_annotators, page_events)
    dense = sorted({it["page"] for it in items if page_events[it["page"]] >= MIN_EVENTS_PER_PAGE})
    sparse = sorted({it["page"] for it in items if page_events[it["page"]] < MIN_EVENTS_PER_PAGE}, key=str)
    pooled = None
    if dense and len({str(p) for p in dense}) > 1:
        pooled_items = [it for it in items if it["page"] in dense]
        m = metrics_of(pooled_items, tol)
        why = preliminary_reasons(m["n_fields"], min(page_events[p] for p in dense), n_annotators)
        pooled = dict(m, pages=[str(p) for p in dense], excluded_sparse_pages=[str(p) for p in sparse],
                      label="preliminary" if why else "ok", preliminary_reasons=why)
    return {"per_page": per_page, "pooled_dense_pages": pooled,
            "count_agreement": count_agreement_by_slide(items),
            "by_slide": _slice(items, "slide", tol, n_annotators, page_events),
            "by_field": _slice(items, _slide_field_key, tol, n_annotators, page_events),
            "by_channel": _slice(items, "role", tol, n_annotators, page_events),
            "by_radius": _slice(items, "radius", tol, n_annotators, page_events),
            "by_model_version": _slice(items, "model_version", tol, n_annotators, page_events)}


# ====================================================================== gathering from the store
def _candidate_points(o, role: str) -> Optional[tuple[float, float]]:
    ev = o.red if role == "red" else o.green
    if ev.detected and ev.x is not None and ev.y is not None:
        return float(ev.x), float(ev.y)
    return None


def _polys(a, regions):
    return {r.id: np.asarray(r.polygon, float) for r in regions}


def _assign_fields(points: list[tuple], regions, excl_polys) -> dict:
    """{region_id: [(x,y)]} by polygon membership (first listed wins, exclusions veto)."""
    out: dict = defaultdict(list)
    if not points:
        return out
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    vetoed = np.zeros(len(points), bool)
    for ep in excl_polys:
        vetoed |= points_in_polygon(xs, ys, ep)
    taken = vetoed.copy()
    for r in regions:
        inside = points_in_polygon(xs, ys, np.asarray(r.polygon, float)) & ~taken
        for i in np.nonzero(inside)[0]:
            out[r.id].append((float(xs[i]), float(ys[i])))
        taken |= inside
    return out


def review_minutes(log, idle_cap_s: float = IDLE_CAP_S) -> Optional[float]:
    """Active review time estimated from action timestamps: sum of gaps between consecutive actions, each gap capped at
    ``idle_cap_s`` (a longer pause is treated as a break). None when fewer than two timestamps exist."""
    from datetime import datetime
    ts = []
    for a in log or []:
        try:
            ts.append(datetime.fromisoformat(str(a.ts).replace("Z", "+00:00")))
        except (ValueError, AttributeError):
            continue
    if len(ts) < 2:
        return None
    ts.sort()
    return sum(min((b - a).total_seconds(), idle_cap_s) for a, b in zip(ts, ts[1:])) / 60.0


class Gathered:
    def __init__(self):
        self.items: dict[str, list[dict]] = defaultdict(list)
        self.unavailable: dict[str, str] = {}
        self.warnings: list[str] = []
        self.reviewers: set[str] = set()
        self.acceptance: dict = {}
        self.review_time: list[dict] = []
        self.analyses: list[str] = []
        self.skipped: list[dict] = []
        self.config_info: dict[str, dict] = {}       # configuration name -> identity details (engine runs)
        self.completeness: dict[str, dict] = {}      # slide -> unmarked-but-bright proposals (raw detections available)


def _model_version(run) -> str:
    if run is None:
        return ""
    name = run.model_name or ""
    ver = run.model_version or (run.model_checksum or "")[:8]
    return f"{run.engine.value} {run.engine_version} {name} {ver}".strip()


def _config_for_run(run, registry_ids: set[str]) -> Optional[str]:
    eng = run.engine.value
    if eng == "bigfish":
        sc = (run.params or {}).get("threshold_scale", 1.0)
        try:
            sc = float(sc)
        except (TypeError, ValueError):
            sc = 1.0
        if abs(sc - 1.0) < 1e-9:
            return "bigfish_default"
        if abs(sc - 0.5) < 1e-9:
            return "bigfish_sensitive"
        return None
    if eng == "spotiflow":
        return "spotiflow_finetuned" if (run.model_name in registry_ids) else "spotiflow_pretrained"
    return None


def run_identity(run) -> dict:
    """What makes two Spotiflow runs the same configuration: downscale, probability threshold actually used, normaliser,
    model checksum, crop-to-valid and the crop box. (Two runs that differ in any of these are NOT interchangeable.)"""
    p, pre = dict(run.params or {}), dict(getattr(run, "preprocessing", None) or {})
    thr = p.get("prob_thresh_used", p.get("prob_thresh"))
    try:
        thr = None if thr is None else float(thr)
    except (TypeError, ValueError):
        thr = None
    box = pre.get("crop_box_xyxy")
    return {"downscale": int(p.get("downscale", pre.get("downscale", 1)) or 1), "prob_thresh_used": thr,
            "normaliser": str(p.get("normalizer", p.get("normaliser", "?"))),
            "model": str(run.model_name or ""), "model_checksum": str(run.model_checksum or ""),
            "crop_to_valid": p.get("crop_to_valid"), "crop_box_xyxy": (list(box) if box else None)}


def identity_label(base: str, idn: dict, with_box: bool = False) -> str:
    thr = "?" if idn["prob_thresh_used"] is None else f"{idn['prob_thresh_used']:g}"
    parts = [f"downscale={idn['downscale']}", f"thr={thr}", f"normaliser={idn['normaliser']}",
             f"model={(idn['model_checksum'] or idn['model'] or '?')[:8]}"]
    if idn.get("crop_to_valid") is False:
        parts.append("full page")
    if with_box and idn.get("crop_box_xyxy"):
        parts.append("crop=" + ",".join(str(int(v)) for v in idn["crop_box_xyxy"]))
    return f"{base} [{', '.join(parts)}]"


def gather_analysis(store, aid: str, g: Gathered, *, reference: str, tol: float, numbering: str = "index",
                    registry_ids: Optional[set] = None, configs: Sequence[str] = CONFIGS,
                    run_filter: Optional[set] = None) -> None:
    registry_ids = registry_ids or set()
    with store.view(aid) as st:
        a = st.analysis.model_copy(deep=True)
        user_deleted = list(getattr(st, "user_deleted", []) or [])
        log = list(getattr(st, "log", []) or [])
        events = list(getattr(st, "events", []) or [])
    slide = a.slide.slide_id or a.name or aid
    if a.image is None or not mapping_confirmed(a):
        g.skipped.append({"analysis_id": aid, "reason": "channel mapping not confirmed (page/colour unknown)"})
        g.warnings.append(f"analysis {aid}: skipped - channel mapping is not confirmed, so predictions cannot be assigned to a page")
        return
    mapping = {c.role.value: int(c.page) for c in a.image.channels if c.role is not None and c.role.value in ROLES}
    g.analyses.append(aid)
    rv = reviewers_from(log, events)
    g.reviewers |= set(rv) if rv else {f"unknown:{aid}"}
    reviewed = fully_reviewed_regions(a)
    counting = [r for r in a.regions if r.kind == RegionKind.counting]
    excl = [np.asarray(r.polygon, float) for r in a.regions if r.kind == RegionKind.exclusion and len(r.polygon) >= 3]

    # ---- which fields are evaluated, and the reference points
    ref_pts: dict = {}      # (role, region_id) -> [(x,y)]
    if reference == "analysis":
        fields = [r for r in reviewed]
        rows = classify_objects(a, user_deleted, reviewed)
        for role in ROLES:
            pos = [(r["x"], r["y"]) for r in rows if r["role"] == role and r["status"] in POSITIVE]
            asg = _assign_fields(pos, fields, excl)
            for r in fields:
                ref_pts[(role, r.id)] = asg.get(r.id, [])
    elif reference == "roi":
        from polony_app.fields.reference import load_reference
        ref = load_reference(Path(store.originals_dir(aid)), a.image)
        if not ref.available:
            g.skipped.append({"analysis_id": aid, "reason": f"no reference marks: {ref.reason}"})
            g.warnings.append(f"analysis {aid}: skipped - no embedded reference marks ({ref.reason})")
            return
        g.warnings += [f"analysis {aid}: {w}" for w in ref.warnings]
        fields = []
        for r in counting:
            if r.field_index is None:
                continue
            fields.append(r)
        if not fields:
            g.skipped.append({"analysis_id": aid, "reason": "no field regions (counting regions with a field number)"})
            g.warnings.append(f"analysis {aid}: skipped - no field regions; import the lab fields first")
            return
        missing = set()
        for role, page in mapping.items():
            marks: dict = defaultdict(list)
            for p in ref.points:
                if p.plane == page and ref.field_of(p, numbering) is not None:
                    marks[ref.field_of(p, numbering)].append((p.x, p.y))
            keep = [r for r in fields if r.field_index in marks]
            missing |= {r.field_index for r in fields if r.field_index not in marks}
            for r in keep:
                ref_pts[(role, r.id)] = marks[r.field_index]
        if missing:
            g.warnings.append(f"analysis {aid}: fields without reference marks on a page are NOT evaluated "
                              f"({', '.join(map(str, sorted(missing)))}): the file cannot tell 'counted zero' from 'not counted'")
    else:
        raise MLError("reference must be 'analysis' or 'roi'")
    fields_by_id = {r.id: r for r in a.regions}

    def emit(cfg: str, role: str, model_version: str, pts: list):
        page = mapping.get(role)
        if page is None:
            return
        by_f = _assign_fields(pts, [fields_by_id[rid] for (rl, rid) in ref_pts if rl == role], excl)
        rad = getattr(a.detection_request, role).spot_radius_px
        for (rl, rid), refp in ref_pts.items():
            if rl != role:
                continue
            r = fields_by_id[rid]
            g.items[cfg].append({"config": cfg, "analysis_id": aid, "slide": slide, "unit": (aid, page, role),
                                 "field": (r.field_index if r.field_index is not None else r.name), "page": page,
                                 "role": role, "radius": rad, "model_version": model_version,
                                 "pred": by_f.get(rid, []), "ref": refp})

    # ---- object-based configurations
    prov = f"{a.provenance.engine} (candidates of the stored detection)"
    ud = set(user_deleted)
    autos = [o for o in a.objects if o.origin != Origin.manual and not (o.deleted and o.id not in ud)]
    for role in ROLES:
        cands = [(o, _candidate_points(o, role)) for o in autos]
        cands = [(o, p) for o, p in cands if p is not None]
        if "union_before_review" in configs:
            emit("union_before_review", role, prov, [p for _, p in cands])
        if "agreement_subset" in configs:
            ag = [p for o, p in cands if (o.red if role == "red" else o.green).consensus_status == ConsensusStatus.agreement]
            if any((o.red if role == "red" else o.green).consensus_status is not None for o, _ in cands):
                emit("agreement_subset", role, prov, ag)
            else:
                g.unavailable.setdefault("agreement_subset", "no object carries two-engine consensus status (run engine 'hybrid')")
        if "final_human_reviewed" in configs and reference != "analysis":
            rows = classify_objects(a, user_deleted, reviewed)
            emit("final_human_reviewed", role, "human-reviewed final",
                 [(r["x"], r["y"]) for r in rows if r["role"] == role and r["status"] in POSITIVE])
    if reference == "analysis" and "final_human_reviewed" in configs:
        g.unavailable.setdefault("final_human_reviewed", "identical to the reference when --reference analysis")

    # ---- engine-based configurations (need the raw detections of the store)
    wanted = {"bigfish_default", "bigfish_sensitive", "spotiflow_pretrained", "spotiflow_finetuned"} & set(configs)
    if wanted:
        runs_fn, dets_fn = getattr(store, "get_engine_runs", None), getattr(store, "get_raw_detections", None)
        if runs_fn is None or dets_fn is None:
            for c in wanted:
                g.unavailable[c] = "this store has no engine-run / raw-detection API yet"
        else:
            runs = list(runs_fn(aid))
            cand_runs = []
            for run in runs:
                if run.status != "ok" or (run_filter and run.run_id not in run_filter):
                    continue
                base_cfg = _config_for_run(run, registry_ids)
                if base_cfg not in wanted:
                    continue
                idn = run_identity(run) if run.engine.value == "spotiflow" else None
                core = identity_label(base_cfg, idn) if idn else base_cfg
                cand_runs.append((base_cfg, core, idn, run))
            # runs of one configuration on one page with different crop boxes are different configurations
            boxes: dict = defaultdict(set)
            for base_cfg, core, idn, run in cand_runs:
                boxes[(core, run.page)].add(tuple((idn or {}).get("crop_box_xyxy") or ()))
            latest: dict = {}
            n_same: dict = defaultdict(int)
            for base_cfg, core, idn, run in cand_runs:
                name = identity_label(base_cfg, idn, with_box=True) if (idn and len(boxes[(core, run.page)]) > 1) else core
                k = (base_cfg, name, run.page)
                n_same[k] += 1
                if k not in latest or (run.started_at or "") >= (latest[k][0].started_at or ""):
                    latest[k] = (run, idn)
            for cfg in wanted:
                if not any(k[0] == cfg for k in latest):
                    g.unavailable.setdefault(cfg, "no stored engine run matches this configuration")
            for (base_cfg, name, page), (run, idn) in sorted(latest.items(), key=lambda kv: (kv[0][1], kv[0][2])):
                role = next((rl for rl, pg in mapping.items() if pg == page), None)
                if role is None:
                    continue
                dets = dets_fn(aid, run_id=run.run_id, valid=True)
                emit(name, role, _model_version(run), [(d.x, d.y) for d in dets])
                info = g.config_info.setdefault(name, {"base": base_cfg, "identity": idn, "runs": [], "crop_boxes": {}})
                info["runs"].append({"analysis_id": aid, "page": page, "run_id": run.run_id,
                                     "same_configuration_runs_on_this_page": n_same[(base_cfg, name, page)]})
                if idn and idn.get("crop_box_xyxy"):
                    info["crop_boxes"][f"{aid}:p{page}"] = idn["crop_box_xyxy"]
                if n_same[(base_cfg, name, page)] > 1:
                    g.warnings.append(f"analysis {aid} page {page}: {n_same[(base_cfg, name, page)]} runs share the configuration "
                                      f"'{name}'; the most recent one ({run.run_id}) is used")

    # ---- annotation completeness: raw proposals nobody marked (needs the raw detections of the store)
    if reference == "analysis" and getattr(store, "get_raw_detections", None) is not None:
        try:
            raws = list(store.get_raw_detections(aid))
        except Exception:      # pragma: no cover
            raws = []
        if raws:
            rows_all = classify_objects(a, user_deleted, reviewed)
            c = g.completeness.setdefault(slide, {"n_raw_proposals": 0, "n_unmarked": 0, "n_unmarked_bright": 0,
                                                  "n_confirmed_proposals": 0})
            for role in ROLES:
                fpolys = [np.asarray(r.polygon, float) for r in reviewed]
                raw = []
                for dd in raws:
                    dd = dd.model_dump(mode="json") if hasattr(dd, "model_dump") else (dd if isinstance(dd, dict) else vars(dd))
                    if dd.get("channel_role") == role and dd.get("valid", True) and any(
                            bool(points_in_polygon(np.array([dd["x"]]), np.array([dd["y"]]), q)[0]) for q in fpolys):
                        raw.append((float(dd["x"]), float(dd["y"]), dd.get("score")))
                if not raw:
                    continue
                pos = [(r["x"], r["y"]) for r in rows_all if r["role"] == role and r["status"] in POSITIVE]
                mk = pos + [(r["x"], r["y"]) for r in rows_all if r["role"] == role and r["status"] == "rejected"]
                u = unmarked_proposals(raw, mk, pos)
                for k in ("n_raw_proposals", "n_unmarked", "n_confirmed_proposals"):
                    c[k] += u[k]
                c["n_unmarked_bright"] += u["n_unmarked_bright"] or 0

    # ---- acceptance / review statistics (reviewed regions only)
    rpolys = [np.asarray(r.polygon, float) for r in reviewed]
    acc = g.acceptance.setdefault(slide, {})
    for o in autos:
        for role in ROLES:
            p = _candidate_points(o, role)
            if p is None:
                continue
            ev = o.red if role == "red" else o.green
            if not rpolys or not any(bool(points_in_polygon(np.array([p[0]]), np.array([p[1]]), q)[0]) for q in rpolys):
                continue
            st_ = ev.consensus_status.value if ev.consensus_status else "unlabelled"
            d = acc.setdefault(f"{role}:{st_}", {"candidates": 0, "accepted": 0, "needs_review": 0})
            d["candidates"] += 1
            d["accepted"] += int((not o.deleted) and o.cls != ObjectClass.uncertain)
            d["needs_review"] += int(bool(o.review_reasons) or st_ in ("bigfish_only", "spotiflow_only"))
    rows = classify_objects(a, user_deleted, reviewed)
    fin = [r for r in rows if r["status"] in POSITIVE]
    acc["_final"] = {"final_positives": len(fin), "human_added": sum(1 for r in fin if r["status"] == "human_added")}
    minutes = review_minutes(log)
    g.review_time.append({"analysis_id": aid, "slide": slide, "active_review_minutes": minutes,
                          "fields_reviewed": len(reviewed),
                          "minutes_per_field": (minutes / len(reviewed)) if (minutes is not None and reviewed) else None})


def acceptance_summary(acc: dict) -> dict:
    out = {}
    for slide, d in acc.items():
        s = {}
        for k, v in d.items():
            if k == "_final":
                s["human_added_rate"] = _div(v["human_added"], v["final_positives"])
                s["final_positives"] = v["final_positives"]
                s["human_added"] = v["human_added"]
            else:
                s[k] = dict(v, acceptance_rate=_div(v["accepted"], v["candidates"]),
                            needs_review_rate=_div(v["needs_review"], v["candidates"]))
        out[slide] = s
    return out


def evaluate_analyses(store, analysis_ids: Sequence[str], *, reference: str = "analysis", tolerance_px: float = DEFAULT_TOLERANCE_PX,
                      configs: Sequence[str] = CONFIGS, numbering: str = "index", registry_ids: Optional[set] = None) -> dict:
    """``analysis_ids`` entries are ``<analysis id>`` or ``<analysis id>:<engine run id>[,<run id>...]`` (restrict the
    engine-based configurations to those runs)."""
    g = Gathered()
    filt: dict = defaultdict(set)
    plain = []
    for ent in analysis_ids:
        aid, _, runs = str(ent).partition(":")
        plain.append(aid)
        if runs:
            filt[aid] |= {r for r in runs.split(",") if r}
    analysis_ids = plain
    for aid in sorted(set(analysis_ids)):
        if not store.exists(aid):
            raise MLError(f"analysis '{aid}' not found")
        gather_analysis(store, aid, g, reference=reference, tol=tolerance_px, numbering=numbering,
                        registry_ids=registry_ids, configs=configs, run_filter=filt.get(aid))
    # unknown reviewers are one anonymous session each; several analyses by 'unknown' still cannot establish 2 annotators
    n_ann = len({r for r in g.reviewers if not r.startswith("unknown:")}) or (1 if g.reviewers else 0)
    report = {"generated_at": utc_now(), "reference": reference, "tolerance_px": tolerance_px,
              "tolerances_px": tolerances_for(tolerance_px), "precision_label": PRECISION_LABEL, "numbering": numbering,
              "analyses": g.analyses, "skipped": g.skipped, "n_annotators": n_ann,
              "reviewers": sorted(r for r in g.reviewers if not r.startswith("unknown:")) or ["unknown"],
              "configs": {}, "acceptance": acceptance_summary(g.acceptance), "review_time": g.review_time,
              "warnings": g.warnings, "definitions": DEFINITIONS}
    for c in configs:
        names = sorted(n for n in g.items if n == c or n.startswith(c + " ["))
        if not names:
            report["configs"][c] = {"available": False, "reason": g.unavailable.get(c, "no data")}
        for n in names:
            report["configs"][n] = {"available": True, **summarise_config(g.items[n], tolerance_px, n_ann)}
            if n in g.config_info:
                report["configs"][n]["configuration"] = g.config_info[n]
    families: dict = defaultdict(list)
    for n, i in g.config_info.items():
        families[i["base"]].append(n)
    report["engine_configurations_found"] = {k: sorted(v) for k, v in sorted(families.items())}
    for base, names in families.items():
        if len(names) > 1:
            report["warnings"].append(f"{len(names)} different {base} configurations were found in the stored runs "
                                      f"({'; '.join(sorted(names))}); each is reported separately and none is preferred")
    if g.completeness:
        report["annotation_completeness"] = {
            "per_slide": g.completeness,
            "note": "raw engine proposals inside the reviewed fields that this annotator left unmarked (no confirmed or rejected "
                    f"point within {UNMARKED_RADIUS_PX:g} px); 'bright' = score >= the lower quartile of the confirmed proposals. "
                    "Some of them are real polonies nobody marked, so precision against these marks is a lower bound."}
    report["report_sha256"] = hash_obj({k: v for k, v in report.items() if k not in ("generated_at", "report_sha256")})
    return report


# ====================================================================== markdown
def _f(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _table(slices: dict, first: str) -> list[str]:
    cols = ["n_fields", "n_reference", "n_predicted", "tp", "fp", "fn", "precision", "recall", "f1", "mean_match_distance_px",
            "count_mae", "count_bias", "slope", "intercept", "fp_per_field", "fn_per_field"]
    head = [first] + cols[:6] + ["precision (vs annotator's marks)", "recall", "f1"] + cols[9:] + ["r (within slide)", "r pooled", "label"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for k, m in slices.items():
        pooled = (f"{_f(m.get('pearson_r_pooled'))} (between-slide, not accuracy)" if m.get("n_slides", 1) > 1
                  else "-")
        out.append("| " + " | ".join([str(k)] + [_f(m.get(c)) for c in cols] + [_f(m.get("pearson_r")), pooled, m["label"]]) + " |")
    return out


def _tolerance_table(slices: dict, first: str) -> list[str]:
    tks = sorted({k for m in slices.values() for k in m.get("by_tolerance", {})}, key=float)
    head = [first] + [f"{m} @{k} px" for k in tks for m in ("P", "R", "F1")]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for name, m in slices.items():
        bt = m.get("by_tolerance", {})
        out.append("| " + " | ".join([str(name)] + [_f(bt.get(k, {}).get(x)) for k in tks for x in ("precision", "recall", "f1")]) + " |")
    return out


def _agreement_table(ca: dict) -> list[str]:
    head = ["slide / page", "n_fields", "r (within slide)", "OLS slope", "intercept", "mean bias", "LoA lower", "LoA upper"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for k, r in ca["per_slide"].items():
        out.append("| " + " | ".join([k, str(r["n_fields"])] + [_f(r[x]) for x in ("pearson_r", "slope", "intercept", "bias",
                                                                                      "loa_lower", "loa_upper")]) + " |")
    return out


def report_markdown(rep: dict) -> str:
    tols = ", ".join(f"{t:g}" for t in rep.get("tolerances_px", TOLERANCES))
    L = [f"# Evaluation report", "",
         f"Reference: **{rep['reference']}**; headline matching tolerance **{rep['tolerance_px']:g} px** (one-to-one, matched once per page "
         f"over all its fields); tolerances reported side by side: **{tols} px**. "
         f"Annotators identified: **{rep['n_annotators']}** ({', '.join(rep['reviewers'])}); analyses: {len(rep['analyses'])}.", "",
         "**Precision here is precision vs this annotator's marks**: a real polony the annotator did not mark counts as a false "
         "positive, so it is a lower bound. The model score of Spotiflow is a heatmap value, not a calibrated probability.", "",
         "Rows labelled **preliminary** have too little evidence for a conclusion (fewer than "
         f"{MIN_FIELDS} fields, fewer than {MIN_EVENTS_PER_PAGE} reference events on the page, or one annotator). "
         "Pages are never merged silently: totals are per TIFF page; a pooled row lists the pages it contains and the sparse pages it leaves out. "
         f"A Pearson r over several slides is **{POOLED_R_LABEL}**: it mostly measures that slides differ in density.", ""]
    for name, c in rep["configs"].items():
        L += [f"## {name}", ""]
        if not c["available"]:
            L += [f"_not available: {c['reason']}_", ""]
            continue
        if c.get("configuration"):
            idn = c["configuration"].get("identity") or {}
            L += [f"Configuration: downscale {idn.get('downscale')}, threshold used {idn.get('prob_thresh_used')}, normaliser "
                  f"{idn.get('normaliser')}, model checksum {(idn.get('model_checksum') or '?')[:16]}, crop boxes "
                  f"{c['configuration'].get('crop_boxes') or 'n/a'}.", ""]
        L += ["### per TIFF page", ""] + _table(c["per_page"], "page") + [""]
        L += ["### precision / recall / F1 at 5, 10 and 15 px", ""] + _tolerance_table(c["per_page"], "page") + [""]
        if c["pooled_dense_pages"]:
            p = c["pooled_dense_pages"]
            L += [f"Pooled over pages {', '.join(p['pages'])} only (sparse pages left out: "
                  f"{', '.join(p['excluded_sparse_pages']) or 'none'}):", ""] + _table({"pooled": p}, "scope") + [""] \
                + _tolerance_table({"pooled": p}, "scope") + [""]
        ca = c["count_agreement"]
        L += ["### count agreement per slide (field counts, predicted vs reference)", "",
              f"Median within-slide r: **{_f(ca['median_per_slide_r'])}**. Pooled r over all fields: {_f(ca['pooled_r'])} "
              f"({ca['pooled_r_label']}).", ""] + _agreement_table(ca) + [""]
        for title, key in (("by slide", "by_slide"), ("by slide and field", "by_field"), ("by channel", "by_channel"),
                           ("by spot radius (px)", "by_radius"), ("by model version", "by_model_version")):
            if c[key] and (len(c[key]) > 1 or key in ("by_slide", "by_model_version")):
                L += [f"### {title}", ""] + _table(c[key], key[3:]) + [""]
        why = sorted({r for m in c["per_page"].values() for r in m["preliminary_reasons"]})
        if why:
            L += ["Why preliminary: " + "; ".join(why), ""]
    if rep.get("engine_configurations_found"):
        L += ["## Engine configurations found in the stored runs", ""]
        for base, names in rep["engine_configurations_found"].items():
            L += [f"- {base}: " + "; ".join(names)]
        L += ["", "Every configuration is reported separately; none is preferred or merged."]
        L.append("")
    if rep.get("annotation_completeness"):
        ac = rep["annotation_completeness"]
        L += ["## Annotation completeness (raw proposals nobody marked)", "", ac["note"], "",
              "| slide | raw proposals | unmarked | unmarked and bright |", "|---|---|---|---|"]
        for sl, d in ac["per_slide"].items():
            L.append(f"| {sl} | {d['n_raw_proposals']} | {d['n_unmarked']} | {d['n_unmarked_bright']} |")
        L.append("")
    if rep["acceptance"]:
        L += ["## Candidate acceptance and review load", "",
              "Candidates inside fully reviewed regions only. acceptance = share the reviewer kept; needs_review = share carrying "
              "a review reason or an engine disagreement.", "",
              "| slide | channel:status | candidates | accepted | acceptance | needs review | human-added rate |", "|---|---|---|---|---|---|---|"]
        for slide, d in rep["acceptance"].items():
            for k, v in d.items():
                if isinstance(v, dict):
                    L.append(f"| {slide} | {k} | {v['candidates']} | {v['accepted']} | {_f(v['acceptance_rate'])} | "
                             f"{_f(v['needs_review_rate'])} | {_f(d.get('human_added_rate'))} |")
        L.append("")
    if rep["review_time"]:
        L += ["## Review time (estimate)", "",
              f"From action-log timestamps, gaps above {IDLE_CAP_S:.0f} s count as breaks. Null = not measurable.", "",
              "| slide | active minutes | fields reviewed | minutes per field |", "|---|---|---|---|"]
        for r in rep["review_time"]:
            L.append(f"| {r['slide']} | {_f(r['active_review_minutes'], 1)} | {r['fields_reviewed']} | {_f(r['minutes_per_field'], 2)} |")
        L.append("")
    if rep["skipped"]:
        L += ["## Skipped", ""] + [f"- {s['analysis_id']}: {s['reason']}" for s in rep["skipped"]] + [""]
    if rep["warnings"]:
        L += ["## Warnings", ""] + [f"- {w}" for w in rep["warnings"]] + [""]
    L += ["## Definitions", ""] + [f"- **{k}**: {v}" for k, v in rep["definitions"].items()] + [""]
    return "\n".join(L)


# ====================================================================== model on the locked test crops
# A predictor receives the network INPUT (page-normalised float32, block-mean downscaled by the model's own factor) and
# returns an (N, 2) array of (y, x) in THAT frame. evaluate_on_snapshot prepares the input and maps the points back to
# native pixels with x = x_s * d + (d-1)/2, so a model is never run at a scale or normalisation it was not trained for.
Predictor = Callable[[np.ndarray], np.ndarray]


def make_spotiflow_predictor(base: str, data_dir: str | Path, *, device: str = "cpu", prob_thresh: Optional[float] = None,
                             downscale: int = 1) -> Predictor:
    from .spotiflow_io import load_model, pick_device
    dev = pick_device(device)
    model = load_model(base, data_dir, inference_mode=True, device=dev)

    def predict(img: np.ndarray) -> np.ndarray:
        kw = {"subpix": False} if downscale > 1 else {}     # sub-pixel refinement does not exist on a downscaled image
        # normalizer=None: the input is already normalised with the PAGE's lo/hi; Spotiflow must not re-normalise it
        pts, _ = model.predict(img, prob_thresh=prob_thresh, normalizer=None, verbose=False, device=dev, **kw)
        return np.asarray(pts, float).reshape(-1, 2)
    return predict


def evaluate_on_snapshot(data_dir: str | Path, version: str, predict: Predictor, *, split: str = "test",
                         tolerance_px: float = DEFAULT_TOLERANCE_PX, downscale: int = 1,
                         normaliser: str = DEFAULT_NORMALISER) -> dict:
    """Run a predictor on every crop of ``split`` AT THE MODEL'S CONFIGURATION (``downscale``, ``normaliser``; page-level
    lo/hi of each crop's source page, from the manifest) and score it against the labels (human-confirmed points) in
    native pixels. Metrics are at ``tolerance_px`` and always also at 5 / 10 / 15 px."""
    from .dataset import load_manifest
    downscale = check_downscale(downscale)
    check_normaliser(normaliser)
    m = load_manifest(data_dir, version)
    crops = [c for c in m.crops if c.split == split]
    if not crops:
        raise MLError(f"snapshot {version} has no '{split}' crops")
    items = []
    for c in crops:
        img, lab = load_crop(data_dir, version, c)
        x, _, _, off = prepare_crop(img, lab, lo=c.norm_lo, hi=c.norm_hi, downscale=downscale, normaliser=normaliser,
                                    origin_xy=c.box_xyxy[:2])
        pred_s = np.asarray(predict(x), float).reshape(-1, 2)
        pred = from_downscaled(pred_s, downscale) + np.array(off, float)
        items.append({"slide": c.group_id, "field": c.region_name or c.region_id, "page": c.page, "role": c.channel_role,
                      "radius": "", "model_version": "", "unit": c.crop_id, "pred": [(float(x_), float(y_)) for y_, x_ in pred],
                      "ref": [(float(x_), float(y_)) for y_, x_ in lab]})
    n_groups = len({c.group_id for c in crops})
    why = preliminary_reasons(len(items), sum(len(i["ref"]) for i in items), 2 if n_groups > 1 else 1)
    why = [w for w in why if "annotator" not in w]
    _ensure_matches(items, tolerances_for(tolerance_px))
    met = metrics_of(items, tolerance_px)
    page_events: dict = defaultdict(int)
    for it in items:
        page_events[it["page"]] += len(it["ref"])
    return {"metrics": met, "per_page": _slice(items, "page", tolerance_px, 2, page_events),
            "count_agreement": count_agreement_by_slide(items), "precision_label": PRECISION_LABEL,
            "n_crops": len(items), "n_groups": n_groups, "tolerance_px": tolerance_px, "split": split,
            "config": {"downscale": downscale, "normaliser": normaliser},
            "label": "preliminary" if why else "ok", "preliminary_reasons": why}
