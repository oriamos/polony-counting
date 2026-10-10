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

from .common import MLError, hash_obj, utc_now
from .dataset import POSITIVE, ROLES, classify_objects, fully_reviewed_regions, load_crop, reviewers_from

DEFAULT_TOLERANCE_PX = 5.0
MIN_FIELDS = 30
MIN_EVENTS_PER_PAGE = 30
IDLE_CAP_S = 120.0
CONFIGS = ("bigfish_default", "bigfish_sensitive", "spotiflow_pretrained", "spotiflow_finetuned",
           "agreement_subset", "union_before_review", "final_human_reviewed")

DEFINITIONS = {
    "tp / fp / fn": "matched predictions / predictions without a reference partner / reference points nobody predicted "
                    "(one-to-one matching within the tolerance, minimum total distance)",
    "precision": "tp / (tp + fp): of the predicted points, the share that has a reference partner",
    "recall": "tp / (tp + fn): of the reference points, the share that was predicted",
    "f1": "harmonic mean of precision and recall",
    "mean_match_distance_px": "mean distance between matched prediction and reference (pixels)",
    "count_mae": "mean over fields of |predicted count - reference count|",
    "count_bias": "mean over fields of (predicted - reference); positive = over-counting",
    "slope": "least-squares line through the origin, predicted = slope x reference, over fields",
    "pearson_r": "Pearson correlation of the per-field counts (predicted vs reference)",
    "fp_per_field / fn_per_field": "average false positives / false negatives per field",
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


def metrics_of(items: Iterable[dict], tol: float = DEFAULT_TOLERANCE_PX) -> dict:
    """Aggregate metrics over field items ({'pred': [(x,y)], 'ref': [(x,y)]}). Each item = one field on one page."""
    tp = fp = fn = 0
    dists: list[float] = []
    cp, cr = [], []
    n = 0
    for it in items:
        n += 1
        pairs, up, ur = match_points(np.array(it["pred"], float), np.array(it["ref"], float), tol)
        tp += len(pairs)
        fp += len(up)
        fn += len(ur)
        dists += [p[2] for p in pairs]
        cp.append(len(it["pred"]))
        cr.append(len(it["ref"]))
    out = {"n_fields": n, "n_reference": int(sum(cr)), "n_predicted": int(sum(cp)), "tp": tp, "fp": fp, "fn": fn,
           **prf(tp, fp, fn), "mean_match_distance_px": (float(np.mean(dists)) if dists else None)}
    if n:
        diff = np.asarray(cp, float) - np.asarray(cr, float)
        sxx = float(np.sum(np.asarray(cr, float) ** 2))
        out.update({"count_mae": float(np.mean(np.abs(diff))), "count_bias": float(np.mean(diff)),
                    "slope": (float(np.sum(np.asarray(cr, float) * np.asarray(cp, float)) / sxx) if sxx else None),
                    "pearson_r": pearson(cr, cp), "fp_per_field": fp / n, "fn_per_field": fn / n})
    else:
        out.update({"count_mae": None, "count_bias": None, "slope": None, "pearson_r": None,
                    "fp_per_field": None, "fn_per_field": None})
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
def _slice(items: list[dict], key: str, tol: float, n_annotators: int, page_events: dict) -> dict:
    groups: dict = defaultdict(list)
    for it in items:
        groups[str(it.get(key))].append(it)
    out = {}
    for k, its in sorted(groups.items()):
        m = metrics_of(its, tol)
        pages = {it["page"] for it in its}
        ev = min((page_events.get(p, 0) for p in pages), default=0) if pages else 0
        why = preliminary_reasons(m["n_fields"], ev, n_annotators)
        out[k] = dict(m, label="preliminary" if why else "ok", preliminary_reasons=why)
    return out


def summarise_config(items: list[dict], tol: float, n_annotators: int) -> dict:
    """Per page (never merged across pages), plus a pooled row over the pages that carry enough events only."""
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
            "by_slide": _slice(items, "slide", tol, n_annotators, page_events),
            "by_field": _slice(items, "field", tol, n_annotators, page_events),
            "by_channel": _slice(items, "role", tol, n_annotators, page_events),
            "by_radius": _slice(items, "radius", tol, n_annotators, page_events),
            "by_model_version": _slice(items, "model_version", tol, n_annotators, page_events)}


# ====================================================================== gathering from the store
def _moved(o) -> bool:
    return any(" moved from " in h for h in (o.history or []))


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


def gather_analysis(store, aid: str, g: Gathered, *, reference: str, tol: float, numbering: str = "index",
                    registry_ids: Optional[set] = None, configs: Sequence[str] = CONFIGS) -> None:
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
            g.items[cfg].append({"config": cfg, "analysis_id": aid, "slide": slide,
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
            latest: dict = {}
            for run in runs:
                if run.status != "ok":
                    continue
                cfg = _config_for_run(run, registry_ids)
                if cfg in wanted:
                    k = (cfg, run.page)
                    if k not in latest or (run.started_at or "") >= (latest[k].started_at or ""):
                        latest[k] = run
            for cfg in wanted:
                if not any(k[0] == cfg for k in latest):
                    g.unavailable.setdefault(cfg, "no stored engine run matches this configuration")
            for (cfg, page), run in sorted(latest.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                role = next((rl for rl, pg in mapping.items() if pg == page), None)
                if role is None:
                    continue
                dets = dets_fn(aid, run_id=run.run_id, valid=True)
                emit(cfg, role, _model_version(run), [(d.x, d.y) for d in dets])

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
    g = Gathered()
    for aid in sorted(set(analysis_ids)):
        if not store.exists(aid):
            raise MLError(f"analysis '{aid}' not found")
        gather_analysis(store, aid, g, reference=reference, tol=tolerance_px, numbering=numbering,
                        registry_ids=registry_ids, configs=configs)
    n_ann = len({r for r in g.reviewers if not r.startswith("unknown:")} or ({"x"} if g.reviewers else set()))
    # unknown reviewers are one anonymous session each; several analyses by 'unknown' still cannot establish 2 annotators
    n_ann = len({r for r in g.reviewers if not r.startswith("unknown:")}) or (1 if g.reviewers else 0)
    report = {"generated_at": utc_now(), "reference": reference, "tolerance_px": tolerance_px, "numbering": numbering,
              "analyses": g.analyses, "skipped": g.skipped, "n_annotators": n_ann,
              "reviewers": sorted(r for r in g.reviewers if not r.startswith("unknown:")) or ["unknown"],
              "configs": {}, "acceptance": acceptance_summary(g.acceptance), "review_time": g.review_time,
              "warnings": g.warnings, "definitions": DEFINITIONS}
    for c in configs:
        its = g.items.get(c, [])
        if its:
            report["configs"][c] = {"available": True, **summarise_config(its, tolerance_px, n_ann)}
        else:
            report["configs"][c] = {"available": False, "reason": g.unavailable.get(c, "no data")}
    report["report_sha256"] = hash_obj({k: v for k, v in report.items() if k not in ("generated_at", "report_sha256")})
    return report


# ====================================================================== markdown
def _f(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _table(slices: dict, first: str) -> list[str]:
    cols = ["n_fields", "n_reference", "n_predicted", "tp", "fp", "fn", "precision", "recall", "f1", "mean_match_distance_px",
            "count_mae", "count_bias", "slope", "pearson_r", "fp_per_field", "fn_per_field"]
    head = [first] + cols + ["label"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for k, m in slices.items():
        out.append("| " + " | ".join([str(k)] + [_f(m.get(c)) for c in cols] + [m["label"]]) + " |")
    return out


def report_markdown(rep: dict) -> str:
    L = [f"# Evaluation report", "",
         f"Reference: **{rep['reference']}**; matching tolerance **{rep['tolerance_px']} px** (one-to-one); "
         f"annotators identified: **{rep['n_annotators']}** ({', '.join(rep['reviewers'])}); analyses: {len(rep['analyses'])}.", "",
         "Rows labelled **preliminary** have too little evidence for a conclusion (fewer than "
         f"{MIN_FIELDS} fields, fewer than {MIN_EVENTS_PER_PAGE} reference events on the page, or one annotator). "
         "Pages are never merged silently: totals are per TIFF page; a pooled row lists the pages it contains and the sparse pages it leaves out.", ""]
    for name, c in rep["configs"].items():
        L += [f"## {name}", ""]
        if not c["available"]:
            L += [f"_not available: {c['reason']}_", ""]
            continue
        L += ["### per TIFF page", ""] + _table(c["per_page"], "page") + [""]
        if c["pooled_dense_pages"]:
            p = c["pooled_dense_pages"]
            L += [f"Pooled over pages {', '.join(p['pages'])} only (sparse pages left out: "
                  f"{', '.join(p['excluded_sparse_pages']) or 'none'}):", ""] + _table({"pooled": p}, "scope") + [""]
        for title, key in (("by slide", "by_slide"), ("by field", "by_field"), ("by channel", "by_channel"),
                           ("by spot radius (px)", "by_radius"), ("by model version", "by_model_version")):
            if c[key] and (len(c[key]) > 1 or key in ("by_slide", "by_model_version")):
                L += [f"### {title}", ""] + _table(c[key], key[3:]) + [""]
        why = sorted({r for m in c["per_page"].values() for r in m["preliminary_reasons"]})
        if why:
            L += ["Why preliminary: " + "; ".join(why), ""]
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
Predictor = Callable[[np.ndarray], np.ndarray]    # image -> (N,2) array of (y, x)


def make_spotiflow_predictor(base: str, data_dir: str | Path, *, device: str = "cpu", prob_thresh: Optional[float] = None) -> Predictor:
    from .spotiflow_io import load_model, pick_device
    dev = pick_device(device)
    model = load_model(base, data_dir, inference_mode=True, device=dev)

    def predict(img: np.ndarray) -> np.ndarray:
        pts, _ = model.predict(img, prob_thresh=prob_thresh, verbose=False, device=dev)
        return np.asarray(pts, float).reshape(-1, 2)
    return predict


def evaluate_on_snapshot(data_dir: str | Path, version: str, predict: Predictor, *, split: str = "test",
                         tolerance_px: float = DEFAULT_TOLERANCE_PX) -> dict:
    """Run a predictor on every crop of ``split`` and score it against the labels (human-confirmed points)."""
    from .dataset import load_manifest
    m = load_manifest(data_dir, version)
    crops = [c for c in m.crops if c.split == split]
    if not crops:
        raise MLError(f"snapshot {version} has no '{split}' crops")
    items = []
    for c in crops:
        img, lab = load_crop(data_dir, version, c)
        pred = np.asarray(predict(img), float).reshape(-1, 2)
        items.append({"slide": c.group_id, "field": c.region_name or c.region_id, "page": c.page, "role": c.channel_role,
                      "radius": "", "model_version": "", "pred": [(float(x), float(y)) for y, x in pred],
                      "ref": [(float(x), float(y)) for y, x in lab]})
    n_groups = len({c.group_id for c in crops})
    why = preliminary_reasons(len(items), sum(len(i["ref"]) for i in items), 2 if n_groups > 1 else 1)
    why = [w for w in why if "annotator" not in w]
    met = metrics_of(items, tolerance_px)
    return {"metrics": met, "per_page": _slice(items, "page", tolerance_px, 2, defaultdict(lambda: MIN_EVENTS_PER_PAGE)),
            "n_crops": len(items), "n_groups": n_groups, "tolerance_px": tolerance_px, "split": split,
            "label": "preliminary" if why else "ok", "preliminary_reasons": why}
