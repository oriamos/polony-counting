"""Cross-channel association (red <-> green) as a constrained one-to-one assignment.

Algorithm
* Candidate pairs: all (red, green) pairs with distance <= tolerance_px (kd-tree).
* Per connected component of the candidate graph, solve a Hungarian assignment
  minimising the summed distance, gated at the tolerance (non-candidates cost a
  large constant and are discarded afterwards). So a maximum number of pairs is
  formed first and, among those, the smallest total distance -- each spot is used
  at most once in the one-to-one association (a spot is never in two objects; whether
  two spots are really one physical object, or one object was split into two spots,
  is NOT decided here and is what review reasons / the reviewer are for).
* matched pair -> ``yellow`` at the mean of the two positions; unmatched red /
  green -> ``red`` / ``green``.

Ambiguity rule (documented, tested; margins are keyword-only parameters)
For each matched pair with distance d1, look at every OTHER spot of the opposite
channel within tolerance + max(ambiguity_abs_px, ambiguity_rel * tolerance) of either member
(alternative partner at distance d2; so a 5.9 px pair with a 6.1 px alternative IS ambiguous
although the alternative is beyond the 6 px matching tolerance). The pair is ambiguous when
    d2 - d1 < ambiguity_abs_px (1.5)      or      d2 < (1 + ambiguity_rel) * d1  (25 %).
All objects tied together by such alternatives (the pair and the competing
spot's object) become ``uncertain`` with review reason ``ambiguous_match`` and
``ambiguous_with`` = the other objects' ids. Nothing is resolved automatically.

Determinism: inputs are sorted by (y, x, -peak); spot ids are 'r000001'/'g000001'
in that order, object ids 'o000001'... in (y, x) order of the object position.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..schemas import ChannelEvidence, DetectedObject, ObjectClass, ReviewReason
from .engine import RawSpot

AMBIGUITY_ABS_PX = 1.5
AMBIGUITY_REL = 0.25
_BIG = 1e9


def sort_spots(spots: Sequence[RawSpot]) -> list[RawSpot]:
    """Canonical deterministic order."""
    return sorted(spots, key=lambda s: (round(s.y, 6), round(s.x, 6), -s.peak))


def spot_id(prefix: str, index: int) -> str:
    return f"{prefix}{index + 1:06d}"


def spot_index(sid: Optional[str]) -> Optional[int]:
    return int(sid[1:]) - 1 if sid else None


def evidence_from(spot: RawSpot, sid: str) -> ChannelEvidence:
    bg = None if (spot.background is None or math.isnan(spot.background)) else float(spot.background)
    return ChannelEvidence(detected=True, x=float(spot.x), y=float(spot.y), peak_intensity=float(spot.peak),
                           local_background=bg, snr=None if spot.snr is None else float(spot.snr),
                           saturated=bool(spot.saturated), spot_id=sid)


def associate(red: list[RawSpot], green: list[RawSpot], tolerance_px: float, *,
              ambiguity_abs_px: float = AMBIGUITY_ABS_PX,
              ambiguity_rel: float = AMBIGUITY_REL) -> list[DetectedObject]:
    red, green = sort_spots(red), sort_spots(green)
    nr, ng = len(red), len(green)
    rxy = np.array([[s.x, s.y] for s in red]).reshape(-1, 2)
    gxy = np.array([[s.x, s.y] for s in green]).reshape(-1, 2)

    # ---- candidate edges
    ei = np.zeros(0, int); ej = np.zeros(0, int); ed = np.zeros(0)
    if nr and ng and tolerance_px > 0:
        rec = cKDTree(rxy).sparse_distance_matrix(cKDTree(gxy), tolerance_px, output_type="ndarray")
        ei, ej, ed = rec["i"].astype(int), rec["j"].astype(int), rec["v"].astype(float)
        o = np.lexsort((ej, ei))
        ei, ej, ed = ei[o], ej[o], ed[o]

    # ---- Hungarian per connected component
    match_r = np.full(nr, -1, int)
    match_g = np.full(ng, -1, int)
    if ei.size:
        g = coo_matrix((np.ones(ei.size), (ei, nr + ej)), shape=(nr + ng, nr + ng))
        _, lab = connected_components(g, directed=False)
        comp_of_edge = lab[ei]
        for c in np.unique(comp_of_edge):
            sel = comp_of_edge == c
            ri, gj, dd = ei[sel], ej[sel], ed[sel]
            rr, gg = np.unique(ri), np.unique(gj)
            ridx = {v: k for k, v in enumerate(rr)}
            gidx = {v: k for k, v in enumerate(gg)}
            cost = np.full((len(rr), len(gg)), _BIG)
            for a, b, d in zip(ri, gj, dd):
                cost[ridx[a], gidx[b]] = d
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols):
                if cost[a, b] < _BIG:
                    match_r[rr[a]] = gg[b]
                    match_g[gg[b]] = rr[a]

    # ---- objects (temporary index = creation order)
    objs: list[dict] = []
    obj_of_r = np.full(nr, -1, int)
    obj_of_g = np.full(ng, -1, int)
    dist = {(int(i), int(j)): float(d) for i, j, d in zip(ei, ej, ed)}
    for i in range(nr):
        j = match_r[i]
        if j >= 0:
            obj_of_r[i] = obj_of_g[j] = len(objs)
            objs.append(dict(r=i, g=int(j), x=(red[i].x + green[j].x) / 2, y=(red[i].y + green[j].y) / 2,
                             d=dist[(i, int(j))]))
        else:
            obj_of_r[i] = len(objs)
            objs.append(dict(r=i, g=None, x=red[i].x, y=red[i].y, d=None))
    for j in range(ng):
        if match_g[j] < 0:
            obj_of_g[j] = len(objs)
            objs.append(dict(r=None, g=j, x=green[j].x, y=green[j].y, d=None))

    # ---- ambiguity links between objects
    links: dict[int, set[int]] = {}

    def link(a: int, b: int) -> None:
        if a != b:
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)

    # alternative partners are searched out to tolerance + margin (a 6.1 px alternative of a
    # 5.9 px pair is just as plausible as the pair even though it is beyond the tolerance)
    ext = max(tolerance_px + ambiguity_abs_px, tolerance_px * (1.0 + ambiguity_rel))
    xi_, xj_, xd_ = ei, ej, ed
    if nr and ng and tolerance_px > 0 and ext > tolerance_px:
        rec2 = cKDTree(rxy).sparse_distance_matrix(cKDTree(gxy), ext, output_type="ndarray")
        xi_, xj_, xd_ = rec2["i"].astype(int), rec2["j"].astype(int), rec2["v"].astype(float)
    adj_r: dict[int, list[tuple[int, float]]] = {}
    adj_g: dict[int, list[tuple[int, float]]] = {}
    for i, j, d in zip(xi_, xj_, xd_):
        adj_r.setdefault(int(i), []).append((int(j), float(d)))
        adj_g.setdefault(int(j), []).append((int(i), float(d)))

    def competing(d1: float, d2: float) -> bool:
        return (d2 - d1) < ambiguity_abs_px or d2 < (1.0 + ambiguity_rel) * d1

    for k, o in enumerate(objs):
        if o["r"] is None or o["g"] is None:
            continue
        i, j, d1 = o["r"], o["g"], o["d"]
        for jj, d2 in adj_r.get(i, []):
            if jj != j and competing(d1, d2):
                link(k, int(obj_of_g[jj]))
        for ii, d2 in adj_g.get(j, []):
            if ii != i and competing(d1, d2):
                link(k, int(obj_of_r[ii]))

    # ---- deterministic ids: sort by (y, x)
    order = sorted(range(len(objs)), key=lambda k: (round(objs[k]["y"], 6), round(objs[k]["x"], 6),
                                                    -1 if objs[k]["r"] is not None else 0))
    oid = {k: f"o{n + 1:06d}" for n, k in enumerate(order)}
    out: list[DetectedObject] = []
    for k in order:
        o = objs[k]
        red_ev = evidence_from(red[o["r"]], spot_id("r", o["r"])) if o["r"] is not None \
            else ChannelEvidence(detected=False)
        green_ev = evidence_from(green[o["g"]], spot_id("g", o["g"])) if o["g"] is not None \
            else ChannelEvidence(detected=False)
        if o["r"] is not None and o["g"] is not None:
            cls, hist = ObjectClass.yellow, [f"auto: red {red_ev.spot_id} + green {green_ev.spot_id} "
                                              f"matched at {o['d']:.2f} px"]
        elif o["r"] is not None:
            cls, hist = ObjectClass.red, [f"auto: red-only spot {red_ev.spot_id}"]
        else:
            cls, hist = ObjectClass.green, [f"auto: green-only spot {green_ev.spot_id}"]
        reasons: list[ReviewReason] = []
        amb: list[str] = []
        if k in links:
            cls = ObjectClass.uncertain
            reasons.append(ReviewReason.ambiguous_match)
            amb = sorted(oid[b] for b in links[k])
            hist.append(f"auto: ambiguous association with {', '.join(amb)} -> uncertain")
        out.append(DetectedObject(id=oid[k], x=float(o["x"]), y=float(o["y"]), cls=cls,
                                  red=red_ev, green=green_ev,
                                  association_distance_px=None if o["d"] is None else float(o["d"]),
                                  review_reasons=reasons, ambiguous_with=amb, history=hist))
    return out
