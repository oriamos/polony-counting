"""Deterministic one-to-one spatial matching of Big-FISH and Spotiflow detections of the same channel page.

Neither engine is ground truth: this only records WHERE the two engines agree in space. Nothing is rejected,
no coordinate is averaged (each side keeps its own position; the distance is preserved).

Algorithm (same style as detection/matching.py)
* Groups: (page, channel_role) -- pages / channels are never merged.
* Candidate edges: all (bigfish, spotiflow) pairs with distance <= tolerance_px (kd-tree).
* Per connected component of the candidate graph a Hungarian assignment (scipy linear_sum_assignment)
  minimising the summed distance, gated at the tolerance: a maximum number of pairs is formed first, then the
  smallest total distance. Each detection is used at most once (no duplicate matches).
* Inputs are sorted by (y, x, id) first, so the result is independent of the input order.
* Statuses: agreement (matched) / bigfish_only / spotiflow_only.
* ``ambiguous_with``: ids of OTHER-engine detections that are inside the tolerance of a member of the pair (or of
  an unmatched detection) and whose distance d2 competes with the chosen distance d1:
  d2 - d1 < 1.5 px  or  d2 < 1.25 * d1  (the margins of detection/matching.py). Informational only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..schemas import ConsensusStatus, ConsensusSummary, RawDetection

AMBIGUITY_ABS_PX = 1.5
AMBIGUITY_REL = 0.25
_BIG = 1e9


@dataclass
class ConsensusPair:
    status: ConsensusStatus
    bigfish: Optional[RawDetection] = None
    spotiflow: Optional[RawDetection] = None
    distance_px: Optional[float] = None
    ambiguous_with: list[str] = field(default_factory=list)

    @property
    def page(self) -> int:
        return (self.bigfish or self.spotiflow).page

    @property
    def channel_role(self) -> Optional[str]:
        return (self.bigfish or self.spotiflow).channel_role


def _key(d: RawDetection):
    return (round(d.y, 6), round(d.x, 6), d.id)


def _competing(d1: float, d2: float, abs_px: float, rel: float) -> bool:
    return (d2 - d1) < abs_px or d2 < (1.0 + rel) * d1


def _match_group(bf: list[RawDetection], sf: list[RawDetection], tol: float, abs_px: float,
                 rel: float) -> list[ConsensusPair]:
    bf, sf = sorted(bf, key=_key), sorted(sf, key=_key)
    nb, ns = len(bf), len(sf)
    ei = np.zeros(0, int); ej = np.zeros(0, int); ed = np.zeros(0)
    if nb and ns and tol >= 0:
        bxy = np.array([[d.x, d.y] for d in bf]).reshape(-1, 2)
        sxy = np.array([[d.x, d.y] for d in sf]).reshape(-1, 2)
        rec = cKDTree(bxy).sparse_distance_matrix(cKDTree(sxy), tol, output_type="ndarray")
        ei, ej, ed = rec["i"].astype(int), rec["j"].astype(int), rec["v"].astype(float)
        o = np.lexsort((ej, ei))
        ei, ej, ed = ei[o], ej[o], ed[o]
        keep = ed <= tol
        ei, ej, ed = ei[keep], ej[keep], ed[keep]
    mb = np.full(nb, -1, int)
    ms = np.full(ns, -1, int)
    if ei.size:
        g = coo_matrix((np.ones(ei.size), (ei, nb + ej)), shape=(nb + ns, nb + ns))
        _, lab = connected_components(g, directed=False)
        ce = lab[ei]
        for c in np.unique(ce):
            sel = ce == c
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
                    mb[rr[a]] = gg[b]
                    ms[gg[b]] = rr[a]
    chosen_b = {i: float(d) for i, j, d in zip(ei, ej, ed) if mb[i] == j}   # bf idx -> chosen distance
    chosen_s = {j: float(d) for i, j, d in zip(ei, ej, ed) if mb[i] == j}
    amb_b: dict[int, set[str]] = {}
    amb_s: dict[int, set[str]] = {}
    for i, j, d in zip(ei, ej, ed):
        i, j, d = int(i), int(j), float(d)
        if mb[i] == j:
            continue
        hit = (i in chosen_b and _competing(chosen_b[i], d, abs_px, rel)) or \
              (j in chosen_s and _competing(chosen_s[j], d, abs_px, rel))
        if hit:
            amb_b.setdefault(i, set()).add(sf[j].id)
            amb_s.setdefault(j, set()).add(bf[i].id)
    out: list[ConsensusPair] = []
    for i in range(nb):
        if mb[i] >= 0:
            j = int(mb[i])
            out.append(ConsensusPair(ConsensusStatus.agreement, bf[i], sf[j], chosen_b[i],
                                     sorted(amb_b.get(i, set()) | amb_s.get(j, set()))))
        else:
            out.append(ConsensusPair(ConsensusStatus.bigfish_only, bf[i], None, None, sorted(amb_b.get(i, set()))))
    for j in range(ns):
        if ms[j] < 0:
            out.append(ConsensusPair(ConsensusStatus.spotiflow_only, None, sf[j], None, sorted(amb_s.get(j, set()))))
    out.sort(key=lambda p: (_key(p.bigfish or p.spotiflow), p.status.value))
    return out


def match_engines(bigfish: Sequence[RawDetection], spotiflow: Sequence[RawDetection], tolerance_px: float, *,
                  ambiguity_abs_px: float = AMBIGUITY_ABS_PX, ambiguity_rel: float = AMBIGUITY_REL
                  ) -> list[ConsensusPair]:
    """One consensus candidate per matched pair / unmatched detection. Pass the VALID detections you want
    matched; invalid (outside the counting area) ones are the caller's to keep as raw only."""
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be >= 0")
    groups: dict[tuple, tuple[list, list]] = {}
    for d in bigfish:
        groups.setdefault((d.page, d.channel_role or ""), ([], []))[0].append(d)
    for d in spotiflow:
        groups.setdefault((d.page, d.channel_role or ""), ([], []))[1].append(d)
    out: list[ConsensusPair] = []
    for k in sorted(groups):
        b, s = groups[k]
        out += _match_group(b, s, float(tolerance_px), ambiguity_abs_px, ambiguity_rel)
    return out


def summarize(pairs: Sequence[ConsensusPair]) -> list[ConsensusSummary]:
    """Counts per (page, channel_role); groups are never merged."""
    acc: dict[tuple, ConsensusSummary] = {}
    for p in pairs:
        k = (p.page, p.channel_role)
        s = acc.setdefault(k, ConsensusSummary(page=p.page, channel_role=p.channel_role))
        setattr(s, p.status.value, getattr(s, p.status.value) + 1)
    return [acc[k] for k in sorted(acc, key=lambda k: (k[0], k[1] or ""))]
