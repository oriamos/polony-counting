"""Helpers of the hybrid / Spotiflow-only detection modes (pipeline integration).

* ``make_validity``: validity of raw-detection points under the ONE region rule of detection/masks.py
  (``points_regions``: half-open pixel-centre, first counting region wins, exclusions win) plus the automatic
  saturated exclusion and the green alignment strip.
* ``measure_spots_at``: a RawSpot for a Spotiflow point, with peak / background / snr (and the shape extras)
  MEASURED by the SAME code as every Big-FISH spot (``BigFishEngine._measure``) on the same background-corrected
  image. The Spotiflow x / y are kept exactly (never refined or averaged with Big-FISH). Note: BigFishEngine's
  measure interpolates the peak at its own LoG-refined centre next to the (rounded) Spotiflow pixel, and the
  LoG response is the Big-FISH one; both are measurements of the image, not Spotiflow outputs.
* ``union_spots``: consensus candidates -> RawSpot list for the existing cross-channel association.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..consensus.matching import ConsensusPair
from ..engines.base import EXCLUDED_REASON, PointValidity
from ..schemas import ChannelEvidence, ConsensusStatus, RawDetection, Region
from .engine import RawSpot
from .masks import points_regions

STRIP_REASON = "alignment_strip_no_data"


@dataclass
class SpotInfo:
    """Provenance of one RawSpot that entered the association."""
    status: Optional[ConsensusStatus] = None
    bigfish: Optional[RawDetection] = None
    spotiflow: Optional[RawDetection] = None
    distance_px: Optional[float] = None
    ambiguous_with: tuple = ()


def make_validity(rule_regions: Sequence[Region], shape: tuple[int, int], auto: Optional[np.ndarray],
                  nodata: Optional[np.ndarray]) -> PointValidity:
    h, w = shape

    def check(xs: np.ndarray, ys: np.ndarray):
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        rids = points_regions(xs, ys, rule_regions)
        ok = np.array([r is not None for r in rids], bool) if len(rids) else np.zeros(0, bool)
        why: list[Optional[str]] = [None] * len(rids)
        for i in range(len(rids)):
            yi = min(max(int(math.floor(ys[i] + 0.5)), 0), h - 1)
            xi = min(max(int(math.floor(xs[i] + 0.5)), 0), w - 1)
            if not (-0.5 <= xs[i] < w - 0.5 and -0.5 <= ys[i] < h - 0.5):
                ok[i], why[i] = False, "outside_image"
            elif not ok[i]:
                why[i] = EXCLUDED_REASON
            elif auto is not None and auto[yi, xi]:
                ok[i], why[i] = False, "automatic_saturated_exclusion"
            elif nodata is not None and nodata[yi, xi]:
                ok[i], why[i] = False, STRIP_REASON
        return ok, [r if bool(o) else None for r, o in zip(rids, ok)], why

    return check


def measure_spots_at(engine, params, corrected_full: np.ndarray, measured_full: np.ndarray, valid: np.ndarray,
                     raw: np.ndarray, xs: Sequence[float], ys: Sequence[float], thr_abs: float) -> list[RawSpot]:
    if len(xs) == 0:
        return []
    h, w = corrected_full.shape
    filled = engine._prepare(corrected_full, valid, raw, float(params.spot_radius_px))
    xi = np.clip(np.floor(np.asarray(xs, float) + 0.5), 0, w - 1).astype(np.int64)
    yi = np.clip(np.floor(np.asarray(ys, float) + 0.5), 0, h - 1).astype(np.int64)
    coords = np.stack([yi, xi], axis=1)
    spots = engine._measure(filled, valid, coords, [False] * len(xs), params, raw, float(thr_abs), measured_full, None)
    for s, x, y in zip(spots, xs, ys):
        s.x, s.y = float(x), float(y)
    return spots


def union_spots(pairs: Sequence[ConsensusPair], bf_spot_of: dict, sf_spots: dict) -> tuple[list[RawSpot], dict]:
    """Return (spots, info) where info maps ``id(spot)`` -> SpotInfo. Agreement and bigfish_only use the
    Big-FISH RawSpot (unchanged position / peak / snr); spotiflow_only uses the measured Spotiflow spot
    (``sf_spots``: raw id -> RawSpot)."""
    spots, info = [], {}
    for p in pairs:
        if p.bigfish is not None:
            s = bf_spot_of[p.bigfish.id]
        else:
            s = sf_spots[p.spotiflow.id]
        spots.append(s)
        info[id(s)] = SpotInfo(p.status, p.bigfish, p.spotiflow, p.distance_px, tuple(p.ambiguous_with))
    return spots, info


def fill_evidence(ev: ChannelEvidence, si: SpotInfo) -> None:
    ev.consensus_status = si.status
    if si.bigfish is not None:
        ev.bigfish_raw_id, ev.bigfish_x, ev.bigfish_y, ev.bigfish_score = (
            si.bigfish.id, si.bigfish.x, si.bigfish.y, si.bigfish.score)
    if si.spotiflow is not None:
        ev.spotiflow_raw_id, ev.spotiflow_x, ev.spotiflow_y, ev.spotiflow_score = (
            si.spotiflow.id, si.spotiflow.x, si.spotiflow.y, si.spotiflow.score)
    ev.match_distance_px = None if si.distance_px is None else float(si.distance_px)
