"""Review reasons from MEASURED quantities only (no confidence score).

weak_signal       snr < weak_snr (default 5) in any detected channel
near_threshold    LoG response < near_threshold_factor * threshold (default 1.25)
saturated         raw pixel at dtype max within the spot radius
dense_cluster     another same-channel spot within 2 * minimum_distance_px
                  (or the spot came from dense decomposition)
near_edge         distance to the valid-mask boundary (image border counts) < 2 * spot radius
suspicious_shape  half-max blob elongation (sqrt of second-moment eigenvalue ratio) > elongation_max (1.8)
debris_like       scale ratio (0.3r/0.4r vs 1r/1.4r normalised LoG, 5x5 max) >= 1.0 in any channel.
                  Single-channel objects keep their class (flag only); a matched red+green pair with
                  debris-like evidence in BOTH channels is NOT auto-classed yellow: it becomes
                  ``uncertain`` (a compact bright speck in both channels is more likely autofluorescent
                  debris/dust than a dual-labelled polony). One debris channel: stays yellow, queued.
suspicious_shape  also set for a merged saturated plateau much larger than a spot, elongated (> 1.3)
                  or carrying more than one LoG maximum (touching saturated polonies; merge kept)
(plateau-merged spots: weak_signal / near_threshold are skipped, SNR is undefined there)
ambiguous_match   set by matching.associate
Defaults are starting points exposed for configuration, not validated cut-offs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from ..schemas import DetectedObject, DetectionParams, ObjectClass, ReviewReason
from .engine import RawSpot
from .matching import spot_index

_ORDER = [ReviewReason.weak_signal, ReviewReason.ambiguous_match, ReviewReason.dense_cluster,
          ReviewReason.saturated, ReviewReason.suspicious_shape, ReviewReason.near_edge,
          ReviewReason.near_threshold, ReviewReason.debris_like]


@dataclass
class ReviewConfig:
    weak_snr: float = 5.0
    near_threshold_factor: float = 1.25
    elongation_max: float = 1.8
    edge_radius_factor: float = 2.0
    dense_distance_factor: float = 2.0
    debris_ratio: float = 1.0


def edge_distance_map(valid: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Distance (px) from each pixel centre to the nearest invalid pixel or the
    image border (border treated as a boundary: spots there are truncated)."""
    v = np.ones(shape, bool) if valid is None else valid
    d = ndi.distance_transform_edt(np.pad(v, 1, constant_values=False))[1:-1, 1:-1]
    return d - 0.5  # centre-to-boundary (boundary lies half a pixel beyond the last valid centre)


def apply_review_reasons(objects: list[DetectedObject], red: Sequence[RawSpot], green: Sequence[RawSpot],
                         red_params: DetectionParams, green_params: DetectionParams,
                         valid: Optional[np.ndarray], shape: tuple[int, int],
                         cfg: Optional[ReviewConfig] = None) -> None:
    """Mutates ``objects`` (adds reasons). ``red``/``green`` must be the sorted
    lists used for association (spot ids index into them)."""
    cfg = cfg or ReviewConfig()
    edt = edge_distance_map(valid, shape)
    trees = {}
    for key, spots in (("red", red), ("green", green)):
        trees[key] = cKDTree(np.array([[s.x, s.y] for s in spots])) if len(spots) else None
    params = {"red": red_params, "green": green_params}
    lists = {"red": red, "green": green}

    for o in objects:
        reasons = set(o.review_reasons)
        debris = {"red": False, "green": False}
        for key in ("red", "green"):
            ev = getattr(o, key)
            if not ev.detected:
                continue
            idx = spot_index(ev.spot_id)
            s = lists[key][idx]
            p = params[key]
            plateau = s.plateau_area_px is not None  # merged saturated plateau: SNR / LoG response at the
            # integer position are undefined or meaningless (clipped peak, annulus inside the plateau)
            if not plateau and s.snr is not None and s.snr < cfg.weak_snr:
                reasons.add(ReviewReason.weak_signal)
            if (not plateau and s.threshold and s.log_response is not None
                    and s.log_response < cfg.near_threshold_factor * s.threshold):
                reasons.add(ReviewReason.near_threshold)
            if s.saturated:
                reasons.add(ReviewReason.saturated)
            if s.elongation is not None and s.elongation > cfg.elongation_max:
                reasons.add(ReviewReason.suspicious_shape)
            if s.scale_ratio is not None and s.scale_ratio >= cfg.debris_ratio:
                reasons.add(ReviewReason.debris_like)
                debris[key] = True
            if s.plateau_oversize:
                reasons.add(ReviewReason.suspicious_shape)
            if s.simulated:
                reasons.add(ReviewReason.dense_cluster)
            tree = trees[key]
            if tree is not None:
                near = tree.query_ball_point([s.x, s.y], cfg.dense_distance_factor * p.minimum_distance_px)
                if len(near) > 1:
                    reasons.add(ReviewReason.dense_cluster)
            yi, xi = int(round(s.y)), int(round(s.x))
            yi, xi = min(max(yi, 0), shape[0] - 1), min(max(xi, 0), shape[1] - 1)
            if edt[yi, xi] < cfg.edge_radius_factor * p.spot_radius_px:
                reasons.add(ReviewReason.near_edge)
        o.review_reasons = [r for r in _ORDER if r in reasons]
        if o.cls == ObjectClass.yellow and debris["red"] and debris["green"]:
            o.cls = ObjectClass.uncertain
            o.history.append("auto: debris-like (compact, scale ratio >= "
                             f"{cfg.debris_ratio:g}) in BOTH channels -> uncertain instead of yellow")
