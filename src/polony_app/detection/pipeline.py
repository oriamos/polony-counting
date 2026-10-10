"""End-to-end detection: background correction -> Big-FISH -> association -> review reasons."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import inspect
import math
import platform
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..imaging.preprocess import correct_background
from ..imaging.registration import check_compatible, shifted_valid_region
from ..schemas import DetectedObject, DetectionParams, DetectionRequest, Region, RegionKind
from .bigfish_engine import BigFishEngine
from .engine import SpotEngine
from .errors import DetectionCancelled
from .masks import build_valid_mask, points_regions
from .matching import associate, sort_spots
from .review_reasons import ReviewConfig, apply_review_reasons

DEBRIS_CHANNEL_FRACTION = 0.5   # > this fraction of a channel's spots debris-like => pipeline WARNING
DEBRIS_CHANNEL_MIN = 3          # ... if the channel has at least this many spots

__all__ = ["run_detection", "DetectionResult", "DetectionCancelled", "package_versions"]


@dataclass
class DetectionResult:
    objects: list[DetectedObject]
    package_versions: dict[str, str]
    notes: list[str] = field(default_factory=list)


def package_versions() -> dict[str, str]:
    import bigfish, numpy, scipy, skimage, tifffile
    return {"bigfish": bigfish.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
            "skimage": skimage.__version__, "tifffile": tifffile.__version__,
            "python": platform.python_version()}


def _saturated_outside(raw: np.ndarray, radius: float, min_area: int = 5000) -> np.ndarray:
    """Best-effort exclusion when the user defined no counting region: large
    saturated connected areas (e.g. the slide outside the chamber) dilated by
    ~3 spot radii. Small saturated specks are left alone (they get flagged)."""
    from scipy import ndimage as ndi
    if raw.dtype.kind not in "ui":
        return np.zeros(raw.shape, bool)
    lab, n = ndi.label(raw >= np.iinfo(raw.dtype).max)
    if n == 0:
        return np.zeros(raw.shape, bool)
    area = np.bincount(lab.ravel())
    big = area >= min_area
    big[0] = False
    m = big[lab]
    if not m.any():
        return m
    return ndi.distance_transform_edt(~m) <= max(3.0 * radius, 5.0)


# ---------------------------------------------------------------- threshold cache
_THR_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_THR_LOCK = threading.Lock()
_THR_CACHE_SIZE = 16


def _digest(a: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(a).view(np.uint8).ravel(), digest_size=16).hexdigest()


def threshold_cache_key(raw: np.ndarray, valid: np.ndarray, params: DetectionParams) -> tuple:
    """Key of the cached automatic threshold: image content, effective valid mask (= regions,
    exclusions, alignment strip, automatic saturated exclusion) and every parameter that
    changes the LoG response (radius, minimum distance, background). The cached entry holds the
    automatic threshold AND the noise sigma (the unit of every threshold, manual ones too), both
    functions of the same inputs. threshold_scale, a manual threshold, tiling and dense
    decomposition do not change them."""
    pj = params.model_dump_json(include={"spot_radius_px", "minimum_distance_px", "background"})
    return ("noise-units-v1", raw.shape, str(raw.dtype), _digest(raw), _digest(np.packbits(valid)), pj)


def _cache_get(key):
    with _THR_LOCK:
        v = _THR_CACHE.get(key)
        if v is not None:
            _THR_CACHE.move_to_end(key)
        return v


def _cache_put(key, value: dict) -> None:
    with _THR_LOCK:
        _THR_CACHE[key] = value
        _THR_CACHE.move_to_end(key)
        while len(_THR_CACHE) > _THR_CACHE_SIZE:
            _THR_CACHE.popitem(last=False)


def clear_threshold_cache() -> None:
    with _THR_LOCK:
        _THR_CACHE.clear()


def _margin(p: DetectionParams) -> int:
    bg = p.background
    m = 0
    if bg.method == "gaussian_subtract":
        m = int(4 * bg.sigma_px) + 1
    elif bg.method == "white_tophat":
        m = int(bg.radius_px) + 1
    return m + int(math.ceil(4 * p.spot_radius_px + p.minimum_distance_px)) + 8


def run_detection(red: np.ndarray, green: np.ndarray, regions: list[Region], request: DetectionRequest,
                  progress: Callable[[float, str], None] = lambda f, m: None,
                  cancelled: Callable[[], bool] = lambda: False,
                  window: Optional[tuple[int, int, int, int]] = None,
                  engine: Optional[SpotEngine] = None,
                  review: Optional[ReviewConfig] = None,
                  alignment: tuple[int, int] = (0, 0)) -> DetectionResult:
    """Detect spots in both channels and associate them.

    ``red``/``green`` are raw original-dtype 2-D arrays in the same (registered)
    space as ``regions``. ``alignment`` = (dy, dx) is the integer shift that was applied to
    ``green`` (``registration.apply_shift``): the uncovered strip of the green channel holds
    zeros, not data, so it is removed from the green valid mask (no green spot can be
    detected there). ``window`` = (x0, y0, x1, y1) restricts the result to objects whose
    centre is inside it (preview). The background correction is computed on the full image
    and the automatic threshold of each channel is computed ONCE on the full masked image
    and cached (small LRU keyed by image hash + parameters + valid mask), so a preview
    window reports exactly the objects the full run reports in that window; only the LoG,
    local-maximum search and measurement run on a padded crop. Raises DetectionCancelled.
    Region rule (single, detection/masks.py): an object is kept iff ``point_region`` is not
    None (or, with no counting region, iff it is outside every exclusion).
    """
    def check() -> None:
        if cancelled():
            raise DetectionCancelled()

    engine = engine or BigFishEngine()
    notes: list[str] = []
    progress(0.0, "checking inputs")
    notes += check_compatible(red, green)  # IngestError on shape mismatch
    h, w = red.shape
    valid_full = build_valid_mask((h, w), regions)
    has_counting = any(r.kind == RegionKind.counting for r in regions)
    auto = None
    if not has_counting:
        auto = _saturated_outside(red, request.red.spot_radius_px) | _saturated_outside(green, request.green.spot_radius_px)
        if auto.any():
            valid_full &= ~auto
            notes.append(f"WARNING no counting region defined: {int(auto.sum())} px of large saturated area "
                         "(outside the chamber?) were excluded automatically; draw a counting region for a reliable count")
        else:
            auto = None
            notes.append("WARNING no counting region defined: the whole image is counted")
    if not valid_full.any():
        notes.append("no valid pixels in the counting regions: nothing to detect")
    dy, dx = int(alignment[0]), int(alignment[1])
    valid_by = {"red": valid_full, "green": valid_full}
    nodata = {"red": None, "green": None}
    if (dy, dx) != (0, 0):
        data = shifted_valid_region((h, w), dy, dx)
        nodata["green"] = ~data
        valid_by["green"] = valid_full & data
        notes.append(f"green alignment shift (dy={dy}, dx={dx}): the uncovered green strip "
                     f"({int((valid_full & ~valid_by['green']).sum())} px of the counting area) is excluded from "
                     "green detection (no green data there); red-only objects in it cannot be matched")

    ox = oy = 0
    sl = (slice(0, h), slice(0, w))
    win = None
    if window is not None:
        x0, y0, x1, y1 = (int(v) for v in window)
        x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
        y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"preview window {window} is empty for an image of {w} x {h} (W x H)")
        win = (x0, y0, x1, y1)
        m = max(_margin(request.red), _margin(request.green))
        cy0, cy1, cx0, cx1 = max(0, y0 - m), min(h, y1 + m), max(0, x0 - m), min(w, x1 + m)
        sl = (slice(cy0, cy1), slice(cx0, cx1))
        oy, ox = cy0, cx0
        notes.append(f"preview window {win}: LoG and measurement run on a padded crop; background correction "
                     "and the automatic threshold come from the full image (identical to the full run)")
    check()

    can_inject = "threshold_info" in inspect.signature(engine.detect).parameters and hasattr(engine, "estimate_threshold")
    spots = {}
    infos = {}
    for k, (name, arr, params) in enumerate((("red", red, request.red), ("green", green, request.green))):
        base = 0.05 + 0.4 * k

        def sub(f, msg, base=base, name=name):
            progress(base + 0.4 * f, f"{name}: {msg}")

        check()
        valid_c = valid_by[name]
        sub(0.0, "background correction")
        # Unclipped corrected image: used for noise/SNR statistics (clipping at 0 would
        # zero the robust std). The detection image is clipped if the user asked for it.
        # Always computed on the FULL image (also for previews) so that it equals the full run.
        bg_free = params.background.model_copy(update={"clip_negative": False})
        measured_full = correct_background(arr, bg_free, valid_c)
        corrected_full = np.maximum(measured_full, 0.0) if params.background.clip_negative else measured_full
        if nodata[name] is not None and nodata[name].any():
            # the zero-filled alignment strip is not data: neutralise it (it would otherwise
            # leave a step edge whose LoG ring creates spots inside the valid area)
            fillv = float(np.median(measured_full[valid_c])) if valid_c.any() else 0.0
            measured_full = np.where(nodata[name], np.float32(fillv), measured_full)
            corrected_full = np.where(nodata[name], np.float32(max(fillv, 0.0) if params.background.clip_negative else fillv),
                                      corrected_full)
        check()
        raw = arr[sl]
        measured, corrected, valid = measured_full[sl], corrected_full[sl], valid_c[sl]
        tinfo = None
        key = None
        if can_inject:
            key = threshold_cache_key(arr, valid_c, params)
            tinfo = _cache_get(key)
            if tinfo is None and win is not None:
                sub(0.05, "automatic threshold (full image)")
                tinfo = engine.estimate_threshold(corrected_full, params, valid_c, arr, measured_full)
                _cache_put(key, tinfo)
        kw = {"threshold_info": tinfo} if (tinfo is not None and can_inject) else {}
        sp = engine.detect(corrected, params, valid, raw=raw, progress=lambda f, m, s=sub: s(0.1 + 0.9 * f, m),
                           cancelled=cancelled, measure=measured, **kw)
        info = getattr(engine, "last_info", {}) or {}
        infos[name] = info
        if (key is not None and tinfo is None and info.get("threshold_info") is not None
                and params.threshold is None):
            _cache_put(key, info["threshold_info"])  # full run: remember for later previews
        for s_ in sp:
            s_.x += ox
            s_.y += oy
        spots[name] = sp
        thr = info.get("threshold")
        notes.append(f"{name}: {len(sp)} spot(s); LoG threshold "
                     f"{'n/a' if thr is None else f'{thr:.4g}'} noise units"
                     f"{'' if info.get('auto_threshold') is None else f' (auto {info['auto_threshold']:.4g} x scale {params.threshold_scale:g})'}"
                     f"; tiles {info.get('n_tiles', 1)}")
        notes += [f"{name}: {n}" for n in info.get("notes", [])]
        if params.background.method == "none":
            notes.append(f"{name}: no background correction applied")

    check()
    progress(0.88, "associating channels")
    red_s, green_s = sort_spots(spots["red"]), sort_spots(spots["green"])
    objects = associate(red_s, green_s, request.match_tolerance_px)
    check()
    progress(0.93, "review reasons")
    apply_review_reasons(objects, red_s, green_s, request.red, request.green, valid_full, (h, w), review)
    if win is None:  # (a preview window sees only part of the channel)
        dr = (review or ReviewConfig()).debris_ratio
        for name, sp in (("red", red_s), ("green", green_s)):
            n_deb = sum(1 for s_ in sp if s_.scale_ratio is not None and s_.scale_ratio >= dr)
            if len(sp) >= DEBRIS_CHANNEL_MIN and n_deb > DEBRIS_CHANNEL_FRACTION * len(sp):
                notes.append(
                    f"WARNING {name}: {n_deb} of {len(sp)} detected spots ({100 * n_deb / len(sp):.0f}%) are debris-like "
                    "(compact relative to the spot radius): this channel may be empty (the automatic threshold then "
                    "selects noise or specks), or the spot radius is too large for the real spots. Confirm the channel "
                    "visually and set the threshold / spot radius manually if needed. Classes are not changed.")

    # single region rule: keep iff point_region is not None (virtual whole-image counting
    # region when the user drew none, so the half-open rule still applies at the border)
    if has_counting:
        rule_regions = list(regions)
    else:
        rule_regions = [Region(id="__image__", kind=RegionKind.counting, name="image",
                               polygon=[(0, 0), (w, 0), (w, h), (0, h)])] + list(regions)
    rids = points_regions([o.x for o in objects], [o.y for o in objects], rule_regions)
    kept = []
    dropped_mask = 0
    for o, rid in zip(objects, rids):
        if rid is not None and auto is not None:
            yi, xi = min(max(int(round(o.y)), 0), h - 1), min(max(int(round(o.x)), 0), w - 1)
            if auto[yi, xi]:
                rid = None
        if rid is None:
            dropped_mask += 1
            continue
        if win is not None and not (win[0] <= o.x < win[2] and win[1] <= o.y < win[3]):
            continue
        o.region_id = rid if has_counting else None
        kept.append(o)
    if dropped_mask and win is None:
        notes.append(f"{dropped_mask} object(s) fell outside the counting regions / inside an exclusion and were dropped")
    progress(1.0, "done")
    notes.append(f"{len(kept)} object(s): " + ", ".join(
        f"{c}={sum(1 for o in kept if o.cls.value == c)}" for c in ("red", "green", "yellow", "uncertain")))
    return DetectionResult(objects=kept, package_versions=package_versions(), notes=notes)
