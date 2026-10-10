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
import time

from ..schemas import (ConsensusStatus, ConsensusSummary, DetectedObject, DetectionParams, DetectionRequest,
                       EngineChoice, EngineName, EngineRunInfo, RawDetection, Region, RegionKind, ReviewReason)
from .bigfish_engine import BigFishEngine
from .engine import SpotEngine
from .errors import DetectionCancelled
from .masks import build_valid_mask, points_regions
from .matching import associate, sort_spots, spot_index
from .review_reasons import ReviewConfig, apply_review_reasons
from .two_engine import SpotInfo, fill_evidence, make_validity, measure_spots_at, union_spots
from ..consensus.matching import match_engines, summarize
from ..engines.base import EngineError, EngineUnavailable, new_run_id, utcnow_iso
from ..engines.bigfish_adapter import spots_to_result
from ..engines.spotiflow_adapter import SpotiflowEngine

DEBRIS_CHANNEL_FRACTION = 0.5   # > this fraction of a channel's spots debris-like => pipeline WARNING
DEBRIS_CHANNEL_MIN = 3          # ... if the channel has at least this many spots

__all__ = ["run_detection", "DetectionResult", "DetectionCancelled", "package_versions"]


@dataclass
class DetectionResult:
    objects: list[DetectedObject]
    package_versions: dict[str, str]
    notes: list[str] = field(default_factory=list)
    # two-engine architecture: one EngineRunInfo per engine and channel page, every raw detection of
    # every engine (valid flag set), and per-channel consensus counts (hybrid only)
    engine_runs: list[EngineRunInfo] = field(default_factory=list)
    raw_detections: list[RawDetection] = field(default_factory=list)
    consensus: list[ConsensusSummary] = field(default_factory=list)


def package_versions(with_spotiflow: bool = False) -> dict[str, str]:
    import bigfish, numpy, scipy, skimage, tifffile
    out = {"bigfish": bigfish.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
           "skimage": skimage.__version__, "tifffile": tifffile.__version__,
           "python": platform.python_version()}
    if with_spotiflow:
        for pkg in ("spotiflow", "torch"):
            try:
                out[pkg] = md.version(pkg)
            except Exception:
                pass
    return out


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
                  alignment: tuple[int, int] = (0, 0),
                  analysis_id: str = "", pages: tuple[int, int] = (0, 1),
                  roles: tuple[Optional[str], Optional[str]] = ("red", "green"),
                  image_sha256: Optional[dict] = None, spotiflow_engine=None) -> DetectionResult:
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

    Two-engine architecture (``request.engine``): 'bigfish' (default) is the path above, unchanged.
    'spotiflow' detects with Spotiflow only; 'hybrid' runs Big-FISH and Spotiflow independently on each
    channel page, matches them (consensus/matching.py) and feeds the union candidates to the SAME
    association / review-reason code. ``pages`` are the TIFF page indices of ``red`` / ``green`` and ``roles``
    their colour roles; both come from the caller's confirmed channel mapping (never inferred here; a role of
    None means "unconfirmed"). ``spotiflow_engine`` injects an engine (tests); default SpotiflowEngine().
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

    choice = request.engine
    # same-channel consensus tolerance: an explicit request value is honoured; the default (4 px) is raised to the
    # Spotiflow downscale d, because the AI detector's positions are only good to ~d px at downscale d
    cons_tol = float(request.consensus_tolerance_px)
    cons_tol_note = ""
    if "consensus_tolerance_px" not in request.model_fields_set and float(request.spotiflow.downscale) > cons_tol:
        cons_tol = float(request.spotiflow.downscale)
        cons_tol_note = f", raised from the {request.consensus_tolerance_px:g} px default to the Spotiflow downscale"
    run_bf = choice in (EngineChoice.bigfish, EngineChoice.hybrid)
    run_sf = choice in (EngineChoice.spotiflow, EngineChoice.hybrid)
    if run_sf:
        notes.append(f"engine: {choice.value}" + (" (Big-FISH and Spotiflow run independently on each channel "
                                                    "page; agreement is not ground truth)" if run_bf else ""))
        if win is not None and run_sf:
            notes.append("preview window with Spotiflow: inference runs on the padded window crop, so its automatic "
                         "normalisation can differ slightly from the full run, and preview is slower than Big-FISH")
    # single region rule (virtual whole-image counting region when the user drew none, so the half-open rule
    # still applies at the border); also used to flag raw detections
    if has_counting:
        rule_regions = list(regions)
    else:
        rule_regions = [Region(id="__image__", kind=RegionKind.counting, name="image",
                               polygon=[(0, 0), (w, 0), (w, h), (0, h)])] + list(regions)
    engine_runs: list[EngineRunInfo] = []
    raw_dets: list[RawDetection] = []
    consensus_summaries: list[ConsensusSummary] = []
    # progress layout: Big-FISH alone as before; hybrid gives Spotiflow (the slow engine) the larger share:
    # bf red .02-.07, sf red .07-.40, bf green .40-.45, sf green .45-.78 (association etc. from .88)
    span_bf = 0.4 if choice == EngineChoice.bigfish else 0.05
    span_sf = 0.33 if choice == EngineChoice.hybrid else 0.4
    bf_bases = (0.05, 0.45) if choice == EngineChoice.bigfish else (0.02, 0.40) if choice == EngineChoice.hybrid \
        else (0.05, 0.45)
    sf_bases = (0.07, 0.45) if choice == EngineChoice.hybrid else (0.05, 0.45)
    sf_eng = None
    sf_dead: Optional[str] = None    # unavailable-reason once Spotiflow is known not to run

    can_inject = "threshold_info" in inspect.signature(engine.detect).parameters and hasattr(engine, "estimate_threshold")
    spots = {}
    infos = {}
    spot_info: dict[int, SpotInfo] = {}
    for k, (name, arr, params) in enumerate((("red", red, request.red), ("green", green, request.green))):
        base = bf_bases[k]

        def sub(f, msg, base=base, name=name, span=span_bf):
            progress(base + span * f, f"{name}: {msg}")

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
        validity = make_validity(rule_regions, (h, w), auto, nodata[name])
        page_i, role_i = int(pages[k]), roles[k]
        info = {}
        sp: list = []
        bf_res = None
        if run_bf:
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
            t_bf = time.perf_counter()
            started_bf = utcnow_iso()
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
            thr = info.get("threshold")
            notes.append(f"{name}: {len(sp)} spot(s); LoG threshold "
                         f"{'n/a' if thr is None else f'{thr:.4g}'} noise units"
                         f"{'' if info.get('auto_threshold') is None else f' (auto {info['auto_threshold']:.4g} x scale {params.threshold_scale:g})'}"
                         f"; tiles {info.get('n_tiles', 1)}")
            notes += [f"{name}: {n}" for n in info.get("notes", [])]
            if params.background.method == "none":
                notes.append(f"{name}: no background correction applied")
            bf_res, bf_order = spots_to_result(
                sp, analysis_id=analysis_id, page=page_i, channel_role=role_i, params=params,
                image_sha256=image_sha256, noise_sigma=info.get("noise_sigma"), validity=validity, info=info,
                started_at=started_bf, duration_s=time.perf_counter() - t_bf,
                notes=[n for n in info.get("notes", [])])
            engine_runs.append(bf_res.run)
            raw_dets += bf_res.detections
        else:
            infos[name] = {}

        sf_res = None
        if run_sf:
            if sf_eng is None:
                sf_eng = spotiflow_engine if spotiflow_engine is not None else SpotiflowEngine()
            sf_params = request.spotiflow.model_dump(mode="json")
            sf_valid = valid_c
            if win is not None:
                rect = np.zeros((h, w), bool)
                rect[sl] = True
                sf_valid = valid_c & rect
            sbase = sf_bases[k]

            def sub_sf(f, msg, sbase=sbase, name=name):
                progress(sbase + span_sf * f, f"{name}: {msg}")

            try:
                if sf_dead is not None:
                    raise EngineUnavailable(sf_dead)
                sf_res = sf_eng.detect_page(
                    arr, analysis_id=analysis_id, page=page_i, channel_role=role_i, valid_mask=sf_valid,
                    params=sf_params, image_sha256=dict(image_sha256 or {}), progress=sub_sf, cancelled=cancelled,
                    point_validity=validity)
            except DetectionCancelled:
                raise
            except EngineError as exc:
                if choice == EngineChoice.spotiflow:
                    raise
                if isinstance(exc, EngineUnavailable):
                    sf_dead = str(exc)
                engine_runs.append(EngineRunInfo(
                    run_id=new_run_id(), analysis_id=analysis_id, image_sha256=dict(image_sha256 or {}),
                    engine=EngineName.spotiflow, engine_version=sf_eng.version() if hasattr(sf_eng, "version") else "",
                    model_name=request.spotiflow.model_name, page=page_i, channel_role=role_i, params=sf_params,
                    started_at=utcnow_iso(), finished_at=utcnow_iso(), status="failed", error=str(exc),
                    notes=["Spotiflow failed; the Big-FISH result of this channel page is unaffected"]))
                notes.append(f"WARNING {name}: Spotiflow failed ({exc}); this channel is Big-FISH only "
                             "(no consensus status). The run itself succeeded.")
            else:
                engine_runs.append(sf_res.run)
                raw_dets += sf_res.detections
                n_ok = sum(1 for d in sf_res.detections if d.valid)
                notes.append(f"{name}: Spotiflow {len(sf_res.detections)} raw detection(s), {n_ok} inside the counting "
                             f"area (model {sf_res.run.model_name} {sf_res.run.model_version or ''} on "
                             f"{sf_res.run.device}, {sf_res.run.duration_s or 0:.1f} s)".replace("  ", " "))
                notes += [f"{name}: {n}" for n in sf_res.run.notes]

        # ---- candidate spots of this channel
        if sf_res is not None and choice == EngineChoice.spotiflow:
            sf_valid_dets = [d for d in sf_res.detections if d.valid]
            if not hasattr(engine, "_measure"):
                raise EngineError("the injected Big-FISH engine cannot measure spots; Spotiflow-only needs it")
            thr_abs = float(infos[name].get("threshold_abs") or 0.0)
            ms = measure_spots_at(engine, params, corrected_full, measured_full, valid_c, arr,
                                  [d.x for d in sf_valid_dets], [d.y for d in sf_valid_dets], thr_abs)
            for s_, d in zip(ms, sf_valid_dets):
                spot_info[id(s_)] = SpotInfo(None, None, d, None, ())
            sp = ms
        elif sf_res is not None:   # hybrid
            bf_valid = [(d, sp[i]) for d, i in zip(bf_res.detections, bf_order) if d.valid]
            bf_invalid = [sp[i] for d, i in zip(bf_res.detections, bf_order) if not d.valid]
            bf_spot_of = {d.id: s_ for d, s_ in bf_valid}
            sf_valid_dets = [d for d in sf_res.detections if d.valid]
            pairs = match_engines([d for d, _ in bf_valid], sf_valid_dets, cons_tol)
            need = [p.spotiflow for p in pairs if p.bigfish is None]
            if need and not hasattr(engine, "_measure"):
                raise EngineError("the injected Big-FISH engine cannot measure spots; hybrid needs it")
            thr_abs = float(infos[name].get("threshold_abs") or 0.0)
            ms = measure_spots_at(engine, params, corrected_full, measured_full, valid_c, arr,
                                  [d.x for d in need], [d.y for d in need], thr_abs)
            sf_spots = {d.id: s_ for d, s_ in zip(need, ms)}
            union, ui = union_spots(pairs, bf_spot_of, sf_spots)
            spot_info.update(ui)
            sp = union + bf_invalid          # BF spots outside the valid area flow through exactly as today
            consensus_summaries += summarize(pairs)
            n_by = {st: sum(1 for p in pairs if p.status == st) for st in ConsensusStatus}
            notes.append(f"{name}: consensus (tolerance {cons_tol:g} px{cons_tol_note}) "
                         f"agreement={n_by[ConsensusStatus.agreement]}, "
                         f"bigfish_only={n_by[ConsensusStatus.bigfish_only]}, "
                         f"spotiflow_only={n_by[ConsensusStatus.spotiflow_only]}")
        spots[name] = sp

    check()
    progress(0.88, "associating channels")
    red_s, green_s = sort_spots(spots["red"]), sort_spots(spots["green"])
    objects = associate(red_s, green_s, request.match_tolerance_px)
    check()
    progress(0.93, "review reasons")
    apply_review_reasons(objects, red_s, green_s, request.red, request.green, valid_full, (h, w), review)
    if spot_info:
        _attach_engine_evidence(objects, {"red": red_s, "green": green_s}, spot_info, choice)
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
    return DetectionResult(objects=kept, package_versions=package_versions(run_sf), notes=notes,
                           engine_runs=engine_runs, raw_detections=raw_dets, consensus=consensus_summaries)


def _attach_engine_evidence(objects: list[DetectedObject], lists: dict, spot_info: dict,
                            choice: EngineChoice) -> None:
    """Fill the two-engine provenance of each channel evidence and add the disagreement review reasons
    (nothing is added for a plain agreement). Mutates ``objects``."""
    for o in objects:
        add: list[ReviewReason] = []
        for key in ("red", "green"):
            ev = getattr(o, key)
            if not ev.detected:
                continue
            si = spot_info.get(id(lists[key][spot_index(ev.spot_id)]))
            if si is None:
                continue
            fill_evidence(ev, si)
            if si.status == ConsensusStatus.bigfish_only:
                add.append(ReviewReason.bigfish_only)
                o.history.append(f"auto: {key} spot is Big-FISH only (no Spotiflow detection within the consensus "
                                 "tolerance); not rejected")
            elif si.status == ConsensusStatus.spotiflow_only:
                add.append(ReviewReason.spotiflow_only)
                o.history.append(f"auto: {key} spot is Spotiflow only (no Big-FISH detection within the consensus "
                                 "tolerance); peak / background / SNR measured at the Spotiflow position; not rejected")
            elif si.status == ConsensusStatus.agreement and si.ambiguous_with:
                o.history.append(f"auto: {key} Big-FISH / Spotiflow match is ambiguous with "
                                 f"{', '.join(si.ambiguous_with)}")
        for r in add:
            if r not in o.review_reasons:
                o.review_reasons.append(r)
