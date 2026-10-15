"""Big-FISH expressed in the engine-neutral vocabulary (RawDetection + EngineRunInfo).

The detection itself is the unchanged ``detection.bigfish_engine.BigFishEngine``. The score of a Big-FISH
detection is its LoG response in NOISE UNITS (response / robust noise sigma of the page), score_kind
'bigfish_log_response_noise_units' -- the same unit as the detection threshold; it is not a probability.
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import numpy as np

from ..schemas import DetectionParams, EngineName, EngineRunInfo
from .base import (EngineResult, Progress, PointValidity, build_detections, new_run_id, raster_point_validity,
                   utcnow_iso)

SCORE_KIND = "bigfish_log_response_noise_units"


def bigfish_version() -> str:
    import bigfish
    return bigfish.__version__


def spots_to_result(spots: Sequence, *, analysis_id: str, page: int, channel_role: Optional[str],
                    params: DetectionParams, image_sha256: Optional[dict] = None,
                    noise_sigma: Optional[float] = None, validity: PointValidity,
                    info: Optional[dict] = None, started_at: Optional[str] = None,
                    duration_s: Optional[float] = None, run_id: Optional[str] = None,
                    notes: Optional[list] = None) -> tuple[EngineResult, list[int]]:
    """RawSpot list -> EngineResult. Returns (result, order) where ``order[k]`` is the index in ``spots`` of
    the k-th RawDetection (detections are numbered in (y, x) order, so ids are deterministic)."""
    run_id = run_id or new_run_id()
    xs = [s.x for s in spots]
    ys = [s.y for s in spots]
    sc = [None if s.log_response is None else (s.log_response / noise_sigma if noise_sigma else s.log_response)
          for s in spots]
    dets = build_detections(EngineName.bigfish, run_id, analysis_id, page, channel_role, xs, ys, sc,
                            SCORE_KIND, validity, radii=[float(params.spot_radius_px)] * len(spots))
    # recover the permutation used by build_detections (stable sort by (y, x, index))
    order = sorted(range(len(spots)), key=lambda i: (round(float(ys[i]), 6), round(float(xs[i]), 6), i))
    info = info or {}
    n_valid = sum(1 for d in dets if d.valid)
    pre = {"background": params.background.model_dump(mode="json"),
           "input": "background-corrected float image computed from the ORIGINAL dtype page",
           "score_unit": "LoG response / noise_sigma" if noise_sigma else "raw LoG response (noise sigma unknown)",
           "noise_sigma": noise_sigma, "threshold_noise_units": info.get("threshold")}
    run = EngineRunInfo(
        run_id=run_id, analysis_id=analysis_id, image_sha256=dict(image_sha256 or {}), engine=EngineName.bigfish,
        engine_version=bigfish_version(), page=int(page), channel_role=channel_role,
        params=params.model_dump(mode="json"), preprocessing=pre, started_at=started_at or utcnow_iso(),
        finished_at=utcnow_iso(), device="cpu", duration_s=duration_s, status="ok", n_raw=len(dets),
        n_valid=n_valid, notes=list(notes or []))
    return EngineResult(run=run, detections=dets), order


class BigFishAdapter:
    """Stand-alone page detector (background correction + Big-FISH). The pipeline uses ``spots_to_result``
    on the spots of its own (threshold-cached, window-aware) Big-FISH call instead, so the Big-FISH numbers are
    byte-for-byte the ones of the engine=='bigfish' path."""
    name = EngineName.bigfish

    def available(self) -> tuple[bool, str]:
        try:
            import bigfish  # noqa: F401
        except Exception as exc:  # pragma: no cover
            return False, f"Big-FISH is not importable: {exc}"
        return True, "Big-FISH (classical LoG detector)"

    def version(self) -> str:
        return bigfish_version()

    def detect_page(self, image: np.ndarray, *, analysis_id: str, page: int, channel_role: Optional[str],
                    valid_mask: Optional[np.ndarray], params: dict, image_sha256: dict,
                    progress: Progress = lambda f, m: None, cancelled: Callable[[], bool] = lambda: False,
                    point_validity: Optional[PointValidity] = None) -> EngineResult:
        from ..detection.bigfish_engine import BigFishEngine
        from ..imaging.preprocess import correct_background
        dp = params if isinstance(params, DetectionParams) else DetectionParams(**params)
        t0 = time.perf_counter()
        started = utcnow_iso()
        eng = BigFishEngine()
        measured = correct_background(image, dp.background.model_copy(update={"clip_negative": False}), valid_mask)
        corrected = np.maximum(measured, 0.0) if dp.background.clip_negative else measured
        spots = eng.detect(corrected, dp, valid_mask, raw=image, progress=progress, cancelled=cancelled,
                           measure=measured)
        info = eng.last_info or {}
        res, order = spots_to_result(
            spots, analysis_id=analysis_id, page=page, channel_role=channel_role, params=dp,
            image_sha256=image_sha256, noise_sigma=info.get("noise_sigma"),
            validity=point_validity or raster_point_validity(valid_mask, image.shape), info=info,
            started_at=started, duration_s=time.perf_counter() - t0, notes=list(info.get("notes", [])))
        res.extra["spots"] = [spots[i] for i in order]
        res.extra["info"] = info
        return res
