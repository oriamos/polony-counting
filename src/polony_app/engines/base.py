"""Engine-neutral contract of the two-engine architecture (see docs/app/TWO_ENGINE.md).

An engine turns ONE 2-D page (original dtype, full resolution) into ``RawDetection`` records plus one
``EngineRunInfo`` (what was run, with which model / parameters / preprocessing / device). Raw output is
preserved as-is; nothing here decides what is a real polony.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np

from ..schemas import EngineName, EngineRunInfo, RawDetection

Progress = Callable[[float, str], None]
# (xs, ys) -> (valid bool array, region id per point or None, excluded_reason per point or None)
PointValidity = Callable[[np.ndarray, np.ndarray], "tuple[np.ndarray, list[Optional[str]], list[Optional[str]]]"]

EXCLUDED_REASON = "outside_counting_area_or_in_exclusion"


class EngineError(RuntimeError):
    """An engine could not produce a result (model load / inference failure, bad input...)."""


class EngineUnavailable(EngineError):
    """The engine cannot run in this installation (package, torch or model missing)."""


@dataclass
class EngineResult:
    run: EngineRunInfo
    detections: list[RawDetection]
    extra: dict = field(default_factory=dict)   # engine-private extras (e.g. Big-FISH RawSpot list), not persisted


@runtime_checkable
class DetectionEngine(Protocol):
    name: EngineName

    def available(self) -> tuple[bool, str]: ...

    def version(self) -> str: ...

    def detect_page(self, image: np.ndarray, *, analysis_id: str, page: int, channel_role: Optional[str],
                    valid_mask: Optional[np.ndarray], params: dict, image_sha256: dict,
                    progress: Progress, cancelled: Callable[[], bool]) -> EngineResult: ...


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_run_id() -> str:
    return uuid.uuid4().hex


def raw_id(engine: EngineName, page: int, index: int) -> str:
    """Deterministic id, unique within a run: 'bf-p0-000001' / 'sf-p0-000001' (index is 0-based)."""
    prefix = {"bigfish": "bf", "spotiflow": "sf"}.get(EngineName(engine).value, EngineName(engine).value)
    return f"{prefix}-p{int(page)}-{index + 1:06d}"


def raster_point_validity(valid_mask: Optional[np.ndarray], shape: tuple[int, int]) -> PointValidity:
    """Fallback validity rule when no exact region geometry is supplied: the pixel whose centre is nearest
    (round half up, pixel centre at integer coordinates) must be set in ``valid_mask`` and inside the image.
    The pipeline passes an exact checker built on detection/masks.py (half-open pixel-centre rule) instead."""
    h, w = shape

    def check(xs: np.ndarray, ys: np.ndarray):
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        xi = np.floor(xs + 0.5).astype(np.int64)
        yi = np.floor(ys + 0.5).astype(np.int64)
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        ok = inside.copy()
        if valid_mask is not None and ok.any():
            ok[inside] = valid_mask[yi[inside], xi[inside]]
        return ok, [None] * len(xs), [None if v else EXCLUDED_REASON for v in ok]

    return check


def build_detections(engine: EngineName, run_id: str, analysis_id: str, page: int, channel_role: Optional[str],
                     xs, ys, scores, score_kind: str, validity: PointValidity,
                     radii=None) -> list[RawDetection]:
    """Sort by (y, x), number deterministically, flag validity (never drop an invalid detection)."""
    xs = np.asarray(xs, float).reshape(-1)
    ys = np.asarray(ys, float).reshape(-1)
    n = xs.size
    scores = [None] * n if scores is None else list(np.asarray(scores, float).reshape(-1))
    radii = [None] * n if radii is None else list(radii)
    order = sorted(range(n), key=lambda i: (round(float(ys[i]), 6), round(float(xs[i]), 6), i))
    ok, rids, why = validity(xs, ys) if n else (np.zeros(0, bool), [], [])
    out = []
    for k, i in enumerate(order):
        out.append(RawDetection(
            id=raw_id(engine, page, k), run_id=run_id, analysis_id=analysis_id, engine=engine, page=int(page),
            channel_role=channel_role, x=float(xs[i]), y=float(ys[i]), radius_px=radii[i],
            score=None if scores[i] is None else float(scores[i]), score_kind=score_kind,
            valid=bool(ok[i]), region_id=rids[i] if bool(ok[i]) else None,
            excluded_reason=None if bool(ok[i]) else (why[i] or EXCLUDED_REASON)))
    return out
