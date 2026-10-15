"""Builders for ImageJ-annotated TIFF fixtures (synthetic, deterministic)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tifffile
from roifile import ROI_OPTIONS, ROI_TYPE, ImagejRoi

REPO = Path(__file__).resolve().parents[2]
REAL_TEST_IMG = REPO / "resources" / "dataset_files" / "test" / "test_img.tif"   # synthetic, see make_test_img.py
REAL_TEST_TRUTH = REAL_TEST_IMG.with_name("test_img.truth.json")


def load_test_img_truth() -> dict:
    """Ground truth written by resources/dataset_files/test/make_test_img.py next to the example image."""
    return json.loads(REAL_TEST_TRUTH.read_text())


def grid_overlay(xs: Sequence[float], ys: Sequence[float], width: int, height: int) -> bytes:
    """A composite line-grid overlay like the lab's: one RECT shape ROI (version 228, magenta stroke)
    made of moveTo/lineTo segments, one per grid line. Byte-identical to the lab's overlay layout."""
    segs: list[float] = []
    for x in xs:
        segs += [0, x, 0, 1, x, height]
    for y in ys:
        segs += [0, 0, y, 1, width, y]
    ov = ImagejRoi(roitype=ROI_TYPE.RECT, version=228, left=0, top=0, right=width, bottom=height,
                   stroke_color=b"\xff\x00\xff\xff")
    ov.multi_coordinates = np.array(segs, dtype=">f4")
    ov.shape_roi_size = len(segs)
    return ov.tobytes()


def point_roi(points: Sequence[tuple[float, float]], counters: Optional[Sequence[int]] = None,
              positions: Optional[Sequence[int]] = None, subpixel: bool = True) -> bytes:
    arr = np.array(points, dtype=np.float32)
    roi = ImagejRoi.frompoints(arr if subpixel else arr.astype(np.int32).astype(np.float32))
    roi.roitype = ROI_TYPE.POINT
    if not subpixel:
        roi.options = roi.options & ~ROI_OPTIONS.SUB_PIXEL_RESOLUTION
        roi.subpixel_coordinates = None
    if counters is not None:
        roi.counters = np.array(counters, dtype=np.uint8)
    if positions is not None:
        roi.counter_positions = np.array(positions, dtype=np.uint32)
    return roi.tobytes()


def rect_roi(left: int, top: int, right: int, bottom: int, name: str = "") -> bytes:
    roi = ImagejRoi(roitype=ROI_TYPE.RECT, left=left, top=top, right=right, bottom=bottom, name=name or None)
    return roi.tobytes()


def line_roi() -> bytes:
    roi = ImagejRoi(roitype=ROI_TYPE.LINE, x1=1.0, y1=1.0, x2=9.0, y2=9.0)
    return roi.tobytes()


def write_stack(path: Path, planes: Sequence[np.ndarray], *, roi: Optional[bytes] = None,
                overlays: Optional[Sequence[bytes]] = None, info: Optional[str] = None) -> Path:
    data = np.stack(planes) if len(planes) > 1 else planes[0]
    meta: dict = {}
    if roi is not None:
        meta["ROI"] = roi
    if overlays:
        meta["Overlays"] = list(overlays) if len(overlays) > 1 else overlays[0]
    if info:
        meta["Info"] = info
    tifffile.imwrite(str(path), data, imagej=True, metadata=meta)
    return path


def blank(h: int = 120, w: int = 160, dtype=np.uint16, value: int = 100) -> np.ndarray:
    return np.full((h, w), value, dtype=dtype)


def textured(h: int, w: int, seed: int, dtype=np.uint16, hi: int = 60000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, hi, size=(h, w)).astype(dtype)


# lab-like: 3x3 grid on a 160x120 image (edges in ImageJ frame)
GRID_XS = [10.0, 60.0, 110.0, 160.0]
GRID_YS = [10.0, 45.0, 80.0, 115.0]
