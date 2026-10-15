"""ImageJ ROI reading for the historical annotations (read-only, no interpretation of colour).

What the lab's files contain (verified on 7 research slides + ``test_img.tif``, see
docs/app/DATA_FORMATS.md section 3):

* ``ROI`` blob  = one multi-point ROI (``POINT``, sub-pixel flag set) whose points carry
  an ImageJ *counter* (0..255) and a *position* (1-based plane the point was placed on).
* ``Overlays`` blob = one composite ``RECT`` shape made of axis-aligned line segments:
  the counting grid.

Everything is returned in the raw ImageJ frame (pixel *i* spans [i, i+1)); conversion to
the app's canonical frame is done by :func:`to_canonical`.  Parsing problems become
:class:`RoiIssue` entries instead of exceptions, so one bad blob never hides the rest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import roifile
import tifffile

AREA_TYPES = {"RECT", "OVAL", "POLYGON", "FREEHAND", "TRACED"}
_SEG_ARGS = {0: 2, 1: 2, 2: 4, 3: 6, 4: 0}   # ImageJ shape path: moveTo, lineTo, quadTo, cubicTo, close
_AXIS_TOL = 1e-3
_REGULAR_TOL_PX = 0.5


@dataclass
class RoiIssue:
    code: str
    severity: str          # "info" | "warning" | "error"
    message: str


@dataclass
class RawPoint:
    roi_index: int
    point_index: int
    roi_source: str
    x: float               # raw ImageJ values
    y: float
    subpixel: bool
    counter: Optional[int]
    position: Optional[int]


@dataclass
class RawShape:
    roi_index: int
    roi_source: str
    roi_type: str
    name: Optional[str]
    polygon: list[tuple[float, float]]   # ImageJ frame (edges)
    position: Optional[int]


@dataclass
class RawGrid:
    roi_index: int
    roi_source: str
    xs: list[float]        # x of the vertical lines (ImageJ frame), sorted
    ys: list[float]
    regular: bool
    spacing: Optional[tuple[float, float]]


@dataclass
class RoiBundle:
    points: list[RawPoint] = field(default_factory=list)
    shapes: list[RawShape] = field(default_factory=list)
    grids: list[RawGrid] = field(default_factory=list)
    issues: list[RoiIssue] = field(default_factory=list)
    n_blobs: int = 0


def to_canonical(v: float, subpixel: bool) -> float:
    """ImageJ-frame coordinate -> canonical (pixel centre = integer).

    Sub-pixel ROIs store continuous coordinates in the pixel-edge frame, so the centre of
    pixel *i* is ``i + 0.5`` and the canonical value is ``v - 0.5``.  Integer ROIs
    (no sub-pixel flag) store the *pixel index*, which already is the canonical value.
    """
    return float(v) - 0.5 if subpixel else float(v)


def shape_to_canonical(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Area shapes (rect / polygon / grid cells) live on pixel edges: shift by half a pixel."""
    return [(x - 0.5, y - 0.5) for x, y in poly]


# ---------------------------------------------------------------- blob reading
def embedded_blobs(path: Path) -> tuple[list[tuple[str, bytes]], list[RoiIssue]]:
    """(source, bytes) of every ImageJ ROI / overlay embedded in a TIFF's metadata."""
    issues: list[RoiIssue] = []
    try:
        with tifffile.TiffFile(str(path)) as tif:
            md = tif.imagej_metadata if tif.is_imagej else None
    except Exception as exc:
        issues.append(RoiIssue("tiff_metadata_unreadable", "error",
                               f"cannot read TIFF metadata ({type(exc).__name__}: {exc})"))
        return [], issues
    blobs: list[tuple[str, bytes]] = []
    if md:
        if md.get("ROI"):
            blobs.append(("ROI", bytes(md["ROI"])))
        ov = md.get("Overlays")
        if ov:
            for b in (ov if isinstance(ov, (list, tuple)) else [ov]):
                blobs.append(("Overlays", bytes(b)))
    return blobs, issues


def read_embedded(path: Path) -> RoiBundle:
    blobs, issues = embedded_blobs(path)
    bundle = RoiBundle(issues=list(issues), n_blobs=len(blobs))
    for idx, (source, blob) in enumerate(blobs):
        try:
            roi = roifile.ImagejRoi.frombytes(blob)
        except Exception as exc:
            bundle.issues.append(RoiIssue("roi_unparseable", "error",
                                          f"embedded {source}[{idx}] is not a valid ImageJ ROI "
                                          f"({type(exc).__name__}: {exc}); ignored"))
            continue
        _interpret(roi, source, idx, bundle)
    return bundle


def read_sidecar(path: Path) -> RoiBundle:
    """A standalone ``.roi`` file or a ``.zip`` RoiSet."""
    bundle = RoiBundle()
    try:
        rois = roifile.roiread(str(path))
    except Exception as exc:
        bundle.issues.append(RoiIssue("roi_unparseable", "error",
                                      f"{path.name} is not a valid ImageJ ROI file "
                                      f"({type(exc).__name__}: {exc}); ignored"))
        return bundle
    if isinstance(rois, roifile.ImagejRoi):
        rois = [rois]
    bundle.n_blobs = len(rois)
    for idx, roi in enumerate(rois):
        _interpret(roi, "sidecar", idx, bundle)
    return bundle


# ---------------------------------------------------------------- interpretation
def _interpret(roi: "roifile.ImagejRoi", source: str, idx: int, bundle: RoiBundle) -> None:
    try:
        rtype = getattr(roi.roitype, "name", str(roi.roitype))
        multi = roi.multi_coordinates if getattr(roi, "composite", False) else None
        if multi is not None and len(multi):
            _interpret_composite(roi, multi, source, idx, bundle)
        elif rtype == "POINT":
            _interpret_points(roi, source, idx, bundle)
        elif rtype in AREA_TYPES:
            _interpret_area(roi, rtype, source, idx, bundle)
        else:
            bundle.issues.append(RoiIssue(
                "unsupported_roi_type", "warning",
                f"{source}[{idx}]: ROI type {rtype} is not supported (only POINT, area shapes and the "
                "line grid are converted); ignored"))
    except Exception as exc:
        bundle.issues.append(RoiIssue("roi_unparseable", "error",
                                      f"{source}[{idx}]: could not be decoded ({type(exc).__name__}: {exc}); ignored"))


def _position_of(roi: "roifile.ImagejRoi") -> Optional[int]:
    pos = int(getattr(roi, "position", 0) or 0)
    return pos if pos > 0 else None


def _interpret_points(roi, source: str, idx: int, bundle: RoiBundle) -> None:
    coords = np.asarray(roi.coordinates(), dtype=np.float64).reshape(-1, 2)
    n = len(coords)
    subpixel = bool(roi.options & roifile.ROI_OPTIONS.SUB_PIXEL_RESOLUTION) and roi.subpixel_coordinates is not None
    counters = roi.counters
    cpos = roi.counter_positions
    if counters is not None and len(counters) != n:
        bundle.issues.append(RoiIssue("counter_length_mismatch", "warning",
                                      f"{source}[{idx}]: {len(counters)} counters for {n} points; counters ignored"))
        counters = None
    if cpos is not None and len(cpos) != n:
        bundle.issues.append(RoiIssue("counter_length_mismatch", "warning",
                                      f"{source}[{idx}]: {len(cpos)} counter positions for {n} points; positions ignored"))
        cpos = None
    roi_pos = _position_of(roi)
    if n == 0:
        bundle.issues.append(RoiIssue("empty_point_roi", "warning", f"{source}[{idx}]: point ROI has no points"))
        return
    if counters is None:
        bundle.issues.append(RoiIssue("no_counters", "info",
                                      f"{source}[{idx}]: points carry no ImageJ counters; field membership "
                                      "can only be derived geometrically"))
    for k in range(n):
        position = int(cpos[k]) if cpos is not None else roi_pos
        if position is not None and position <= 0:
            position = None
        bundle.points.append(RawPoint(
            roi_index=idx, point_index=k, roi_source=source,
            x=float(coords[k, 0]), y=float(coords[k, 1]), subpixel=subpixel,
            counter=int(counters[k]) if counters is not None else None, position=position))


def _interpret_area(roi, rtype: str, source: str, idx: int, bundle: RoiBundle) -> None:
    if rtype == "RECT":
        l, t, r, b = float(roi.left), float(roi.top), float(roi.right), float(roi.bottom)
        poly = [(l, t), (r, t), (r, b), (l, b)]
    elif rtype == "OVAL":
        cx, cy = (roi.left + roi.right) / 2.0, (roi.top + roi.bottom) / 2.0
        a, b = (roi.right - roi.left) / 2.0, (roi.bottom - roi.top) / 2.0
        poly = [(cx + a * math.cos(2 * math.pi * k / 64), cy + b * math.sin(2 * math.pi * k / 64)) for k in range(64)]
    else:
        poly = [(float(x), float(y)) for x, y in np.asarray(roi.coordinates(), dtype=np.float64).reshape(-1, 2)]
    if len(poly) < 3 or _area(poly) <= 0:
        bundle.issues.append(RoiIssue("degenerate_shape", "warning",
                                      f"{source}[{idx}]: {rtype} has zero area; ignored"))
        return
    bundle.shapes.append(RawShape(idx, source, rtype, roi.name or None, poly, _position_of(roi)))


def _area(poly: list[tuple[float, float]]) -> float:
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def parse_shape_path(multi: np.ndarray) -> list[list[tuple[float, float]]]:
    """Split an ImageJ composite-shape coordinate array into sub-paths (vertex lists)."""
    arr = np.asarray(multi, dtype=np.float64).ravel()
    paths: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    i = 0
    while i < len(arr):
        cmd = arr[i]
        if cmd != int(cmd) or int(cmd) not in _SEG_ARGS:
            raise ValueError(f"invalid shape command {cmd!r} at offset {i}")
        cmd = int(cmd)
        n = _SEG_ARGS[cmd]
        args = arr[i + 1:i + 1 + n]
        if len(args) != n:
            raise ValueError("truncated shape path")
        i += 1 + n
        if cmd == 0:
            if cur:
                paths.append(cur)
            cur = [(float(args[0]), float(args[1]))]
        elif cmd in (1, 2, 3):
            cur.append((float(args[-2]), float(args[-1])))
        elif cmd == 4 and cur:
            paths.append(cur)
            cur = []
    if cur:
        paths.append(cur)
    return paths


def _interpret_composite(roi, multi: np.ndarray, source: str, idx: int, bundle: RoiBundle) -> None:
    paths = parse_shape_path(multi)
    vx: list[float] = []
    hy: list[float] = []
    other = 0
    for p in paths:
        if len(p) == 2 and abs(p[0][0] - p[1][0]) < _AXIS_TOL and abs(p[0][1] - p[1][1]) > _AXIS_TOL:
            vx.append(round(p[0][0], 3))
        elif len(p) == 2 and abs(p[0][1] - p[1][1]) < _AXIS_TOL and abs(p[0][0] - p[1][0]) > _AXIS_TOL:
            hy.append(round(p[0][1], 3))
        else:
            other += 1
    xs, ys = sorted(set(vx)), sorted(set(hy))
    if len(xs) >= 2 and len(ys) >= 2:
        if other:
            bundle.issues.append(RoiIssue("grid_with_extra_geometry", "warning",
                                          f"{source}[{idx}]: grid overlay also contains {other} non-grid sub-path(s); ignored"))
        dx, dy = np.diff(xs), np.diff(ys)
        regular = bool(np.ptp(dx) <= _REGULAR_TOL_PX and np.ptp(dy) <= _REGULAR_TOL_PX)
        if not regular:
            bundle.issues.append(RoiIssue("irregular_grid", "warning",
                                          f"{source}[{idx}]: grid spacing is not uniform "
                                          f"(x spacings {dx.min():.2f}..{dx.max():.2f}, y {dy.min():.2f}..{dy.max():.2f}); "
                                          "cells are built from the actual lines"))
        bundle.grids.append(RawGrid(idx, source, xs, ys, regular,
                                    (float(np.median(dx)), float(np.median(dy)))))
        return
    # composite shape that is not a line grid: expose its sub-paths as unclassified regions
    kept = 0
    for p in paths:
        if len(p) >= 3 and _area(p) > 0:
            bundle.shapes.append(RawShape(idx, source, "COMPOSITE", roi.name or None, p, _position_of(roi)))
            kept += 1
    bundle.issues.append(RoiIssue("unrecognised_composite", "warning",
                                  f"{source}[{idx}]: composite shape is not an axis-aligned line grid; "
                                  f"{kept} closed sub-path(s) kept as unclassified regions"))
