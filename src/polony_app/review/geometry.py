"""Polygon helpers. Membership and rasterisation delegate to detection/masks.py (the one
rule used everywhere); the remaining helpers are validation/overlap utilities."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..detection.masks import points_in_polygon, rasterize_polygon as _rasterize

Poly = Sequence[Sequence[float]]


def point_in_polygon(x: float, y: float, poly: Poly) -> bool:
    """Single app-wide rule (detection/masks.py): even-odd with half-open edges, pixel
    centres at integer coordinates. NOT boundary-inclusive any more."""
    if len(poly) < 3:
        return False
    p = np.asarray(poly, dtype=np.float64)
    return bool(points_in_polygon(np.array([x], float), np.array([y], float), p)[0])


def polygon_area(poly: Poly) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i][0], poly[i][1]
        x2, y2 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def bbox(poly: Poly) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _orient(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _segments_intersect(p1, p2, p3, p4) -> bool:
    d1 = _orient(*p3, *p4, *p1)
    d2 = _orient(*p3, *p4, *p2)
    d3 = _orient(*p1, *p2, *p3)
    d4 = _orient(*p1, *p2, *p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 * d2 != 0 and d3 * d4 != 0:
        return True

    def on_seg(a, b, c) -> bool:
        return (
            abs(_orient(*a, *b, *c)) < 1e-9
            and min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    return on_seg(p3, p4, p1) or on_seg(p3, p4, p2) or on_seg(p1, p2, p3) or on_seg(p1, p2, p4)


def polygons_overlap(a: Poly, b: Poly) -> bool:
    """True if the two polygons share any area/boundary point."""
    if len(a) < 3 or len(b) < 3:
        return False
    ax0, ay0, ax1, ay1 = bbox(a)
    bx0, by0, bx1, by1 = bbox(b)
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return False
    na, nb = len(a), len(b)
    for i in range(na):
        for j in range(nb):
            if _segments_intersect(a[i], a[(i + 1) % na], b[j], b[(j + 1) % nb]):
                return True
    return point_in_polygon(a[0][0], a[0][1], b) or point_in_polygon(b[0][0], b[0][1], a)


def validate_polygon(poly: Poly) -> str | None:
    if len(poly) < 3:
        return "polygon needs at least 3 points"
    for p in poly:
        if len(p) != 2 or not all(math.isfinite(v) for v in p):
            return "polygon has non-finite coordinates"
    if polygon_area(poly) <= 0:
        return "polygon has zero area"
    return None


def rasterize_polygon(poly: Poly, shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask (h, w) of pixel centres inside the polygon (detection/masks.py rule)."""
    return _rasterize(np.asarray(poly, dtype=np.float64), shape)
