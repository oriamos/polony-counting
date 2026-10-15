"""Region geometry: valid mask, point-in-region, areas.

Convention (consistent between raster mask and point queries): pixel (r, c) has
its centre at (x=c, y=r). A pixel centre is INSIDE a polygon by the even-odd
rule with half-open edges: scanline y hits edges with y_lo <= y < y_hi and the
centre is inside when x_a <= x < x_b for crossing pairs. So a rectangle
[x0,x1) x [y0,y1) contains exactly the pixels with x0 <= c < x1, y0 <= r < y1.
To cover a whole W x H image with a rectangle use (0,0)-(W,H).
`point_region` uses the identical rule, so mask and point queries never disagree.

This module is the ONE geometry rule of the app: detection (pipeline keep/drop), the
review engine (membership, areas) and the export all call it (review/geometry.py only
re-exports). Consequences: a 10x10 rectangle (10,10)-(20,20) has area exactly 100 px; the
point (10,15) is inside and (20,15) is outside; two rectangles sharing an edge partition
the pixels without double counting; the boundary of an exclusion polygon belongs to the
exclusion only on its low (x0/y0) edges (half-open), so (10,15) is excluded and (20,15)
is not. Overlapping counting regions: the FIRST listed region wins both for membership
(`point_region`) and for area (`region_areas`: area net of earlier counting regions and of
all exclusions), so per-region areas always sum to the union area.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np

from ..schemas import Region, RegionKind


def _poly_array(region: Region) -> Optional[np.ndarray]:
    p = np.asarray(region.polygon, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3:
        return None
    return p


def rasterize_polygon(poly: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """bool (H,W) mask of pixel centres inside the polygon (even-odd, half-open)."""
    h, w = shape
    mask = np.zeros((h, w), bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    keep = y1 != y2  # horizontal edges never cross a scanline
    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    if x1.size == 0:
        return mask
    ylo, yhi = np.minimum(y1, y2), np.maximum(y1, y2)
    r0 = max(int(np.ceil(ylo.min())), 0)
    r1 = min(int(np.ceil(yhi.max())) - 1, h - 1)
    for r in range(r0, r1 + 1):
        hit = (ylo <= r) & (r < yhi)
        if not hit.any():
            continue
        xs = x1[hit] + (r - y1[hit]) * (x2[hit] - x1[hit]) / (y2[hit] - y1[hit])
        xs.sort()
        for a, b in zip(xs[0::2], xs[1::2]):
            ca, cb = max(int(np.ceil(a)), 0), min(int(np.ceil(b)), w)
            if cb > ca:
                mask[r, ca:cb] = True
    return mask


def points_in_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd test, same rule as rasterize_polygon."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inside = np.zeros(x.shape, bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for ax, ay, bx, by in zip(x1, y1, x2, y2):
        if ay == by:
            continue
        lo, hi = (ay, by) if ay < by else (by, ay)
        cond = (lo <= y) & (y < hi)
        if not cond.any():
            continue
        xc = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= cond & (xc > x)
    return inside


def build_valid_mask(shape: tuple[int, int], regions: Sequence[Region]) -> np.ndarray:
    """union(counting) minus union(exclusion); no counting regions => whole image
    minus exclusions. Degenerate polygons (<3 vertices) are ignored."""
    counting = [r for r in regions if r.kind == RegionKind.counting]
    exclusion = [r for r in regions if r.kind == RegionKind.exclusion]
    if counting:
        valid = np.zeros(shape, bool)
        for r in counting:
            p = _poly_array(r)
            if p is not None:
                valid |= rasterize_polygon(p, shape)
    else:
        valid = np.ones(shape, bool)
    for r in exclusion:
        p = _poly_array(r)
        if p is not None:
            valid &= ~rasterize_polygon(p, shape)
    return valid


def point_region(x: float, y: float, regions: Sequence[Region]) -> Optional[str]:
    """Id of the first (in list order) counting region containing the point and
    not inside any exclusion; else None (also None when there are no counting
    regions at all)."""
    xa, ya = np.array([x]), np.array([y])
    for r in regions:
        if r.kind == RegionKind.exclusion:
            p = _poly_array(r)
            if p is not None and points_in_polygon(xa, ya, p)[0]:
                return None
    for r in regions:
        if r.kind == RegionKind.counting:
            p = _poly_array(r)
            if p is not None and points_in_polygon(xa, ya, p)[0]:
                return r.id
    return None


def points_regions(xs: Iterable[float], ys: Iterable[float], regions: Sequence[Region]) -> list[Optional[str]]:
    """Vectorised point_region for many points (same semantics)."""
    xs = np.asarray(list(xs), dtype=np.float64)
    ys = np.asarray(list(ys), dtype=np.float64)
    out: list[Optional[str]] = [None] * xs.size
    if xs.size == 0:
        return out
    excluded = np.zeros(xs.shape, bool)
    for r in regions:
        if r.kind == RegionKind.exclusion:
            p = _poly_array(r)
            if p is not None:
                excluded |= points_in_polygon(xs, ys, p)
    assigned = excluded.copy()
    for r in regions:
        if r.kind == RegionKind.counting:
            p = _poly_array(r)
            if p is None:
                continue
            inside = points_in_polygon(xs, ys, p) & ~assigned
            for i in np.nonzero(inside)[0]:
                out[i] = r.id
            assigned |= inside
    return out


def region_area_px(region_or_regions: Union[Region, Sequence[Region]], shape: tuple[int, int]) -> float:
    """Area in pixels. A single Region: its own rasterised polygon clipped to the
    image. A list: net valid area = build_valid_mask(...).sum() (counting union
    minus exclusions)."""
    if isinstance(region_or_regions, Region):
        p = _poly_array(region_or_regions)
        return 0.0 if p is None else float(rasterize_polygon(p, shape).sum())
    return float(build_valid_mask(shape, list(region_or_regions)).sum())


def region_areas(regions: Sequence[Region], shape: tuple[int, int]) -> dict[str, float]:
    """Per-region pixel area under the single geometry rule.

    Counting region: pixels inside it, minus all exclusions, minus pixels of EARLIER
    counting regions (first listed wins, same as `point_region`), so the counting areas
    add up to the union area. Exclusion region: its own rasterised area (clipped to the
    image). Duplicate ids: the later entry overwrites (ids are unique in practice)."""
    exclusion = np.zeros(shape, bool)
    for r in regions:
        if r.kind == RegionKind.exclusion:
            p = _poly_array(r)
            if p is not None:
                exclusion |= rasterize_polygon(p, shape)
    taken = np.zeros(shape, bool)
    out: dict[str, float] = {}
    for r in regions:
        p = _poly_array(r)
        m = np.zeros(shape, bool) if p is None else rasterize_polygon(p, shape)
        if r.kind == RegionKind.counting:
            net = m & ~exclusion & ~taken
            taken |= m & ~exclusion
            out[r.id] = float(net.sum())
        else:
            out[r.id] = float(m.sum())
    return out
