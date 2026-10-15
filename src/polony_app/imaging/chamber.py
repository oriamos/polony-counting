"""Automatic hybridisation-chamber region (preview only).

On the validated GenePix stacks the scanner background OUTSIDE the chamber is clipped (value = dtype max) in
one channel (page 1: ~18 % of the frame) while the chamber interior is dark. The mask is therefore:
  outside = smoothed fraction of clipped pixels (8x8 blocks) > 0.5 ; chamber = largest non-outside component,
  holes filled, boundary refined at full resolution ; counting area = chamber eroded by ``rim_px`` (Euclidean).
Deterministic (no randomness). Adapted from scripts/validation/chamber.py (docs/app/REAL_DATA_VALIDATION.md 1.4).
Also proposes, as a SUGGESTION only, a dark-bubble exclusion where the interior background is much lower
than elsewhere.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

BIN = 8
BUBBLE_BIN = 16
CLIP_FRACTION = 0.915          # >= 60000 for uint16, >= 234 for uint8
MIN_CLIPPED_FRACTION = 0.02
MAX_CLIPPED_FRACTION = 0.9
BUBBLE_RATIO = 0.6             # block background below this fraction of the chamber median = dark
BUBBLE_MIN_AREA_FRACTION = 0.01


class ChamberError(ValueError):
    pass


def _clipped(arr: np.ndarray) -> np.ndarray:
    return arr >= int(np.iinfo(arr.dtype).max * CLIP_FRACTION)


def chamber_mask(channels: Sequence[np.ndarray]) -> tuple[np.ndarray, int, list[str]]:
    """(mask, index of the channel used, warnings). Raises ChamberError when no clipped background is found."""
    warns: list[str] = []
    fracs = [float(_clipped(c).mean()) for c in channels]
    k = int(np.argmax(fracs))
    if fracs[k] < MIN_CLIPPED_FRACTION or fracs[k] > MAX_CLIPPED_FRACTION:
        raise ChamberError(
            f"No clipped scanner background found (largest clipped fraction {fracs[k]*100:.1f} %); the chamber "
            "cannot be detected automatically for this image. Draw the counting region by hand.")
    arr = channels[k]
    h, w = arr.shape
    hb, wb = h // BIN, w // BIN
    if hb < 4 or wb < 4:
        raise ChamberError("Image is too small for automatic chamber detection.")
    sat = _clipped(arr[: hb * BIN, : wb * BIN]).reshape(hb, BIN, wb, BIN).mean(axis=(1, 3))
    inside = ~(ndi.uniform_filter(sat, 3) > 0.5)
    lab, n = ndi.label(inside)
    if n == 0:
        raise ChamberError("The whole image looks like clipped background; no chamber found.")
    sizes = ndi.sum(inside, lab, range(1, n + 1))
    m = ndi.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))
    full = np.zeros((h, w), bool)
    full[: hb * BIN, : wb * BIN] = np.kron(m, np.ones((BIN, BIN), bool))
    if hb * BIN < h:
        full[hb * BIN:, :] = full[hb * BIN - 1: hb * BIN, :]
    if wb * BIN < w:
        full[:, wb * BIN:] = full[:, wb * BIN - 1: wb * BIN]
    band = ndi.binary_dilation(full, iterations=2 * BIN) & ~ndi.binary_erosion(full, iterations=2 * BIN)
    sm = ndi.median_filter(arr, size=5)
    full = (full & ~band) | (band & ~_clipped(sm))
    lab, n = ndi.label(full)
    if n > 1:
        sizes = ndi.sum(full, lab, range(1, n + 1))
        full = lab == 1 + int(np.argmax(sizes))
    full = ndi.binary_fill_holes(full)
    if full.mean() > 0.97:
        warns.append("The detected chamber covers almost the whole image; check the result.")
    return full, k, warns


def counting_mask(mask: np.ndarray, rim_px: int) -> np.ndarray:
    if rim_px <= 0:
        return mask
    dist = ndi.distance_transform_edt(mask)
    cm = dist > rim_px
    lab, n = ndi.label(cm)
    if n > 1:
        sizes = ndi.sum(cm, lab, range(1, n + 1))
        cm = lab == 1 + int(np.argmax(sizes))
    return ndi.binary_fill_holes(cm)


def mask_to_polygon(mask: np.ndarray, tol: float = 2.0) -> list[list[float]]:
    """Outer contour of the largest component as a simple polygon in canonical coordinates (x, y)."""
    pad = np.pad(mask.astype(np.uint8), 1)
    best: Optional[np.ndarray] = None
    for c in measure.find_contours(pad, 0.5):
        if best is None or len(c) > len(best):
            best = c
    if best is None or len(best) < 4:
        raise ChamberError("Region is empty after applying the rim.")
    best = best - 1.0                                  # undo padding; (row, col) in pixel-centre coordinates
    for t in (tol, tol / 2, 0.5, 0.0):
        pts = measure.approximate_polygon(best, t) if t > 0 else best
        pts = pts[:-1] if len(pts) > 1 and np.allclose(pts[0], pts[-1]) else pts
        poly = [[float(round(p[1], 2)), float(round(p[0], 2))] for p in pts]
        if len(poly) >= 3 and _simple(poly):
            return poly
    raise ChamberError("Could not build a valid polygon from the chamber mask.")


def _simple(poly: list[list[float]]) -> bool:
    """No two non-adjacent edges intersect (O(n^2) on a few hundred vertices at most; bounding-box prefilter)."""
    p = np.asarray(poly, float)
    n = len(p)
    a, b = p, np.roll(p, -1, axis=0)
    lo, hi = np.minimum(a, b), np.maximum(a, b)

    def ccw(x, y, z):
        return (z[1] - x[1]) * (y[0] - x[0]) - (y[1] - x[1]) * (z[0] - x[0])

    for i in range(n):
        ov = np.nonzero((lo[:, 0] <= hi[i, 0]) & (hi[:, 0] >= lo[i, 0]) & (lo[:, 1] <= hi[i, 1]) & (hi[:, 1] >= lo[i, 1]))[0]
        for j in ov:
            if j <= i or j == (i + 1) % n or (j + 1) % n == i:
                continue
            d1, d2 = ccw(a[i], b[i], a[j]), ccw(a[i], b[i], b[j])
            d3, d4 = ccw(a[j], b[j], a[i]), ccw(a[j], b[j], b[i])
            if d1 * d2 < 0 and d3 * d4 < 0:
                return False
    return True


def polygon_area(poly: list[list[float]]) -> float:
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def suggest_bubble(arr: np.ndarray, count_mask: np.ndarray) -> list[dict]:
    """Dark-region (bubble) suggestions inside the counting mask from the block-wise low background of `arr`."""
    h, w = arr.shape
    hb, wb = h // BUBBLE_BIN, w // BUBBLE_BIN
    if hb < 4 or wb < 4 or count_mask.sum() < 100 * BUBBLE_BIN * BUBBLE_BIN:
        return []
    a = arr[: hb * BUBBLE_BIN, : wb * BUBBLE_BIN].astype(np.float32)
    blocks = a.reshape(hb, BUBBLE_BIN, wb, BUBBLE_BIN).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
    low = np.percentile(blocks, 20, axis=2)                       # robust to polonies
    cov = count_mask[: hb * BUBBLE_BIN, : wb * BUBBLE_BIN].reshape(hb, BUBBLE_BIN, wb, BUBBLE_BIN).mean(axis=(1, 3))
    ok = cov > 0.99
    if ok.sum() < 20:
        return []
    ref = float(np.median(low[ok]))
    if ref <= 0:
        return []
    dark = ok & (ndi.uniform_filter(low, 3) < BUBBLE_RATIO * ref)
    dark = ndi.binary_opening(dark, iterations=1)
    lab, n = ndi.label(dark)
    out = []
    for i in range(1, n + 1):
        comp = ndi.binary_fill_holes(lab == i)
        if comp.sum() * BUBBLE_BIN ** 2 < BUBBLE_MIN_AREA_FRACTION * count_mask.sum():
            continue
        full = np.zeros((h, w), bool)
        full[: hb * BUBBLE_BIN, : wb * BUBBLE_BIN] = np.kron(comp, np.ones((BUBBLE_BIN, BUBBLE_BIN), bool))
        full &= count_mask
        try:
            poly = mask_to_polygon(full, tol=BUBBLE_BIN / 2)
        except ChamberError:
            continue
        out.append({"polygon": poly, "area_px2": polygon_area(poly),
                    "reason": f"Block background below {int(BUBBLE_RATIO*100)} % of the chamber median "
                              "(possible bubble or dry area). Suggestion only; check the image."})
    out.sort(key=lambda d: -d["area_px2"])
    return out[:3]


def auto_chamber(channels: Sequence[np.ndarray], rim_px: int = 40) -> dict:
    """Preview of the chamber counting region. Does not touch any analysis."""
    if rim_px < 0 or rim_px > 500:
        raise ChamberError("rim_px must be between 0 and 500.")
    mask, k, warns = chamber_mask(channels)
    cm = counting_mask(mask, rim_px)
    if not cm.any():
        raise ChamberError(f"The rim of {rim_px} px leaves no area; use a smaller rim.")
    poly = mask_to_polygon(cm)
    other = [i for i in range(len(channels)) if i != k]
    bubble: list[dict] = []
    if other:
        bubble = suggest_bubble(channels[min(other, key=lambda i: float(_clipped(channels[i]).mean()))], cm)
    return {
        "polygon": poly, "area_px2": float(cm.sum()), "rim_px": int(rim_px),
        "method": (f"clipped scanner background of channel index {k} marks the outside; chamber = largest "
                   f"inside component; eroded by {rim_px} px"),
        "chamber_area_px2": float(mask.sum()), "mask_channel_index": k,
        "suggested_exclusions": bubble, "warnings": warns,
    }
