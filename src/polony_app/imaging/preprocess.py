"""Analytical background correction (independent of display rendering).

All work is float32. ``valid_mask`` (optional extension of the contract
signature) keeps pixels outside the counting area -- bright edge structures,
debris -- from biasing the background estimate of valid pixels. Saturated raw pixels
(integer dtype maximum) additionally get weight 0 in the gaussian background estimate,
inside or outside the mask.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, opening

from ..schemas import BackgroundParams

_MAD_TO_SIGMA = 1.4826


def _masked_gaussian_direct(a: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: gauss(a*w)/gauss(w), reflect boundaries."""
    num = ndi.gaussian_filter(a * w, sigma=sigma, mode="reflect")
    den = ndi.gaussian_filter(w, sigma=sigma, mode="reflect")
    with np.errstate(divide="ignore", invalid="ignore"):
        est = num / den
    bad = ~(den > 1e-4)
    if bad.any():
        sel = w > 0
        est[bad] = float(np.median(a[sel])) if sel.any() else 0.0
    return est.astype(np.float32, copy=False)


def _masked_gaussian(a: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Masked gaussian; for large sigma (>= 16 px) computed on a block-averaged
    grid (factor f = floor(sigma/8), so the filter is still >= 8 low-res px wide)
    and bilinearly interpolated back -- the estimate is smooth by construction, so
    this is accurate to a small fraction of the noise and ~20x faster / lighter."""
    f = int(sigma // 8)
    if f < 2:
        return _masked_gaussian_direct(a, w, sigma)
    h, wd = a.shape
    hp, wp = -(-h // f) * f, -(-wd // f) * f
    aw = np.zeros((hp, wp), np.float32)
    ww = np.zeros((hp, wp), np.float32)
    aw[:h, :wd] = a * w
    ww[:h, :wd] = w
    low_aw = aw.reshape(hp // f, f, wp // f, f).sum(axis=(1, 3), dtype=np.float64)
    low_w = ww.reshape(hp // f, f, wp // f, f).sum(axis=(1, 3), dtype=np.float64)
    s = sigma / f
    num = ndi.gaussian_filter(low_aw, s, mode="reflect")
    den = ndi.gaussian_filter(low_w, s, mode="reflect")
    with np.errstate(divide="ignore", invalid="ignore"):
        low = num / den
    sel = w > 0
    fallback = float(np.median(a[sel])) if sel.any() else 0.0
    low = np.where(den > 1e-4 * f * f, low, fallback)
    up = ndi.zoom(low.astype(np.float32), (f, f), order=1, mode="nearest", grid_mode=True)
    return np.ascontiguousarray(up[:h, :wd], dtype=np.float32)


def saturated_mask(arr: np.ndarray) -> np.ndarray:
    """Pixels at the integer dtype maximum (clipped, so their true value is unknown).
    All-False for float arrays (no saturation information)."""
    if arr.dtype.kind in "ui":
        return arr >= np.iinfo(arr.dtype).max
    return np.zeros(arr.shape, bool)


def gaussian_background(arr: np.ndarray, sigma_px: float, valid_mask: Optional[np.ndarray] = None,
                        exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """Large-sigma gaussian background estimate, robust to spots.

    Pixels in ``exclude`` (saturated raw pixels, see ``correct_background``) get weight 0
    in the normalised convolution in every case, valid mask or not: a clipped pixel is a
    lower bound of the true value and would otherwise bias the estimate (e.g. a saturated
    slide rim lifting the background under adjacent valid pixels).

    Pass 1: masked gaussian of all valid pixels. Pass 2: pixels more than 3
    robust sigmas above pass 1 (spots, bright structures) get weight 0 and the
    estimate is recomputed, so dense bright objects do not lift the background
    under themselves.
    """
    a = np.asarray(arr, dtype=np.float32)
    w = np.ones(a.shape, np.float32) if valid_mask is None else valid_mask.astype(np.float32)
    if exclude is not None:
        w = w * (~exclude).astype(np.float32)
    if not w.any():
        return np.zeros_like(a)
    bg = _masked_gaussian(a, w, sigma_px)
    sel = w > 0
    resid = (a - bg)[sel]
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    sig = _MAD_TO_SIGMA * mad
    if sig > 0:
        w2 = w * ((a - bg) <= (med + 3.0 * sig)).astype(np.float32)
        if w2.sum() > 0.2 * w.sum():  # keep enough support to be meaningful
            bg = _masked_gaussian(a, w2, sigma_px)
    return bg


def correct_background(arr: np.ndarray, p: BackgroundParams,
                       valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return a float32 background-corrected image (same shape)."""
    if arr.ndim != 2:
        raise ValueError("correct_background expects a 2-D array")
    a = np.asarray(arr, dtype=np.float32)
    if valid_mask is not None and valid_mask.shape != a.shape:
        raise ValueError("valid_mask shape differs from image shape")
    if p.method == "none":
        return a.copy()
    if p.method == "gaussian_subtract":
        if p.sigma_px <= 0:
            raise ValueError("sigma_px must be > 0")
        # saturated raw pixels carry weight 0 in the background estimate (in all cases)
        out = a - gaussian_background(a, float(p.sigma_px), valid_mask, exclude=saturated_mask(arr))
    elif p.method == "white_tophat":
        if p.radius_px < 1:
            raise ValueError("radius_px must be >= 1")
        src = a
        if valid_mask is not None and valid_mask.any() and not valid_mask.all():
            src = a.copy()
            src[~valid_mask] = np.median(a[valid_mask])  # neutral fill, see module doc
        opened = opening(src, disk(int(p.radius_px), decomposition="crosses"))
        out = a - opened
    else:  # pragma: no cover
        raise ValueError(f"unknown background method {p.method!r}")
    if p.clip_negative:
        np.maximum(out, 0.0, out=out)
    return out.astype(np.float32, copy=False)
