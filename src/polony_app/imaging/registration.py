"""Channel registration helpers. Nothing here resamples: shifts are integer
pixel moves applied only when the user asks for them."""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft

from .io import IngestError


def check_compatible(red: np.ndarray, green: np.ndarray) -> list[str]:
    """Warnings for a channel pair; IngestError if shapes differ."""
    if red.ndim != 2 or green.ndim != 2:
        raise IngestError("Both channels must be 2-D images.")
    if red.shape != green.shape:
        raise IngestError(
            f"Channel sizes differ: {red.shape[0]} x {red.shape[1]} vs {green.shape[0]} x {green.shape[1]} (H x W). "
            "The app never resamples or crops silently. Re-export both channels with identical dimensions."
        )
    warns: list[str] = []
    if red.dtype != green.dtype:
        warns.append(f"Channels have different dtypes ({red.dtype} vs {green.dtype}).")
    for name, a in (("first", red), ("second", green)):
        if a.min() == a.max():
            warns.append(f"The {name} channel is constant (no signal).")
    if red.dtype == green.dtype and np.array_equal(red, green):
        warns.append("Both channels are pixel-identical - check that two different channels were exported.")
    return warns


def _prep(a: np.ndarray, win: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    a = np.log1p(np.maximum(a - float(a.min()), 0.0))  # compress saturated edges
    a -= a.mean()
    return a * win


def estimate_shift(red: np.ndarray, green: np.ndarray, max_shift_px: int = 20) -> tuple[float, float, float]:
    """(dy, dx, peak_corr): the shift to apply to GREEN (apply_shift(green, dy, dx))
    so it lines up with RED, from windowed phase correlation, sub-pixel refined
    by a parabola. peak_corr is the Pearson correlation of the two images over
    their overlap at the rounded shift (0..1; low values mean the estimate is
    not trustworthy -- e.g. channels that share no structure)."""
    if red.shape != green.shape:
        raise IngestError("Cannot register channels of different size.")
    h, w = red.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    fr, fg = sfft.rfft2(_prep(red, win)), sfft.rfft2(_prep(green, win))
    cp = fr * np.conj(fg)
    cp /= np.maximum(np.abs(cp), 1e-12)
    corr = sfft.fftshift(sfft.irfft2(cp, s=(h, w)))
    cy, cx = h // 2, w // 2
    m = int(max_shift_px)
    y0, y1, x0, x1 = max(cy - m, 1), min(cy + m + 1, h - 1), max(cx - m, 1), min(cx + m + 1, w - 1)
    sub = corr[y0:y1, x0:x1]
    iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
    py, px = iy + y0, ix + x0

    def _para(m1, c, p1):
        d = m1 - 2 * c + p1
        return 0.0 if d == 0 else float(np.clip(0.5 * (m1 - p1) / d, -0.5, 0.5))

    sy = _para(corr[py - 1, px], corr[py, px], corr[py + 1, px])
    sx = _para(corr[py, px - 1], corr[py, px], corr[py, px + 1])
    dy, dx = float(py - cy + sy), float(px - cx + sx)
    ry, rx = int(round(dy)), int(round(dx))
    return dy, dx, _pearson_overlap(red, apply_shift(green, ry, rx), ry, rx)


def _pearson_overlap(ref: np.ndarray, shifted: np.ndarray, dy: int, dx: int) -> float:
    h, w = ref.shape
    ys, ye = max(dy, 0), h + min(dy, 0)
    xs, xe = max(dx, 0), w + min(dx, 0)
    if ye - ys < 2 or xe - xs < 2:
        return 0.0
    a = ref[ys:ye, xs:xe].astype(np.float64).ravel()
    b = shifted[ys:ye, xs:xe].astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def apply_shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Move image content by (dy rows down, dx columns right), integer pixels
    only (no interpolation, no wrap-around). Uncovered pixels are 0 and the
    array keeps its shape and dtype; use ``shifted_valid_region`` to know which
    pixels carry real data."""
    if float(dy) != int(dy) or float(dx) != int(dx):
        raise ValueError("apply_shift accepts integer pixel shifts only (the app never resamples)")
    dy, dx = int(dy), int(dx)
    out = np.zeros_like(arr)
    h, w = arr.shape[:2]
    if abs(dy) >= h or abs(dx) >= w:
        return out
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        arr[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def shifted_valid_region(shape: tuple[int, int], dy: int, dx: int) -> np.ndarray:
    """bool mask of pixels that hold real data after apply_shift(.., dy, dx)."""
    m = np.zeros(shape, bool)
    h, w = shape
    if abs(dy) < h and abs(dx) < w:
        m[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = True
    return m
