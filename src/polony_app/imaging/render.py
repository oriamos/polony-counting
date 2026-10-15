"""Display rendering. Purely for the human viewer: never feeds detection.

Pixel coordinates follow schemas.py: x = column, y = row; ``crop`` is
(x0, y0, x1, y1) with x1/y1 exclusive, in image pixels.
"""
from __future__ import annotations

import io
from typing import Literal, Optional

import numpy as np
from PIL import Image

Crop = tuple[int, int, int, int]


def percentile_range(arr: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.8) -> tuple[float, float]:
    """Display range from percentiles (a viewing aid, not a measurement)."""
    a = np.asarray(arr)
    if a.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _crop(arr: np.ndarray, crop: Optional[Crop]) -> np.ndarray:
    if crop is None:
        return arr
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in crop)
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"crop {crop} is empty for an image of {w} x {h} (W x H)")
    return arr[y0:y1, x0:x1]


def normalize(arr: np.ndarray, lo: float, hi: float, gamma: float = 1.0) -> np.ndarray:
    """float32 in [0,1]: clip((arr-lo)/(hi-lo)) ** (1/gamma). gamma > 1 brightens."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if hi <= lo:
        hi = lo + 1e-9
    v = (np.asarray(arr, dtype=np.float32) - np.float32(lo)) / np.float32(hi - lo)
    np.clip(v, 0.0, 1.0, out=v)
    if gamma != 1.0:
        np.power(v, np.float32(1.0 / gamma), out=v)
    return v


def _downscale(v: np.ndarray, max_dim: Optional[int]) -> np.ndarray:
    """Area-average downscale of a float32 HxW or HxWx3 array so max side <= max_dim."""
    if not max_dim:
        return v
    h, w = v.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return v
    s = max_dim / m
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    chans = [v] if v.ndim == 2 else [v[..., i] for i in range(v.shape[2])]
    out = [np.asarray(Image.fromarray(np.ascontiguousarray(c), mode="F").resize((nw, nh), Image.BOX)) for c in chans]
    return out[0] if v.ndim == 2 else np.stack(out, axis=-1)


def _png(rgb_or_l: np.ndarray) -> bytes:
    img = Image.fromarray(rgb_or_l)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def _to_u8(v: np.ndarray) -> np.ndarray:
    return np.rint(v * 255.0).astype(np.uint8)


def render_png(arr: np.ndarray, lo: float, hi: float, gamma: float = 1.0,
               crop: Optional[Crop] = None, max_dim: Optional[int] = None,
               colormap: Literal["gray", "red", "green"] = "gray") -> bytes:
    """8-bit PNG of one 2-D channel with a linear [lo,hi] window and gamma."""
    if arr.ndim != 2:
        raise ValueError("render_png expects a 2-D array")
    v = _downscale(normalize(_crop(arr, crop), lo, hi, gamma), max_dim)
    u = _to_u8(v)
    if colormap == "gray":
        return _png(u)
    z = np.zeros_like(u)
    if colormap == "red":
        return _png(np.stack([u, z, z], axis=-1))
    if colormap == "green":
        return _png(np.stack([z, u, z], axis=-1))
    raise ValueError(f"unknown colormap {colormap!r}")


def render_composite_png(red: Optional[np.ndarray], green: Optional[np.ndarray],
                         red_range: tuple[float, float], green_range: tuple[float, float],
                         gamma: float = 1.0, crop: Optional[Crop] = None,
                         max_dim: Optional[int] = None) -> bytes:
    """Additive red+green display overlay (blue = 0); either may be None."""
    ref = red if red is not None else green
    if ref is None:
        raise ValueError("need at least one channel")
    h, w = _crop(ref, crop).shape
    out = np.zeros((h, w, 3), np.float32)
    if red is not None:
        out[..., 0] = normalize(_crop(red, crop), *red_range, gamma)
    if green is not None:
        out[..., 1] = normalize(_crop(green, crop), *green_range, gamma)
    return _png(_to_u8(_downscale(out, max_dim)))


def render_rgb_png(rgb: np.ndarray, lo: Optional[float] = None, hi: Optional[float] = None,
                   gamma: float = 1.0, crop: Optional[Crop] = None,
                   max_dim: Optional[int] = None) -> bytes:
    """Display an HxWx3 composite (dtype range used when lo/hi are None)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected HxWx3")
    if lo is None:
        lo = 0.0
    if hi is None:
        hi = float(np.iinfo(rgb.dtype).max) if rgb.dtype.kind in "ui" else 1.0
    v = _downscale(normalize(_crop(rgb, crop), lo, hi, gamma), max_dim)
    return _png(_to_u8(v))
