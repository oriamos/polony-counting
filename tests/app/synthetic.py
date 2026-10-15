"""Synthetic image generator for software checks (NOT accuracy evidence).

Known gaussian spots on an uneven background with noise, optional saturation
and a bright saturated edge band. Everything is seeded and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tifffile

Spot = tuple[float, float, float]  # x, y, amplitude (counts above background)


@dataclass
class Scene:
    red: np.ndarray
    green: np.ndarray
    red_spots: list[Spot]
    green_spots: list[Spot]
    sigma: float
    shape: tuple[int, int]
    edge_band: int = 0


def _gauss(shape, x, y, amp, sigma):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))


def uneven_background(shape, base=1500.0, slope=800.0, bump=1200.0, bump_sigma=90.0) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    bg = base + slope * (xx / w) + 0.5 * slope * (yy / h)
    bg += bump * np.exp(-((xx - 0.7 * w) ** 2 + (yy - 0.3 * h) ** 2) / (2 * bump_sigma ** 2))
    return bg


def make_channel(shape, spots: Sequence[Spot], sigma=3.0, noise=12.0, seed=0, base=1500.0,
                 edge_band=0, dtype=np.uint16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = uneven_background(shape, base=base)
    for x, y, a in spots:
        img += _gauss(shape, x, y, a, sigma)
    img += rng.normal(0.0, noise, shape)
    if edge_band:
        # ragged saturated structure along the left edge, as a slide rim would be
        ragged = edge_band + (np.sin(np.arange(shape[0]) / 7.0) * 3).astype(int)
        for r in range(shape[0]):
            img[r, :max(ragged[r], 0)] = 1e6
    top = np.iinfo(dtype).max
    return np.clip(np.rint(img), 0, top).astype(dtype)


def make_scene(shape=(300, 400), red_spots: Optional[Sequence[Spot]] = None,
               green_spots: Optional[Sequence[Spot]] = None, sigma=3.0, noise=12.0, seed=1,
               edge_band=0) -> Scene:
    red_spots = list(red_spots or [])
    green_spots = list(green_spots or [])
    red = make_channel(shape, red_spots, sigma, noise, seed, edge_band=edge_band)
    green = make_channel(shape, green_spots, sigma, noise, seed + 1000, base=900.0, edge_band=edge_band)
    return Scene(red, green, red_spots, green_spots, sigma, shape, edge_band)


def default_scene(seed=1, edge_band=0) -> Scene:
    """Well separated spots: 5 red-only, 4 green-only, 4 shared (yellow, tiny
    offset). Away from borders; amplitudes well above noise."""
    red = [(60, 50, 3000), (140, 80, 2500), (300, 60, 4000), (80, 220, 2800), (330, 250, 3500)]
    green = [(250, 150, 3000), (120, 160, 2600), (200, 250, 3200), (350, 130, 2400)]
    shared = [(100.0, 120.0), (220.0, 80.0), (180.0, 200.0), (290.0, 200.0)]
    red += [(x + 0.4, y - 0.3, 3000) for x, y in shared]
    green += [(x - 0.4, y + 0.5, 2800) for x, y in shared]
    return make_scene((300, 400), red, green, seed=seed, edge_band=edge_band)


def saturated_spot_scene() -> Scene:
    """One spot whose peak exceeds uint16 (clipped), one normal spot."""
    return make_scene((200, 200), [(60, 100, 90000), (140, 100, 3000)], [], seed=3)


# ------------------------------------------------------------------ file writers
def write_two_channel(path: Path, red: np.ndarray, green: np.ndarray, imagej: bool = True,
                      info: Optional[str] = None, roi_points: Optional[np.ndarray] = None) -> Path:
    data = np.stack([red, green])
    if imagej:
        meta: dict = {}
        if info is not None:
            meta["Info"] = info
        if roi_points is not None:
            import roifile
            roi = roifile.ImagejRoi.frompoints(roi_points.astype(np.float32))
            roi.roitype = roifile.ROI_TYPE.POINT
            meta["ROI"] = roi.tobytes()
        tifffile.imwrite(str(path), data, imagej=True, metadata=meta or None)
    else:
        tifffile.imwrite(str(path), data)  # plain multi-page TIFF
    return path


def write_single(path: Path, arr: np.ndarray, info: Optional[str] = None) -> Path:
    if info is not None:
        tifffile.imwrite(str(path), arr, imagej=True, metadata={"Info": info})
    else:
        tifffile.imwrite(str(path), arr)
    return path


def write_rgb(path: Path, red: np.ndarray, green: np.ndarray) -> Path:
    rgb = np.zeros(red.shape + (3,), np.uint8)
    rgb[..., 0] = (red >> 8).astype(np.uint8)
    rgb[..., 1] = (green >> 8).astype(np.uint8)
    tifffile.imwrite(str(path), rgb, photometric="rgb")
    return path
