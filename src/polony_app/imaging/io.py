"""Reading pixel data back from stored originals (never writes).

Plane addressing: ``ChannelInfo.page`` is the *plane index* inside the stored
file: TIFF page index, or (for ImageJ hyperstacks written with a single IFD)
the plane index of the first image series. For RGB composites it is the colour
plane (0=R, 1=G, 2=B) of the stored composite -- a structural fact, not a claim
about which marker is in which colour.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from ..schemas import ImageKind, ImageMeta


class IngestError(ValueError):
    """Input files cannot be used as-is. The message is meant for the end user."""


def read_stack(path: Path) -> np.ndarray:
    """Read the first image series of a TIFF as stored.

    Returns (H, W) for one plane, (n, H, W) for a plane stack or (H, W, 3|4)
    for interleaved RGB(A) (planar RGB is moved to the last axis). dtype is the
    file's dtype (no scaling).
    """
    try:
        with tifffile.TiffFile(str(path)) as tif:
            if not tif.series:
                raise IngestError(f"{path.name}: no image data found in this TIFF.")
            series = tif.series[0]
            arr = series.asarray()
            axes = series.axes
    except IngestError:
        raise
    except Exception as exc:  # tifffile raises many types (codec missing, corrupt...)
        raise IngestError(
            f"{path.name}: cannot read as TIFF ({type(exc).__name__}: {exc}). "
            "If it is compressed with an unusual codec, re-save it uncompressed."
        ) from exc
    return _normalize_axes(arr, axes)


def _normalize_axes(arr: np.ndarray, axes: str) -> np.ndarray:
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    if arr.ndim == 3 and axes[:1] == "S" and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)  # planar RGB -> interleaved
    return arr


def read_plane(path: Path, page: int) -> np.ndarray:
    """Read one 2-D plane without loading unrelated planes when possible."""
    try:
        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            if series.ndim == 3 and len(tif.pages) == series.shape[0] and series.axes[-2:] == "YX":
                return tif.pages[page].asarray()
    except Exception:
        pass
    arr = read_stack(path)
    if arr.ndim == 2:
        if page != 0:
            raise IngestError(f"{path.name}: has a single plane, page {page} requested.")
        return arr
    return np.ascontiguousarray(arr[page])


def load_rgb(meta: ImageMeta, dest_dir: Path) -> np.ndarray:
    """H x W x 3 composite as stored (original dtype). Only for rgb_composite."""
    if meta.kind != ImageKind.rgb_composite:
        raise ValueError("load_rgb is only valid for rgb_composite images")
    arr = read_stack(Path(dest_dir) / meta.channels[0].source_file)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise IngestError("stored composite is not an RGB image")
    return np.ascontiguousarray(arr[..., :3])


def load_channel(meta: ImageMeta, dest_dir: Path, channel_index: int) -> np.ndarray:
    """2-D array of one channel in its original dtype and values.

    For rgb_composite this returns the stored colour plane ``page`` (0=R,1=G,2=B)
    of the composite -- NOT a raw channel (see ``ImageMeta.composite_only``).
    """
    info = next((c for c in meta.channels if c.index == channel_index), None)
    if info is None:
        raise IndexError(f"no channel with index {channel_index}")
    if meta.kind == ImageKind.rgb_composite:
        return np.ascontiguousarray(load_rgb(meta, dest_dir)[..., info.page])
    arr = read_plane(Path(dest_dir) / info.source_file, info.page)
    if arr.ndim != 2:
        raise IngestError(f"{info.source_file}: plane {info.page} is not 2-D")
    return arr
