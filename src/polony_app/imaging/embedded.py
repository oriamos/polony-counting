"""Raw access to embedded ImageJ point ROIs (not interpreted)."""
from __future__ import annotations

from pathlib import Path

import tifffile


def embedded_points(path: Path) -> list[tuple[float, float]]:
    """(x, y) coordinates of every POINT ROI / overlay embedded in the TIFF's
    ImageJ metadata, in image pixels (sub-pixel if stored). What the points
    mean (e.g. a manual annotation) is NOT known to the app; show them only as
    user-supplied reference marks."""
    import roifile
    pts: list[tuple[float, float]] = []
    try:
        with tifffile.TiffFile(str(path)) as tif:
            md = tif.imagej_metadata if tif.is_imagej else None
    except Exception:
        return pts
    if not md:
        return pts
    blobs = []
    if md.get("ROI"):
        blobs.append(md["ROI"])
    ov = md.get("Overlays")
    if ov:
        blobs += list(ov) if isinstance(ov, (list, tuple)) else [ov]
    for b in blobs:
        try:
            roi = roifile.ImagejRoi.frombytes(b)
            if getattr(roi.roitype, "name", "") == "POINT":
                pts += [(float(x), float(y)) for x, y in roi.coordinates()]
        except Exception:
            continue
    return pts
