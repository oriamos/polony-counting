"""Ingestion of user TIFF files into a data dir, preserving original bytes.

Rules (see docs/app/CONTRACTS.md):
* originals are copied byte-for-byte (the source is only read); every stored
  original is sha256-hashed (the hash of the *stored copy* is verified against
  the hash of the source stream);
* pixel values and bit depth are never altered (no scaling, no resampling);
* channel colours are NEVER inferred from metadata or file names:
  ``role`` stays None and ``role_confirmed`` False. ``label_hint`` is raw text;
* embedded ImageJ ROI / Overlays are parsed to ``EmbeddedRoi`` and not interpreted.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from ..schemas import ChannelInfo, ChannelRole, EmbeddedRoi, ImageKind, ImageMeta
from .io import IngestError, read_stack

__all__ = ["ingest_files", "IngestError", "parse_info_label_hint", "channel_stats"]

_SUPPORTED_DTYPES = (np.dtype("uint8"), np.dtype("uint16"))
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


# ------------------------------------------------------------------ helpers
def _safe_name(idx: int, original: str) -> str:
    base = Path(original).name or f"file{idx}"
    stem, ext = Path(base).stem, Path(base).suffix.lower()
    stem = _SAFE.sub("_", stem).strip("._") or f"file{idx}"
    if ext not in (".tif", ".tiff"):
        ext = ".tif"
    return f"{idx:02d}_{stem[:80]}{ext}"


def _copy_and_hash(src: Path, dst: Path) -> str:
    """Copy src -> dst in chunks; return sha256 of the bytes written, verified
    against the bytes read. The source is opened read-only."""
    h_src = hashlib.sha256()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            chunk = fi.read(1 << 20)
            if not chunk:
                break
            h_src.update(chunk)
            fo.write(chunk)
    h_dst = hashlib.sha256()
    with open(dst, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h_dst.update(chunk)
    if h_src.hexdigest() != h_dst.hexdigest():
        raise IngestError(f"{src.name}: copy verification failed (disk error?). Try again.")
    return h_dst.hexdigest()


def parse_info_label_hint(info: Optional[str]) -> Optional[str]:
    """Raw hint from an ImageJ 'Info' text: the value of an 'ImageDescription:'
    line (e.g. '635 [W1]'). Returned verbatim; never interpreted as a colour."""
    if not info:
        return None
    for line in info.splitlines():
        m = re.match(r"\s*ImageDescription\s*[:=]\s*(.+?)\s*$", line)
        if m and m.group(1):
            return m.group(1)
    return None


def channel_stats(plane: np.ndarray) -> tuple[int, int, float]:
    """(min, max, fraction of pixels at the dtype maximum)."""
    dmax = np.iinfo(plane.dtype).max
    return int(plane.min()), int(plane.max()), float(np.count_nonzero(plane == dmax) / plane.size)


def _roi_from_bytes(blob: bytes, source: str, warnings: list[str]) -> list[EmbeddedRoi]:
    import roifile

    out: list[EmbeddedRoi] = []
    try:
        roi = roifile.ImagejRoi.frombytes(blob)
    except Exception as exc:
        warnings.append(f"Embedded ImageJ {source} could not be parsed ({type(exc).__name__}); ignored.")
        return out
    try:
        n = int(roi.n_coordinates) or None
        name = roi.name if roi.name else None
        out.append(
            EmbeddedRoi(
                source=source,  # type: ignore[arg-type]
                roi_type="COMPOSITE" if getattr(roi, "composite", False) else getattr(roi.roitype, "name", str(roi.roitype)),
                name=name,
                left=int(roi.left), top=int(roi.top),
                right=int(roi.right), bottom=int(roi.bottom),
                n_points=n,
                c_position=int(roi.c_position), z_position=int(roi.z_position),
                t_position=int(roi.t_position),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"Embedded ImageJ {source} has unreadable fields ({exc}); ignored.")
    return out


def _read_metadata(path: Path, warnings: list[str]) -> dict:
    """Metadata only; no pixel decoding."""
    info: dict = {"info": None, "labels": None, "rois": [], "description": None,
                  "res_note": None, "is_imagej": False, "n_series": 1}
    try:
        with tifffile.TiffFile(str(path)) as tif:
            info["is_imagej"] = bool(tif.is_imagej)
            info["n_series"] = len(tif.series)
            page0 = tif.pages[0]
            desc = page0.description if hasattr(page0, "description") else None
            info["description"] = desc
            md = tif.imagej_metadata if tif.is_imagej else None
            if md:
                info["info"] = md.get("Info") if isinstance(md.get("Info"), str) else None
                lab = md.get("Labels")
                info["labels"] = list(lab) if isinstance(lab, (list, tuple)) else None
                if md.get("ROI"):
                    info["rois"] += _roi_from_bytes(md["ROI"], "ROI", warnings)
                ov = md.get("Overlays")
                if ov:
                    for blob in (ov if isinstance(ov, (list, tuple)) else [ov]):
                        info["rois"] += _roi_from_bytes(blob, "Overlays", warnings)
                unit = str(md.get("unit", "") or "")
            else:
                unit = ""
            # resolution tag: informational only, never applied
            try:
                xr = page0.tags.get("XResolution")
                if xr is not None and md is not None and unit:
                    num, den = xr.value
                    if num and den:
                        info["res_note"] = (
                            f"{path.name}: TIFF resolution tag = {num/den:g} pixels per '{unit}' "
                            "(informational; pixel size is NOT applied or assumed - set it explicitly if known)."
                        )
            except Exception:
                pass
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"{path.name}: cannot read as TIFF ({type(exc).__name__}: {exc}).") from exc
    return info


def _plain_description_hint(desc: Optional[str], is_imagej: bool) -> Optional[str]:
    if is_imagej or not desc:
        return None
    d = desc.strip()
    if d.startswith("<") or "\n" in d or len(d) > 80:
        return None
    return d


# ------------------------------------------------------------------ main
def ingest_files(files: list[tuple[str, Path]], dest_dir: Path) -> ImageMeta:
    """See module docstring and CONTRACTS.md. Raises IngestError on bad input."""
    if not files:
        raise IngestError("No files were provided. Upload one 2-channel TIFF, two single-channel TIFFs, or one RGB TIFF.")
    if len(files) > 2:
        raise IngestError(f"{len(files)} files were provided; at most 2 are supported (one per channel).")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    stored: list[tuple[str, Path, str]] = []  # (orig name, stored path, sha)
    for i, (orig, src) in enumerate(files):
        src = Path(src)
        if not src.is_file():
            raise IngestError(f"{orig}: file not found.")
        dst = dest_dir / _safe_name(i, orig)
        stored.append((orig, dst, _copy_and_hash(src, dst)))
    sha = {p.name: h for _, p, h in stored}

    metas = [_read_metadata(p, warnings) for _, p, _ in stored]
    arrays = [read_stack(p) for _, p, _ in stored]
    for (orig, _, _), arr in zip(stored, arrays):
        if arr.dtype not in _SUPPORTED_DTYPES:
            raise IngestError(
                f"{orig}: pixel type {arr.dtype} is not supported (only 8-bit and 16-bit unsigned integer). "
                "Export the raw scanner image without converting to floating point."
            )

    for (orig, _, _), m in zip(stored, metas):
        if m["n_series"] > 1:
            raise IngestError(
                f"{orig} contains {m['n_series']} separate image series (pages of differing shape/type). "
                "Export only the channels of interest with identical dimensions."
            )

    rois: list[EmbeddedRoi] = [r for m in metas for r in m["rois"]]
    infos = [m["info"] for m in metas if m["info"]]
    tiff_info = None
    if infos:
        tiff_info = infos[0] if len(infos) == 1 and len(files) == 1 else "\n".join(
            f"--- {stored[i][1].name} ---\n{m['info']}" for i, m in enumerate(metas) if m["info"])
    for m in metas:
        if m["res_note"]:
            warnings.append(m["res_note"])

    if len(files) == 1:
        kind, channels, shape, dtype = _ingest_single(stored[0], arrays[0], metas[0], warnings)
    else:
        kind, channels, shape, dtype = _ingest_pair(stored, arrays, metas, warnings)

    for ch in channels:
        if ch.saturated_fraction > 0.001:
            warnings.append(
                f"Channel {ch.index}: {ch.saturated_fraction*100:.2f}% of pixels are at the maximum value "
                f"({np.iinfo(dtype).max}) - saturated; intensities there are clipped."
            )
    if kind == ImageKind.rgb_composite:
        warnings.append(
            "RGB composite: raw per-channel intensities are NOT available. Colour planes of a rendered "
            "composite may have been scaled, clipped or mixed by the exporting software, so counts from "
            "it are approximate. Prefer the raw two-channel image if you have it."
        )
    return ImageMeta(
        kind=kind, height=int(shape[0]), width=int(shape[1]),
        dtype=str(dtype), bit_depth=int(dtype.itemsize * 8), sha256=sha,
        channels=channels, pixel_size_um=None, tiff_info_text=tiff_info,
        embedded_rois=rois, warnings=warnings,
        composite_only=(kind == ImageKind.rgb_composite),
    )


def _ingest_single(st, arr, meta, warnings):
    orig, path, _ = st
    name = path.name
    if arr.ndim == 2:
        raise IngestError(
            f"{orig} contains a single-channel image. Provide the second channel as another file "
            "(two single-channel TIFFs) or upload one multi-page 2-channel TIFF."
        )
    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] > 4:
        h, w = arr.shape[:2]
        if arr.shape[-1] == 4:
            warnings.append("RGB image has an alpha channel; it is ignored.")
        chans = []
        info = meta["info"]
        for k, nm in enumerate("RGB"):
            plane = arr[..., k]
            mn, mx, sf = channel_stats(plane)
            chans.append(ChannelInfo(index=k, source_file=name, page=k,
                                     label_hint=f"composite colour plane {nm}",
                                     min=mn, max=mx, saturated_fraction=sf))
        return ImageKind.rgb_composite, chans, (h, w), arr.dtype
    if arr.ndim == 3 and arr.shape[0] == 2:
        h, w = arr.shape[1:]
        labels = meta["labels"]
        hint0 = parse_info_label_hint(meta["info"]) or _plain_description_hint(meta["description"], meta["is_imagej"])
        chans = []
        for k in range(2):
            mn, mx, sf = channel_stats(arr[k])
            hint = None
            if labels and len(labels) == 2 and labels[k]:
                hint = str(labels[k])
            elif k == 0:
                hint = hint0
            chans.append(ChannelInfo(index=k, source_file=name, page=k, label_hint=hint,
                                     min=mn, max=mx, saturated_fraction=sf))
        if meta["is_imagej"] and hint0 and "635" in hint0 and not (labels and len(labels) == 2):
            # suggestion only: role stays None / role_confirmed False
            chans[0].suggested_role = ChannelRole.red
            chans[1].suggested_role = ChannelRole.green
            warnings.append(
                "Suggested from 8 validated GenePix stacks; confirm with your scanner settings. "
                "RGB composites are clipped; do not use for detection."
            )
        if hint0 and not (labels and len(labels) == 2):
            warnings.append(
                f"The file carries one metadata label ('{hint0}') for the whole stack; it is shown as a hint on "
                "page 0 only and cannot tell which marker/colour either channel is. Confirm the channel mapping yourself."
            )
        return ImageKind.raw_two_channel, chans, (h, w), arr.dtype
    if arr.ndim == 3:
        raise IngestError(
            f"{orig} has {arr.shape[0]} planes (shape {tuple(arr.shape)}); exactly 2 channels (or one RGB image) "
            "are supported. Export only the two channels of interest."
        )
    raise IngestError(f"{orig}: unsupported image layout (shape {tuple(arr.shape)}).")


def _ingest_pair(stored, arrays, metas, warnings):
    shapes = [a.shape for a in arrays]
    for (orig, _, _), a in zip(stored, arrays):
        if a.ndim != 2:
            raise IngestError(
                f"{orig}: when uploading two files each must be a single-channel, single-page image "
                f"(got shape {tuple(a.shape)})."
            )
    if shapes[0] != shapes[1]:
        raise IngestError(
            f"The two files have different sizes: {stored[0][0]} is {shapes[0][0]} x {shapes[0][1]} (H x W) but "
            f"{stored[1][0]} is {shapes[1][0]} x {shapes[1][1]}. The app never resamples. Export both channels "
            "with identical dimensions (same scan area / crop) and upload again."
        )
    if arrays[0].dtype != arrays[1].dtype:
        raise IngestError(
            f"The two files have different bit depths ({arrays[0].dtype} vs {arrays[1].dtype}); "
            "export both channels with the same bit depth."
        )
    chans = []
    for k, ((orig, path, _), a, m) in enumerate(zip(stored, arrays, metas)):
        mn, mx, sf = channel_stats(a)
        hint = parse_info_label_hint(m["info"]) or _plain_description_hint(m["description"], m["is_imagej"])
        chans.append(ChannelInfo(index=k, source_file=path.name, page=0, label_hint=hint,
                                 min=mn, max=mx, saturated_fraction=sf))
    warnings.append("Two separate files: file names are not used to decide which is red or green; confirm the mapping.")
    return ImageKind.raw_separate_files, chans, shapes[0], arrays[0].dtype
