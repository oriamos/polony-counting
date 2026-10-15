"""Discover historical files, convert them into the canonical :class:`Dataset`, and report.

Nothing here writes anywhere: :func:`build_dataset` only reads the source tree (dry-run
by construction); :func:`polony_app.legacy_import.schema.write_dataset` commits.

Hard rules (see docs/app/DATA_FORMATS.md):

* pixel data is only read, never rescaled, down-converted or resampled;
* no channel/colour/class is ever inferred: ``class_label`` stays ``None`` unless the
  caller passes an explicit ``class_map``; planes keep their raw hints;
* out-of-bounds points are flagged, never clamped;
* the CSV is compared with the ROI counts as *evidence only*.
"""
from __future__ import annotations

import fnmatch
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from ..imaging.ingest import channel_stats, parse_info_label_hint
from ..imaging.io import IngestError, read_stack
from ..store.csv_import import parse_csv
from . import roi as roimod
from .identity import parse_identity
from .schema import (
    Dataset, DatasetManifest, FieldRecord, FieldStatus, ImageKind, ImageRecord, PlaneInfo,
    PlaneMatch, PointLabel, PointStatus, Reconciliation, RegionRecord, Severity, WarningRecord,
    compute_dataset_id,
)

SUPPORTED_DTYPES = ("uint8", "uint16")
_TIF = {".tif", ".tiff"}


@dataclass
class ImportConfig:
    """Explicit user decisions. Defaults make NO scientific assumption."""
    class_map: dict[int, str] = field(default_factory=dict)         # ImageJ position (1-based) -> class label
    reserve: dict[str, str] = field(default_factory=dict)           # glob on rel path, or sha256 prefix -> split
    valid_region_names: tuple[str, ...] = ()
    excluded_region_names: tuple[str, ...] = ()
    duplicate_tol_px: float = 1.0
    close_pair_px: float = 3.0
    check_signal: bool = True
    check_composites: bool = True
    reserve_legacy_test_dirs: bool = True
    created_at: Optional[str] = None                                # inject for deterministic manifests

    def to_json(self) -> dict:
        return {
            "class_map": {str(k): v for k, v in sorted(self.class_map.items())},
            "reserve": dict(sorted(self.reserve.items())),
            "valid_region_names": list(self.valid_region_names),
            "excluded_region_names": list(self.excluded_region_names),
            "duplicate_tol_px": self.duplicate_tol_px, "close_pair_px": self.close_pair_px,
            "reserve_legacy_test_dirs": self.reserve_legacy_test_dirs,
        }


@dataclass
class Discovery:
    root: Path
    tifs: list[Path] = field(default_factory=list)
    roi_files: list[Path] = field(default_factory=list)
    csvs: list[Path] = field(default_factory=list)
    archives: list[Path] = field(default_factory=list)      # *.tif.zip etc.: NOT extracted
    ignored: list[Path] = field(default_factory=list)


@dataclass
class ImportReport:
    root: str
    files: dict[str, int] = field(default_factory=dict)
    images_loaded: int = 0
    images_failed: list[str] = field(default_factory=list)
    slides: list[str] = field(default_factory=list)
    roi_images_converted: int = 0
    labels_by_plane: dict[str, int] = field(default_factory=dict)
    labels_not_ok: dict[str, int] = field(default_factory=dict)
    fields_found: int = 0
    fields_with_points: int = 0
    regions_found: int = 0
    excluded_regions: int = 0
    pairing_issues: list[str] = field(default_factory=list)
    signal_evidence: list[str] = field(default_factory=list)
    warnings: list[WarningRecord] = field(default_factory=list)
    skipped_archives: list[str] = field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return sum(1 for w in self.warnings if w.severity == Severity.error)

    @property
    def n_warnings(self) -> int:
        return sum(1 for w in self.warnings if w.severity == Severity.warning)

    def render_text(self, max_warnings: int = 60) -> str:
        L = [f"Import report for {self.root}", "=" * 60]
        L.append("Files discovered: " + ", ".join(f"{k}={v}" for k, v in sorted(self.files.items())))
        L.append(f"Images loaded: {self.images_loaded}   failed: {len(self.images_failed)}")
        for f in self.images_failed:
            L.append(f"   FAILED  {f}")
        L.append(f"Slides reconstructed: {len(self.slides)}  ({', '.join(self.slides) or '-'})")
        L.append(f"Images with ROI annotations converted: {self.roi_images_converted}")
        L.append("Labels per plane (ImageJ position): " + (
            ", ".join(f"{k}={v}" for k, v in sorted(self.labels_by_plane.items())) or "none"))
        if self.labels_not_ok:
            L.append("Labels flagged (kept, not usable): " + ", ".join(f"{k}={v}" for k, v in sorted(self.labels_not_ok.items())))
        L.append(f"Grid fields: {self.fields_found} found, {self.fields_with_points} with points")
        L.append(f"Other regions: {self.regions_found}   excluded regions (by explicit name): {self.excluded_regions}")
        if self.skipped_archives:
            L.append(f"Archives NOT extracted (extract first): {', '.join(self.skipped_archives)}")
        if self.pairing_issues:
            L.append("Missing / ambiguous pairings:")
            L += [f"   - {p}" for p in self.pairing_issues]
        if self.signal_evidence:
            L.append("Signal evidence (median pixel value at the points, annotated plane vs other planes):")
            L += [f"   - {p}" for p in self.signal_evidence]
        L.append(f"Warnings: {self.n_warnings}   Errors: {self.n_errors}")
        shown = sorted(self.warnings, key=lambda w: (w.severity != Severity.error, w.severity != Severity.warning))
        for w in shown[:max_warnings]:
            where = f" [{w.file}]" if w.file else ""
            L.append(f"   {w.severity.value.upper():7s} {w.code}{where}: {w.message}")
        if len(shown) > max_warnings:
            L.append(f"   ... {len(shown) - max_warnings} more (see warnings.jsonl after commit)")
        return "\n".join(L)


# ---------------------------------------------------------------- discovery
def discover(root: Path) -> Discovery:
    root = Path(root)
    if root.is_file():
        d = Discovery(root=root.parent)
        paths = [root]
    else:
        d = Discovery(root=root)
        paths = sorted(p for p in root.rglob("*") if p.is_file())
    for p in paths:
        rel_parts = p.relative_to(d.root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        name = p.name.lower()
        if name.endswith((".tif.zip", ".tiff.zip")):
            d.archives.append(p)
        elif p.suffix.lower() in _TIF:
            d.tifs.append(p)
        elif p.suffix.lower() == ".roi" or p.suffix.lower() == ".zip":
            d.roi_files.append(p)
        elif p.suffix.lower() == ".csv":
            d.csvs.append(p)
        else:
            d.ignored.append(p)
    return d


def _rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Warn:
    def __init__(self) -> None:
        self.items: list[WarningRecord] = []

    def add(self, code: str, sev: str, msg: str, file: Optional[str] = None, image_id: Optional[str] = None) -> None:
        self.items.append(WarningRecord(code=code, severity=Severity(sev), message=msg, file=file, image_id=image_id))


# ---------------------------------------------------------------- scan metadata
_SCAN_RE = {"DateTime": re.compile(r"DateTime:\s*(.+)"), "PMT": re.compile(r"PMT=([\d.]+)"),
            "LaserPower": re.compile(r"LaserPower=([\d.]+)")}


def scan_settings(info: Optional[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, rx in _SCAN_RE.items():
        m = rx.search(info or "")
        if m:
            out[k] = m.group(1).strip()
    return out


def scan_session(settings: dict[str, str]) -> Optional[str]:
    """Inferred session key: scanner date + PMT + laser power. Not a lab-confirmed concept."""
    dt = settings.get("DateTime", "")
    m = re.match(r"(\d{4})/(\d{2})/(\d{2})", dt)
    if not m:
        return None
    day = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return f"{day}|PMT={settings.get('PMT', '?')}|LaserPower={settings.get('LaserPower', '?')}"


# ---------------------------------------------------------------- one image
def _describe_image(path: Path, rel: str, sha: str, warn: _Warn) -> Optional[tuple[ImageRecord, np.ndarray]]:
    """Load pixels exactly as stored and describe them. None (+ an error warning) on failure."""
    try:
        arr = read_stack(path)
        with tifffile.TiffFile(str(path)) as tif:
            axes = tif.series[0].axes
            md = tif.imagej_metadata if tif.is_imagej else None
            page0 = tif.pages[0]
            labels = md.get("Labels") if md else None
            info = md.get("Info") if md and isinstance(md.get("Info"), str) else None
            res_note = None
            try:
                xr = page0.tags.get("XResolution")
                unit = str((md or {}).get("unit", "") or "")
                if xr is not None and unit:
                    num, den = xr.value
                    if num and den:
                        res_note = f"TIFF resolution tag = {num / den:g} pixels per '{unit}' (informational; not applied)"
            except Exception:
                pass
    except (IngestError, Exception) as exc:  # noqa: BLE001 - tifffile raises many types
        hint = ""
        if "imagecodecs" in str(exc):
            hint = " (install the optional codec package: pip install imagecodecs)"
        warn.add("image_unreadable", "error", f"cannot load as TIFF: {exc}{hint}", file=rel)
        return None
    if str(arr.dtype) not in SUPPORTED_DTYPES:
        warn.add("unsupported_dtype", "error",
                 f"pixel type {arr.dtype} is not supported (only uint8/uint16); it is NOT converted", file=rel)
        return None
    if arr.ndim == 2:
        kind, planes_arr, h, w = ImageKind.single_plane, [arr], arr.shape[0], arr.shape[1]
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] > 4:
        kind, h, w = ImageKind.rgb_composite, arr.shape[0], arr.shape[1]
        planes_arr = [arr[..., k] for k in range(3)]
    elif arr.ndim == 3:
        kind, planes_arr, h, w = ImageKind.stack, [arr[k] for k in range(arr.shape[0])], arr.shape[1], arr.shape[2]
    else:
        warn.add("unsupported_layout", "error", f"unsupported image layout {tuple(arr.shape)}", file=rel)
        return None
    hint0 = parse_info_label_hint(info)
    planes: list[PlaneInfo] = []
    for k, pl in enumerate(planes_arr):
        mn, mx, sf = channel_stats(pl)
        if kind == ImageKind.rgb_composite:
            hint = f"composite colour plane {'RGB'[k]}"
        elif labels and len(labels) == len(planes_arr) and labels[k]:
            hint = str(labels[k])
        else:
            hint = hint0 if k == 0 else None
        planes.append(PlaneInfo(page=k, label_hint=hint, min=mn, max=mx, saturated_fraction=sf))
        if sf > 0.001:
            warn.add("saturation", "info", f"plane {k}: {sf * 100:.2f}% of pixels at the dtype maximum "
                     f"({np.iinfo(arr.dtype).max}); clipped", file=rel)
    if kind == ImageKind.stack and hint0 and not (labels and len(labels) == len(planes_arr)):
        warn.add("single_channel_label", "info",
                 f"the file carries one metadata label ('{hint0}') for the whole stack; it cannot tell which "
                 "marker/colour each plane is", file=rel)
    if kind == ImageKind.rgb_composite:
        warn.add("rgb_composite", "info", "rendered RGB composite: raw 16-bit intensities are NOT available from it", file=rel)
    elif arr.dtype == np.uint8:
        warn.add("eight_bit_data", "warning", "8-bit data: any scaling done by the exporting software is not recoverable", file=rel)
    settings = scan_settings(info)
    rec = ImageRecord(
        image_id="img_" + sha[:12], file_name=path.name, rel_path=rel, sha256=sha,
        size_bytes=path.stat().st_size, kind=kind, width=int(w), height=int(h), n_planes=len(planes_arr),
        dtype=str(arr.dtype), bit_depth=int(arr.dtype.itemsize * 8), axes=axes, planes=planes,
        info_text=info, scan_settings=settings, resolution_note=res_note,
        scan_session=scan_session(settings))
    return rec, arr


def _apply_identity(rec: ImageRecord, stem: str) -> None:
    ident = parse_identity(stem)
    rec.assay, rec.specimen_id, rec.replicate = ident.assay, ident.specimen_id, ident.replicate
    rec.slide_id, rec.file_date, rec.filename_suffix = ident.slide_id, ident.file_date, ident.suffix
    rec.identity_source = ident.source
    rec.identity_confirmed = False


def _check_file_date(rec: ImageRecord, warn: _Warn) -> None:
    """File-name date vs scanner date: informational (they differ legitimately)."""
    if rec.file_date and rec.scan_settings.get("DateTime"):
        d, m, y = rec.file_date.split("-")
        if rec.scan_settings["DateTime"].replace("/", "-")[:10] != f"{y}-{m}-{d}":
            warn.add("file_date_vs_scan_date", "info",
                     f"file-name date {rec.file_date} differs from the scanner date {rec.scan_settings['DateTime']}; "
                     "the file-name date is NOT used as scan date", file=rec.rel_path, image_id=rec.image_id)


def _reserve(rec: ImageRecord, rel: str, cfg: ImportConfig, extra_dirs: tuple[str, ...] = ()) -> None:
    for pat, split in cfg.reserve.items():
        if fnmatch.fnmatch(rel, pat) or rec.sha256.startswith(pat):
            rec.reserved_split, rec.reserved_reason = split, f"import config: '{pat}'"
            return
    # directory names are taken from the path below the import root (plus the file's own parent
    # directory when a single file was given) -- never from unrelated directories above the root.
    if cfg.reserve_legacy_test_dirs and "test" in (*Path(rel).parts[:-1], *extra_dirs):
        rec.reserved_split, rec.reserved_reason = "test", "lives in a 'test' directory (historical test image)"


# ---------------------------------------------------------------- annotations
def _cell_polygons(grid: roimod.RawGrid) -> list[tuple[int, int, tuple[float, float, float, float]]]:
    out = []
    for r in range(len(grid.ys) - 1):
        for c in range(len(grid.xs) - 1):
            out.append((r, c, (grid.xs[c], grid.ys[r], grid.xs[c + 1], grid.ys[r + 1])))
    return out


def _convert_annotations(rec: ImageRecord, bundle: roimod.RoiBundle, src_files: dict[str, str],
                         cfg: ImportConfig, warn: _Warn) -> tuple[list[PointLabel], list[FieldRecord], list[RegionRecord]]:
    iid, file = rec.image_id, rec.rel_path
    for iss in bundle.issues:
        warn.add(iss.code, iss.severity, iss.message, file=file, image_id=iid)
    grid = bundle.grids[0] if bundle.grids else None
    if len(bundle.grids) > 1:
        warn.add("multiple_grids", "warning", f"{len(bundle.grids)} grid overlays found; only the first is used", file=file, image_id=iid)
    cells = _cell_polygons(grid) if grid else []
    xs = np.array(grid.xs) if grid else None
    ys = np.array(grid.ys) if grid else None

    # --- points
    pts: list[PointLabel] = []
    for rp in bundle.points:
        cx, cy = roimod.to_canonical(rp.x, rp.subpixel), roimod.to_canonical(rp.y, rp.subpixel)
        ijx, ijy = (rp.x, rp.y) if rp.subpixel else (rp.x + 0.5, rp.y + 0.5)    # centre in the ImageJ frame
        finite = bool(np.isfinite([rp.x, rp.y]).all())
        in_bounds = finite and 0.0 <= ijx < rec.width and 0.0 <= ijy < rec.height
        page: Optional[int] = None
        if rp.position is not None:
            if 1 <= rp.position <= rec.n_planes:
                page = rp.position - 1
            else:
                warn.add("position_out_of_range", "warning",
                         f"point {rp.roi_index}:{rp.point_index} has plane position {rp.position} but the image has "
                         f"{rec.n_planes} plane(s); plane left unspecified", file=file, image_id=iid)
        field_id = None
        if in_bounds and grid is not None:
            ci = int(np.searchsorted(xs, ijx, side="right")) - 1
            ri = int(np.searchsorted(ys, ijy, side="right")) - 1
            if 0 <= ci < len(xs) - 1 and 0 <= ri < len(ys) - 1:
                field_id = f"{iid}:r{ri}c{ci}"
        label = cfg.class_map.get(rp.position) if rp.position is not None else None
        pts.append(PointLabel(
            label_id=f"{iid}:{rp.roi_index}:{rp.point_index}", image_id=iid,
            source_file=src_files.get(rp.roi_source, file), roi_source=rp.roi_source,
            roi_index=rp.roi_index, point_index=rp.point_index,
            x=cx, y=cy, x_imagej=rp.x, y_imagej=rp.y, subpixel=rp.subpixel,
            counter=rp.counter, position=rp.position, page=page, field_id=field_id,
            class_label=label, class_confirmed=False,
            status=PointStatus.ok if in_bounds else PointStatus.out_of_bounds))
    n_oob = sum(1 for p in pts if p.status == PointStatus.out_of_bounds)
    if n_oob:
        warn.add("points_out_of_bounds", "warning",
                 f"{n_oob} point(s) lie outside the {rec.width}x{rec.height} image; flagged out_of_bounds, not clamped",
                 file=file, image_id=iid)
    # duplicates / near duplicates per plane
    by_page: dict[Optional[int], list[PointLabel]] = defaultdict(list)
    for p in pts:
        if p.status == PointStatus.ok:
            by_page[p.page].append(p)
    n_dup = n_close = 0
    for page, group in by_page.items():
        if len(group) < 2:
            continue
        xy = np.array([[p.x, p.y] for p in group])
        tree = cKDTree(xy)
        for i, j in sorted(tree.query_pairs(cfg.duplicate_tol_px)):
            if group[i].status == PointStatus.ok and group[j].status == PointStatus.ok:
                group[j].status = PointStatus.duplicate
                n_dup += 1
        ok = [p for p in group if p.status == PointStatus.ok]
        if len(ok) > 1:
            t2 = cKDTree(np.array([[p.x, p.y] for p in ok]))
            n_close += len(t2.query_pairs(cfg.close_pair_px))
    if n_dup:
        warn.add("duplicate_points", "warning",
                 f"{n_dup} point(s) within {cfg.duplicate_tol_px:g}px of an earlier point on the same plane; flagged duplicate",
                 file=file, image_id=iid)
    if n_close:
        warn.add("overlapping_points", "info",
                 f"{n_close} pair(s) of distinct points closer than {cfg.close_pair_px:g}px (possible touching polonies); kept",
                 file=file, image_id=iid)
    ok_pts = [p for p in pts if p.status == PointStatus.ok]
    n_out_grid = sum(1 for p in ok_pts if p.field_id is None)
    if pts and grid is None:
        warn.add("no_grid", "warning", "points found but no counting-grid overlay: fields cannot be derived", file=file, image_id=iid)
    elif n_out_grid:
        warn.add("points_outside_grid", "warning", f"{n_out_grid} point(s) lie outside every grid cell", file=file, image_id=iid)

    # --- fields
    fields: list[FieldRecord] = []
    if grid is not None:
        counter_cells: dict[int, Counter] = defaultdict(Counter)
        for p in ok_pts:
            if p.counter is not None and p.field_id is not None:
                counter_cells[p.counter][p.field_id] += 1
        majority = {c: cnt.most_common(1)[0][0] for c, cnt in counter_cells.items()}
        for c, cnt in sorted(counter_cells.items()):
            if len(cnt) > 1:
                off = sum(cnt.values()) - cnt.most_common(1)[0][1]
                warn.add("counter_spans_cells", "info",
                         f"counter {c} has {off} point(s) outside its majority cell (clicked across a cell border?)",
                         file=file, image_id=iid)
        # index once (cells x points would be quadratic for dense grids / many points)
        page_counts: dict[str, Counter] = defaultdict(Counter)
        for p in ok_pts:
            if p.field_id is not None:
                page_counts[p.field_id]["unspecified" if p.page is None else str(p.page)] += 1
        counters_of: dict[str, list[int]] = defaultdict(list)
        for k, v in majority.items():
            counters_of[v].append(k)
        for r, c, (x0, y0, x1, y1) in cells:
            fid = f"{iid}:r{r}c{c}"
            counts: Counter = page_counts.get(fid) or Counter()
            fields.append(FieldRecord(
                field_id=fid, image_id=iid, grid_row=r, grid_col=c,
                polygon=[(a - 0.5, b - 0.5) for a, b in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]],
                counters=sorted(counters_of.get(fid, [])),
                n_points_by_page=dict(sorted(counts.items())),
                status=FieldStatus.has_points if counts else FieldStatus.no_points))
        if not grid.regular:
            warn.add("irregular_grid", "warning", "grid lines are not evenly spaced", file=file, image_id=iid)
        if sum(1 for f in fields if f.status == FieldStatus.no_points):
            warn.add("empty_fields_ambiguous", "info",
                     "cells without points cannot be told apart (not counted vs counted with zero) from the ROI alone",
                     file=file, image_id=iid)
    # --- regions
    regions: list[RegionRecord] = []
    for k, sh in enumerate(bundle.shapes):
        role = role_src = None
        if sh.name and sh.name in cfg.excluded_region_names:
            role, role_src = "excluded", "import config (name)"
        elif sh.name and sh.name in cfg.valid_region_names:
            role, role_src = "valid", "import config (name)"
        regions.append(RegionRecord(
            region_id=f"{iid}:reg{sh.roi_index}:{k}", image_id=iid, roi_type=sh.roi_type, name=sh.name,
            polygon=roimod.shape_to_canonical(sh.polygon), role=role, role_source=role_src,
            position=sh.position, source_file=src_files.get(sh.roi_source, file), roi_source=sh.roi_source,
            roi_index=sh.roi_index))
    if regions and not any(r.role for r in regions):
        warn.add("unclassified_regions", "warning",
                 f"{len(regions)} area ROI(s) found whose role (valid vs excluded) is unknown; "
                 "pass --excluded-name / --valid-name to classify them", file=file, image_id=iid)
    return pts, fields, regions


def _signal_evidence(rec: ImageRecord, arr: np.ndarray, pts: list[PointLabel], warn: _Warn) -> list[str]:
    out: list[str] = []
    if rec.kind != ImageKind.stack:
        return out
    for pos in sorted({p.position for p in pts if p.position and p.status == PointStatus.ok}):
        sel = [p for p in pts if p.position == pos and p.status == PointStatus.ok and p.page is not None]
        if len(sel) < 5:
            continue
        page = pos - 1
        xi = np.clip(np.floor([p.x_imagej if p.subpixel else p.x_imagej + 0.5 for p in sel]).astype(int), 0, rec.width - 1)
        yi = np.clip(np.floor([p.y_imagej if p.subpixel else p.y_imagej + 0.5 for p in sel]).astype(int), 0, rec.height - 1)
        on = float(np.median(arr[page][yi, xi]))
        others = [float(np.median(arr[k][yi, xi])) for k in range(arr.shape[0]) if k != page]
        off = max(others) if others else float("nan")
        ratio = on / off if off and off > 0 else float("inf")
        out.append(f"{rec.file_name}: position {pos} ({len(sel)} pts): plane {page} median={on:g}, other plane(s) max-median={off:g} (x{ratio:.1f})")
        if ratio < 1.5:
            warn.add("weak_plane_signal", "warning",
                     f"points on position {pos} ({len(sel)}) are not brighter on plane {page} than on the other plane "
                     f"(x{ratio:.2f}); the plane a point was placed on may not be the plane that shows it",
                     file=rec.rel_path, image_id=rec.image_id)
    return out


# ---------------------------------------------------------------- CSV reconciliation
def _reconcile(ds: Dataset, csv_paths: list[Path], root: Path, warn: _Warn) -> None:
    by_slide: dict[str, list[ImageRecord]] = defaultdict(list)
    for im in ds.images:
        if im.slide_id and im.kind != ImageKind.rgb_composite:
            by_slide[im.slide_id].append(im)
    seen_slides: set[str] = set()
    unmatched_rows: list[str] = []
    ok_points_by_image: dict[str, list[PointLabel]] = defaultdict(list)     # one pass instead of images x points
    for p in ds.points:
        if p.status == PointStatus.ok:
            ok_points_by_image[p.image_id].append(p)
    for cp in csv_paths:
        data = cp.read_bytes()
        imp = parse_csv(data, cp.name, import_id=cp.stem)
        series = cp.stem.split("_")[0]
        for row in imp.rows:
            imgs = by_slide.get(row.slide_id)
            if not imgs:
                unmatched_rows.append(f"{series}:{row.slide_id}")
                continue
            if len(imgs) > 1:
                warn.add("ambiguous_csv_match", "warning",
                         f"CSV row {series}:{row.slide_id} matches {len(imgs)} images ({', '.join(i.file_name for i in imgs)}); "
                         "reconciled against the first only", file=_rel(root, cp))
            seen_slides.add(row.slide_id)
            im = imgs[0]
            seq = list(row.fields)
            while seq and seq[-1] is None:
                seq.pop()
            has_gap = any(v is None for v in seq)
            vals = [v for v in seq if v is not None]
            planes: list[PlaneMatch] = []
            for pos in range(1, im.n_planes + 1):
                pts = [p for p in ok_points_by_image.get(im.image_id, []) if p.position == pos]
                cnt = Counter(p.counter for p in pts if p.counter is not None)
                by_idx = [cnt.get(c, 0) for c in range(max(cnt) + 1)] if cnt else []
                if len(by_idx) < len(vals):          # fields counted beyond the last marked one are zeros
                    by_idx = by_idx + [0] * (len(vals) - len(by_idx))
                by_order = [cnt[c] for c in sorted(cnt)]
                if not has_gap and vals == by_idx and (vals or not pts):
                    m = "exact_by_counter_index"
                elif not has_gap and vals == by_order and (vals or not pts):
                    m = "exact_by_present_order"
                elif sum(vals) == len(pts):
                    m = "total_only"
                else:
                    m = "mismatch"
                first = None
                if m not in ("exact_by_counter_index", "exact_by_present_order"):
                    for i in range(max(len(vals), len(by_idx))):
                        a = vals[i] if i < len(vals) else None
                        b = by_idx[i] if i < len(by_idx) else None
                        if a != b:
                            first = i + 1
                            break
                planes.append(PlaneMatch(position=pos, roi_total=len(pts), match=m, first_mismatch_field=first))
            ds.reconciliations.append(Reconciliation(
                image_id=im.image_id, slide_id=row.slide_id, csv_file=_rel(root, cp), csv_sha256=imp.sha256,
                series=series, csv_total=sum(vals), csv_counted_fields=len(vals),
                csv_blank_fields=sum(1 for v in row.fields if v is None), planes=planes))
    if unmatched_rows:
        warn.add("csv_rows_without_image", "info",
                 f"{len(unmatched_rows)} CSV row(s) have no image in this import: {', '.join(sorted(set(unmatched_rows))[:12])}"
                 + (" ..." if len(set(unmatched_rows)) > 12 else ""))
    for sid in sorted(by_slide):
        if csv_paths and sid not in seen_slides:
            warn.add("image_without_csv_row", "info", f"slide {sid} has no row in the supplied CSV file(s)",
                     file=by_slide[sid][0].rel_path, image_id=by_slide[sid][0].image_id)


# ---------------------------------------------------------------- composites
def _link_composites(ds: Dataset, arrays: dict[str, np.ndarray], cfg: ImportConfig, warn: _Warn, report: ImportReport) -> None:
    stacks = [i for i in ds.images if i.kind != ImageKind.rgb_composite]
    for comp in [i for i in ds.images if i.kind == ImageKind.rgb_composite]:
        ident = parse_identity(Path(comp.rel_path).stem)
        cands = [s for s in stacks if Path(s.rel_path).parent == Path(comp.rel_path).parent
                 and Path(s.rel_path).stem == ident.base_stem]
        if not cands:
            msg = f"RGB composite {comp.file_name} has no raw stack with the same base name"
            report.pairing_issues.append(msg)
            warn.add("unpaired_composite", "warning", msg, file=comp.rel_path, image_id=comp.image_id)
            continue
        if len(cands) > 1:
            msg = f"RGB composite {comp.file_name} matches {len(cands)} raw stacks; not paired"
            report.pairing_issues.append(msg)
            warn.add("ambiguous_composite_pairing", "warning", msg, file=comp.rel_path, image_id=comp.image_id)
            continue
        st = cands[0]
        if (st.height, st.width) != (comp.height, comp.width):
            msg = f"{comp.file_name} is {comp.height}x{comp.width} but {st.file_name} is {st.height}x{st.width}: not registered, not paired"
            report.pairing_issues.append(msg)
            warn.add("composite_size_mismatch", "warning", msg, file=comp.rel_path, image_id=comp.image_id)
            continue
        comp.composite_of = st.image_id
        comp.slide_id, comp.specimen_id, comp.assay, comp.replicate = st.slide_id, st.specimen_id, st.assay, st.replicate
        comp.identity_source = st.identity_source
        if cfg.check_composites and comp.image_id in arrays and st.image_id in arrays:
            rgb, stack = arrays[comp.image_id], arrays[st.image_id]
            rng = np.random.default_rng(0)
            n = comp.height * comp.width
            idx = rng.choice(n, size=min(100_000, n), replace=False)
            for k in range(3):
                for pg in range(min(stack.shape[0], 2)):
                    a, b = rgb[..., k].ravel()[idx], stack[pg].ravel()[idx]
                    if a.min() == a.max() or b.min() == b.max():     # constant plane: correlation undefined
                        continue
                    comp.composite_evidence[f"rgb{'RGB'[k]}_vs_page{pg}"] = round(float(spearmanr(a, b)[0]), 4)
            warn.add("composite_paired", "info",
                     f"{comp.file_name} paired with {st.file_name} by name + size; rank correlations recorded in "
                     "composite_evidence (evidence for plane<->colour association, not a confirmed role)",
                     file=comp.rel_path, image_id=comp.image_id)
    for st in stacks:
        if st.kind == ImageKind.stack and not any(c.composite_of == st.image_id for c in ds.images):
            report.pairing_issues.append(f"{st.file_name}: no RGB composite found (optional)")


# ---------------------------------------------------------------- main entry
def build_dataset(root: Path, cfg: Optional[ImportConfig] = None) -> tuple[Dataset, ImportReport]:
    """Read ``root`` (a directory or a single TIFF) and build the dataset in memory."""
    cfg = cfg or ImportConfig()
    single_file = Path(root).is_file()
    disc = discover(Path(root))
    root = disc.root
    warn = _Warn()
    report = ImportReport(root=str(root))
    report.files = {"tiff": len(disc.tifs), "roi_sidecar": len(disc.roi_files), "csv": len(disc.csvs),
                    "archives_not_extracted": len(disc.archives), "ignored": len(disc.ignored)}
    for a in disc.archives:
        report.skipped_archives.append(_rel(root, a))
        warn.add("archive_not_extracted", "warning",
                 "archive was not opened (extract it into its own directory first; archives are never auto-extracted)",
                 file=_rel(root, a))

    sidecars: dict[Path, list[Path]] = defaultdict(list)
    orphan = []
    tif_stems = {(t.parent, t.stem): t for t in disc.tifs}
    for rf in disc.roi_files:
        t = tif_stems.get((rf.parent, rf.stem))
        (sidecars[t].append(rf) if t is not None else orphan.append(rf))
    for rf in orphan:
        msg = f"ROI file {rf.name} has no TIFF with the same name next to it; not used"
        report.pairing_issues.append(msg)
        warn.add("orphan_roi_file", "warning", msg, file=_rel(root, rf))

    ds = Dataset()
    arrays: dict[str, np.ndarray] = {}          # kept only for composite pairing checks (memory)
    composite_bases = {(t.parent, parse_identity(t.stem).base_stem) for t in disc.tifs
                       if parse_identity(t.stem).composite_wavelengths}
    seen_sha: dict[str, ImageRecord] = {}
    for tif in disc.tifs:
        rel = _rel(root, tif)
        sha = _sha256(tif)
        if sha in seen_sha:
            seen_sha[sha].alias_paths.append(rel)
            warn.add("duplicate_file", "warning", f"byte-identical to {seen_sha[sha].rel_path}; second copy ignored",
                     file=rel, image_id=seen_sha[sha].image_id)
            continue
        got = _describe_image(tif, rel, sha, warn)
        if got is None:
            report.images_failed.append(rel)
            continue
        rec, arr = got
        seen_sha[sha] = rec
        _apply_identity(rec, tif.stem)
        _check_file_date(rec, warn)
        _reserve(rec, rel, cfg, (tif.parent.name,) if single_file else ())
        if cfg.check_composites and (rec.kind == ImageKind.rgb_composite or (tif.parent, tif.stem) in composite_bases):
            arrays[rec.image_id] = arr
        # annotations: embedded first, then sidecars
        bundles: list[tuple[roimod.RoiBundle, dict[str, str]]] = []
        if rec.kind != ImageKind.rgb_composite:
            bundles.append((roimod.read_embedded(tif), {}))
        for sc in sidecars.get(tif, []):
            bundles.append((roimod.read_sidecar(sc), {"sidecar": _rel(root, sc)}))
        merged = roimod.RoiBundle()
        src_files: dict[str, str] = {}
        for k, (b, srcs) in enumerate(bundles):
            src_files.update(srcs)
            offset = 1000 * k
            for pt in b.points:
                pt.roi_index += offset
            for sh in b.shapes:
                sh.roi_index += offset
            for g in b.grids:
                g.roi_index += offset
            merged.points += b.points
            merged.shapes += b.shapes
            merged.grids += b.grids
            merged.issues += b.issues
            merged.n_blobs += b.n_blobs
        rec.n_roi_blobs = merged.n_blobs
        pts, fields, regions = _convert_annotations(rec, merged, src_files, cfg, warn)
        rec.grid_found = bool(merged.grids)
        rec.has_annotations = bool(pts or regions or merged.grids)
        if rec.kind != ImageKind.rgb_composite and not rec.has_annotations:
            warn.add("no_annotations", "warning",
                     "no ROI annotations found: usable for inference / evaluation inputs, not as labelled data",
                     file=rel, image_id=rec.image_id)
        if cfg.check_signal and rec.kind == ImageKind.stack:
            report.signal_evidence += _signal_evidence(rec, arr, pts, warn)
        ds.images.append(rec)
        ds.points += pts
        ds.fields += fields
        ds.regions += regions

    _link_composites(ds, arrays, cfg, warn, report)
    if disc.csvs:
        _reconcile(ds, disc.csvs, root, warn)
    ds.warnings = warn.items

    # --- report
    report.images_loaded = len(ds.images)
    report.slides = sorted({i.slide_id for i in ds.images if i.slide_id})
    by_slide: dict[str, list[str]] = defaultdict(list)
    for i in ds.images:
        if i.kind != ImageKind.rgb_composite and i.slide_id:
            by_slide[i.slide_id].append(i.file_name)
    for sid, names in sorted(by_slide.items()):
        if len(names) > 1:
            msg = f"slide {sid} is claimed by {len(names)} raw images ({', '.join(names)}); which one is authoritative is ambiguous"
            report.pairing_issues.append(msg)
            ds.warnings.append(WarningRecord(code="ambiguous_slide_images", severity=Severity.warning, message=msg))
    unknown = [i.file_name for i in ds.images if i.kind != ImageKind.rgb_composite and not i.slide_id]
    if unknown:
        report.pairing_issues.append(f"slide identity could not be reconstructed from the file name: {', '.join(unknown)}")
    report.roi_images_converted = len({p.image_id for p in ds.points} | {r.image_id for r in ds.regions})
    for p in ds.points:
        key = f"position {p.position}" if p.position else "unspecified"
        if p.status == PointStatus.ok:
            report.labels_by_plane[key] = report.labels_by_plane.get(key, 0) + 1
        else:
            report.labels_not_ok[p.status.value] = report.labels_not_ok.get(p.status.value, 0) + 1
    report.fields_found = len(ds.fields)
    report.fields_with_points = sum(1 for f in ds.fields if f.status == FieldStatus.has_points)
    report.regions_found = len(ds.regions)
    report.excluded_regions = sum(1 for r in ds.regions if r.role == "excluded")
    report.warnings = ds.warnings

    ds.manifest = DatasetManifest(
        created_at=cfg.created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_root=str(root), config=cfg.to_json(),
        counts={"images": len(ds.images), "points": len(ds.points),
                "points_ok": sum(1 for p in ds.points if p.status == PointStatus.ok),
                "fields": len(ds.fields), "regions": len(ds.regions), "reconciliations": len(ds.reconciliations),
                "warnings": len(ds.warnings), "errors": report.n_errors},
        notes=["Identity fields are inferred from file names and are NOT confirmed (identity_confirmed=false).",
               "class_label is only set from an explicit class map; plane roles (red/green) are never inferred.",
               "Cells without points are ambiguous: not counted vs counted with zero."])
    ds.manifest.dataset_id = compute_dataset_id(ds)
    return ds, report
