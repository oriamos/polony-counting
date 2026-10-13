"""Bridge: historical, ImageJ-annotated TIFFs -> the CURRENT Spotiflow fine-tuning lifecycle (``ml import-legacy``).

The output is an ordinary immutable dataset snapshot (``datasets/<version>/``) that ``ml verify``, ``ml split --lock-test``,
``ml train``, ``ml eval-model``, ``ml compare`` and ``ml promote`` understand. The bridge adds NO new training path.

THE SCIENTIFIC RULE (overrides any "use every grid cell" behaviour; there is no override flag)
    Train ONLY on (source TIFF, field, page) tuples the researcher actually EXAMINED, decided independently per tuple by
    :mod:`polony_app.fields.examined`:

    * embedded ImageJ marks on THAT page inside THAT field      -> ``positive_marked`` (that page only);
    * an explicit sidecar / manifest entry that states the researcher counted THAT field on THAT page, including
      ``count: 0``                                             -> ``verified_zero`` (crop with an EMPTY label file);
    * grid overlay alone, marks on the other page, application detections, "no marks" -> ``unknown``: excluded, never a
      negative crop, never zero; a stated count that contradicts the marks -> ``conflict``: excluded.

    A point is exported as a label only if its geometric grid cell is itself examined for that page. A cell whose only
    points belong to ANOTHER cell's ImageJ counter (clicked across a border) is not evidence for this cell. Points that are
    invalid (out of bounds, duplicate), outside the grid, on an unmapped page, with an unspecified plane or in a
    non-examined cell are never labels: they are listed in ``points.csv`` with status ``dropped`` and the reason, and counted
    in the manifest. Nothing is ever clamped. Before a snapshot is assembled a final assertion refuses (``MLError``) any crop
    whose provenance does not say ``researcher_examined == true``; :func:`verify_examination_provenance` re-checks this on
    any manifest.

Phases
    A  scan (read-only, no pixel reads): deterministic discovery, streaming sha256, byte-identical de-duplication, TIFF header,
       ImageJ ROI -> compact per-file arrays (``legacy_import.roi`` / ``legacy_import.importer._convert_annotations``).
    B  mapping: channel roles per page from the operator's confirmed channel map (never inferred), identity from the operator's
       identity map (file-name inference is only an UNCONFIRMED fallback), optional examined-fields manifest / lab CSV.
    C  eligibility per (file, field, page) -> lazy crops (one page plane read per source and page, cut with ``loader``).
    D  :func:`polony_app.ml.dataset.assemble_snapshot` (group split, resumable atomic write, version hash).

``import_summary`` (manifest key; every number is deterministic for identical inputs and options)
    kind, mode, input_root_name, options{...}, input_files{name, sha256 of channel map / identity map / manifest / lab CSVs},
    tiffs{discovered, unique, accepted, rejected, deduplicated, rejected_by_reason, rejected_files, deduplicated_paths},
    sources{with_crops, without_crops, without_crops_reasons}, channel_mappings{origin_counts, by_page_role, per_file},
    fields{eligible_examined_fields, positive_marked_fields, verified_zero_fields, unknown_fields_excluded,
           conflict_fields_excluded, reasons (ExaminationTally.as_dict), excluded_unproven, ambiguous_zero_vs_unreviewed,
           tuples_total, by_page}, crops{created, skipped, skipped_reasons}, positives_per_page[{page, role, positives, crops}],
    annotations{points_total, labels_exported, dropped_by_reason, straddling_labels_in_examined_cells},
    splits{seed, level, fractions, groups, by_split{train,val,test: crops, positives, groups, sources}, locked_sources},
    estimates{native_disk_bytes, training{steps_per_epoch, optimizer_steps, prepared_bytes, ...}}, identity{...},
    unresolved_manifest_entries, assumptions.
"""
from __future__ import annotations

import csv
import difflib
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import tifffile

from polony_app import schemas
from polony_app.fields import examined as ex
from polony_app.imaging.io import read_plane
from polony_app.legacy_import import importer as imp
from polony_app.legacy_import import roi as roimod
from polony_app.legacy_import.identity import parse_identity
from polony_app.legacy_import.schema import ImageKind, ImageRecord, PointStatus

from . import dataset as ds
from .common import (MLError, PAGE_KEY_RE, canonical_json, check_downscale, check_normaliser, hash_obj, loads_strict, page_bounds,
                     pixel_box, printable_name, read_json_strict, sha256_bytes, utf8_name_problem, DEFAULT_PERCENTILES,
                     LO_HI_POLICY)
from .estimates import estimate_training
from .manifest import SnapshotManifest
from .splits import SourceInfo, assign_groups, load_lock, resolve_identity

Say = Callable[[str], None]

# ---------------------------------------------------------------------------------------------- constants
CHANNEL_SCHEMA = "polony-channel-map"
CHANNEL_VERSION = 1
KIND = "legacy_imagej"
VALID_MASKS = ("chamber", "grid-cells", "eligible-fields")
DEFAULT_VALID_MASK = "chamber"
CHAMBER_RIM_PX = 40                      # the app's default rim for the automatic chamber (api AutoChamberBody.rim_px)
# bump when the labelling / eligibility / normalisation logic changes: part of the resume key, so a staging directory left by
# other logic is never resumed (the source text of this module, dataset.py and common.py is hashed as well)
LOGIC_VERSION = 2
STRADDLE_REASON = "only straddling points from another counter"
PLANE_CONFLICT_REASON = ("a mark without a usable plane (ImageJ position 0 / missing / beyond the file's planes) lies in this "
                         "cell: its polony may belong to any page, so no page of the cell can be proven examined or zero")
COORDINATE_FRAME = ("ImageJ ROI points (pixel-edge frame; sub-pixel ROIs: canonical = imagej - 0.5, integer ROIs: pixel index) "
                    "in canonical image pixel coordinates (pixel centre = integer, x = column, y = row); label CSV 'y,x' relative "
                    "to the top-left pixel of the exported native-resolution crop (box_xyxy)")
SOURCE_ORIGIN = "legacy_imagej"
LABEL_BASIS = "legacy_imagej_mark"

# machine-readable reasons a TIFF is rejected / a source contributes nothing
R_UNREADABLE = "tiff_unreadable"
R_DTYPE = "unsupported_dtype"
R_LAYOUT = "unsupported_layout"
R_CORRUPT = "corrupt_metadata"
R_NO_ROI = "missing_roi"
R_NO_GRID = "missing_grid"
R_MULTI_ROI = "multiple_roi_blobs"
R_MULTI_GRID = "multiple_grids"
R_NO_COUNTERS = "no_counters"
R_MAP = "channel_mapping_unresolved"
R_MAP_PAGE = "channel_mapping_page_out_of_range"
R_NO_PAGES = "no_mapped_pages"
R_NO_ELIGIBLE = "no_eligible_fields"
R_LOCKED = "locked_test_set"
R_NAME = "filename_not_utf8"
NAME_NOT_UTF8 = (f"{R_NAME}: the file name is not valid UTF-8 (undecodable bytes / non-ASCII under a C locale); it is skipped, never "
                 "imported under a mangled name. Rename the file, or run with a UTF-8 locale (LC_ALL=C.UTF-8)")

_ID_RE = re.compile(r"r(\d+)c(\d+)$")
_PSTATUS = {"ok": 0, "duplicate": 1, "out_of_bounds": 2}


# ---------------------------------------------------------------------------------------------- options
@dataclass
class BridgeOptions:
    mode: str = ex.MODE_MARKED_ONLY
    acknowledge_no_counters: bool = False
    allow_unconfirmed_identity: bool = False
    acknowledge_field_equals_counter_plus_one: bool = False
    acknowledge_lab_csv_field_numbering: bool = False
    seed: int = 0
    fractions: Optional[dict] = None
    level: str = "auto"
    downscale: int = 1
    normaliser: str = "valid_unsaturated"
    percentiles: tuple = DEFAULT_PERCENTILES
    min_crop_px: int = 32
    batch_size: int = 4
    epochs: int = 30
    workers: int = 1
    created_at: Optional[str] = None
    duplicate_tol_px: float = 1.0
    valid_mask: str = DEFAULT_VALID_MASK

    def check(self) -> "BridgeOptions":
        if self.mode not in ex.MODES:
            raise MLError(f"--mode must be one of {list(ex.MODES)} (got {self.mode!r}); there is no mode that treats unmarked "
                          "cells as negatives")
        self.downscale = check_downscale(self.downscale)
        self.normaliser = check_normaliser(self.normaliser)
        if self.valid_mask not in VALID_MASKS:
            raise MLError(f"--valid-mask must be one of {list(VALID_MASKS)} (got {self.valid_mask!r})")
        self.workers = max(1, int(self.workers))
        if self.batch_size < 1 or self.epochs < 1:
            raise MLError("--batch-size and --epochs must be >= 1")
        return self

    def public(self) -> dict:
        """Option values recorded in the manifest (``workers`` is excluded: it never changes the output)."""
        return {"mode": self.mode, "acknowledge_no_counters": self.acknowledge_no_counters,
                "allow_unconfirmed_identity": self.allow_unconfirmed_identity,
                "acknowledge_field_equals_counter_plus_one": self.acknowledge_field_equals_counter_plus_one,
                "acknowledge_lab_csv_field_numbering": self.acknowledge_lab_csv_field_numbering,
                "seed": self.seed, "fractions": self.fractions, "level": self.level, "downscale": self.downscale,
                "normaliser": self.normaliser, "percentiles": [float(p) for p in self.percentiles],
                "min_crop_px": self.min_crop_px, "batch_size": self.batch_size, "epochs": self.epochs,
                "valid_mask": self.valid_mask, "chamber_rim_px": CHAMBER_RIM_PX,
                "duplicate_tol_px": self.duplicate_tol_px}


# ---------------------------------------------------------------------------------------------- file helpers
def sha256_stream(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def discover_tiffs(root: Path, skipped: Optional[list] = None) -> list[tuple[str, Path]]:
    """[(relative posix path, path)] of every .tif/.tiff below ``root``: sorted, dot-directories and dot-files skipped,
    symlinked directories not followed, symlinked FILES whose target lies outside ``root`` skipped, archives never opened.
    Every skipped link is appended to ``skipped`` (when given) as ``(relative path, reason)``. A file or directory whose name does
    not survive a strict UTF-8 round trip (latin-1 bytes on a share, non-ASCII under a C locale) is skipped with the structured
    reason ``filename_not_utf8`` (its path is shown with escaped bytes): source names are written to the manifest as UTF-8 and
    are never mangled or escaped silently."""
    root = Path(root)
    if root.is_file():
        if utf8_name_problem(root.name):
            if skipped is not None:
                skipped.append((printable_name(root.name), NAME_NOT_UTF8))
            return []
        return [(root.name, root)] if root.suffix.lower() in (".tif", ".tiff") else []
    out: list[tuple[str, Path]] = []
    real_root = Path(os.path.realpath(root))
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        keep = []
        for d in sorted(dirnames):
            if d.startswith("."):
                continue
            if utf8_name_problem(d):
                if skipped is not None:
                    skipped.append((printable_name((Path(dirpath) / d).relative_to(root).as_posix()), NAME_NOT_UTF8))
                continue
            if (Path(dirpath) / d).is_symlink():
                if skipped is not None:
                    skipped.append(((Path(dirpath) / d).relative_to(root).as_posix(),
                                    "symlinked directory: not followed"))
                continue
            keep.append(d)
        dirnames[:] = keep
        for fn in sorted(filenames):
            if fn.startswith(".") or Path(fn).suffix.lower() not in (".tif", ".tiff"):
                continue
            if fn.lower().endswith((".tif.zip", ".tiff.zip")):
                continue
            if utf8_name_problem(fn):
                if skipped is not None:
                    skipped.append((printable_name((Path(dirpath) / fn).relative_to(root).as_posix()), NAME_NOT_UTF8))
                continue
            p = Path(dirpath) / fn
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                target = Path(os.path.realpath(p))
                try:
                    target.relative_to(real_root)
                except ValueError:
                    if skipped is not None:
                        skipped.append((rel, "symlink resolving outside --input: not imported"))
                    continue
                if not target.is_file():
                    if skipped is not None:
                        skipped.append((rel, "broken or non-file symlink: not imported"))
                    continue
            out.append((rel, p))
    out.sort(key=lambda t: t[0])
    return out


def _file_info(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    p = Path(path)
    return {"name": p.name, "sha256": sha256_stream(p)}


# ---------------------------------------------------------------------------------------------- scan structures
@dataclass
class SourceScan:
    """Compact, pixel-free description of one TIFF (no pydantic objects are kept)."""
    source_id: str
    rel: str
    path: Path
    sha256: str
    size: int
    mtime_ns: int
    ino: int = 0                                           # identity of the file as scanned (st_ino / st_ctime_ns), see _PlaneCache
    ctime_ns: int = 0
    alias_paths: list = field(default_factory=list)
    reasons: list = field(default_factory=list)            # [(code, message)] -> rejected when non-empty
    notes: list = field(default_factory=list)              # [(code, message)] informational
    width: int = 0
    height: int = 0
    n_planes: int = 0
    dtype: str = ""
    axes: str = ""
    xs: tuple = ()                                         # grid lines, ImageJ frame
    ys: tuple = ()
    grid_regular: bool = True
    cell_counters: dict = field(default_factory=dict)      # cell index -> [counters whose majority cell it is]
    counter_cell: dict = field(default_factory=dict)       # counter -> majority cell index
    x: Optional[np.ndarray] = None                         # canonical point coordinates
    y: Optional[np.ndarray] = None
    counter: Optional[np.ndarray] = None                   # -1 = none
    page: Optional[np.ndarray] = None                      # -1 = unspecified
    position: Optional[np.ndarray] = None                  # ImageJ position as stored (-1 = none); tells 'beyond the planes' from 'missing'
    pstatus: Optional[np.ndarray] = None                   # 0 ok, 1 duplicate, 2 out_of_bounds
    cell: Optional[np.ndarray] = None                      # -1 = outside the grid / unusable
    object_ids: list = field(default_factory=list)
    n_blobs: int = 0
    n_point_rois: int = 0
    identity: Optional[dict] = None
    info_hint: Optional[str] = None                        # ImageJ Info label hint of page 0 (information only)

    @property
    def rejected(self) -> bool:
        return bool(self.reasons)

    @property
    def n_cols(self) -> int:
        return max(0, len(self.xs) - 1)

    @property
    def n_rows(self) -> int:
        return max(0, len(self.ys) - 1)

    @property
    def n_cells(self) -> int:
        return self.n_cols * self.n_rows

    @property
    def n_points(self) -> int:
        return 0 if self.x is None else int(len(self.x))

    def cell_rc(self, idx: int) -> tuple[int, int]:
        return divmod(int(idx), self.n_cols)

    def cell_rect_ij(self, idx: int) -> tuple:
        r, c = self.cell_rc(idx)
        return (self.xs[c], self.ys[r], self.xs[c + 1], self.ys[r + 1])

    def cell_rect(self, idx: int) -> tuple:
        """Canonical rectangle (pixel-centre frame) of a grid cell."""
        x0, y0, x1, y1 = self.cell_rect_ij(idx)
        return (x0 - 0.5, y0 - 0.5, x1 - 0.5, y1 - 0.5)

    def cell_name(self, idx: int) -> str:
        r, c = self.cell_rc(idx)
        return f"r{r}c{c}"

    @property
    def has_counters(self) -> bool:
        return self.counter is not None and bool((self.counter >= 0).any())


@dataclass
class ScanResult:
    root: Path
    discovered: int
    sources: list                                          # SourceScan, kept (unique) files, sorted by rel path
    aliases: dict                                          # kept rel -> [alias rels]
    skipped_links: list = field(default_factory=list)      # [(rel, reason)] symlinks that were not followed / imported


def _header(path: Path) -> dict:
    with tifffile.TiffFile(str(path)) as tif:
        s = tif.series[0]
        return {"shape": tuple(int(v) for v in s.shape), "axes": str(s.axes), "dtype": str(np.dtype(s.dtype)),
                "n_pages": len(tif.pages)}


def _layout(h: dict) -> tuple[Optional[tuple], Optional[str]]:
    """((planes, height, width), None) for 2-D pages (a single plane or a stack of planes), else (None, why)."""
    shape, axes = h["shape"], h["axes"]
    if len(shape) == 2 and axes.endswith("YX"):
        return (1, shape[0], shape[1]), None
    if len(shape) == 3 and axes.endswith("YX"):
        return (shape[0], shape[1], shape[2]), None
    if "S" in axes:
        return None, f"RGB / multi-sample pages (axes {axes}, shape {shape}) are rendered composites, not raw 2-D planes"
    return None, f"unsupported axes/shape {axes} {shape} (need one 2-D plane or a stack of 2-D planes)"


def _info_hint(path: Path) -> Optional[str]:
    """Raw ImageJ Info label hint (e.g. '635 [W1]'): INFORMATION ONLY, never used to decide a role."""
    from polony_app.imaging.ingest import parse_info_label_hint
    try:
        with tifffile.TiffFile(str(path)) as tif:
            md = tif.imagej_metadata if tif.is_imagej else None
        info = md.get("Info") if md else None
        return parse_info_label_hint(info if isinstance(info, str) else None)
    except Exception:  # noqa: BLE001
        return None


def _clean(msg: str, path: Path, rel: str) -> str:
    """Error texts must not embed absolute paths (the manifest is location independent)."""
    return msg.replace(str(path), rel).replace(str(path.parent), "<dir>")


def pixel_footprint_cells(x, y, xs, ys, pstatus) -> np.ndarray:
    """Cell index (row * n_cols + col) of every mark, -1 = outside the grid / unusable.

    A mark belongs to the cell whose integer PIXEL footprint (``pixel_box`` of the cell rectangle, i.e. the pixels whose centres
    lie in the cell) contains the pixel the mark rounds to (``floor(x + 0.5)``; pixel centres are integers). The footprints of
    neighbouring cells tile the pixel grid exactly, so a mark is never inside one crop's pixels while counted for another
    cell, also on a fractional grid. Out-of-bounds marks (pstatus 2) have no cell."""
    n = len(x)
    out = np.full(n, -1, np.int64)
    if len(xs) < 2 or len(ys) < 2 or n == 0:
        return out
    ex_ = np.array([math.ceil(float(v) - 0.5) for v in xs], np.int64)
    ey_ = np.array([math.ceil(float(v) - 0.5) for v in ys], np.int64)
    nc, nr = len(xs) - 1, len(ys) - 1
    fx, fy = np.asarray(x, float), np.asarray(y, float)
    for k in range(n):
        if pstatus[k] == 2 or not (np.isfinite(fx[k]) and np.isfinite(fy[k])):
            continue
        ix, iy = int(math.floor(fx[k] + 0.5)), int(math.floor(fy[k] + 0.5))
        c = int(np.searchsorted(ex_, ix, side="right")) - 1
        r = int(np.searchsorted(ey_, iy, side="right")) - 1
        if 0 <= c < nc and 0 <= r < nr and ix < ex_[c + 1] and iy < ey_[r + 1]:
            out[k] = r * nc + c
    return out


def pixel_of(v: float) -> int:
    """The pixel a canonical coordinate falls in (pixel centres are integers)."""
    return int(math.floor(float(v) + 0.5))


def _stat_sig(st) -> tuple:
    """What identifies the file version that was hashed: size, mtime, inode and ctime (ctime cannot be set back by a writer)."""
    return (int(st.st_size), int(st.st_mtime_ns), int(getattr(st, "st_ino", 0) or 0), int(getattr(st, "st_ctime_ns", 0) or 0))


def _scan_one(rel: str, path: Path, sha: str, source_id: str, opts: BridgeOptions, st=None) -> SourceScan:
    st = st if st is not None else path.stat()
    sc = SourceScan(source_id=source_id, rel=rel, path=path, sha256=sha, size=int(st.st_size), mtime_ns=int(st.st_mtime_ns),
                    ino=int(getattr(st, "st_ino", 0) or 0), ctime_ns=int(getattr(st, "st_ctime_ns", 0) or 0))
    try:
        h = _header(path)
    except Exception as exc:  # noqa: BLE001 - tifffile raises many types
        sc.reasons.append((R_UNREADABLE, _clean(f"cannot read the TIFF header ({type(exc).__name__}: {exc})", path, rel)))
        return sc
    sc.dtype, sc.axes = h["dtype"], h["axes"]
    lay, why = _layout(h)
    if lay is None:
        sc.reasons.append((R_LAYOUT, why or "unsupported layout"))
        return sc
    sc.n_planes, sc.height, sc.width = lay
    if sc.dtype not in imp.SUPPORTED_DTYPES:
        sc.reasons.append((R_DTYPE, f"pixel type {sc.dtype} is not supported (only uint8/uint16); it is never converted"))
        return sc
    sc.info_hint = _info_hint(path)
    bundle = roimod.read_embedded(path)
    sc.n_blobs = bundle.n_blobs
    damaged = [i for i in bundle.issues if i.code in ("tiff_metadata_unreadable", "roi_unparseable")]
    if damaged:
        sc.reasons.append((R_CORRUPT, _clean("; ".join(i.message for i in damaged[:3]), path, rel)))
        return sc
    point_rois = sorted({p.roi_index for p in bundle.points})
    sc.n_point_rois = len(point_rois)
    if len(point_rois) > 1:
        sc.reasons.append((R_MULTI_ROI, f"{len(point_rois)} separate point ROI blobs: which one is the researcher's count is "
                                        "ambiguous"))
        return sc
    if len(bundle.grids) > 1:
        sc.reasons.append((R_MULTI_GRID, f"{len(bundle.grids)} grid overlays: which one defines the fields is ambiguous"))
        return sc
    if not bundle.points and opts.mode == ex.MODE_MARKED_ONLY:
        sc.reasons.append((R_NO_ROI, "no embedded ImageJ point ROI (no researcher marks)"))
    if not bundle.grids:
        sc.reasons.append((R_NO_GRID, "no counting-grid overlay: fields cannot be derived (no cell, no label)"))
    if sc.reasons:
        return sc
    if bundle.points and not any(p.counter is not None for p in bundle.points) and not opts.acknowledge_no_counters:
        sc.reasons.append((R_NO_COUNTERS, "the points carry no ImageJ counters: counter-based validation of the fields is "
                                          "impossible (pass --acknowledge-no-counters to accept geometric cells only)"))
        return sc
    # ---- convert with the existing importer logic, then keep only compact arrays
    rec = ImageRecord(image_id=source_id, file_name=path.name, rel_path=rel, sha256=sha, size_bytes=sc.size, kind=ImageKind.stack,
                      width=sc.width, height=sc.height, n_planes=sc.n_planes, dtype=sc.dtype,
                      bit_depth=int(np.dtype(sc.dtype).itemsize * 8), axes=sc.axes)
    warn = imp._Warn()
    cfg = imp.ImportConfig(duplicate_tol_px=opts.duplicate_tol_px, check_signal=False, check_composites=False,
                           reserve_legacy_test_dirs=False)
    pts, fields, _regions = imp._convert_annotations(rec, bundle, {}, cfg, warn)
    grid = bundle.grids[0]
    sc.xs, sc.ys, sc.grid_regular = tuple(float(v) for v in grid.xs), tuple(float(v) for v in grid.ys), bool(grid.regular)
    nc = sc.n_cols
    n = len(pts)
    sc.x = np.array([p.x for p in pts], float)
    sc.y = np.array([p.y for p in pts], float)
    sc.counter = np.array([-1 if p.counter is None else int(p.counter) for p in pts], np.int32)
    sc.page = np.array([-1 if p.page is None else int(p.page) for p in pts], np.int32)
    sc.position = np.array([-1 if p.position is None else int(p.position) for p in pts], np.int32)
    sc.pstatus = np.array([_PSTATUS[p.status.value] for p in pts], np.int8)
    sc.cell = pixel_footprint_cells(sc.x, sc.y, sc.xs, sc.ys, sc.pstatus)
    sc.object_ids = [f"{p.roi_index}:{p.point_index}" for p in pts]
    # a counter's field = the cell holding most of its (ok) marks; ties go to the first cell met (as the importer does)
    counter_cells: dict[int, Counter] = defaultdict(Counter)
    for k in range(n):
        if sc.pstatus[k] == 0 and sc.counter[k] >= 0 and sc.cell[k] >= 0:
            counter_cells[int(sc.counter[k])][int(sc.cell[k])] += 1
    for ctr, cnt in sorted(counter_cells.items()):
        cell = cnt.most_common(1)[0][0]
        sc.counter_cell[ctr] = cell
        sc.cell_counters.setdefault(cell, []).append(ctr)
    sc.notes = [(w.code, w.message) for w in warn.items if w.severity.value != "info"][:20]
    return sc


def scan_tree(root: Path, opts: BridgeOptions, say: Say = print, *, progress_every: int = 100) -> ScanResult:
    """Phase A. Streams every file once for its sha256, drops byte-identical copies (first path in sorted order wins), reads
    headers and ROI blobs. Never reads pixel planes."""
    root = Path(root)
    skipped_links: list = []
    files = discover_tiffs(root, skipped_links)
    say(f"discovered {len(files)} TIFF file(s) under {root}")
    for rel, why in skipped_links:
        say(f"  SKIPPED {rel}: {why}")
    kept: dict[str, SourceScan] = {}
    aliases: dict[str, list] = {}
    ids: dict[str, str] = {}            # id -> sha256 (collision guard)
    for n, (rel, path) in enumerate(files, 1):
        st0 = path.stat()
        sha = sha256_stream(path)
        if _stat_sig(path.stat()) != _stat_sig(st0):
            raise MLError(f"{rel} changed while it was being read (size / modification time / inode differ before and after "
                          "hashing); nothing was written. Make sure nothing writes to the input directory and rerun")
        if sha in kept:
            kept[sha].alias_paths.append(rel)
            aliases.setdefault(kept[sha].rel, []).append(rel)
        else:
            k = 12
            while f"img_{sha[:k]}" in ids and ids[f"img_{sha[:k]}"] != sha:
                k += 4
            sid = f"img_{sha[:k]}"
            ids[sid] = sha
            kept[sha] = _scan_one(rel, path, sha, sid, opts, st0)
        if progress_every and n % progress_every == 0:
            say(f"  scanned {n}/{len(files)} file(s)")
    sources = sorted(kept.values(), key=lambda s: s.rel)
    return ScanResult(root=root, discovered=len(files), sources=sources, aliases=aliases,
                      skipped_links=sorted(skipped_links))


# ---------------------------------------------------------------------------------------------- channel map
@dataclass
class ChannelMap:
    path: Optional[Path]
    sha256: str
    confirmed_by: str
    confirmed_on: str
    source_of_truth: str
    defaults: list                  # [(id, pages)]
    rules: list                     # [(id, match, pages)]
    files: dict                     # rel path -> pages (keys exactly as written: checked against the scanned files)

    def describe(self) -> dict:
        return {"name": None if self.path is None else Path(self.path).name, "sha256": self.sha256,
                "confirmed_by": self.confirmed_by, "confirmed_on": self.confirmed_on,
                "source_of_truth": self.source_of_truth, "n_rules": len(self.rules), "n_files": len(self.files)}


_MATCH_KEYS = ("glob", "assay", "sha256", "specimen_id", "slide_id")


_PAGE_KEYS = ("role", "train")


def _unknown_keys(d: dict, allowed: Iterable[str], where: str, *, comments: bool = False) -> None:
    """Typos are errors: an unknown key (``trian``) must never be silently ignored. Keys starting with '_' are comments when
    ``comments`` is true (top level only)."""
    bad = sorted(str(k) for k in d if k not in allowed and not (comments and str(k).startswith("_")))
    if bad:
        raise MLError(f"channel map: {where}: unknown key(s) {bad}; allowed: {sorted(allowed)}")


def _pages_block(raw, where: str) -> dict:
    if not isinstance(raw, dict) or not raw:
        raise MLError(f"channel map: {where}: 'pages' must be a non-empty object {{\"0\": {{\"role\": ..., \"train\": true}}}}")
    pages: dict[int, dict] = {}
    roles: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not PAGE_KEY_RE.fullmatch(k):
            raise MLError(f"channel map: {where}: page keys are 0-based integers of 1-4 ASCII digits (\"0\", \"1\", ...), got "
                          f"{str(k)[:20]!r}")
        if not isinstance(v, dict):
            raise MLError(f"channel map: {where}: page {k} must be an object with 'role' and 'train'")
        _unknown_keys(v, _PAGE_KEYS, f"{where}: page {k}")
        role = v.get("role")
        if not isinstance(role, str) or not role.strip():
            raise MLError(f"channel map: {where}: page {k} has no role (got {role!r}). The map must state, for every page, what "
                          "it shows (e.g. 'red'); roles are never inferred. Fill in the template.")
        if "train" not in v:
            raise MLError(f"channel map: {where}: page {k} has no 'train'. State explicitly whether this page is trained on "
                          "(true) or not (false); there is no default.")
        train = v["train"]
        if not isinstance(train, bool):
            raise MLError(f"channel map: {where}: page {k}: 'train' must be true or false (got {train!r})")
        role = role.strip()
        if role in roles:
            raise MLError(f"channel map: {where}: role '{role}' is used for pages {roles[role]} and {k}; roles must be unique "
                          "within a file")
        roles[role] = str(k)
        pages[int(k)] = {"role": role, "train": train}
    return dict(sorted(pages.items()))


def load_channel_map(path: str | Path) -> ChannelMap:
    p = Path(path)
    d, raw = read_json_strict(p, "channel map")
    if not isinstance(d, dict) or d.get("schema") != CHANNEL_SCHEMA:
        raise MLError(f"channel map {p}: must be a JSON object whose 'schema' is '{CHANNEL_SCHEMA}'")
    if d.get("version") != CHANNEL_VERSION or isinstance(d.get("version"), bool):
        raise MLError(f"channel map {p}: unsupported version {str(d.get('version'))[:30]!r} (supported: {CHANNEL_VERSION})")
    _unknown_keys(d, ("schema", "version", "confirmed_by", "confirmed_on", "source_of_truth", "defaults", "rules", "files"),
                  f"{p}", comments=True)
    for k in ("confirmed_by", "confirmed_on", "source_of_truth"):
        if d.get(k) is not None and not isinstance(d.get(k), str):
            raise MLError(f"channel map {p}: '{k}' must be text")
    if not str(d.get("confirmed_by") or "").strip():
        raise MLError(f"channel map {p}: 'confirmed_by' is empty. A channel map is the researcher's statement of what each page "
                      "shows; fill in who confirmed this mapping and on what basis ('confirmed_by', 'confirmed_on', 'source_of_truth').")
    if not str(d.get("source_of_truth") or "").strip():
        raise MLError(f"channel map {p}: 'source_of_truth' is empty. Fill in who confirmed this mapping and on what basis "
                      "(lab notebook, scanner protocol, ...): a default mapping is never active unconfirmed.")
    defaults = []
    draw = d.get("defaults")
    if draw:
        if not isinstance(draw, (list, dict)):
            raise MLError(f"channel map {p}: 'defaults' must be an object {{\"pages\": ...}} or a list of such objects")
        items = draw if isinstance(draw, list) else [draw]
        for k, it in enumerate(items):
            if not isinstance(it, dict):
                raise MLError(f"channel map {p}: defaults[{k}] must be an object")
            _unknown_keys(it, ("id", "pages"), f"defaults[{k}]")
            defaults.append((str(it.get("id") or f"default{k}"), _pages_block(it.get("pages"), f"defaults[{k}]")))
    rules, seen = [], set()
    rraw = d.get("rules")
    if rraw is not None and not isinstance(rraw, list):
        raise MLError(f"channel map {p}: 'rules' must be a list of rule objects (got {type(rraw).__name__})")
    for k, r in enumerate(rraw or []):
        if not isinstance(r, dict):
            raise MLError(f"channel map {p}: rules[{k}] must be an object with 'id', 'match' and 'pages'")
        _unknown_keys(r, ("id", "match", "pages"), f"rules[{k}]")
        rid = str(r.get("id") or "").strip()
        if not rid:
            raise MLError(f"channel map {p}: rules[{k}] needs a non-empty 'id' (it is recorded as the origin of the mapping)")
        if rid in seen:
            raise MLError(f"channel map {p}: rule id '{rid}' is used twice")
        seen.add(rid)
        match = r.get("match")
        if not isinstance(match, dict) or not match or set(match) - set(_MATCH_KEYS):
            raise MLError(f"channel map {p}: rule '{rid}': 'match' must use only {list(_MATCH_KEYS)} (got {str(match)[:80]!r})")
        if any(isinstance(v, (dict, list)) for v in match.values()):
            raise MLError(f"channel map {p}: rule '{rid}': 'match' values must be text")
        rules.append((rid, {k2: str(v) for k2, v in match.items()}, _pages_block(r.get("pages"), f"rule '{rid}'")))
    files = {}
    fraw = d.get("files")
    if fraw is not None and not isinstance(fraw, dict):
        raise MLError(f"channel map {p}: 'files' must be an object {{\"relative/path.tif\": {{\"pages\": ...}}}} (got "
                      f"{type(fraw).__name__})")
    for k, v in (fraw or {}).items():
        why = _bad_map_path(k)
        if why:
            raise MLError(f"channel map {p}: files[{k[:80]!r}]: {why}")
        if not isinstance(v, dict):
            raise MLError(f"channel map {p}: files['{k}'] must be an object with 'pages'")
        _unknown_keys(v, ("pages",), f"files['{k}']")
        files[str(k)] = _pages_block(v.get("pages"), f"files['{k}']")
    if not (defaults or rules or files):
        raise MLError(f"channel map {p}: it defines no defaults, rules or files")
    return ChannelMap(p, sha256_bytes(raw), str(d["confirmed_by"]), str(d.get("confirmed_on") or ""),
                      str(d.get("source_of_truth") or ""), defaults, rules, files)


def _bad_map_path(key) -> Optional[str]:
    """Why a file key of a mapping file is not a relative posix path below --input (None = well-formed)."""
    if not isinstance(key, str) or not key.strip():
        return "the file key must be a non-empty relative path"
    if "\x00" in key:
        return "the file key contains a NUL character"
    if key.startswith("/") or re.match(r"^[A-Za-z]:", key) or key.startswith("\\"):
        return "absolute paths are not allowed: use the path relative to --input (posix separators)"
    if any(part == ".." for part in key.replace("\\", "/").split("/")):
        return "'..' leaves the --input directory: use the path relative to --input"
    return None


def _rule_matches(match: dict, sc: SourceScan) -> bool:
    ident = sc.identity or {}
    for k, v in match.items():
        if k == "glob":
            if not any(fnmatch.fnmatch(r, v) or ("/" not in v and fnmatch.fnmatch(Path(r).name, v)) for r in [sc.rel, *sc.alias_paths]):
                return False
        elif k == "sha256":
            if not sc.sha256.startswith(v.lower()) or len(v) < 8:
                return False
        elif k == "assay":
            if (ident.get("assay") or None) != v:
                return False
        elif k == "specimen_id":
            if (ident.get("specimen_id") or None) != v:
                return False
        elif k == "slide_id":
            if (ident.get("slide_id") or None) != v:
                return False
    return True


def resolve_mapping(cm: ChannelMap, sc: SourceScan) -> Optional[tuple[dict, str]]:
    """(pages, origin) with precedence files > rules > defaults, or None when nothing matches. Conflicting matches at the
    same precedence level raise an actionable ``MLError``."""
    for rel in [sc.rel, *sc.alias_paths]:
        if rel in cm.files:
            return cm.files[rel], "file"
    hits = [(rid, pages) for rid, match, pages in cm.rules if _rule_matches(match, sc)]
    if hits:
        first = hits[0]
        for other in hits[1:]:
            if other[1] != first[1]:
                raise MLError(f"channel map: the rules '{first[0]}' and '{other[0]}' both match {sc.rel} but assign different "
                              f"pages/roles ({_brief(first[1])} vs {_brief(other[1])}). Make the rules disjoint, or list this "
                              "file explicitly under 'files'.")
        return first[1], f"rule:{first[0]}"
    if cm.defaults:
        first = cm.defaults[0]
        for other in cm.defaults[1:]:
            if other[1] != first[1]:
                raise MLError(f"channel map: the defaults '{first[0]}' and '{other[0]}' conflict for {sc.rel} "
                              f"({_brief(first[1])} vs {_brief(other[1])}); keep one default.")
        return first[1], "default"
    return None


def channel_map_unmatched(cm: ChannelMap, scans: list) -> list[str]:
    """Blocker texts: every ``files`` entry that is not EXACTLY the relative posix path (or byte-identical alias path) of a
    scanned TIFF, and every rule that matches no scanned TIFF. Nothing is normalised silently: a key such as './a.tif', 'A.tif'
    or 'sub\\a.tif' that would have fallen through to the default mapping is reported with the nearest existing file."""
    rels = sorted({r for s in scans for r in [s.rel, *s.alias_paths]})
    relset = set(rels)
    out: list[str] = []
    for key in sorted(cm.files):
        if key in relset:
            continue
        norm = key.replace("\\", "/")
        while norm.startswith("./"):
            norm = norm[2:]
        fold = {r.casefold(): r for r in rels}
        near = norm if norm in relset else fold.get(norm.casefold())
        if near is None:
            cands = difflib.get_close_matches(norm, rels, n=1, cutoff=0.0) or difflib.get_close_matches(
                Path(norm).name, [Path(r).name for r in rels], n=1, cutoff=0.0)
            near = next((r for r in rels if Path(r).name == cands[0]), cands[0]) if cands else None
        out.append(f"channel map files['{key}'] matches no scanned TIFF (it must be the exact relative posix path below --input, "
                   "no './', no backslashes, same case)" + (f"; nearest existing file: '{near}'" if near else "; no TIFF was scanned"))
    for rid, match, _pages in cm.rules:
        if not any(_rule_matches(match, s) for s in scans):
            out.append(f"channel map rule '{rid}' ({json.dumps(match, sort_keys=True)}) matches no scanned TIFF")
    return out


def _brief(pages: dict) -> str:
    return ", ".join(f"page {k}={v['role']}" for k, v in pages.items())


# ---------------------------------------------------------------------------------------------- identity map
@dataclass
class IdentityTable:
    path: Optional[Path]
    sha256: str
    rows: list
    warnings: list = field(default_factory=list)


_TRUE = {"1", "true", "yes", "y", "t"}
_FALSE = {"0", "false", "no", "n", "f"}


def _idkey(v: Optional[str]) -> Optional[str]:
    """Grouping/lock key of an identity string: stripped, whitespace-collapsed, case-folded."""
    return None if v is None else " ".join(str(v).split()).casefold()


def _canonicalise(rows: list, key: str) -> list[str]:
    """Rows whose ``key`` differs only by case/whitespace get ONE spelling (the lexicographically smallest), so replicates
    spelled 'AM' and 'am' stay in one group. Returns human-readable notes."""
    clusters: dict[str, set] = defaultdict(set)
    for r in rows:
        if r.get(key):
            clusters[_idkey(r[key])].add(" ".join(r[key].split()))
    notes = []
    canon = {k: sorted(v)[0] for k, v in clusters.items()}
    for k, v in sorted(clusters.items()):
        if len(v) > 1:
            notes.append(f"{key} spellings {sorted(v)} differ only by case/spacing: treated as one ('{canon[k]}')")
    for r in rows:
        if r.get(key):
            r[key] = canon[_idkey(r[key])]
    return notes


_ID_ROW_KEYS = ("file", "specimen_id", "slide_id", "scan_session", "assay", "replicate", "confirmed", "inferred_from")
_ID_TOP_KEYS = ("rows", "identities", "files")


def _identity_rows(p: Path) -> tuple[list, bytes]:
    """(rows, raw bytes) of an identity map (JSON or CSV); every malformed shape is an actionable :class:`MLError`."""
    if p.suffix.lower() == ".csv":
        try:
            size = p.stat().st_size
            if size > 64 * 1024 * 1024:
                raise MLError(f"identity map {p}: larger than 64 MB - not an identity map")
            raw = p.read_bytes()
            text = raw.decode("utf-8-sig")
        except OSError as exc:
            raise MLError(f"identity map {p}: cannot be read ({exc.strerror or exc})")
        except UnicodeDecodeError as exc:
            raise MLError(f"identity map {p}: is not valid UTF-8 ({exc})")
        try:
            rd = csv.DictReader(text.splitlines())
            unknown = [c for c in (rd.fieldnames or []) if c not in _ID_ROW_KEYS and c is not None]
            if unknown:
                raise MLError(f"identity map {p}: unknown column(s) {unknown}; allowed: {list(_ID_ROW_KEYS)}")
            rows = list(rd)
        except csv.Error as exc:
            raise MLError(f"identity map {p}: not a valid CSV ({exc})")
        if any(None in r for r in rows):
            raise MLError(f"identity map {p}: a CSV row has more cells than the header")
        return rows, raw
    d, raw = read_json_strict(p, "identity map")
    if isinstance(d, dict):
        unknown = sorted(str(k) for k in d if k not in _ID_TOP_KEYS and not str(k).startswith("_"))
        if unknown:
            raise MLError(f"identity map {p}: unknown key(s) {unknown}; allowed: {list(_ID_TOP_KEYS)} (keys starting with '_' are "
                          "comments)")
        rows = d.get("rows") or d.get("identities") or d.get("files") or []
    elif isinstance(d, list):
        rows = d
    else:
        raise MLError(f"identity map {p}: must be a JSON object with 'rows' (or a list of rows), got {type(d).__name__}")
    if isinstance(rows, dict):
        if not all(isinstance(v, dict) for v in rows.values()):
            raise MLError(f"identity map {p}: when 'rows' is an object, every value must be a row object "
                          "{\"specimen_id\": ..., \"confirmed\": ...}")
        rows = [dict(v, file=k) for k, v in rows.items()]
    if not isinstance(rows, list):
        raise MLError(f"identity map {p}: 'rows' must be a list of row objects (got {type(rows).__name__})")
    for k, r in enumerate(rows):
        if isinstance(r, dict):
            bad = sorted(str(c) for c in r if c not in _ID_ROW_KEYS and not str(c).startswith("_"))
            if bad:
                raise MLError(f"identity map {p}: row {k + 1}: unknown key(s) {bad}; allowed: {list(_ID_ROW_KEYS)}")
            if any(isinstance(v, (dict, list)) for v in r.values()):
                raise MLError(f"identity map {p}: row {k + 1}: values must be text / true / false, not nested objects or lists")
    return rows, raw


def load_identity_map(path: str | Path, *, level: str = "auto") -> IdentityTable:
    p = Path(path)
    rows, raw = _identity_rows(p)
    out = []
    for k, r in enumerate(rows):
        if not isinstance(r, dict) or not str(r.get("file") or "").strip():
            raise MLError(f"identity map {p}: row {k + 1} has no 'file'")
        fname = str(r["file"]).strip()
        conf = r.get("confirmed")
        if isinstance(conf, bool):
            pass
        elif isinstance(conf, str) and conf.strip().lower() in _TRUE | _FALSE:
            conf = conf.strip().lower() in _TRUE
        else:
            raise MLError(f"identity map {p}: row {k + 1} ({fname}): 'confirmed' must be explicitly true or false (got "
                          f"{'nothing / an empty cell' if conf is None or conf == '' else repr(conf)}). A missing value is never read as "
                          "confirmed: set true only for rows you checked against the lab's records.")
        row = {"file": fname.replace("\\", "/"), "confirmed": bool(conf)}
        for key in ("specimen_id", "slide_id", "scan_session", "assay", "replicate"):
            v = r.get(key)
            row[key] = None if v is None or str(v).strip() == "" else str(v).strip()
        if row["confirmed"] and not (row["specimen_id"] or row["slide_id"]):
            raise MLError(f"identity map {p}: row {k + 1} ({row['file']}) is marked confirmed but has neither specimen_id nor slide_id")
        if row["confirmed"] and not row["specimen_id"] and level != "slide":
            raise MLError(f"identity map {p}: row {k + 1} ({row['file']}) is confirmed but has no specimen_id: replicate slides of one "
                          "specimen could then land in different splits (leakage). Fill in specimen_id, or pass --level slide to "
                          "state that each slide is an independent unit.")
        if row["confirmed"] and level == "slide" and not row["slide_id"]:
            raise MLError(f"identity map {p}: row {k + 1} ({row['file']}): --level slide needs a slide_id on every confirmed row")
        out.append(row)
    notes: list[str] = []
    confirmed = [r for r in out if r["confirmed"]]
    for key in ("specimen_id", "slide_id", "assay"):
        notes += _canonicalise(confirmed, key)
    # one specimen must not be described with a set assay on one slide and a blank/different one on another: the assay is part
    # of the grouping key, so the replicates would silently be split apart
    by_spec: dict[str, list] = defaultdict(list)
    for r in confirmed:
        if r["specimen_id"]:
            by_spec[_idkey(r["specimen_id"])].append(r)
    for sk, rs in sorted(by_spec.items()):
        assays = {r["assay"] for r in rs}
        if None in assays and len(assays) > 1:
            blank = sorted(r["file"] for r in rs if not r["assay"])
            raise MLError(f"identity map {p}: specimen '{rs[0]['specimen_id']}' has an assay on some rows "
                          f"({sorted(a for a in assays if a)}) and none on others ({blank}); the assay is part of the grouping key, so "
                          "its replicate slides would land in different groups. Fill in the same assay on every row of the specimen "
                          "(or clear it everywhere).")
        if len(assays - {None}) > 1:
            raise MLError(f"identity map {p}: specimen '{rs[0]['specimen_id']}' appears with different assays {sorted(assays)}; if "
                          "these slides come from the same sample, give every row the same assay (otherwise replicates of one "
                          "sample could land in different splits: leakage); if the code is reused for different samples, give "
                          "them distinct specimen ids.")
    return IdentityTable(p, sha256_bytes(raw), out, notes)


def apply_identity_map(table: Optional[IdentityTable], scans: list) -> tuple[dict, dict]:
    """(identity_map {source_id: {...}} for ``ml.splits``, report). Only rows marked confirmed enter the map."""
    report = {"rows": 0, "matched": 0, "unconfirmed_rows": [], "ambiguous_basename": [], "missing_on_disk": [],
              "unlisted_files": [], "conflicting_rows": [], "normalisation_notes": list(table.warnings) if table else []}
    if table is None:
        report["unlisted_files"] = [s.rel for s in scans]
        return {}, report
    by_rel: dict[str, SourceScan] = {}
    by_base: dict[str, set] = defaultdict(set)
    for s in scans:
        for r in [s.rel, *s.alias_paths]:
            by_rel[r] = s
            by_base[Path(r).name].add(s.source_id)
    srcs = {s.source_id: s for s in scans}
    imap: dict[str, dict] = {}
    matched: set[str] = set()
    report["rows"] = len(table.rows)
    for r in table.rows:
        f = r["file"]
        s = by_rel.get(f)
        if s is None and "/" not in f:
            cands = by_base.get(f, set())
            if len(cands) > 1:
                report["ambiguous_basename"].append(f)
                continue
            if cands:
                s = srcs[next(iter(cands))]
        if s is None:
            report["missing_on_disk"].append(f)
            continue
        report["matched"] += 1
        matched.add(s.source_id)
        if not r["confirmed"]:
            report["unconfirmed_rows"].append(f)
            continue
        entry = {k: r[k] for k in ("assay", "specimen_id", "replicate", "slide_id", "scan_session")}
        if s.source_id in imap and imap[s.source_id] != entry:
            report["conflicting_rows"].append(f)
            raise MLError(f"identity map: two rows give different identities for the same file ({s.rel}); fix the map")
        imap[s.source_id] = entry
    report["unlisted_files"] = [s.rel for s in scans if s.source_id not in matched]
    for k in report:
        if isinstance(report[k], list):
            report[k] = sorted(set(report[k]))
    return imap, report


# ---------------------------------------------------------------------------------------------- examined fields
def load_examined(path: str | Path) -> ex.ExaminedManifest:
    try:
        return ex.load_manifest(path)
    except ex.ExaminedError as exc:
        raise MLError(str(exc)) from exc
    except OSError as exc:
        raise MLError(f"examined-fields manifest {path}: {exc}")


def _lab_series_map(spec: Optional[Sequence[str]]) -> dict:
    out: dict[str, int] = {}
    for item in spec or []:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            name, sep, pg = part.partition("=")
            if not sep or not name.strip() or not PAGE_KEY_RE.fullmatch(pg.strip()):
                raise MLError(f"--lab-series expects SERIES=PAGE (e.g. T7B=0,T7A=1), got {part!r}")
            if name.strip() in out and out[name.strip()] != int(pg):
                raise MLError(f"--lab-series: series {name.strip()} is given two different pages")
            out[name.strip()] = int(pg)
    return out


def lab_csv_manifest(csv_paths: Sequence[Path], series_spec: Optional[Sequence[str]], scans: list, opts: BridgeOptions
                     ) -> tuple[ex.ExaminedManifest, dict]:
    """In-memory examined-fields manifest (source_kind sidecar_csv) from the lab's per-slide CSV(s): blank = not counted
    (no entry), 0 = counted zero, N = counted N. Field N is taken as ImageJ counter N-1 ONLY because the operator acknowledged
    it (--acknowledge-lab-csv-field-numbering); SERIES -> page ONLY as given on the command line."""
    from polony_app.store.csv_import import parse_csv
    if not opts.acknowledge_lab_csv_field_numbering:
        raise MLError("--lab-csv needs --acknowledge-lab-csv-field-numbering: it records that you confirm 'field N of the lab CSV = "
                      "ImageJ counter N-1' (this mapping was inferred from the data, not stated by the lab).")
    series = _lab_series_map(series_spec)
    if not series:
        raise MLError("--lab-csv needs --lab-series SERIES=PAGE (e.g. T7B=0,T7A=1): which TIFF page each CSV series counts is "
                      "never inferred")
    by_slide: dict[str, list] = defaultdict(list)
    for s in scans:
        sid = (s.identity or {}).get("slide_id")
        if sid:
            by_slide[sid].append(s)
    files: dict[str, dict] = {}
    info = {"csv_files": [], "rows": 0, "applied_rows": 0, "ambiguous_rows": [], "unmatched_rows": [], "series": series}
    used_pages: dict[str, str] = {}
    for cp in sorted(Path(c) for c in csv_paths):
        data = cp.read_bytes()
        imp_ = parse_csv(data, cp.name, import_id=cp.stem)
        ser = cp.stem.split("_")[0]
        if ser not in series:
            raise MLError(f"--lab-csv {cp.name}: its series '{ser}' (file-name prefix before the first '_') is not in --lab-series "
                          f"{sorted(series)}")
        page = series[ser]
        if str(page) in used_pages and used_pages[str(page)] != ser:
            raise MLError(f"--lab-series: series {used_pages[str(page)]} and {ser} both count page {page}")
        used_pages[str(page)] = ser
        info["csv_files"].append({"name": cp.name, "sha256": imp_.sha256, "series": ser, "page": page})
        for row in imp_.rows:
            info["rows"] += 1
            tg = by_slide.get(row.slide_id, [])
            if not tg:
                info["unmatched_rows"].append(f"{ser}:{row.slide_id}")
                continue
            if len(tg) > 1:
                info["ambiguous_rows"].append(f"{ser}:{row.slide_id}")
                continue
            info["applied_rows"] += 1
            ents = [ex.FieldEntry(counter=j, count=int(v)) for j, v in enumerate(row.fields) if v is not None]
            files.setdefault(tg[0].rel, {"sha": tg[0].sha256, "pages": {}})["pages"].setdefault(str(page), []).extend(ents)
    ev = ("lab CSV " + ", ".join(f"{c['name']} (sha256 {c['sha256'][:12]}, series {c['series']} = page {c['page']})"
                                 for c in info["csv_files"]) +
          ": blank = not counted, 0 = counted zero; field N = ImageJ counter N-1 (acknowledged by the operator)")
    try:
        m = ex.ExaminedManifest(evidence=ev, source_kind=ex.SRC_SIDECAR_CSV, examined_by="lab CSV", recorded_on="",
                                files=[ex.FileEntry(file=r, sha256=d["sha"], pages=d["pages"]) for r, d in sorted(files.items())],
                                sha256=hash_obj(info["csv_files"]))
    except ValueError as exc:      # pydantic ValidationError: e.g. one slide listed twice in the CSV(s)
        raise MLError(f"--lab-csv: the lab CSV(s) do not describe each (slide, series, field) exactly once: {exc}") from exc
    for k in ("ambiguous_rows", "unmatched_rows"):
        info[k] = sorted(set(info[k]))
    return m, info


def manifest_entries_for(manifest: ex.ExaminedManifest, scans: list, *, all_scans: Optional[list] = None) -> tuple[dict, list]:
    """{source_id: {page: {key: FieldEntry}}} and a list of problems. A manifest file entry names a source by relative path
    (or alias), else by base name when that name is unique; a recorded sha256 must match."""
    srcs = {s.source_id: s for s in scans}
    by_rel = {r: s for s in scans for r in [s.rel, *s.alias_paths]}
    by_base: dict[str, set] = defaultdict(set)
    for s in scans:
        for r in [s.rel, *s.alias_paths]:
            by_base[Path(r).name].add(s.source_id)
    known = {r for s in (all_scans if all_scans is not None else scans) for r in [s.rel, *s.alias_paths]}
    known_base = {Path(r).name for r in known}
    out: dict[str, dict] = {}
    problems: list[str] = []
    for fe in manifest.files:
        f = fe.file.replace("\\", "/")
        s = by_rel.get(f)
        if s is None:
            cands = by_base.get(Path(f).name, set()) if "/" not in f else set()
            if len(cands) > 1:
                raise MLError(f"examined-fields manifest: '{fe.file}' is ambiguous: {len(cands)} files share that base name "
                              f"({', '.join(sorted(srcs[c].rel for c in cands))}); use the relative path")
            if cands:
                s = srcs[next(iter(cands))]
        if s is None and (f in known or ("/" not in f and f in known_base)):
            problems.append(f"{fe.file}: listed in the examined-fields manifest but the file was rejected (see the rejected files)")
            continue
        if s is None:
            rels = sorted(known)
            near = (difflib.get_close_matches(f, rels, n=1, cutoff=0.0) or difflib.get_close_matches(
                Path(f).name, [Path(r).name for r in rels], n=1, cutoff=0.0)) if rels else []
            problems.append(f"'{printable_name(fe.file)[:200]}' is listed in the examined-fields manifest but matches no scanned TIFF"
                            + (f" (nearest existing file: '{near[0]}')" if near else ""))
            continue
        if fe.sha256 and fe.sha256.lower() != s.sha256.lower():
            raise MLError(f"examined-fields manifest: {fe.file} lists sha256 {fe.sha256[:12]}... but {s.rel} has {s.sha256[:12]}...; "
                          "refusing to apply evidence to a different file")
        tgt = out.setdefault(s.source_id, {})
        for pg, ents in fe.pages.items():
            d = tgt.setdefault(int(pg), {})
            for e in ents:
                if e.key() in d:
                    raise MLError(f"examined-fields manifest: {s.rel}: page {pg}: field {e.key()} is listed twice")
                d[e.key()] = e
    return out, problems


def resolve_entries(sc: SourceScan, page_entries: dict, opts: BridgeOptions) -> tuple[dict, list]:
    """{(page, cell): FieldEntry | _ConflictingEntries} and unresolved descriptions."""
    resolved: dict = {}
    unresolved: list[dict] = []
    clash: set = set()
    for page, ents in sorted(page_entries.items()):
        for key, e in sorted(ents.items(), key=lambda kv: repr(kv[0])):
            cell = None
            why = ""
            if key[0] == "counter":
                cell = sc.counter_cell.get(key[1])
                why = f"counter {key[1]} has no usable point on the grid, so its cell cannot be located"
            elif key[0] == "cell":
                r, c = key[1], key[2]
                if 0 <= r < sc.n_rows and 0 <= c < sc.n_cols:
                    cell = r * sc.n_cols + c
                else:
                    why = f"cell row {r} col {c} is outside the {sc.n_rows}x{sc.n_cols} grid"
            else:   # field N
                if opts.acknowledge_field_equals_counter_plus_one:
                    cell = sc.counter_cell.get(key[1] - 1)
                    why = f"field {key[1]} (= counter {key[1] - 1}) has no usable point on the grid"
                else:
                    why = (f"entry names the lab field {key[1]}; field N is only read as counter N-1 with "
                           "--acknowledge-field-equals-counter-plus-one")
            if cell is None:
                unresolved.append({"file": sc.rel, "page": page, "entry": list(key), "reason": why})
                continue
            if (page, cell) in resolved:
                clash.add((page, cell))
            resolved[(page, cell)] = e
    for pc in clash:
        resolved[pc] = None          # two entries address one cell: conflict
    return resolved, unresolved


# ---------------------------------------------------------------------------------------------- eligibility
@dataclass
class TupleResult:
    page: int
    cell: int
    exam: ex.Examination
    n_marks: int                 # researcher marks attributed to this field on this page (counter-based)
    labels: list                 # indices of points exported as labels (geometric cell == this cell)
    n_stray: int                 # labels that belong to another cell's counter
    clash: bool = False


def classify_source(sc: SourceScan, pages: Iterable[int], *, mode: str, manifest: Optional[ex.ExaminedManifest],
                    entries: dict) -> dict:
    """{(page, cell): TupleResult} for EVERY grid cell and every page in ``pages`` (independently), via
    :func:`polony_app.fields.examined.classify`. ``entries`` = {(page, cell): FieldEntry | None(=clash)}."""
    out: dict = {}
    n = sc.n_points
    # usable marks whose plane is unspecified (ImageJ position 0 = "all planes" / missing) or beyond the file's planes: their
    # polony may belong to ANY page, so every (page, cell) tuple of a cell holding one is a CONFLICT (never a verified zero,
    # never a crop with that polony unlabelled inside it)
    unplaced: dict[int, int] = defaultdict(int)
    for i in range(n):
        if sc.pstatus[i] != 0 or int(sc.cell[i]) < 0:
            continue
        pg = int(sc.page[i])
        if pg < 0 or pg >= sc.n_planes:
            unplaced[int(sc.cell[i])] += 1
    for page in pages:
        own: dict[int, list] = defaultdict(list)
        stray: dict[int, list] = defaultdict(list)
        extra: dict[int, list] = defaultdict(list)       # marks of the cell's counter that physically lie elsewhere
        for i in range(n):
            if sc.page[i] != page or sc.pstatus[i] != 0:
                continue
            c = int(sc.cell[i])
            ctr = int(sc.counter[i])
            owner = sc.counter_cell.get(ctr) if ctr >= 0 else None
            if c >= 0:
                if ctr < 0 or owner == c:
                    own[c].append(i)
                else:
                    stray[c].append(i)
            if owner is not None and owner != c:
                extra[owner].append(i)
        for cell in range(sc.n_cells):
            if cell in unplaced:
                out[(page, cell)] = TupleResult(
                    page, cell, ex.Examination(ex.CONFLICT, unplaced[cell], (), "", PLANE_CONFLICT_REASON), unplaced[cell],
                    [], 0)
                continue
            o, s, e = own.get(cell, []), stray.get(cell, []), extra.get(cell, [])
            has_entry = (page, cell) in entries
            entry = entries.get((page, cell))
            if has_entry and entry is None:
                res = TupleResult(page, cell, ex.Examination(ex.CONFLICT, len(o) + len(e), (), "",
                                                              "two manifest entries address the same cell"), len(o) + len(e),
                                  [], 0, clash=True)
                out[(page, cell)] = res
                continue
            n_attr = len(o) + len(e) if o else 0
            exam = ex.classify(n_attr, entry, mode=mode, manifest=manifest)
            if exam.status == ex.VERIFIED_ZERO and (o or s or e):
                # a stated zero is contradicted by marks of this field's counter or marks lying in the cell
                exam = ex.classify(len(o) + len(s) + len(e), entry, mode=mode, manifest=manifest)
                n_attr = len(o) + len(s) + len(e)
            elif not o and s and exam.status == ex.UNKNOWN:
                exam = ex.Examination(ex.UNKNOWN, 0, reason=STRADDLE_REASON)
            labels = sorted(o + s) if exam.researcher_examined else []
            out[(page, cell)] = TupleResult(page, cell, exam, n_attr, labels, len(s) if exam.researcher_examined else 0)
    return out


# ---------------------------------------------------------------------------------------------- crop building
class _PlaneCache:
    """Per-source cache of page planes; ``release`` frees them. Planes are read ONCE per page."""

    def __init__(self, sc: SourceScan, rehash: bool = True):
        self.sc = sc
        self.rehash = rehash                 # real import: hash the file again at the first plane read (dry runs never read planes)
        self.verified = False
        self.planes: dict[int, np.ndarray] = {}
        self.lock = threading.Lock()

    def _changed(self, what: str) -> MLError:
        return MLError(f"{self.sc.rel} changed on disk {what}; the crops would no longer match the recorded source sha256. "
                       "Nothing was published; rerun the import (the staging directory is kept and resumes).")

    def _check_unchanged(self, phase: str) -> None:
        st = self.sc.path.stat()
        sig = (self.sc.size, self.sc.mtime_ns, self.sc.ino, self.sc.ctime_ns)
        if _stat_sig(st) != sig and not (self.sc.ino == 0 and self.sc.ctime_ns == 0 and _stat_sig(st)[:2] == sig[:2]):
            raise self._changed(f"{phase} (size / modification time / inode / change time differ from the scan)")

    def plane(self, page: int) -> np.ndarray:
        with self.lock:
            if page not in self.planes:
                self._check_unchanged("after it was scanned")
                if self.rehash and not self.verified:
                    # the cheap stat check cannot see a same-size rewrite whose mtime was restored: hash the bytes again
                    # (streaming) at the first plane read of the source, and compare with the sha256 that is recorded
                    if sha256_stream(self.sc.path) != self.sc.sha256:
                        raise self._changed("after it was scanned (its content hash differs from the one recorded in the scan)")
                    self.verified = True
                a = read_plane(self.sc.path, page)
                self._check_unchanged("while its pixels were being read")
                if a.shape != (self.sc.height, self.sc.width) or str(a.dtype) != self.sc.dtype:
                    raise MLError(f"{self.sc.rel} page {page}: pixels {a.shape} {a.dtype} differ from the header "
                                  f"({self.sc.height}x{self.sc.width} {self.sc.dtype})")
                self.planes[page] = a
            return self.planes[page]

    def drop(self, page: int) -> None:
        with self.lock:
            self.planes.pop(page, None)

    def release(self) -> None:
        with self.lock:
            self.planes.clear()


@dataclass
class SourcePlan:
    scan: SourceScan
    ext: ds._Extract
    tuples: dict
    pages: dict                      # page -> {role, train}
    origin: str
    dropped: Counter
    straddling_kept: int
    n_edge: int
    skipped: list                    # [(page, cell, reason)]
    unresolved_entries: list
    n_positive_labels: int


def _assumptions_for(opts: BridgeOptions, sc: SourceScan, n_no_counter: int) -> list[str]:
    a = []
    if opts.acknowledge_no_counters and not sc.has_counters and sc.n_points:
        a.append("acknowledge_no_counters: the points carry no ImageJ counters; fields are geometric grid cells only")
    if opts.allow_unconfirmed_identity:
        a.append("allow_unconfirmed_identity")
    if opts.acknowledge_field_equals_counter_plus_one:
        a.append("acknowledge_field_equals_counter_plus_one: manifest 'field N' read as ImageJ counter N-1")
    if opts.acknowledge_lab_csv_field_numbering:
        a.append("acknowledge_lab_csv_field_numbering: lab CSV 'field N' read as ImageJ counter N-1")
    if n_no_counter:
        a.append(f"{n_no_counter} point(s) without a counter are attributed to their geometric cell")
    return a


VALID_MASK_TEXT = {
    "chamber": ("the app's automatic chamber (imaging.chamber.auto_chamber, rim 40 px, all planes of the file) as one counting "
                "region -> detection.masks.build_valid_mask; saturated pixels excluded: the SAME mask the app uses at inference"),
    "grid-cells": "union of ALL grid cells of the file (fallback when no chamber is found, or --valid-mask grid-cells)",
    "eligible-fields": ("union of the pixel boxes of the researcher-examined fields on this page (--valid-mask eligible-fields; "
                        "NOT the inference mask: a biased, marked-only subset)")}


def grid_cells_mask(shape_hw: tuple, boxes: Iterable[tuple]) -> np.ndarray:
    valid = np.zeros(shape_hw, bool)
    for x0, y0, x1, y1 in boxes:
        valid[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return valid


def chamber_valid_mask(planes: Sequence[np.ndarray]) -> tuple:
    """(mask | None, 'chamber' | 'grid-cells', why-not, extra info) exactly as the app's auto chamber -> counting region ->
    ``build_valid_mask``. ``mask`` is None (and 'grid-cells' returned) when no chamber could be found."""
    from polony_app.detection.masks import build_valid_mask
    from polony_app.imaging import chamber as chm
    try:
        res = chm.auto_chamber(list(planes), CHAMBER_RIM_PX)
    except chm.ChamberError as exc:
        return None, "grid-cells", str(exc), {}
    region = schemas.Region(id="chamber", kind=schemas.RegionKind.counting, name="automatic chamber",
                            polygon=[(float(px), float(py)) for px, py in res["polygon"]])
    mask = build_valid_mask(tuple(planes[0].shape), [region])
    if not mask.any():
        return None, "grid-cells", "the automatic chamber polygon covers no pixel", {}
    return mask, "chamber", None, {"chamber_area_px2": float(res["area_px2"]), "chamber_mask_channel_index": int(res["mask_channel_index"]),
                                    "chamber_polygon_vertices": len(res["polygon"])}


def plan_source(sc: SourceScan, *, pages: dict, origin: str, opts: BridgeOptions, manifest: Optional[ex.ExaminedManifest],
                page_entries: dict, ident: dict, identity_map_given: bool, valid_mask_report: Optional[dict] = None) -> SourcePlan:
    """Phase C for one source: eligibility, labels, crop candidates (lazy) and the points / regions rows."""
    valid_mask_report = {} if valid_mask_report is None else valid_mask_report
    train_pages = sorted(p for p, v in pages.items() if v["train"])
    entries, unresolved = resolve_entries(sc, page_entries, opts)
    tuples = classify_source(sc, train_pages, mode=opts.mode, manifest=manifest, entries=entries)
    cache = _PlaneCache(sc)
    n_no_counter = int((sc.counter < 0).sum()) if sc.n_points else 0
    assumptions = _assumptions_for(opts, sc, n_no_counter)
    mapping = {pages[p]["role"]: {"file": sc.rel, "page": int(p)} for p in train_pages}
    src = ds.SourceAnalysis(
        analysis_id=sc.source_id, name=sc.rel, status="historical", review_status="historical_imagej_marks",
        tiff_sha256={sc.rel: sc.sha256}, image_kind="stack" if sc.n_planes > 1 else "single_plane", dtype=sc.dtype,
        shape_hw=[sc.height, sc.width], channel_mapping=mapping or None, reviewers=["unknown"],
        reviewer_identity_source="unknown", provenance_engine="",
        n_regions_total=sc.n_cells, identity={})
    info = SourceInfo(analysis_id=sc.source_id, name=sc.rel, sha256=hash_obj({sc.rel: sc.sha256}), file_names=[sc.rel],
                      file_sha256s=[sc.sha256], width=sc.width, height=sc.height, dtype=sc.dtype, bit_depth=int(np.dtype(sc.dtype).itemsize * 8),
                      n_planes=sc.n_planes)
    warnings: list[str] = [f"{sc.rel}: {m}" for _c, m in sc.notes]
    if n_no_counter and sc.has_counters:
        warnings.append(f"{sc.rel}: {n_no_counter} point(s) carry no counter (attributed to their geometric cell)")
    ext = ds._Extract(src=src, info=info, rows=[], regions=[], cands=[], warnings=warnings, skipped_crops=[])

    role_of = {p: pages[p]["role"] for p in train_pages}
    dropped: Counter = Counter()
    label_cell: dict[int, tuple] = {}
    skipped: list = []
    cell_status: dict[int, dict] = defaultdict(dict)
    exported_cells: set = set()
    n_edge_total = 0
    n_positive = 0
    straddling_kept = 0
    cands = []
    for (page, cell), tr in sorted(tuples.items()):
        e = tr.exam
        rname = sc.cell_name(cell)
        cell_status[cell][str(page)] = {"role": role_of[page], "status": e.status, "reason": e.reason or None,
                                        "n_marks": int(tr.n_marks), "source": e.examination_source or None,
                                        "stated_count": e.stated_count}
        if not e.researcher_examined:
            continue
        rect = sc.cell_rect(cell)
        bx0, by0, bx1, by1 = pixel_box(rect)
        why = None
        if bx0 < 0 or by0 < 0 or bx1 > sc.width or by1 > sc.height:
            why = "crop box leaves the image (the grid cell extends beyond the pixel data; never clamped)"
        elif bx1 - bx0 < opts.min_crop_px or by1 - by0 < opts.min_crop_px:
            why = f"crop smaller than {opts.min_crop_px}px"
        if why:
            skipped.append((page, cell, why))
            ext.skipped_crops.append({"analysis_id": sc.source_id, "region_id": rname, "page": int(page), "reason": why})
            cell_status[cell][str(page)]["crop"] = {"skipped": why}
            for i in tr.labels:
                label_cell[i] = ("crop_not_exported", cell)
            continue
        w, h = bx1 - bx0, by1 - by0
        pos, edge = [], 0
        for i in tr.labels:
            cx, cy = float(sc.x[i]) - bx0, float(sc.y[i]) - by0
            if bx0 <= pixel_of(sc.x[i]) < bx1 and by0 <= pixel_of(sc.y[i]) < by1:
                pos.append((cy, cx, sc.object_ids[i]))
                label_cell[i] = ("label", cell)
            else:
                edge += 1
                label_cell[i] = ("outside_crop_footprint", cell)
        pos.sort(key=lambda t: (t[0], t[1], t[2]))
        n_edge_total += edge
        n_positive += len(pos)
        straddling_kept += tr.n_stray
        counters = sorted(sc.cell_counters.get(cell, []))
        r, c = sc.cell_rc(cell)
        prov = {
            "researcher_examined": True, "examination_source": e.examination_source, "evidence": e.evidence,
            "examination_status": e.status, "n_researcher_marks": int(tr.n_marks),
            "source_tiff": sc.rel, "source_sha256": sc.sha256, "page": int(page), "imagej_position": int(page) + 1,
            "channel_role": role_of[page], "channel_mapping_origin": origin,
            "field_counters": counters, "field_counter": (counters[0] if len(counters) == 1 else None),
            "grid_row": int(r), "grid_col": int(c), "field_id": f"{sc.source_id}:{rname}",
            "cell_polygon_canonical": [[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]],
            "coordinate_frame": COORDINATE_FRAME, "n_points_dropped_edge": int(edge),
            "n_labels_from_other_counters": int(tr.n_stray), "n_labels": len(tr.labels), "stated_count": e.stated_count,
            "assumptions": list(assumptions),
            "alias_paths": list(sc.alias_paths), "mode": opts.mode}
        region = schemas.Region(id=rname, kind=schemas.RegionKind.counting, name=f"field {rname}",
                                field_index=(counters[0] + 1 if len(counters) == 1 else None),
                                polygon=[(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])])
        cand = ds._CropCand(region=region, role=role_of[page], page=int(page), source_file=sc.rel, rect_ref=rect,
                            box_file=(bx0, by0, bx1, by1), bbox_only=False, points=[(a, b) for a, b, _o in pos], n_rejected=0,
                            n_edge=edge, crop=None, dy_dx=(0, 0), source_sha=sc.sha256, shape_hw=(h, w), dtype=sc.dtype,
                            provenance=prov)
        cand.loader = _make_loader(cache, page, bx0, by0, bx1, by1)
        cands.append(cand)
        exported_cells.add(cell)
    ext.cands = cands
    src.n_regions_fully_reviewed = len({cell for (_p, cell), tr in tuples.items() if tr.exam.researcher_examined})

    # ---- regions (one record per grid cell, auditable)
    for cell in range(sc.n_cells):
        r, c = sc.cell_rc(cell)
        rect = sc.cell_rect(cell)
        ext.regions.append({
            "analysis_id": sc.source_id, "region_id": sc.cell_name(cell), "name": f"field {sc.cell_name(cell)}",
            "kind": "counting", "field_index": (sc.cell_counters[cell][0] + 1 if len(sc.cell_counters.get(cell, [])) == 1 else None),
            "polygon": [[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]],
            "approved": False, "fully_reviewed": False, "rectangular": True, "crops": {},
            "grid_row": int(r), "grid_col": int(c), "counters": sorted(sc.cell_counters.get(cell, [])),
            "examination": cell_status.get(cell, {}), "origin": SOURCE_ORIGIN})

    # ---- point rows (confirmed labels, or dropped with a reason)
    cell_state = {(p, c): t.exam.status for (p, c), t in tuples.items()}
    for i in range(sc.n_points):
        pg, cl, ps = int(sc.page[i]), int(sc.cell[i]), int(sc.pstatus[i])
        role = role_of.get(pg)
        reason = None
        status = "dropped"
        if ps == 2:
            reason = "out_of_bounds"
        elif ps == 1:
            reason = "duplicate"
        elif cl < 0:
            reason = "outside_grid"
        elif pg < 0:
            # the importer leaves page unset for a position beyond the planes as well as for a missing / zero one
            reason = "plane_out_of_range" if int(sc.position[i]) > sc.n_planes else "plane_unspecified"
        elif pg >= sc.n_planes:
            reason = "plane_out_of_range"
        elif pg not in role_of:
            reason = "page_not_trained" if pg in pages else "page_unmapped"
        elif i in label_cell:
            kind, _c = label_cell[i]
            if kind == "label":
                status, reason = "confirmed", None
            else:
                reason = kind
        else:
            st = cell_state.get((pg, cl))
            tr = tuples.get((pg, cl))
            if tr is not None and tr.exam.reason == STRADDLE_REASON:
                reason = "cell_not_examined:straddling_other_counter"
            else:
                reason = f"cell_not_examined:{st or 'unknown'}"
        if status == "dropped":
            dropped[reason] += 1
        ext.rows.append({"object_id": sc.object_ids[i], "role": role, "page": (None if pg < 0 else pg),
                         "x": float(sc.x[i]), "y": float(sc.y[i]), "status": status,
                         "basis": LABEL_BASIS if status == "confirmed" else f"dropped:{reason}", "cls": "",
                         "origin": SOURCE_ORIGIN, "region_id": (sc.cell_name(cl) if cl >= 0 else None),
                         "consensus_status": None, "footprint_cell": True})

    if not cands:
        why = (R_NO_ELIGIBLE, "no (field, page) tuple of this file is proven examined (" + "; ".join(
            f"{k}: {v}" for k, v in sorted(Counter(t.exam.status for t in tuples.values()).items())) + ")")
        src.skipped = f"{why[0]}: {why[1]}"
        ext.warnings.append(f"{sc.rel}: skipped - {src.skipped}")

    # ---- hooks: read each page plane once, page normalisation over the app's valid mask, then release
    boxes_by_page: dict[int, list] = defaultdict(list)
    for (page, cell), tr in tuples.items():
        if tr.exam.researcher_examined:
            boxes_by_page[page].append(pixel_box(sc.cell_rect(cell)))
    all_boxes = [pixel_box(sc.cell_rect(cell)) for cell in range(sc.n_cells)]

    def prepare() -> None:
        """Statistics only (no label and no test-set information enters): lo/hi of every trained page over the SAME valid mask
        the app uses at inference, computed ONCE per source (planes are read once and released afterwards)."""
        pages_needed = sorted({c.page for c in cands})
        mask, used, why, extra = (None, None, None, {})
        if opts.valid_mask == "chamber":
            planes = [cache.plane(p) for p in range(sc.n_planes)]
            mask, used, why, extra = chamber_valid_mask(planes)
            if used != "chamber":
                ext.late_warnings.append(f"{sc.rel}: no automatic chamber ({why}); page lo/hi use the union of all "
                                         "grid cells instead (valid_mask_used = grid-cells)")
            for p in range(sc.n_planes):
                if p not in pages_needed:
                    cache.drop(p)                      # only the planes that become crops stay in memory
            if used != "chamber":
                mask = grid_cells_mask((sc.height, sc.width), all_boxes)
        elif opts.valid_mask == "grid-cells":
            used = "grid-cells"
            mask = grid_cells_mask((sc.height, sc.width), all_boxes)
        done: dict[str, dict] = {}
        for page in pages_needed:
            plane = cache.plane(page)
            if opts.valid_mask == "eligible-fields":
                valid = grid_cells_mask(plane.shape, boxes_by_page[page])
                used = "eligible-fields"
            else:
                valid = mask
            b = page_bounds(plane, valid, opts.percentiles)
            role = role_of[page]
            b.update({"file": sc.rel, "page": int(page), "mode": opts.normaliser, "policy": LO_HI_POLICY,
                      "valid_mask_option": opts.valid_mask, "valid_mask_used": used, "valid_mask_fallback": why,
                      "valid_mask": VALID_MASK_TEXT[used], "valid_mask_pixels": int(valid.sum())})
            b.update(extra)
            if b["fallback"]:
                ext.late_warnings.append(f"{sc.rel} page {page} ({role}): {b['fallback']}")
            done[role] = b
            for cd in cands:
                if cd.page == page:
                    cd.norm = b
        src.page_normalization = done

    def finalize() -> None:
        vm = valid_mask_report.setdefault("by_source", {})
        if src.page_normalization:
            vm[sc.rel] = {role: {k: b.get(k) for k in ("valid_mask_option", "valid_mask_used", "valid_mask_fallback", "fallback",
                                                         "lo", "hi", "n_used", "n_valid", "chamber_area_px2")}
                          for role, b in sorted(src.page_normalization.items())}
            uc = valid_mask_report.setdefault("used_counts", {})
            used = sorted({b.get("valid_mask_used") for b in src.page_normalization.values()})
            for u in used:
                uc[u] = uc.get(u, 0) + 1
            for role, b in sorted(src.page_normalization.items()):
                if b.get("valid_mask_fallback"):
                    valid_mask_report.setdefault("chamber_fallbacks", []).append(
                        {"file": sc.rel, "used": b.get("valid_mask_used"), "why": b["valid_mask_fallback"]})
                    break
            for role, b in sorted(src.page_normalization.items()):
                if b.get("fallback"):
                    valid_mask_report.setdefault("page_bounds_fallbacks", []).append(
                        {"file": sc.rel, "page": b.get("page"), "role": role, "why": b["fallback"]})

    ext.prepare = prepare if cands else None
    ext.release = cache.release
    ext.finalize = finalize if cands else None

    return SourcePlan(scan=sc, ext=ext, tuples=tuples, pages=pages, origin=origin, dropped=dropped,
                      straddling_kept=straddling_kept, n_edge=n_edge_total, skipped=skipped, unresolved_entries=unresolved,
                      n_positive_labels=n_positive)


def _make_loader(cache: _PlaneCache, page: int, x0: int, y0: int, x1: int, y1: int) -> Callable[[], np.ndarray]:
    def load() -> np.ndarray:
        return np.ascontiguousarray(cache.plane(page)[y0:y1, x0:x1])
    return load


# ---------------------------------------------------------------------------------------------- provenance gate
_EXAMINED_STATUSES = (ex.POSITIVE, ex.VERIFIED_ZERO)
_REQUIRED_PROV = ("examination_source", "evidence", "source_tiff", "source_sha256", "page", "imagej_position", "channel_role",
                  "channel_mapping_origin", "grid_row", "grid_col", "field_id", "cell_polygon_canonical", "coordinate_frame",
                  "n_points_dropped_edge", "n_researcher_marks")


def _check_prov(cid: str, prov: dict) -> list[str]:
    bad = []
    if prov.get("researcher_examined") is not True:
        bad.append(f"crop {cid}: provenance does not say researcher_examined == true")
        return bad
    if prov.get("examination_status") not in _EXAMINED_STATUSES:
        bad.append(f"crop {cid}: examination_status {prov.get('examination_status')!r} is not positive_marked / verified_zero")
    for k in _REQUIRED_PROV:
        if k not in prov or prov[k] in (None, ""):
            if k in ("evidence",) and prov.get("examination_status") == ex.POSITIVE:
                continue
            bad.append(f"crop {cid}: provenance lacks '{k}'")
    return bad


def assert_examined(cands: Iterable) -> None:
    """Final gate before assembling: refuse any crop candidate without researcher_examined == true."""
    for c in cands:
        bad = _check_prov(f"{c.source_file}:p{c.page}:{c.region.id}", c.provenance or {})
        if bad:
            raise MLError("REFUSED to export a crop that is not proven researcher-examined: " + "; ".join(bad))


_ZERO_SOURCES = (ex.SRC_SIDECAR_CSV, ex.SRC_SIDECAR_XLSX, ex.SRC_MANIFEST)


def _check_consistency(cid: str, prov: dict, n_positive: int, crop_sha: Optional[str], strict: bool = True) -> list[str]:
    """Status, counts and evidence of one crop must agree (a verified zero with positives, or a positive crop without marks, is
    a corrupted or hand-edited manifest)."""
    bad = []
    status = prov.get("examination_status")
    marks = prov.get("n_researcher_marks")
    labels = prov.get("n_labels")
    edge = prov.get("n_points_dropped_edge") or 0
    if not isinstance(marks, int) or isinstance(marks, bool):
        bad.append(f"crop {cid}: n_researcher_marks is not an integer")
        return bad
    if labels is not None and n_positive + int(edge) != labels:
        bad.append(f"crop {cid}: {n_positive} positive(s) + {edge} edge-dropped != {labels} label(s) inside the field")
    if status == ex.POSITIVE:
        if marks <= 0:
            bad.append(f"crop {cid}: positive_marked but n_researcher_marks is {marks}")
        if n_positive <= 0 and not edge:
            bad.append(f"crop {cid}: positive_marked but it has no positive label")
        if ex.SRC_EMBEDDED not in str(prov.get("examination_source") or ""):
            bad.append(f"crop {cid}: positive_marked but the evidence source is {prov.get('examination_source')!r}, not embedded marks")
    elif status == ex.VERIFIED_ZERO:
        if marks != 0 or n_positive != 0 or (labels not in (None, 0)):
            bad.append(f"crop {cid}: verified_zero but it has {marks} mark(s) / {n_positive} positive(s) / {labels} label(s)")
        if (strict or "stated_count" in prov) and prov.get("stated_count") != 0:
            bad.append(f"crop {cid}: verified_zero but no stated zero is recorded (stated_count {prov.get('stated_count')!r})")
        if not any(k in str(prov.get("examination_source") or "").split("+") for k in _ZERO_SOURCES):
            bad.append(f"crop {cid}: verified_zero but the evidence source {prov.get('examination_source')!r} is not a sidecar / manifest")
        if not str(prov.get("evidence") or "").strip():
            bad.append(f"crop {cid}: verified_zero without a stated evidence text")
    if crop_sha is not None and prov.get("source_sha256") != crop_sha:
        bad.append(f"crop {cid}: provenance source_sha256 differs from the crop's source_sha256")
    return bad


def is_legacy_import(manifest) -> bool:
    """True when a snapshot manifest (SnapshotManifest or dict) is - or claims to have been - a bridge import: it records
    ``options.legacy_import`` OR a non-empty ``import_summary`` OR any of its own (not carried-over frozen) crops carries
    provenance. Deleting one of these marks from a hand-edited manifest therefore never turns an import into a plain snapshot."""
    d = manifest.model_dump(mode="json") if hasattr(manifest, "model_dump") else dict(manifest)
    if "legacy_import" in (d.get("options") or {}) or d.get("import_summary"):
        return True
    return any(c.get("provenance") for c in d.get("crops", []) if not c.get("frozen_from"))


def verify_examination_provenance(manifest) -> list[str]:
    """Problems found in the crops of a legacy-import snapshot manifest (empty list = every train / val / test crop is proven
    researcher-examined AND its status, counts, evidence and source hash agree). ``manifest`` = SnapshotManifest or its dict.
    Snapshots that are not legacy imports (see :func:`is_legacy_import`) are out of scope and return []."""
    d = manifest.model_dump(mode="json") if hasattr(manifest, "model_dump") else dict(manifest)
    if not is_legacy_import(d):
        return []
    bad: list[str] = []
    if (d.get("import_summary") or {}).get("kind") != KIND:
        bad.append("the snapshot is a historical-TIFF import but its import_summary is missing or not a legacy-import summary "
                   "(the import can no longer be cross-checked)")
    if "legacy_import" not in (d.get("options") or {}):
        bad.append("the snapshot is a historical-TIFF import but options.legacy_import is missing")
    for c in d.get("crops", []):
        cid = c.get("crop_id", "?")
        prov = c.get("provenance") or {}
        found = _check_prov(cid, prov)
        bad += found
        if not found:
            bad += _check_consistency(cid, prov, int(c.get("n_positive") or 0), c.get("source_sha256"),
                                      strict=not c.get("frozen_from"))
    return bad


# ---------------------------------------------------------------------------------------------- the plan
@dataclass
class ImportPlan:
    scan: ScanResult
    opts: BridgeOptions
    exts: list                                  # _Extract of every kept source (rejected ones with a skip reason)
    plans: list                                 # SourcePlan of accepted sources
    summary: dict
    blockers: list
    identity_map: dict
    ga: object = None
    warnings: list = field(default_factory=list)
    extra_assumptions: list = field(default_factory=list)
    resume_key: str = ""
    n_cands: int = 0
    mapped: bool = True
    valid_mask_report: dict = field(default_factory=dict)
    hard_refusals: list = field(default_factory=list)


def _identity_of(sc: SourceScan, imap: dict) -> dict:
    return resolve_identity(SourceInfo(analysis_id=sc.source_id, name=sc.rel, sha256="", file_names=[sc.rel]), imap)


def _rejected_ext(sc: SourceScan) -> ds._Extract:
    code, msg = sc.reasons[0]
    src = ds.SourceAnalysis(analysis_id=sc.source_id, name=sc.rel, status="historical", review_status="historical_imagej_marks",
                            tiff_sha256={sc.rel: sc.sha256}, dtype=sc.dtype, shape_hw=[sc.height, sc.width],
                            skipped=f"{code}: {msg}")
    return ds._Extract(src=src, info=None, rows=[], regions=[], cands=[], warnings=[f"{sc.rel}: rejected - {code}: {msg}"],
                       skipped_crops=[])


def build_plan(root: Path, opts: BridgeOptions, *, channel_map: Optional[Path] = None, identity_map: Optional[Path] = None,
               examined_fields: Optional[Path] = None, lab_csv: Sequence[Path] = (), lab_series: Optional[Sequence[str]] = None,
               dry_run: bool = False, data_dir: Optional[Path] = None, say: Say = print) -> ImportPlan:
    opts.check()
    root = Path(root)
    if not root.exists():
        raise MLError(f"--input {printable_name(str(root))} does not exist")
    if utf8_name_problem(root.name):
        raise MLError(f"--input {printable_name(str(root))}: the name of the input directory is not valid UTF-8 and is recorded in "
                      "the manifest; rename it or use a UTF-8 locale")
    if opts.mode == ex.MODE_MANIFEST and not (examined_fields or lab_csv):
        raise MLError("--mode examined-fields-manifest needs --examined-fields FILE (or --lab-csv): only the tuples the manifest "
                      "lists can be used; nothing is inferred")
    if examined_fields and lab_csv:
        raise MLError("give ONE examined-fields source: --examined-fields or --lab-csv, not both")
    series = _lab_series_map(lab_series)              # a malformed --lab-series is refused even when it would not be used
    if lab_series and not lab_csv:
        raise MLError("--lab-series was given without --lab-csv: it only says which TIFF page each lab-CSV series counts; "
                      "give the CSV (and --acknowledge-lab-csv-field-numbering) or drop the option")
    if lab_csv and not series:
        raise MLError("--lab-csv needs --lab-series SERIES=PAGE (e.g. T7B=0,T7A=1): which TIFF page each CSV series counts is "
                      "never inferred")
    if lab_csv and not opts.acknowledge_lab_csv_field_numbering:
        raise MLError("--lab-csv needs --acknowledge-lab-csv-field-numbering: it records that you confirm 'field N of the lab CSV = "
                      "ImageJ counter N-1' (this mapping was inferred from the data, not stated by the lab).")
    if opts.acknowledge_lab_csv_field_numbering and not lab_csv:
        raise MLError("--acknowledge-lab-csv-field-numbering was given without --lab-csv: there is no lab CSV whose field numbering "
                      "it could confirm (the flag would be recorded as an assumption that nothing uses)")
    cm = load_channel_map(channel_map) if channel_map else None
    table = load_identity_map(identity_map, level=opts.level) if identity_map else None
    man = load_examined(examined_fields) if examined_fields else None

    scan = scan_tree(root, opts, say)
    kept = scan.sources
    # ---- identity (map > file-name inference, which stays UNCONFIRMED)
    imap, idrep = apply_identity_map(table, kept)
    for sc in kept:
        sc.identity = _identity_of(sc, imap)
    lab_info = None
    if lab_csv:
        man, lab_info = lab_csv_manifest([Path(p) for p in lab_csv], lab_series, [s for s in kept if not s.rejected], opts)
    entries_by_src, man_problems = (manifest_entries_for(man, [s for s in kept if not s.rejected], all_scans=kept)
                                    if man else ({}, []))
    nameless = [x for x in man_problems if "matches no scanned TIFF" in x] if examined_fields else []

    # ---- mapping + eligibility
    blockers: list[str] = []
    mapped = cm is not None
    if cm is None:
        blockers.append("no --channel-map: which page shows which channel is never inferred; supply a confirmed channel map "
                        "(see --write-templates)")
    if cm is not None:
        blockers += channel_map_unmatched(cm, kept)
    exts, plans = [], []
    valid_mask_report: dict = {"option": opts.valid_mask, "chamber_rim_px": CHAMBER_RIM_PX, "by_source": {}}
    map_origin: Counter = Counter()
    per_file_map: dict = {}
    for sc in kept:
        if sc.rejected:
            exts.append(_rejected_ext(sc))
            continue
        pages = origin = None
        if cm is not None:
            got = resolve_mapping(cm, sc)
            if got is None:
                sc.reasons.append((R_MAP, "no default, rule or file entry of the channel map covers this file"))
            else:
                pages, origin = got
                bad = [p for p in pages if p >= sc.n_planes]
                if bad:
                    sc.reasons.append((R_MAP_PAGE, f"the channel map ({origin}) maps page(s) {bad} but the file has {sc.n_planes} plane(s)"))
                elif not any(v["train"] for v in pages.values()):
                    sc.reasons.append((R_NO_PAGES, "no page of this file is mapped with train: true"))
        else:
            pages, origin = {p: {"role": f"page{p}", "train": True} for p in range(sc.n_planes)}, "unmapped"
        if sc.rejected:
            exts.append(_rejected_ext(sc))
            continue
        map_origin[origin] += 1
        per_file_map[sc.rel] = {"origin": origin, "pages": {str(p): dict(v) for p, v in pages.items()}}
        sp = plan_source(sc, pages=pages, origin=origin, opts=opts, manifest=man, page_entries=entries_by_src.get(sc.source_id, {}),
                         ident=sc.identity, identity_map_given=table is not None, valid_mask_report=valid_mask_report)
        plans.append(sp)
        exts.append(sp.ext)
    # a mapping-less dry run must not look like a plan: nothing can be exported
    n_cands = sum(len(p.ext.cands) for p in plans) if cm is not None else 0
    if cm is None:
        for p in plans:
            p.ext.cands = []
            p.ext.prepare = None
    assert_examined(c for p in plans for c in p.ext.cands)

    plan = ImportPlan(scan=scan, opts=opts, exts=exts, plans=plans, summary={}, blockers=blockers, identity_map=imap,
                      n_cands=n_cands, mapped=mapped, valid_mask_report=valid_mask_report)
    # mapping entries that name no TIFF at all (a stale or mistyped file list): `import_legacy` refuses them even in a dry run
    # (a read-only caller such as the preflight can still report them from here)
    plan.hard_refusals = [b for b in blockers if b.startswith("channel map files[")]
    if nameless:
        plan.hard_refusals.append("examined-fields manifest " + str(examined_fields) + ": " + "; ".join(nameless[:5])
                                  + (f" (+{len(nameless) - 5} more)" if len(nameless) > 5 else "")
                                  + ". Use the exact relative path below --input (or a unique base name).")
    live = [p for p in plans if p.ext.cands]
    # ---- identity gate (every source with at least one examined tuple; known even before a channel map exists)
    eligible = [p for p in plans if any(t.exam.researcher_examined for t in p.tuples.values())]
    uncon = [p.scan for p in eligible if not p.scan.identity["confirmed"]]
    if uncon and not opts.allow_unconfirmed_identity:
        blockers.append(f"{len(uncon)} file(s) have an UNCONFIRMED identity (inferred from the file name, or not in the identity "
                        "map): a split on it cannot exclude leakage between replicates. Supply --identity-map (see "
                        "--write-templates) or pass --allow-unconfirmed-identity (recorded)")
    if cm is not None and not live:
        blockers.append("no crop can be exported: no (file, field, page) tuple is proven researcher-examined and mapped")
    for k in ("ambiguous_basename", "missing_on_disk", "unlisted_files", "unconfirmed_rows"):
        pass
    plan.warnings += [f"identity map: {n}" for n in (table.warnings if table else [])]
    plan.warnings += [f"{rel}: {why}" for rel, why in scan.skipped_links]
    plan.warnings += [w for p in plans for w in p.ext.warnings]
    plan.extra_assumptions = _plan_assumptions(opts, plans, uncon)
    _fill_summary(plan, cm, table, man, lab_info, idrep, man_problems, per_file_map, map_origin, root, examined_fields,
                  channel_map, identity_map, lab_csv, data_dir)
    return plan


def _plan_assumptions(opts: BridgeOptions, plans: list, uncon: list) -> list[str]:
    a = ["historical ImageJ marks are one unknown person's clicks: precision is measured against that annotator; "
         "unmarked polonies inside a crop count as false positives",
         "a field/page is used only with proof the researcher examined it; nothing outside such a field is a negative"]
    if opts.acknowledge_no_counters and any(not p.scan.has_counters and p.scan.n_points for p in plans):
        a.append("--acknowledge-no-counters: files without any ImageJ counter were accepted with geometric grid cells only")
    if opts.allow_unconfirmed_identity and uncon:
        a.append(f"--allow-unconfirmed-identity: {len(uncon)} file(s) were split on identities inferred from file names")
    if opts.acknowledge_field_equals_counter_plus_one:
        a.append("--acknowledge-field-equals-counter-plus-one: manifest 'field N' = ImageJ counter N-1")
    if opts.acknowledge_lab_csv_field_numbering:
        a.append("--acknowledge-lab-csv-field-numbering: lab CSV field N = ImageJ counter N-1")
    a.append("reviewer identity of the historical marks is not recorded in the files (unknown)")
    return a


def _fill_summary(plan: ImportPlan, cm, table, man, lab_info, idrep, man_problems, per_file_map, map_origin, root, examined_path,
                  channel_path, identity_path, lab_csv, data_dir) -> None:
    opts, scan = plan.opts, plan.scan
    kept = scan.sources
    rejected = [s for s in kept if s.rejected]
    n_alias = sum(len(s.alias_paths) for s in kept)
    by_reason = Counter(s.reasons[0][0] for s in rejected)
    live = [p for p in plan.plans if p.ext.cands]
    tally = ex.ExaminationTally()
    by_page: dict = defaultdict(lambda: Counter())
    ambiguous = 0
    for p in plan.plans:
        for (page, cell), tr in p.tuples.items():
            tally.add(tr.exam)
            by_page[page][tr.exam.status] += 1
            if tr.exam.status == ex.UNKNOWN and tr.n_marks == 0 and tr.exam.reason != STRADDLE_REASON:
                ambiguous += 1
    dropped: Counter = Counter()
    n_points = labels = straddle = 0
    for p in plan.plans:
        dropped.update(p.dropped)
        n_points += p.scan.n_points
        labels += p.n_positive_labels
        straddle += p.straddling_kept
        n_edge = p.n_edge
        if n_edge:
            dropped["outside_crop_footprint"] += 0   # already counted through label_cell
    crops_by_pr: dict = defaultdict(lambda: {"positives": 0, "crops": 0})
    skipped_reasons: Counter = Counter()
    for p in plan.plans:
        for c in p.ext.cands:
            d = crops_by_pr[(c.page, c.role)]
            d["crops"] += 1
            d["positives"] += len(c.points)
        if cm is None:       # no channel map: the roles are unknown, the labels per page are still known
            for (pg, _cell), tr in p.tuples.items():
                if tr.exam.researcher_examined:
                    d = crops_by_pr[(pg, None)]
                    d["crops"] += 1
                    d["positives"] += len(tr.labels)
        for _pg, _cell, why in p.skipped:
            skipped_reasons[why] += 1
    # ---- splits (the same call assemble_snapshot will make)
    splits = None
    ga = None
    if live:
        infos = [p.ext.info for p in live]
        lock = load_lock(data_dir) if data_dir else None
        ga = assign_groups(infos, seed=opts.seed, fractions=opts.fractions, level=opts.level, allow_unconfirmed=True,
                           identity_map=plan.identity_map, lock=lock)
        plan.ga = ga
        by_split: dict = {}
        for p in live:
            aid = p.scan.source_id
            if aid in ga.locked_analyses:
                continue
            sp = ga.split_of[aid]
            d = by_split.setdefault(sp, {"crops": 0, "positives": 0, "groups": set(), "sources": 0})
            d["crops"] += len(p.ext.cands)
            d["positives"] += sum(len(c.points) for c in p.ext.cands)
            d["groups"].add(ga.group_of[aid])
            d["sources"] += 1
        splits = {"seed": ga.seed, "level": ga.level, "fractions": ga.fractions,
                  "groups": len({ga.group_of[p.scan.source_id] for p in live}),
                  "by_split": {k: {"crops": v["crops"], "positives": v["positives"], "groups": len(v["groups"]),
                                   "sources": v["sources"]} for k, v in sorted(by_split.items())},
                  "locked_sources": len(ga.locked_analyses), "unconfirmed_identity": bool(ga.unconfirmed)}
    # ---- estimates
    cands = [c for p in live for c in p.ext.cands]
    disk = sum(c.size_bytes() + 400 for c in cands)
    est_train = None
    if ga is not None:
        def shapes(sp):
            return [c.shape_hw for p in live if p.scan.source_id not in ga.locked_analyses and ga.split_of[p.scan.source_id] == sp
                    for c in p.ext.cands]
        trs, vas = shapes("train"), shapes("val")
        est_train = estimate_training(len(trs), len(vas), opts.batch_size, opts.epochs, None, opts.downscale,
                                      train_shapes_hw=trs, val_shapes_hw=vas)
    elig = [p for p in plan.plans if any(t.exam.researcher_examined for t in p.tuples.values())]
    id_src = Counter((p.scan.identity or {}).get("source", "unknown") for p in elig)
    id_conf = sum(1 for p in elig if (p.scan.identity or {}).get("confirmed"))
    unres_entries = [u for p in plan.plans for u in p.unresolved_entries]
    page_role = Counter()
    for rel, d in per_file_map.items():
        for pg, v in d["pages"].items():
            page_role[(int(pg), v["role"], bool(v["train"]))] += 1
    summary = {
        "kind": KIND, "mode": opts.mode, "input_root_name": Path(root).name, "options": opts.public(),
        "input_files": {"channel_map": _file_info(channel_path), "identity_map": _file_info(identity_path),
                        "examined_fields": _file_info(examined_path),
                        "lab_csv": [_file_info(Path(p)) for p in sorted(lab_csv)] if lab_csv else []},
        "tiffs": {"discovered": scan.discovered, "unique": len(kept), "accepted": len(kept) - len(rejected),
                  "rejected": len(rejected), "deduplicated": n_alias,
                  "rejected_by_reason": dict(sorted(by_reason.items())),
                  "rejected_files": {s.rel: [c for c, _m in s.reasons] for s in rejected},
                  "rejected_messages": {s.rel: [m for _c, m in s.reasons] for s in rejected},
                  "deduplicated_paths": {k: sorted(v) for k, v in sorted(scan.aliases.items())},
                  "skipped_links": [{"path": r, "reason": w} for r, w in scan.skipped_links]},
        "sources": {"with_crops": len(live), "without_crops": len(plan.plans) - len(live),
                    "without_crops_reasons": {R_NO_ELIGIBLE: len(plan.plans) - len(live)} if len(plan.plans) - len(live) else {}},
        "channel_mappings": {"mapped": cm is not None, "map": (cm.describe() if cm else None),
                             "origin_counts": dict(sorted(map_origin.items())),
                             "by_page_role": [{"page": pg, "role": role, "train": tr, "files": n}
                                              for (pg, role, tr), n in sorted(page_role.items())],
                             "per_file": per_file_map},
        "fields": dict(tally.as_dict(), excluded_unproven=tally.excluded_unproven, ambiguous_zero_vs_unreviewed=ambiguous,
                       tuples_total=tally.positive_marked + tally.verified_zero + tally.unknown + tally.conflict,
                       by_page={str(k): dict(sorted(v.items())) for k, v in sorted(by_page.items())}),
        "crops": {"created": len(cands), "skipped": sum(skipped_reasons.values()),
                  "skipped_reasons": dict(sorted(skipped_reasons.items()))},
        "positives_per_page": [{"page": pg, "role": role, "positives": v["positives"], "crops": v["crops"]}
                               for (pg, role), v in sorted(crops_by_pr.items())],
        "annotations": {"points_total": n_points, "labels_exported": labels,
                        "dropped_by_reason": dict(sorted((k, v) for k, v in dropped.items() if v)),
                        "dropped_total": sum(dropped.values()),
                        "straddling_labels_in_examined_cells": straddle},
        "splits": splits,
        "estimates": {"native_disk_bytes": int(disk), "native_disk": ds.fmt_bytes(disk), "training": est_train},
        "identity": {"sources": dict(sorted(id_src.items())), "confirmed": id_conf, "unconfirmed": len(elig) - id_conf,
                     "map": idrep, "allow_unconfirmed_identity": opts.allow_unconfirmed_identity},
        "valid_mask": plan.valid_mask_report,
        "unresolved_manifest_entries": unres_entries + [{"problem": m} for m in man_problems],
        "lab_csv": lab_info,
        "info_label_check": _info_label_check(plan, per_file_map),
        "assumptions": plan.extra_assumptions,
    }
    plan.summary = _jsonable(summary)
    plan.summary["valid_mask"] = plan.valid_mask_report       # the SAME object: filled in while the sources are written
    lock = load_lock(data_dir) if data_dir else None
    key = {"files": sorted((s.rel, s.sha256) for s in kept), "options": {k: v for k, v in opts.public().items()
                                                                          if k not in ("batch_size", "epochs")},
           "inputs": plan.summary["input_files"], "app": f"legacy-bridge-{LOGIC_VERSION}", "code": code_fingerprint(),
           "lock": (lock or {}).get("lock_sha256")}
    plan.resume_key = hash_obj(key)[:20]
    plan.summary["plan_key"] = plan.resume_key


def code_fingerprint() -> str:
    """sha256 over the source text of the modules that decide labels, eligibility and normalisation (this file, dataset.py,
    common.py) plus LOGIC_VERSION: part of the resume key, so a staging directory written by other code is never reused."""
    h = hashlib.sha256(f"logic-{LOGIC_VERSION}".encode())
    here = Path(__file__).resolve().parent
    for name in ("legacy_bridge.py", "dataset.py", "common.py"):
        try:
            h.update(name.encode() + b"\0" + (here / name).read_bytes())
        except OSError:           # pragma: no cover (zipped install): fall back to the version constant alone
            h.update(name.encode() + b"\0-")
    return h.hexdigest()[:20]


def _info_label_check(plan: ImportPlan, per_file_map: dict) -> dict:
    """INFORMATION ONLY: is the ImageJ Info label hint ('635 [W1]') of page 0 present / absent / different from the usual hint of
    the files that map the same role to page 0? Never decides anything; lists deviating files so overrides can be added."""
    by_role: dict[str, Counter] = defaultdict(Counter)
    rows = []
    for p in plan.plans:
        pm = per_file_map.get(p.scan.rel)
        if not pm or "0" not in pm["pages"]:
            continue
        role = pm["pages"]["0"]["role"]
        rows.append((p.scan.rel, role, p.scan.info_hint))
        if p.scan.info_hint:
            by_role[role][p.scan.info_hint] += 1
    usual = {r: c.most_common(1)[0][0] for r, c in by_role.items()}
    absent = sorted(rel for rel, _r, h in rows if not h)
    deviating = sorted(rel for rel, r, h in rows if h and r in usual and h != usual[r])
    return {"note": "information only; never used to decide a role", "usual_hint_by_page0_role": usual,
            "present": len(rows) - len(absent), "absent": absent, "different": deviating,
            "hints": {r: dict(sorted(c.items())) for r, c in by_role.items()}}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return o


# ---------------------------------------------------------------------------------------------- run
@dataclass
class ImportOutcome:
    plan: ImportPlan
    result: Optional[ds.BuildResult]
    blockers: list
    already_imported: Optional[str] = None       # version of a published snapshot built from this exact plan (nothing written)


def find_published(data_dir: Path, plan_key: str) -> Optional[str]:
    """Version of a published snapshot whose ``import_summary.plan_key`` equals ``plan_key`` and that verifies, else None."""
    for v in ds.list_snapshots(data_dir):
        try:
            m = ds.load_manifest(data_dir, v)
        except Exception:  # noqa: BLE001 - unreadable snapshots are ignored here (ml verify reports them)
            continue
        if (m.import_summary or {}).get("plan_key") == plan_key and not ds.verify_snapshot(data_dir, v):
            return v
    return None


def import_legacy(data_dir: str | Path, root: str | Path, opts: BridgeOptions, *, channel_map=None, identity_map=None,
                  examined_fields=None, lab_csv=(), lab_series=None, dry_run: bool = False, resume: bool = True,
                  say: Say = print) -> ImportOutcome:
    """Scan + eligibility + (unless ``dry_run``) assemble the snapshot. Raises MLError when a real import is not allowed;
    a dry run never raises for identity / mapping gaps (they are returned as ``blockers``) and never writes anything."""
    data_dir = Path(data_dir)
    plan = build_plan(Path(root), opts, channel_map=channel_map, identity_map=identity_map, examined_fields=examined_fields,
                      lab_csv=lab_csv, lab_series=lab_series, dry_run=dry_run, data_dir=data_dir, say=say)
    if plan.hard_refusals:
        raise MLError("; ".join(plan.hard_refusals + [b for b in plan.blockers if "matches no scanned TIFF" in b
                                                        and b not in plan.hard_refusals]))
    for b in plan.blockers:
        say(f"  BLOCKER: {b}")
    if not dry_run and plan.blockers:
        raise MLError("; ".join(plan.blockers))
    live = [p.ext for p in plan.plans if p.ext.cands]
    if not live:
        return ImportOutcome(plan, None, list(plan.blockers))
    if not dry_run:
        done = find_published(data_dir, plan.resume_key)
        if done:
            # identical inputs, options, code and test lock: nothing is rewritten; a staging directory left by an interrupted
            # run of the SAME plan is obsolete now
            ds.remove_staging_if_free(data_dir, f".staging-{ds.safe_id(plan.resume_key)}")
            say(f"already imported as {done} (identical plan {plan.resume_key}); nothing was written")
            return ImportOutcome(plan, None, list(plan.blockers), already_imported=done)
    bo = ds.BuildOptions(seed=opts.seed, fractions=opts.fractions, level=opts.level,
                         allow_unconfirmed_identity=opts.allow_unconfirmed_identity or dry_run, roles=tuple(sorted(
                             {c.role for e in live for c in e.cands})), identity_map=plan.identity_map or None,
                         min_crop_px=opts.min_crop_px, created_at=opts.created_at, include_raw=False,
                         downscale=opts.downscale, normaliser=opts.normaliser, percentiles=tuple(opts.percentiles))
    extra_options = {"import_mode": opts.mode,
                     "legacy_import": {"kind": KIND, "valid_mask": opts.valid_mask, "mode": opts.mode,
                                       "resume_key": plan.resume_key}}
    res = ds.assemble_snapshot(
        data_dir, plan.exts, bo, say=say, dry_run=dry_run, warnings=plan.warnings, extra_options=extra_options,
        extra_assumptions=plan.extra_assumptions, import_summary=plan.summary,
        resume_key=(plan.resume_key if resume else None), workers=opts.workers, materialize_on_dry_run=False)
    if not dry_run and res.manifest is not None:
        bad = verify_examination_provenance(res.manifest)
        if bad:   # pragma: no cover - the gate above makes this unreachable
            raise MLError("snapshot provenance check failed: " + "; ".join(bad[:3]))
    return ImportOutcome(plan, res, list(plan.blockers))


# ---------------------------------------------------------------------------------------------- templates
def write_templates(out_dir: str | Path, root: Optional[str | Path] = None) -> list[Path]:
    """Write identities.template.json, channels.template.json and examined-fields.template.json. Only names and TIFF
    HEADERS are read (never pixel data); inferred values are pre-filled as UNCONFIRMED, roles are null (the operator must decide)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = discover_tiffs(Path(root)) if root else []
    idrows, fentries, counters_seen, n_planes_seen = [], [], {}, set()
    for rel, path in files:
        ident = parse_identity(Path(rel).stem)
        idrows.append({"file": rel, "specimen_id": ident.specimen_id, "slide_id": ident.slide_id, "assay": ident.assay,
                       "replicate": ident.replicate, "scan_session": None, "confirmed": False,
                       "inferred_from": ("file name pattern" if ident.source != "unknown" else "nothing (no pattern matched)")})
        try:
            h = _header(path)
            lay, _why = _layout(h)
            npl = lay[0] if lay else 0
        except Exception:  # noqa: BLE001
            npl = 0
        n_planes_seen.add(npl)
        fentries.append({"file": rel, "sha256": None, "pages": {str(k): [] for k in range(max(npl, 1))}})
    ident_doc = {"_comment": ["Review every row against the lab's records. Set \"confirmed\": true only for rows you checked; "
                              "rows left false are treated as INFERRED and a real import refuses to split on them.",
                              "specimen_id groups replicate slides of one sample (kept in one split); slide_id is one physical slide.",
                              "file = relative path below --input (or a unique base name)."],
                 "rows": idrows or [{"file": "relative/path.tif", "specimen_id": None, "slide_id": None, "assay": None,
                                     "replicate": None, "scan_session": None, "confirmed": False}]}
    page_ex = {"0": {"role": None, "train": True}, "1": {"role": None, "train": True}}
    # the operator's default rule (layout validated on the 8 research stacks; NOT inferred from any file): inactive until confirmed
    page_default = {"0": {"role": "red", "train": True}, "1": {"role": "green", "train": False}}
    chan_doc = {"_comment": ["A channel map is the researcher's statement of what each TIFF page shows. Nothing is inferred from file "
                             "names or from the '635 [W1]' label. Fill in role (free label, e.g. 'red') for every page; set \"train\": "
                             "false for pages that must not be trained on. The default below (page 0 red/train, page 1 green/not trained) is the operator's "
                             "rule for all files; it is refused until confirmed_by / source_of_truth are filled in. Per-file and rule entries override it.",
                             "precedence: files > rules > defaults. Two different rules matching one file are refused."],
                "schema": CHANNEL_SCHEMA, "version": CHANNEL_VERSION, "confirmed_by": "", "confirmed_on": "",
                "source_of_truth": "", "defaults": {"pages": page_default},
                "rules": [], "files": {},
                "_example_rule": {"id": "example-rule", "match": {"glob": "subdir/*.tif"}, "pages": page_ex}}
    exam_doc = {"_comment": ["Only list what the researcher can prove they examined: a field entry per page, named by counter | "
                             "row+col | field. \"count\": 0 means counted zero (verified zero); omit count for 'examined' without a number "
                             "(then zero is NOT assumed). Blank / missing = unknown = excluded.",
                             "The field number is only read as counter N-1 with --acknowledge-field-equals-counter-plus-one."],
                "schema": ex.SCHEMA, "version": ex.SCHEMA_VERSION, "evidence": "", "source_kind": ex.SRC_MANIFEST,
                "examined_by": "", "recorded_on": "", "files": fentries or [
                    {"file": "relative/path.tif", "sha256": None, "pages": {"0": [{"counter": 0, "count": 12}], "1": []}}]}
    paths = []
    for name, doc in (("identities.template.json", ident_doc), ("channels.template.json", chan_doc),
                      ("examined-fields.template.json", exam_doc)):
        p = out / name
        p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(p)
    return paths
