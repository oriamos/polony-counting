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
import fnmatch
import hashlib
import json
import os
import re
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
from .common import (MLError, canonical_json, check_downscale, check_normaliser, hash_obj, page_bounds, pixel_box,
                     sha256_bytes, DEFAULT_PERCENTILES, LO_HI_POLICY)
from .estimates import estimate_training
from .manifest import SnapshotManifest
from .splits import SourceInfo, assign_groups, load_lock, resolve_identity

Say = Callable[[str], None]

# ---------------------------------------------------------------------------------------------- constants
CHANNEL_SCHEMA = "polony-channel-map"
CHANNEL_VERSION = 1
KIND = "legacy_imagej"
VALID_MASK = "eligible-fields"
STRADDLE_REASON = "only straddling points from another counter"
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

    def check(self) -> "BridgeOptions":
        if self.mode not in ex.MODES:
            raise MLError(f"--mode must be one of {list(ex.MODES)} (got {self.mode!r}); there is no mode that treats unmarked "
                          "cells as negatives")
        self.downscale = check_downscale(self.downscale)
        self.normaliser = check_normaliser(self.normaliser)
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
                "valid_mask": VALID_MASK, "duplicate_tol_px": self.duplicate_tol_px}


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


def discover_tiffs(root: Path) -> list[tuple[str, Path]]:
    """[(relative posix path, path)] of every .tif/.tiff below ``root``: sorted, dot-directories and dot-files skipped,
    symlinked directories not followed, archives never opened."""
    root = Path(root)
    if root.is_file():
        return [(root.name, root)] if root.suffix.lower() in (".tif", ".tiff") else []
    out: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if fn.startswith(".") or Path(fn).suffix.lower() not in (".tif", ".tiff"):
                continue
            if fn.lower().endswith((".tif.zip", ".tiff.zip")):
                continue
            p = Path(dirpath) / fn
            out.append((p.relative_to(root).as_posix(), p))
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


def _scan_one(rel: str, path: Path, sha: str, source_id: str, opts: BridgeOptions) -> SourceScan:
    st = path.stat()
    sc = SourceScan(source_id=source_id, rel=rel, path=path, sha256=sha, size=int(st.st_size), mtime_ns=int(st.st_mtime_ns))
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
    sc.pstatus = np.array([_PSTATUS[p.status.value] for p in pts], np.int8)
    cells = np.full(n, -1, np.int64)
    for k, p in enumerate(pts):
        if p.field_id:
            m = _ID_RE.search(p.field_id)
            cells[k] = int(m.group(1)) * nc + int(m.group(2))
    sc.cell = cells
    sc.object_ids = [f"{p.roi_index}:{p.point_index}" for p in pts]
    for f in fields:
        idx = f.grid_row * nc + f.grid_col
        if f.counters:
            sc.cell_counters[idx] = [int(c) for c in f.counters]
            for c in f.counters:
                sc.counter_cell[int(c)] = idx
    sc.notes = [(w.code, w.message) for w in warn.items if w.severity.value != "info"][:20]
    return sc


def scan_tree(root: Path, opts: BridgeOptions, say: Say = print, *, progress_every: int = 100) -> ScanResult:
    """Phase A. Streams every file once for its sha256, drops byte-identical copies (first path in sorted order wins), reads
    headers and ROI blobs. Never reads pixel planes."""
    root = Path(root)
    files = discover_tiffs(root)
    say(f"discovered {len(files)} TIFF file(s) under {root}")
    kept: dict[str, SourceScan] = {}
    aliases: dict[str, list] = {}
    ids: dict[str, str] = {}            # id -> sha256 (collision guard)
    for n, (rel, path) in enumerate(files, 1):
        sha = sha256_stream(path)
        if sha in kept:
            kept[sha].alias_paths.append(rel)
            aliases.setdefault(kept[sha].rel, []).append(rel)
        else:
            k = 12
            while f"img_{sha[:k]}" in ids and ids[f"img_{sha[:k]}"] != sha:
                k += 4
            sid = f"img_{sha[:k]}"
            ids[sid] = sha
            kept[sha] = _scan_one(rel, path, sha, sid, opts)
        if progress_every and n % progress_every == 0:
            say(f"  scanned {n}/{len(files)} file(s)")
    sources = sorted(kept.values(), key=lambda s: s.rel)
    return ScanResult(root=root, discovered=len(files), sources=sources, aliases=aliases)


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
    files: dict                     # rel path -> pages

    def describe(self) -> dict:
        return {"name": None if self.path is None else Path(self.path).name, "sha256": self.sha256,
                "confirmed_by": self.confirmed_by, "confirmed_on": self.confirmed_on,
                "source_of_truth": self.source_of_truth, "n_rules": len(self.rules), "n_files": len(self.files)}


_MATCH_KEYS = ("glob", "assay", "sha256", "specimen_id", "slide_id")


def _pages_block(raw, where: str) -> dict:
    if not isinstance(raw, dict) or not raw:
        raise MLError(f"channel map: {where}: 'pages' must be a non-empty object {{\"0\": {{\"role\": ..., \"train\": true}}}}")
    pages: dict[int, dict] = {}
    roles: dict[str, str] = {}
    for k, v in raw.items():
        if not str(k).isdigit():
            raise MLError(f"channel map: {where}: page keys are 0-based integers, got {k!r}")
        if not isinstance(v, dict):
            raise MLError(f"channel map: {where}: page {k} must be an object with 'role' (and optional 'train')")
        role = v.get("role")
        if not isinstance(role, str) or not role.strip():
            raise MLError(f"channel map: {where}: page {k} has no role (got {role!r}). The map must state, for every page, what "
                          "it shows (e.g. 'red'); roles are never inferred. Fill in the template.")
        train = v.get("train", True)
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
    try:
        raw = p.read_bytes()
        d = json.loads(raw.decode("utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise MLError(f"channel map {p}: cannot be read as JSON ({exc})")
    if not isinstance(d, dict) or d.get("schema") != CHANNEL_SCHEMA:
        raise MLError(f"channel map {p}: 'schema' must be '{CHANNEL_SCHEMA}'")
    if d.get("version") != CHANNEL_VERSION:
        raise MLError(f"channel map {p}: unsupported version {d.get('version')!r} (supported: {CHANNEL_VERSION})")
    if not str(d.get("confirmed_by") or "").strip():
        raise MLError(f"channel map {p}: 'confirmed_by' is empty. A channel map is the researcher's statement of what each page "
                      "shows; fill in who confirmed this mapping and on what basis ('confirmed_by', 'confirmed_on', 'source_of_truth').")
    if not str(d.get("source_of_truth") or "").strip():
        raise MLError(f"channel map {p}: 'source_of_truth' is empty. Fill in who confirmed this mapping and on what basis "
                      "(lab notebook, scanner protocol, ...): a default mapping is never active unconfirmed.")
    defaults = []
    draw = d.get("defaults")
    if draw:
        items = draw if isinstance(draw, list) else [draw]
        for k, it in enumerate(items):
            if not isinstance(it, dict):
                raise MLError(f"channel map {p}: defaults[{k}] must be an object")
            defaults.append((str(it.get("id") or f"default{k}"), _pages_block(it.get("pages"), f"defaults[{k}]")))
    rules, seen = [], set()
    for k, r in enumerate(d.get("rules") or []):
        rid = str(r.get("id") or "").strip() if isinstance(r, dict) else ""
        if not rid:
            raise MLError(f"channel map {p}: rules[{k}] needs a non-empty 'id' (it is recorded as the origin of the mapping)")
        if rid in seen:
            raise MLError(f"channel map {p}: rule id '{rid}' is used twice")
        seen.add(rid)
        match = r.get("match")
        if not isinstance(match, dict) or not match or set(match) - set(_MATCH_KEYS):
            raise MLError(f"channel map {p}: rule '{rid}': 'match' must use only {list(_MATCH_KEYS)} (got {match!r})")
        rules.append((rid, {k2: str(v) for k2, v in match.items()}, _pages_block(r.get("pages"), f"rule '{rid}'")))
    files = {str(k): _pages_block((v or {}).get("pages") if isinstance(v, dict) else None, f"files['{k}']")
             for k, v in (d.get("files") or {}).items()}
    if not (defaults or rules or files):
        raise MLError(f"channel map {p}: it defines no defaults, rules or files")
    return ChannelMap(p, sha256_bytes(raw), str(d["confirmed_by"]), str(d.get("confirmed_on") or ""),
                      str(d.get("source_of_truth") or ""), defaults, rules, files)


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


def _brief(pages: dict) -> str:
    return ", ".join(f"page {k}={v['role']}" for k, v in pages.items())


# ---------------------------------------------------------------------------------------------- identity map
@dataclass
class IdentityTable:
    path: Optional[Path]
    sha256: str
    rows: list


_TRUE = {"1", "true", "yes", "y", "t"}
_FALSE = {"0", "false", "no", "n", "f", ""}


def load_identity_map(path: str | Path) -> IdentityTable:
    p = Path(path)
    try:
        raw = p.read_bytes()
        text = raw.decode("utf-8-sig")
        if p.suffix.lower() == ".csv":
            rows = list(csv.DictReader(text.splitlines()))
        else:
            d = json.loads(text)
            rows = d if isinstance(d, list) else (d.get("rows") or d.get("identities") or d.get("files") or [])
            if isinstance(rows, dict):
                rows = [dict(v, file=k) for k, v in rows.items()]
    except (OSError, ValueError) as exc:
        raise MLError(f"identity map {p}: cannot be read ({exc})")
    out = []
    for k, r in enumerate(rows):
        if not isinstance(r, dict) or not str(r.get("file") or "").strip():
            raise MLError(f"identity map {p}: row {k + 1} has no 'file'")
        conf = r.get("confirmed", True)
        if isinstance(conf, str):
            s = conf.strip().lower()
            if s not in _TRUE | _FALSE:
                raise MLError(f"identity map {p}: row {k + 1}: 'confirmed' must be true/false (got {conf!r})")
            conf = s in _TRUE if s != "" else True
        row = {"file": str(r["file"]).strip().replace("\\", "/"), "confirmed": bool(conf)}
        for key in ("specimen_id", "slide_id", "scan_session", "assay", "replicate"):
            v = r.get(key)
            row[key] = None if v is None or str(v).strip() == "" else str(v).strip()
        if row["confirmed"] and not (row["specimen_id"] or row["slide_id"]):
            raise MLError(f"identity map {p}: row {k + 1} ({row['file']}) is marked confirmed but has neither specimen_id nor slide_id")
        out.append(row)
    return IdentityTable(p, sha256_bytes(raw), out)


def apply_identity_map(table: Optional[IdentityTable], scans: list) -> tuple[dict, dict]:
    """(identity_map {source_id: {...}} for ``ml.splits``, report). Only rows marked confirmed enter the map."""
    report = {"rows": 0, "matched": 0, "unconfirmed_rows": [], "ambiguous_basename": [], "missing_on_disk": [],
              "unlisted_files": [], "conflicting_rows": []}
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
            if not sep or not name.strip() or not pg.strip().isdigit():
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
    m = ex.ExaminedManifest(evidence=ev, source_kind=ex.SRC_SIDECAR_CSV, examined_by="lab CSV", recorded_on="",
                            files=[ex.FileEntry(file=r, sha256=d["sha"], pages=d["pages"]) for r, d in sorted(files.items())],
                            sha256=hash_obj(info["csv_files"]))
    for k in ("ambiguous_rows", "unmatched_rows"):
        info[k] = sorted(set(info[k]))
    return m, info


def manifest_entries_for(manifest: ex.ExaminedManifest, scans: list) -> tuple[dict, list]:
    """{source_id: {page: {key: FieldEntry}}} and a list of problems. A manifest file entry names a source by relative path
    (or alias), else by base name when that name is unique; a recorded sha256 must match."""
    srcs = {s.source_id: s for s in scans}
    by_rel = {r: s for s in scans for r in [s.rel, *s.alias_paths]}
    by_base: dict[str, set] = defaultdict(set)
    for s in scans:
        for r in [s.rel, *s.alias_paths]:
            by_base[Path(r).name].add(s.source_id)
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
        if s is None:
            problems.append(f"{fe.file}: listed in the examined-fields manifest but not found among the TIFFs")
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

    def __init__(self, sc: SourceScan):
        self.sc = sc
        self.planes: dict[int, np.ndarray] = {}
        self.lock = threading.Lock()

    def plane(self, page: int) -> np.ndarray:
        with self.lock:
            if page not in self.planes:
                st = self.sc.path.stat()
                if st.st_size != self.sc.size or st.st_mtime_ns != self.sc.mtime_ns:
                    raise MLError(f"{self.sc.rel} changed on disk after it was scanned (size/mtime differ); rerun the import")
                a = read_plane(self.sc.path, page)
                if a.shape != (self.sc.height, self.sc.width) or str(a.dtype) != self.sc.dtype:
                    raise MLError(f"{self.sc.rel} page {page}: pixels {a.shape} {a.dtype} differ from the header "
                                  f"({self.sc.height}x{self.sc.width} {self.sc.dtype})")
                self.planes[page] = a
            return self.planes[page]

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


def plan_source(sc: SourceScan, *, pages: dict, origin: str, opts: BridgeOptions, manifest: Optional[ex.ExaminedManifest],
                page_entries: dict, ident: dict, identity_map_given: bool) -> SourcePlan:
    """Phase C for one source: eligibility, labels, crop candidates (lazy) and the points / regions rows."""
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
            if -0.5 <= cx < w - 0.5 and -0.5 <= cy < h - 0.5:
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
            "n_labels_from_other_counters": int(tr.n_stray), "assumptions": list(assumptions),
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
            reason = "plane_unspecified"
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
                         "consensus_status": None})

    if not cands:
        why = (R_NO_ELIGIBLE, "no (field, page) tuple of this file is proven examined (" + "; ".join(
            f"{k}: {v}" for k, v in sorted(Counter(t.exam.status for t in tuples.values()).items())) + ")")
        src.skipped = f"{why[0]}: {why[1]}"
        ext.warnings.append(f"{sc.rel}: skipped - {src.skipped}")

    # ---- hooks: read each page plane once, page normalisation on the eligible fields only, then release
    boxes_by_page: dict[int, list] = defaultdict(list)
    for (page, cell), tr in tuples.items():
        if tr.exam.researcher_examined:
            boxes_by_page[page].append(pixel_box(sc.cell_rect(cell)))

    def prepare() -> None:
        done: dict[str, dict] = {}
        for page in sorted({c.page for c in cands}):
            plane = cache.plane(page)
            valid = np.zeros(plane.shape, bool)
            for x0, y0, x1, y1 in boxes_by_page[page]:
                valid[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
            b = page_bounds(plane, valid, opts.percentiles)
            role = role_of[page]
            b.update({"file": sc.rel, "page": int(page), "mode": opts.normaliser, "policy": LO_HI_POLICY,
                      "valid_mask": "union of the pixel boxes of the researcher-examined fields on this page (eligible-fields)"})
            if b["fallback"]:
                ext.late_warnings.append(f"{sc.rel} page {page} ({role}): {b['fallback']}")
            done[role] = b
            for cd in cands:
                if cd.page == page:
                    cd.norm = b
        src.page_normalization = done

    ext.prepare = prepare if cands else None
    ext.release = cache.release

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


def verify_examination_provenance(manifest) -> list[str]:
    """Problems found in the crops of a legacy-import snapshot manifest (empty list = every train / val / test crop is proven
    researcher-examined). ``manifest`` = SnapshotManifest or its dict. Snapshots that are not legacy imports (no
    ``import_summary``) are out of scope and return []."""
    d = manifest.model_dump(mode="json") if hasattr(manifest, "model_dump") else dict(manifest)
    summary = d.get("import_summary") or {}
    if summary.get("kind") != KIND:
        return []
    bad: list[str] = []
    for c in d.get("crops", []):
        bad += _check_prov(c.get("crop_id", "?"), c.get("provenance") or {})
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
        raise MLError(f"--input {root} does not exist")
    if opts.mode == ex.MODE_MANIFEST and not (examined_fields or lab_csv):
        raise MLError("--mode examined-fields-manifest needs --examined-fields FILE (or --lab-csv): only the tuples the manifest "
                      "lists can be used; nothing is inferred")
    if examined_fields and lab_csv:
        raise MLError("give ONE examined-fields source: --examined-fields or --lab-csv, not both")
    cm = load_channel_map(channel_map) if channel_map else None
    table = load_identity_map(identity_map) if identity_map else None
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
    entries_by_src, man_problems = (manifest_entries_for(man, [s for s in kept if not s.rejected]) if man else ({}, []))

    # ---- mapping + eligibility
    blockers: list[str] = []
    mapped = cm is not None
    if cm is None:
        blockers.append("no --channel-map: which page shows which channel is never inferred; supply a confirmed channel map "
                        "(see --write-templates)")
    exts, plans = [], []
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
                         ident=sc.identity, identity_map_given=table is not None)
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
                      n_cands=n_cands, mapped=mapped)
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
                  "deduplicated_paths": {k: sorted(v) for k, v in sorted(scan.aliases.items())}},
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
        "unresolved_manifest_entries": unres_entries + [{"problem": m} for m in man_problems],
        "lab_csv": lab_info,
        "info_label_check": _info_label_check(plan, per_file_map),
        "assumptions": plan.extra_assumptions,
    }
    plan.summary = _jsonable(summary)
    key = {"files": sorted((s.rel, s.sha256) for s in kept), "options": {k: v for k, v in opts.public().items()
                                                                          if k not in ("batch_size", "epochs")},
           "inputs": plan.summary["input_files"], "app": "legacy-bridge-1"}
    plan.resume_key = hash_obj(key)[:20]


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


def import_legacy(data_dir: str | Path, root: str | Path, opts: BridgeOptions, *, channel_map=None, identity_map=None,
                  examined_fields=None, lab_csv=(), lab_series=None, dry_run: bool = False, resume: bool = True,
                  say: Say = print) -> ImportOutcome:
    """Scan + eligibility + (unless ``dry_run``) assemble the snapshot. Raises MLError when a real import is not allowed;
    a dry run never raises for identity / mapping gaps (they are returned as ``blockers``) and never writes anything."""
    data_dir = Path(data_dir)
    plan = build_plan(Path(root), opts, channel_map=channel_map, identity_map=identity_map, examined_fields=examined_fields,
                      lab_csv=lab_csv, lab_series=lab_series, dry_run=dry_run, data_dir=data_dir, say=say)
    for b in plan.blockers:
        say(f"  BLOCKER: {b}")
    if not dry_run and plan.blockers:
        raise MLError("; ".join(plan.blockers))
    live = [p.ext for p in plan.plans if p.ext.cands]
    if not live:
        return ImportOutcome(plan, None, list(plan.blockers))
    bo = ds.BuildOptions(seed=opts.seed, fractions=opts.fractions, level=opts.level,
                         allow_unconfirmed_identity=opts.allow_unconfirmed_identity or dry_run, roles=tuple(sorted(
                             {c.role for e in live for c in e.cands})), identity_map=plan.identity_map or None,
                         min_crop_px=opts.min_crop_px, created_at=opts.created_at, include_raw=False,
                         downscale=opts.downscale, normaliser=opts.normaliser, percentiles=tuple(opts.percentiles))
    extra_options = {"import_mode": opts.mode,
                     "legacy_import": {"kind": KIND, "valid_mask": VALID_MASK, "mode": opts.mode,
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
