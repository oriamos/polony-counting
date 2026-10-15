"""``python -m polony_app.ml preflight`` - a READ-ONLY audit of the historical TIFF directory BEFORE ``import-legacy``.

Nothing is ever written into ``--input`` (files are opened ``rb``; symlinks are reported, never followed outside it) and no
snapshot is written. The only optional output is a report in ``--report-dir``. Five groups of checks, each finding is
``PASS`` / ``WARN`` / ``FAIL`` (or ``INFO``) with a count and the first N offending files:

A  ENVIRONMENT      python / torch / spotiflow / lightning versions, ROCm (``torch.version.hip``) and GPU visibility, free
                    disk vs the estimated snapshot (+10 %), RAM vs the prepared training size, file descriptors, write
                    permission and case sensitivity, ``/dev/kfd`` hints (Linux only), long paths (Windows only).
B  INPUT SANITY     header-only (no pixel decode), thread pool: counts by extension, zero-byte / truncated / unreadable files,
                    compression, dtype, planes, shape / axes, size outliers, ImageJ metadata, ROI / Overlays blobs and parse
                    errors, grid regularity and ORIGIN CLUSTERS (the shifted grid of one real file), counters (missing /
                    out of order / split), positions, points outside the image or grid, (near-)duplicate points, byte-identical
                    duplicates (streaming sha256, bounded memory), odd names, symlinks, dot-directories, archives (never
                    opened), scanner ``Info`` clusters, ``635 [W1]`` label consistency.
C  MAPPING FILES    channels.json / identities.json / examined-fields are read by the SAME loaders as ``import-legacy``; files
                    without a mapping, mapping / identity entries for non-existing files, basename collisions, unconfirmed
                    identity rows, conflicting rules, manifest entries that cannot be resolved.
D  INDEPENDENT CROSS-CHECK  for a deterministic sample (default 40 files + every odd file; ``--sample all``) the marks, grid
                    cells, counter -> majority cell, per-(page, cell) examination status, labels, dropped points and crop boxes
                    are re-derived DIRECTLY with tifffile + roifile (this module shares no scan / eligibility code with
                    ``legacy_bridge``) and compared with what the bridge's own dry-run plan reports. Any disagreement is a FAIL
                    with the exact tuple. Also: median distance of the labels on page 0 to the nearest local intensity maximum.
E  POLICY SUMMARY   tuples examined vs excluded (unknown / conflict by reason), projected crops / labels per split, positives per
                    page, projected disk / RAM / steps / validation passes, identity blockers and a final VERDICT.

Exit codes: 0 = no FAIL (READY / READY WITH WARNINGS); 1 = at least one FAIL (NOT READY); 2 = refused (bad arguments, e.g.
``--input`` missing or ``--report-dir`` inside ``--input``). ``--json`` prints one JSON document (schema ``polony-preflight``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import re
import shlex
import shutil
import sys
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .common import MLError

SCHEMA = "polony-preflight"
VERSION = 1
PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
EXIT_OK, EXIT_NOT_READY, EXIT_REFUSED = 0, 1, 2
V_READY, V_WARN, V_NOT = "READY", "READY WITH WARNINGS", "NOT READY"

TIFF_SUFFIXES = (".tif", ".tiff")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".zst")
OTHER_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".czi", ".lsm", ".nd2", ".lif", ".btf", ".tf8", ".ome", ".stk", ".roi")
SUPPORTED_DTYPES = ("uint8", "uint16")
DUP_TOL_PX = 1.0           # the bridge's duplicate tolerance (same plane)
NEAR_PX = 3.0              # information only: touching polonies / double clicks
MIN_CROP_PX = 32
EPS = 1e-6
BRIGHT_WARN_PX = 12.0
BRIGHT_RADIUS = 15


# ====================================================================================================== report model
@dataclass
class Finding:
    id: str
    check: str                 # A..E
    title: str
    level: str                 # PASS | WARN | FAIL | INFO
    summary: str
    count: int = 0
    examples: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"id": self.id, "check": self.check, "title": self.title, "level": self.level, "summary": self.summary,
                "count": int(self.count), "examples": self.examples, "detail": self.detail}


class Report:
    def __init__(self, max_examples: int = 10):
        self.max_examples = max(1, int(max_examples))
        self.findings: list[Finding] = []
        self.sections: dict = {}

    def add(self, check: str, fid: str, title: str, level: str, summary: str, *, count: int = 0, examples=(), detail=None
            ) -> Finding:
        ex = list(examples)
        det = dict(detail or {})
        if len(ex) > self.max_examples:
            det["examples_total"] = len(ex)
            ex = ex[:self.max_examples]
        f = Finding(fid, check, title, level, summary, count, [_jsonable(e) for e in ex], _jsonable(det))
        self.findings.append(f)
        return f

    def of(self, level: str) -> list:
        return [f for f in self.findings if f.level == level]

    def counts(self) -> dict:
        c = Counter(f.level for f in self.findings)
        return {k: int(c.get(k, 0)) for k in (PASS, WARN, FAIL, INFO)}

    def verdict(self) -> tuple[str, list]:
        fails, warns = self.of(FAIL), self.of(WARN)
        if fails:
            return V_NOT, [f"{f.id}: {f.title}" + (f" ({f.count})" if f.count else "") for f in fails]
        if warns:
            return V_WARN, [f"{f.id}: {f.title}" + (f" ({f.count})" if f.count else "") for f in warns]
        return V_READY, []


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


# ====================================================================================================== parameters
@dataclass
class PreflightParams:
    input: Path
    data_dir: Path
    channel_map: Optional[Path] = None
    identity_map: Optional[Path] = None
    examined_fields: Optional[Path] = None
    lab_csv: list = field(default_factory=list)
    lab_series: list = field(default_factory=list)
    mode: str = "marked-only"
    acknowledge_no_counters: bool = False
    allow_unconfirmed_identity: bool = False
    acknowledge_field_equals_counter_plus_one: bool = False
    acknowledge_lab_csv_field_numbering: bool = False
    seed: int = 0
    fractions: Optional[dict] = None
    level: str = "auto"
    valid_mask: str = "chamber"
    downscale: int = 1
    normaliser: str = "valid_unsaturated"
    batch_size: int = 4
    epochs: int = 30
    workers: int = 0                 # 0 = automatic
    sample: str = "40"
    max_examples: int = 10
    report_dir: Optional[Path] = None
    pixels: bool = True              # D: read page 0 of the sampled files for the local-maximum sanity check
    progress: Optional[Callable[[str], None]] = None

    def n_workers(self) -> int:
        return int(self.workers) if int(self.workers) > 0 else max(1, min(8, (os.cpu_count() or 2)))

    def say(self, msg: str) -> None:
        if self.progress is not None:
            self.progress(msg)


def _is_within(child: Path, parent: Path) -> bool:
    try:
        Path(os.path.realpath(child)).relative_to(Path(os.path.realpath(parent)))
        return True
    except ValueError:
        return False


# ====================================================================================================== walking the input
def walk_input(root: Path) -> dict:
    """Everything below ``root`` (never following symlinked directories, never opening archives). ``tiffs`` follows the same
    discovery rules as ``legacy_bridge.discover_tiffs`` (and is compared with it in check D)."""
    root = Path(root)
    out = {"tiffs": [], "ext_counts": Counter(), "n_files": 0, "dot_entries": [], "symlink_files": [], "symlink_dirs": [],
           "archives": [], "other_images": [], "n_dirs": 0}
    if root.is_file():
        out["n_files"] = 1
        out["ext_counts"][root.suffix.lower() or "(none)"] += 1
        if root.suffix.lower() in TIFF_SUFFIXES:
            out["tiffs"].append((root.name, root))
        return out
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        keep = []
        for d in sorted(dirnames):
            full = Path(dirpath) / d
            rel = full.relative_to(root).as_posix()
            if d.startswith("."):
                out["dot_entries"].append(rel + "/")
                continue
            if full.is_symlink():
                out["symlink_dirs"].append(rel)
                continue
            keep.append(d)
        dirnames[:] = keep
        out["n_dirs"] += len(keep)
        for fn in sorted(filenames):
            full = Path(dirpath) / fn
            rel = full.relative_to(root).as_posix()
            if fn.startswith("."):
                out["dot_entries"].append(rel)
                continue
            out["n_files"] += 1
            suf = Path(fn).suffix.lower() or "(none)"
            out["ext_counts"][suf] += 1
            if full.is_symlink():
                out["symlink_files"].append(rel)
            if suf in ARCHIVE_SUFFIXES:
                out["archives"].append(rel)
                continue
            if suf in OTHER_IMAGE_SUFFIXES:
                out["other_images"].append(rel)
            if suf in TIFF_SUFFIXES:
                out["tiffs"].append((rel, full))
    out["tiffs"].sort(key=lambda t: t[0])
    return out


# ====================================================================================================== ROI / TIFF parsing
_SEG_ARGS = {0: 2, 1: 2, 2: 4, 3: 6, 4: 0}


def _shape_paths(multi) -> list:
    """Sub-paths (vertex lists) of an ImageJ composite shape (moveTo / lineTo / quadTo / cubicTo / close)."""
    arr = np.asarray(multi, dtype=np.float64).ravel()
    paths, cur, i = [], [], 0
    while i < len(arr):
        cmd = arr[i]
        if cmd != int(cmd) or int(cmd) not in _SEG_ARGS:
            raise ValueError(f"invalid shape command {cmd!r} at offset {i}")
        cmd = int(cmd)
        n = _SEG_ARGS[cmd]
        args = arr[i + 1:i + 1 + n]
        if len(args) != n:
            raise ValueError("truncated shape path")
        i += 1 + n
        if cmd == 0:
            if cur:
                paths.append(cur)
            cur = [(float(args[0]), float(args[1]))]
        elif cmd in (1, 2, 3):
            cur.append((float(args[-2]), float(args[-1])))
        elif cmd == 4 and cur:
            paths.append(cur)
            cur = []
    if cur:
        paths.append(cur)
    return paths


def _grid_of(paths) -> Optional[tuple]:
    vx, hy = [], []
    for p in paths:
        if len(p) == 2 and abs(p[0][0] - p[1][0]) < 1e-3 and abs(p[0][1] - p[1][1]) > 1e-3:
            vx.append(round(p[0][0], 3))
        elif len(p) == 2 and abs(p[0][1] - p[1][1]) < 1e-3 and abs(p[0][0] - p[1][0]) > 1e-3:
            hy.append(round(p[0][1], 3))
    xs, ys = sorted(set(vx)), sorted(set(hy))
    if len(xs) >= 2 and len(ys) >= 2:
        return xs, ys
    return None


def parse_blobs(md: Optional[dict]) -> dict:
    """The ImageJ ROI / Overlays blobs of a TIFF, decoded with roifile only. Returns {n_blobs, blobs: [{source, kind, error}],
    points: dict of arrays or None, n_point_blobs, grids: [(xs, ys)], errors: [str]}."""
    import roifile
    raw: list = []
    if md:
        if md.get("ROI"):
            raw.append(("ROI", bytes(md["ROI"])))
        ov = md.get("Overlays")
        if ov:
            for b in (ov if isinstance(ov, (list, tuple)) else [ov]):
                raw.append(("Overlays", bytes(b)))
    res = {"n_blobs": len(raw), "blobs": [], "grids": [], "errors": [], "n_point_blobs": 0, "notes": [],
           "has_roi_key": bool(md and md.get("ROI")), "has_overlay_key": bool(md and md.get("Overlays"))}
    px, py, psub, pctr, ppos, pblob = [], [], [], [], [], []
    for idx, (source, blob) in enumerate(raw):
        try:
            roi = roifile.ImagejRoi.frombytes(blob)
        except Exception as exc:  # noqa: BLE001
            res["blobs"].append({"source": source, "kind": "unparseable", "error": f"{type(exc).__name__}: {exc}"})
            res["errors"].append(f"{source}[{idx}] unparseable ({type(exc).__name__}: {exc})")
            continue
        try:
            rtype = getattr(roi.roitype, "name", str(roi.roitype))
            multi = roi.multi_coordinates if getattr(roi, "composite", False) else None
            if multi is not None and len(multi):
                g = _grid_of(_shape_paths(multi))
                if g is not None:
                    res["grids"].append(g)
                    res["blobs"].append({"source": source, "kind": "grid", "error": None})
                else:
                    res["blobs"].append({"source": source, "kind": "composite_other", "error": None})
                    res["notes"].append(f"{source}[{idx}] composite shape that is not a line grid")
            elif rtype == "POINT":
                coords = np.asarray(roi.coordinates(), dtype=np.float64).reshape(-1, 2)
                n = len(coords)
                sub = bool(roi.options & roifile.ROI_OPTIONS.SUB_PIXEL_RESOLUTION) and roi.subpixel_coordinates is not None
                ctr, cpos = roi.counters, roi.counter_positions
                if ctr is not None and len(ctr) != n:
                    res["notes"].append(f"{source}[{idx}] {len(ctr)} counters for {n} points (counters ignored)")
                    ctr = None
                if cpos is not None and len(cpos) != n:
                    res["notes"].append(f"{source}[{idx}] {len(cpos)} counter positions for {n} points (positions ignored)")
                    cpos = None
                rpos = int(getattr(roi, "position", 0) or 0)
                if n == 0:
                    res["blobs"].append({"source": source, "kind": "empty_points", "error": None})
                    continue
                res["n_point_blobs"] += 1
                res["blobs"].append({"source": source, "kind": "points", "error": None, "n": n})
                for k in range(n):
                    position = int(cpos[k]) if cpos is not None else rpos
                    px.append(coords[k, 0])
                    py.append(coords[k, 1])
                    psub.append(sub)
                    pctr.append(-1 if ctr is None else int(ctr[k]))
                    ppos.append(position if position > 0 else 0)
                    pblob.append(idx)
            else:
                res["blobs"].append({"source": source, "kind": f"shape:{rtype}", "error": None})
        except Exception as exc:  # noqa: BLE001
            res["blobs"].append({"source": source, "kind": "undecodable", "error": f"{type(exc).__name__}: {exc}"})
            res["errors"].append(f"{source}[{idx}] could not be decoded ({type(exc).__name__}: {exc})")
    res["points"] = ({"x": np.array(px), "y": np.array(py), "sub": np.array(psub, bool), "counter": np.array(pctr, np.int32),
                      "position": np.array(ppos, np.int32), "blob": np.array(pblob, np.int32)} if px else None)
    return res


_INFO_RES = {"label": re.compile(r"ImageDescription:\s*([^\n\r]+)"), "datetime": re.compile(r"DateTime:\s*(\d{4})[/-](\d{2})[/-](\d{2})[ T](\d{2}:\d{2}:\d{2})"),
             "pmt": re.compile(r"\bPMT=([^;\s]+)"), "laser": re.compile(r"\bLaserPower=([^;\s]+)"),
             "focus": re.compile(r"\bFocusPosition=([^;\s]+)"), "averaging": re.compile(r"\bAveraging=([^;\s]+)"),
             "software": re.compile(r"Software:\s*([^\n\r]+)")}


def parse_info(info) -> dict:
    out: dict = {}
    if not isinstance(info, str):
        return out
    for k, rx in _INFO_RES.items():
        m = rx.search(info)
        if m:
            out[k] = (f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}" if k == "datetime" else m.group(1).strip())
    return out


def point_table(pts: dict, planes: int, w: int, h: int, grid: Optional[tuple]) -> dict:
    """Canonical coordinates, plane, status (0 ok, 1 duplicate, 2 out of bounds) and grid cell of every mark.

    canonical = ImageJ - 0.5 for sub-pixel ROIs, = the pixel index for integer ROIs. A mark is inside a cell when its centre
    satisfies the half-open rule  x0 - 0.5 <= cx < x1 - 0.5  (same for y); cells are numbered row * n_cols + col."""
    n = len(pts["x"])
    cx = np.where(pts["sub"], pts["x"] - 0.5, pts["x"])
    cy = np.where(pts["sub"], pts["y"] - 0.5, pts["y"])
    finite = np.isfinite(cx) & np.isfinite(cy)
    inb = finite & (cx >= -0.5) & (cx < w - 0.5) & (cy >= -0.5) & (cy < h - 0.5)
    pos = pts["position"]
    page = np.where((pos >= 1) & (pos <= planes), pos - 1, -1).astype(np.int32)
    status = np.where(inb, 0, 2).astype(np.int8)
    ncl = 0
    cell = np.full(n, -1, np.int64)
    if grid is not None:
        # a mark belongs to the cell whose integer PIXEL footprint contains the pixel it rounds to, floor(c + 0.5); the footprint
        # of a cell [x0, x1) in the canonical frame is the pixel range ceil(x0 - 0.5) .. ceil(x1 - 0.5) - 1
        ex_ = [math.ceil(float(v) - 0.5) for v in grid[0]]
        ey_ = [math.ceil(float(v) - 0.5) for v in grid[1]]
        ncl = len(ex_) - 1
        for k in range(n):
            if not inb[k]:
                continue
            ix, iy = math.floor(cx[k] + 0.5), math.floor(cy[k] + 0.5)
            ci = next((c for c in range(len(ex_) - 1) if ex_[c] <= ix < ex_[c + 1]), -1)
            ri = next((r for r in range(len(ey_) - 1) if ey_[r] <= iy < ey_[r + 1]), -1)
            if ci >= 0 and ri >= 0:
                cell[k] = ri * ncl + ci
    near = 0
    for pg in sorted(set(page.tolist())):
        idx = np.nonzero((page == pg) & (status == 0))[0]
        m = len(idx)
        if m < 2:
            continue
        gx, gy = cx[idx], cy[idx]
        alive = np.ones(m, bool)
        for i in range(m - 1):
            if not alive[i]:
                continue
            d = np.hypot(gx[i + 1:] - gx[i], gy[i + 1:] - gy[i])
            hit = np.nonzero((d <= DUP_TOL_PX) & alive[i + 1:])[0] + i + 1
            alive[hit] = False
        keep = idx[alive]
        status[idx[~alive]] = 1
        gx, gy = cx[keep], cy[keep]
        for i in range(len(keep) - 1):
            near += int((np.hypot(gx[i + 1:] - gx[i], gy[i + 1:] - gy[i]) <= NEAR_PX).sum())
    return {"cx": cx, "cy": cy, "page": page, "status": status, "cell": cell, "counter": pts["counter"],
            "position": pos, "n_close_pairs": int(near)}


def counter_stats(counter: np.ndarray) -> dict:
    """Counters in click order: which are absent from 0..max, which appear after a larger counter (out of order) and which
    appear in more than one separate run."""
    vals = [int(c) for c in counter if c >= 0]
    out = {"n_with": len(vals), "n_without": int(len(counter) - len(vals)), "absent": [], "out_of_order": [], "split_runs": [],
           "max": (max(vals) if vals else None), "distinct": len(set(vals))}
    if not vals:
        return out
    seen = set(vals)
    out["absent"] = [c for c in range(0, max(vals) + 1) if c not in seen]
    first = []
    runs: Counter = Counter()
    last = None
    for c in vals:
        if c != last:
            runs[c] += 1
            last = c
        if c not in first:
            first.append(c)
    out["split_runs"] = sorted(c for c, n in runs.items() if n > 1)
    hi = -1
    for c in first:
        if c < hi:
            out["out_of_order"].append(c)
        hi = max(hi, c)
    return out


def inspect_tiff(rel: str, path: Path, *, outside_symlink: bool = False) -> dict:
    """Header-only description of one TIFF (no pixel decode). Never raises."""
    import tifffile
    rec: dict = {"rel": rel, "size": 0, "error": None, "error_kind": None, "outside_symlink": bool(outside_symlink),
                 "layout_ok": False, "layout_why": None, "is_imagej": False, "planes": 0, "h": 0, "w": 0, "dtype": None,
                 "axes": None, "shape": None, "compression": None, "photometric": None, "n_pages": 0, "ij_images": None,
                 "truncated": False, "info": {}, "info_raw": None, "blobs": None, "pt": None, "grid": None, "counters": None,
                 "points": 0, "parse_errors": [], "n_point_blobs": 0, "n_grids": 0}
    if outside_symlink:
        rec.update(error="symlink resolving outside --input or broken (never opened; the importer skips it)", error_kind="symlink_outside")
        return rec
    try:
        rec["size"] = int(os.stat(path).st_size)
    except OSError as exc:
        rec.update(error=f"cannot stat ({type(exc).__name__}: {exc})", error_kind="unreadable")
        return rec
    if rec["size"] == 0:
        rec.update(error="zero-byte file", error_kind="zero_byte")
        return rec
    try:
        with tifffile.TiffFile(str(path)) as tif:
            s = tif.series[0]
            shape = tuple(int(v) for v in s.shape)
            axes = str(s.axes)
            rec.update(shape=shape, axes=axes, dtype=str(np.dtype(s.dtype)), n_pages=len(tif.pages), is_imagej=bool(tif.is_imagej))
            p0 = tif.pages[0]
            rec["compression"] = getattr(p0.compression, "name", str(p0.compression))
            rec["photometric"] = getattr(p0.photometric, "name", str(p0.photometric))
            if len(shape) == 2 and axes.endswith("YX"):
                rec.update(planes=1, h=shape[0], w=shape[1], layout_ok=True)
            elif len(shape) == 3 and axes.endswith("YX"):
                rec.update(planes=shape[0], h=shape[1], w=shape[2], layout_ok=True)
            elif "S" in axes:
                rec["layout_why"] = f"RGB / multi-sample pages (axes {axes}, shape {shape})"
            else:
                rec["layout_why"] = f"unsupported axes/shape {axes} {shape}"
            md = tif.imagej_metadata if tif.is_imagej else None
            # ---- truncation: the declared pixel data must fit into the file
            try:
                images = int((md or {}).get("images") or 0) if md else 0
                rec["ij_images"] = images or None
                itemsize = np.dtype(s.dtype).itemsize
                if tif.is_imagej and images and p0.dataoffsets:
                    spp = int(getattr(p0, "samplesperpixel", 1) or 1)
                    need = int(p0.dataoffsets[0]) + images * int(p0.imagelength) * int(p0.imagewidth) * itemsize * spp \
                        if (p0.compression.name == "NONE") else 0
                    if need and need > rec["size"]:
                        rec["truncated"] = True
                    if rec["layout_ok"] and rec["planes"] < images:
                        rec["truncated"] = True
                else:
                    last = max((int(o) + int(b) for pg in list(tif.pages)[-2:] for o, b in zip(pg.dataoffsets, pg.databytecounts)), default=0)
                    if last > rec["size"]:
                        rec["truncated"] = True
            except Exception:  # noqa: BLE001
                pass
            if md:
                rec["info_raw"] = md.get("Info") if isinstance(md.get("Info"), str) else None
                rec["info"] = parse_info(rec["info_raw"])
    except Exception as exc:  # noqa: BLE001
        rec.update(error=f"{type(exc).__name__}: {exc}", error_kind="unreadable", layout_ok=False)
        return rec
    if rec["truncated"]:
        rec["error_kind"] = "truncated"
        rec["error"] = (f"truncated: the file has {rec['size']} bytes but the header declares more pixel data "
                        f"(ImageJ images={rec['ij_images']}, readable planes={rec['planes']})")
    # ---- blobs (ROI / Overlays): in a second open, so a damaged blob never hides the header facts
    try:
        with tifffile.TiffFile(str(path)) as tif:
            md = tif.imagej_metadata if tif.is_imagej else None
        b = parse_blobs(md)
    except Exception as exc:  # noqa: BLE001
        rec["parse_errors"].append(f"TIFF metadata unreadable ({type(exc).__name__}: {exc})")
        return rec
    rec["blobs"] = [{"source": x["source"], "kind": x["kind"]} for x in b["blobs"]]
    rec["n_blobs"] = b["n_blobs"]
    rec["parse_errors"] = list(b["errors"])
    rec["parse_notes"] = list(b["notes"])
    rec["n_point_blobs"] = b["n_point_blobs"]
    rec["n_grids"] = len(b["grids"])
    rec["has_roi_key"], rec["has_overlay_key"] = b["has_roi_key"], b["has_overlay_key"]
    if b["grids"]:
        rec["grid"] = b["grids"][0]
    if b["points"] is not None:
        rec["points"] = int(len(b["points"]["x"]))
        rec["raw_points"] = b["points"]
        if rec["layout_ok"]:
            rec["pt"] = point_table(b["points"], rec["planes"], rec["w"], rec["h"], rec["grid"])
            rec["counters"] = counter_stats(b["points"]["counter"])
    return rec


# ====================================================================================================== hashing
def sha256_stream(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_duplicates(files: list, workers: int, *, hash_all: bool = False) -> tuple[dict, dict]:
    """({sha256: [rel, ...]} of byte-identical groups with >= 2 members, {rel: sha256} of every hashed file). Only files that
    share their size with another file are hashed (identical bytes imply identical size), streamed in 1 MiB chunks."""
    by_size: dict[int, list] = defaultdict(list)
    sizes: dict[str, int] = {}
    for rel, path in files:
        try:
            sz = os.stat(path).st_size
        except OSError:
            continue
        sizes[rel] = sz
        by_size[sz].append((rel, path))
    todo = [t for sz, g in by_size.items() for t in g if (hash_all or len(g) > 1)]
    shas: dict[str, str] = {}

    def one(t):
        try:
            return t[0], sha256_stream(t[1])
        except OSError:
            return t[0], None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for rel, sha in ex.map(one, todo):
            if sha:
                shas[rel] = sha
    groups: dict[str, list] = defaultdict(list)
    for rel in sorted(shas):
        groups[shas[rel]].append(rel)
    return {k: v for k, v in groups.items() if len(v) > 1}, shas


# ====================================================================================================== A: environment
def _nearest_existing(p: Path) -> Path:
    p = Path(p)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _fmt(n) -> str:
    from .estimates import format_bytes
    return format_bytes(n)


def check_environment_static(rep: Report, p: PreflightParams, walk: dict) -> dict:
    from . import spotiflow_io as sio
    env: dict = {"platform": platform.platform(), "python": platform.python_version()}
    rep.add("A", "A.python", "python", INFO, f"python {platform.python_version()} on {platform.platform()}")
    try:
        diag = sio.gpu_diagnosis()
        hw = sio.hardware_info("cuda" if diag["kind"] == "ok" else "cpu")
    except Exception as exc:  # noqa: BLE001
        diag, hw = {"kind": "error", "message": f"{type(exc).__name__}: {exc}", "hints": []}, {}
    env.update({k: hw.get(k) for k in ("torch", "torch_hip", "torch_cuda", "cuda_available", "spotiflow", "lightning", "device_names",
                                       "gcn_arch", "cpu_count")})
    env["gpu_diagnosis"] = diag
    missing = [k for k in ("torch", "spotiflow", "lightning") if not hw.get(k)]
    if missing:
        rep.add("A", "A.ml_stack", "ML libraries", WARN,
                f"not importable here: {', '.join(missing)}. import-legacy does not need them, but train / eval-model do "
                "(install the optional ML dependencies)", count=len(missing), detail={"versions": env})
    else:
        rep.add("A", "A.ml_stack", "ML libraries", PASS,
                f"torch {hw['torch']} (hip={hw.get('torch_hip')}, cuda={hw.get('torch_cuda')}), spotiflow {hw['spotiflow']}, "
                f"lightning {hw['lightning']}", detail={"versions": env})
    if diag["kind"] == "ok":
        rep.add("A", "A.gpu", "GPU visibility", PASS,
                f"a GPU is visible to PyTorch: {', '.join(hw.get('device_names') or []) or 'device 0'}"
                f"{' (' + hw['gcn_arch'] + ')' if hw.get('gcn_arch') else ''}; run `python -m polony_app.ml gpu-smoke --device rocm` "
                "once to prove matmul / conv backward", detail={"diagnosis": diag})
    else:
        rep.add("A", "A.gpu", "GPU visibility", WARN,
                f"CPU only: {diag['message']}. Training on the CPU will be slow; check `python -m polony_app.ml gpu-smoke "
                f"--device rocm` for the details", examples=list(diag.get("hints", [])), detail={"diagnosis": diag})
    # ---- file descriptors
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        need = 256 + 8 * p.n_workers()
        lvl = PASS if soft == resource.RLIM_INFINITY or soft >= max(1024, need) else WARN
        rep.add("A", "A.fd_limit", "file-descriptor limit", lvl,
                f"soft limit {soft} (hard {hard}); about {need} are needed for {p.n_workers()} worker(s)"
                + ("" if lvl == PASS else "; raise it with `ulimit -n 4096`"), detail={"soft": soft, "hard": hard})
    except Exception:  # noqa: BLE001 - Windows
        rep.add("A", "A.fd_limit", "file-descriptor limit", INFO, "not available on this platform")
    # ---- data dir: inside the input? writable? case sensitivity
    dd, root = Path(p.data_dir), Path(p.input)
    if _is_within(dd, root if root.is_dir() else root.parent) and root.is_dir():
        rep.add("A", "A.data_dir_in_input", "--data-dir inside --input", FAIL,
                f"{dd} is inside the input directory: snapshots would be written into the historical data", count=1)
    base = _nearest_existing(dd)
    try:
        with tempfile.TemporaryDirectory(prefix=".polony-preflight-", dir=str(base)) as td:
            tdp = Path(td)
            (tdp / "a").write_text("x")
            case_insensitive = (tdp / "A").exists()
        rep.add("A", "A.data_dir_writable", "write permission to --data-dir", PASS,
                f"{base} is writable" + ("" if dd.exists() else f" ({dd} will be created by the import)"))
        env["data_dir_case_insensitive"] = bool(case_insensitive)
        rep.add("A", "A.fs_case", "filesystem case sensitivity (data dir)", INFO,
                "case-INsensitive filesystem" if case_insensitive else "case-sensitive filesystem")
    except Exception as exc:  # noqa: BLE001
        rep.add("A", "A.data_dir_writable", "write permission to --data-dir", FAIL,
                f"cannot write below {base}: {type(exc).__name__}: {exc}", count=1)
    try:      # input volume: no writes - compare a name with its case-swapped twin
        sample = next((Path(rp) for _rel, rp in walk["tiffs"] if rp.name.swapcase() != rp.name), None)
        if sample is not None:
            twin = sample.with_name(sample.name.swapcase())
            ci = twin.exists() and os.path.samefile(sample, twin)
            env["input_case_insensitive"] = bool(ci)
            rep.add("A", "A.fs_case_input", "filesystem case sensitivity (input)", INFO,
                    "the input volume is case-INsensitive (names differing only by case cannot coexist)" if ci
                    else "the input volume is case-sensitive")
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":
        longest = max((len(str(rp)) for _rel, rp in walk["tiffs"]), default=0) + len(str(dd)) // 4
        lp = None
        try:
            import winreg      # type: ignore
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
                lp = bool(winreg.QueryValueEx(k, "LongPathsEnabled")[0])
        except Exception:  # noqa: BLE001
            pass
        lvl = WARN if (longest > 240 and not lp) else PASS
        rep.add("A", "A.long_paths", "Windows long paths", lvl,
                f"longest input path {longest} characters; LongPathsEnabled={lp}"
                + ("; enable long paths or move the data to a shorter path" if lvl == WARN else ""))
    return env


def check_environment_resources(rep: Report, p: PreflightParams, summary: Optional[dict]) -> dict:
    """Disk and RAM against the dry-run estimates (needs the plan)."""
    from . import dataset as ds
    from .estimates import LAZY_ABSOLUTE_BYTES, LAZY_FRACTION_OF_RAM, available_memory_bytes, choose_load_mode
    out: dict = {}
    if not summary:
        rep.add("A", "A.disk", "free disk", INFO, "not evaluated: no plan could be built")
        return out
    est = int(summary["estimates"]["native_disk_bytes"])
    base = _nearest_existing(Path(p.data_dir) / "datasets")
    free = int(shutil.disk_usage(base).free)
    need = int(est * 1.1)
    out.update({"free_bytes": free, "snapshot_bytes": est, "required_bytes": need})
    if est == 0:
        rep.add("A", "A.disk", "free disk", INFO, f"no crop is planned; {_fmt(free)} free under {base}")
    elif free < need:
        rep.add("A", "A.disk", "free disk", FAIL,
                f"the snapshot needs about {_fmt(est)} (+10 % margin = {_fmt(need)}; the staging directory is renamed, not copied) "
                f"but only {_fmt(free)} are free under {base}", count=1, detail=out)
    elif free < 2 * need:
        rep.add("A", "A.disk", "free disk", WARN,
                f"{_fmt(free)} free vs {_fmt(need)} needed: enough for one snapshot but not for a second one (re-import with other "
                "options, interrupted staging directories)", count=1, detail=out)
    else:
        rep.add("A", "A.disk", "free disk", PASS, f"{_fmt(free)} free vs about {_fmt(need)} needed (snapshot {_fmt(est)} + 10 %)",
                detail=out)
    tr = summary["estimates"].get("training")
    avail = available_memory_bytes()
    out["available_ram_bytes"] = avail
    if tr:
        prep = int(tr["prepared_bytes"])
        mode, why = choose_load_mode(prep, avail)
        out.update({"prepared_bytes": prep, "load_mode": mode})
        lvl = PASS if mode == "eager" else WARN
        rep.add("A", "A.ram", "RAM vs prepared training size", lvl,
                f"prepared float32 network inputs at downscale {tr['downscale']}: {_fmt(prep)}; available RAM {_fmt(avail)}; train "
                f"would use the {mode} load mode ({why})" + ("" if mode == "eager" else "; a larger --downscale or lazy loading "
                                                              "keeps it bounded"), detail=out)
    else:
        rep.add("A", "A.ram", "RAM vs prepared training size", INFO, "no training crops planned")
    # peak RAM of the import itself (one page plane cached per worker)
    return out


# ====================================================================================================== B: input sanity
def _modal(counter: Counter):
    return counter.most_common(1)[0][0] if counter else None


def check_input(rep: Report, p: PreflightParams, walk: dict, recs: list, dups: dict, shas: dict) -> dict:
    ex = rep.max_examples
    n = len(walk["tiffs"])
    root = Path(p.input)
    facts: dict = {"n_tiffs": n, "extensions": dict(sorted(walk["ext_counts"].items()))}
    rep.add("B", "B.counts", "files by extension", PASS if n else FAIL,
            f"{walk['n_files']} file(s) in {walk['n_dirs']} sub-director{'y' if walk['n_dirs'] == 1 else 'ies'}; "
            f"{n} TIFF(s); by extension: " + ", ".join(f"{k}={v}" for k, v in sorted(walk["ext_counts"].items())) if n
            else f"no .tif / .tiff file below {root}", count=n, detail={"extensions": facts["extensions"]})
    others = [r for r in walk["other_images"]]
    if others:
        rep.add("B", "B.other_images", "non-TIFF image / ROI files", WARN,
                f"{len(others)} file(s) with image-like extensions are IGNORED by import-legacy", count=len(others), examples=others)
    if walk["archives"]:
        rep.add("B", "B.archives", "archives", WARN,
                f"{len(walk['archives'])} archive(s) are never opened (extract them first if they hold TIFFs)",
                count=len(walk["archives"]), examples=walk["archives"])
    if walk["dot_entries"]:
        rep.add("B", "B.dot_entries", "dot-files / dot-directories", WARN,
                f"{len(walk['dot_entries'])} hidden entr(y/ies) are skipped by import-legacy (e.g. .git, ._* macOS forks, .DS_Store)",
                count=len(walk["dot_entries"]), examples=walk["dot_entries"])
    if walk["symlink_dirs"]:
        rep.add("B", "B.symlink_dirs", "symlinked directories", WARN,
                f"{len(walk['symlink_dirs'])} symlinked director(y/ies) are not followed: TIFFs below them are NOT imported",
                count=len(walk["symlink_dirs"]), examples=walk["symlink_dirs"])
    outside = [r["rel"] for r in recs if r["outside_symlink"]]
    if walk["symlink_files"]:
        rep.add("B", "B.symlink_files", "symlinked files", WARN if outside else INFO,
                f"{len(walk['symlink_files'])} symlinked TIFF(s); {len(outside)} resolve outside --input or are broken: they are "
                "NEVER opened or followed (import-legacy skips them too, see tiffs.skipped_links) and are not imported"
                if outside else f"{len(walk['symlink_files'])} symlinked TIFF(s), all resolving inside --input",
                count=len(walk["symlink_files"]), examples=outside or walk["symlink_files"],
                detail={"skipped": outside})
    # ---- readable?
    zero = [r["rel"] for r in recs if r["error_kind"] == "zero_byte"]
    unread = [f"{r['rel']}: {r['error']}" for r in recs if r["error_kind"] == "unreadable"]
    trunc = [f"{r['rel']}: {r['error']}" for r in recs if r["error_kind"] == "truncated"]
    for fid, title, items, what in (("B.zero_byte", "zero-byte files", zero, "empty file(s)"),
                                    ("B.unreadable", "unreadable TIFFs", unread, "file(s) that cannot be opened as TIFF"),
                                    ("B.truncated", "truncated TIFFs", trunc,
                                     "file(s) whose pixel data does not fit in the file (the bridge would silently read fewer "
                                     "planes)")):
        if items:
            rep.add("B", fid, title, FAIL, f"{len(items)} {what}", count=len(items), examples=items)
        else:
            rep.add("B", fid, title, PASS, "none")
    ok = [r for r in recs if r["error_kind"] in (None, "truncated") and r["shape"] is not None]
    if not ok:
        return facts
    # ---- compression / dtype / planes / shape / size
    comp = Counter(r["compression"] for r in ok)
    rep.add("B", "B.compression", "compression", INFO if len(comp) == 1 else WARN,
            "; ".join(f"{k}: {v}" for k, v in comp.most_common()), count=len(ok), detail={"by_compression": dict(comp)},
            examples=[r["rel"] for r in ok if r["compression"] != _modal(comp)] if len(comp) > 1 else [])
    dt = Counter(r["dtype"] for r in ok)
    bad_dt = [f"{r['rel']}: {r['dtype']}" for r in ok if r["dtype"] not in SUPPORTED_DTYPES]
    rep.add("B", "B.dtype", "pixel dtype", WARN if bad_dt else PASS,
            "; ".join(f"{k}: {v}" for k, v in dt.most_common()) + (f"; {len(bad_dt)} file(s) have an unsupported type and are rejected "
                                                                  "(only uint8 / uint16; never converted)" if bad_dt else ""),
            count=len(bad_dt), examples=bad_dt, detail={"by_dtype": dict(dt)})
    lay_bad = [f"{r['rel']}: {r['layout_why']}" for r in ok if not r["layout_ok"]]
    if lay_bad:
        rep.add("B", "B.layout", "page layout", WARN, f"{len(lay_bad)} file(s) are not 2-D plane stacks (RGB composites, ...) and are "
                "rejected by import-legacy (unsupported_layout)", count=len(lay_bad), examples=lay_bad)
    good = [r for r in ok if r["layout_ok"]]
    planes = Counter(r["planes"] for r in good)
    mp = _modal(planes)
    odd_planes = [f"{r['rel']}: {r['planes']} plane(s)" for r in good if r["planes"] != mp]
    rep.add("B", "B.planes", "planes per file", WARN if odd_planes else PASS,
            f"modal {mp} plane(s); " + ", ".join(f"{k}x{v}" for k, v in planes.most_common()), count=len(odd_planes), examples=odd_planes,
            detail={"by_planes": dict(planes)})
    axes = Counter((r["axes"], len(r["shape"])) for r in good)
    odd_ax = [f"{r['rel']}: axes {r['axes']}" for r in good if (r["axes"], len(r["shape"])) != _modal(axes)]
    rep.add("B", "B.axes", "axes / shape", WARN if odd_ax else PASS, ", ".join(f"{a[0]}={v}" for a, v in axes.most_common()),
            count=len(odd_ax), examples=odd_ax)
    sizes = Counter((r["h"], r["w"]) for r in good)
    ms = _modal(sizes)
    odd_sz = [f"{r['rel']}: {r['h']}x{r['w']}" for r in good if (r["h"], r["w"]) != ms]
    rep.add("B", "B.size", "image size", WARN if odd_sz else PASS,
            f"modal {ms[0]}x{ms[1]} (HxW) in {sizes[ms]} of {len(good)} file(s)" + (f"; {len(odd_sz)} outlier(s): " + ", ".join(
                f"{k[0]}x{k[1]}={v}" for k, v in sizes.most_common()[1:6]) if odd_sz else ""),
            count=len(odd_sz), examples=odd_sz, detail={"by_size": {f"{k[0]}x{k[1]}": v for k, v in sizes.items()}})
    # ---- ImageJ / blobs
    non_ij = [r["rel"] for r in good if not r["is_imagej"]]
    rep.add("B", "B.imagej", "ImageJ metadata", WARN if non_ij else PASS,
            f"{len(good) - len(non_ij)} of {len(good)} file(s) carry ImageJ metadata" + ("; the others have no marks / grid" if non_ij else ""),
            count=len(non_ij), examples=non_ij)
    no_roi = [r["rel"] for r in good if r["is_imagej"] and not r["points"]]
    no_ov = [r["rel"] for r in good if r["is_imagej"] and r["n_grids"] == 0]
    rep.add("B", "B.roi_missing", "ROI (marks) blob", WARN if (no_roi or non_ij) else PASS,
            f"{len(no_roi) + len(non_ij)} file(s) have no point marks (rejected in marked-only mode: missing_roi)"
            if (no_roi or non_ij) else "every ImageJ file has point marks", count=len(no_roi) + len(non_ij), examples=no_roi + non_ij)
    rep.add("B", "B.grid_missing", "Overlays (grid) blob", WARN if (no_ov or non_ij) else PASS,
            f"{len(no_ov) + len(non_ij)} file(s) have no counting grid (rejected: missing_grid)" if (no_ov or non_ij)
            else "every ImageJ file has a counting grid", count=len(no_ov) + len(non_ij), examples=no_ov + non_ij)
    perr = [f"{r['rel']}: {'; '.join(r['parse_errors'])}" for r in good if r["parse_errors"]]
    perr += [f"{r['rel']}: {'; '.join(r['parse_errors'])}" for r in recs if r["parse_errors"] and r not in good]
    rep.add("B", "B.roi_parse", "ROI / Overlays parse errors", FAIL if perr else PASS,
            f"{len(perr)} file(s) with an unparseable ROI / Overlays blob (rejected: corrupt_metadata)" if perr else "all blobs parse",
            count=len(perr), examples=perr)
    multi_roi = [f"{r['rel']}: {r['n_point_blobs']} point ROI blobs" for r in good if r["n_point_blobs"] > 1]
    multi_grid = [f"{r['rel']}: {r['n_grids']} grids" for r in good if r["n_grids"] > 1]
    rep.add("B", "B.multiple_blobs", "multiple ROI / grid blobs", WARN if (multi_roi or multi_grid) else PASS,
            f"{len(multi_roi)} file(s) with several point ROIs and {len(multi_grid)} with several grids are rejected (ambiguous)"
            if (multi_roi or multi_grid) else "one point ROI and one grid per file", count=len(multi_roi) + len(multi_grid),
            examples=multi_roi + multi_grid)
    with_grid = [r for r in good if r["grid"] is not None]
    # ---- grid regularity / origin clusters
    if with_grid:
        def spec(r):
            xs, ys = r["grid"]
            return {"origin": (round(xs[0], 1), round(ys[0], 1)), "shape": (len(xs) - 1, len(ys) - 1),
                    "step": (round(float(np.median(np.diff(xs))), 1), round(float(np.median(np.diff(ys))), 1)),
                    "regular": bool(np.ptp(np.diff(xs)) <= 0.5 and np.ptp(np.diff(ys)) <= 0.5)}
        specs = {r["rel"]: spec(r) for r in with_grid}
        org = Counter(s["origin"] for s in specs.values())
        mo = _modal(org)
        odd = [f"{k}: origin {v['origin']}" for k, v in specs.items() if v["origin"] != mo]
        rep.add("B", "B.grid_origin", "grid origin clusters", WARN if len(org) > 1 else PASS,
                f"{len(org)} distinct grid origin(s) (x, y in ImageJ px): " + ", ".join(f"{k}x{v}" for k, v in org.most_common(8))
                + (f"; modal {mo}. Files on a non-modal origin (their counters are only consistent with their OWN grid; the bridge "
                   "uses the file's own overlay):" if len(org) > 1 else ""),
                count=len(odd), examples=odd, detail={"origins": {str(k): v for k, v in org.items()},
                                                      "modal": mo, "off_modal_files": [k for k, v in specs.items() if v["origin"] != mo]})
        shp = Counter((s["shape"], s["step"]) for s in specs.values())
        oddg = [f"{k}: {v['shape'][0]}x{v['shape'][1]} cells, step {v['step']}" for k, v in specs.items()
                if (v["shape"], v["step"]) != _modal(shp)]
        rep.add("B", "B.grid_shape", "grid shape / spacing", WARN if len(shp) > 1 else PASS,
                ", ".join(f"{k[0][0]}x{k[0][1]} cells step {k[1]}: {v}" for k, v in shp.most_common(5)), count=len(oddg), examples=oddg)
        irr = [k for k, v in specs.items() if not v["regular"]]
        rep.add("B", "B.grid_regular", "grid regularity", WARN if irr else PASS,
                f"{len(irr)} irregular grid(s) (spacing varies by more than 0.5 px)" if irr else "all grids are evenly spaced",
                count=len(irr), examples=irr)
        beyond = []
        for r in with_grid:
            xs, ys = r["grid"]
            if xs[0] - 0.5 < -1e-6 or ys[0] - 0.5 < -1e-6 or xs[-1] - 0.5 > r["w"] + 1e-6 or ys[-1] - 0.5 > r["h"] + 1e-6:
                beyond.append(r["rel"])
        rep.add("B", "B.grid_in_image", "grid inside the image", WARN if beyond else PASS,
                f"{len(beyond)} grid(s) extend beyond the pixel data (those cells cannot be exported)" if beyond else "all grids lie inside the image",
                count=len(beyond), examples=beyond)
        facts["grid_origins"] = {str(k): v for k, v in org.items()}
        facts["grid_off_modal"] = [k for k, v in specs.items() if v["origin"] != mo]
    # ---- counters
    with_pts = [r for r in good if r["pt"] is not None]
    if with_pts:
        nocount = [f"{r['rel']}: {r['counters']['n_without']} of {r['points']} point(s)" for r in with_pts if r["counters"]["n_without"]]
        allno = [r["rel"] for r in with_pts if r["counters"]["n_with"] == 0]
        rep.add("B", "B.counters_missing", "counters present", WARN if nocount else PASS,
                f"{len(nocount)} file(s) have marks without an ImageJ counter ({len(allno)} have none at all: rejected unless "
                "--acknowledge-no-counters)" if nocount else "every mark carries a counter", count=len(nocount), examples=nocount)
        gaps = [f"{r['rel']}: absent {r['counters']['absent'][:8]}" for r in with_pts if r["counters"]["absent"]]
        rep.add("B", "B.counters_absent", "absent counters", INFO if gaps else PASS,
                f"{len(gaps)} file(s) skip counter numbers (a field was not marked or numbering has gaps; lab-CSV field numbers will "
                "NOT equal counter + 1 after a gap)" if gaps else "counters are contiguous from 0", count=len(gaps), examples=gaps)
        ooo = [f"{r['rel']}: out of order {r['counters']['out_of_order'][:6]}, split runs {r['counters']['split_runs'][:6]}"
               for r in with_pts if r["counters"]["out_of_order"] or r["counters"]["split_runs"]]
        rep.add("B", "B.counters_order", "counter order", WARN if ooo else PASS,
                f"{len(ooo)} file(s) have counters clicked out of order or in several separate runs" if ooo else "counters appear in order",
                count=len(ooo), examples=ooo)
        # a counter whose marks lie in several cells
        span = []
        for r in with_pts:
            pt = r["pt"]
            by: dict = defaultdict(Counter)
            for k in range(len(pt["cx"])):
                if pt["status"][k] == 0 and pt["counter"][k] >= 0 and pt["cell"][k] >= 0:
                    by[int(pt["counter"][k])][int(pt["cell"][k])] += 1
            multi = {c: dict(v) for c, v in by.items() if len(v) > 1}
            if multi:
                span.append(f"{r['rel']}: counters {sorted(multi)[:6]} span several cells")
        rep.add("B", "B.counters_span", "counters spanning cells", WARN if span else PASS,
                f"{len(span)} file(s) have a counter whose marks lie in more than one grid cell (clicked across a border, or a "
                "shifted grid); minority marks are 'straddling'" if span else "every counter lies in one cell",
                count=len(span), examples=span)
        # positions
        pos_all: Counter = Counter()
        bad_pos, no_pos = [], []
        for r in with_pts:
            pp = r["pt"]["position"]
            pos_all.update(int(v) for v in pp)
            nb = int(((pp > r["planes"])).sum())
            nz = int((pp == 0).sum())
            if nb:
                bad_pos.append(f"{r['rel']}: {nb} point(s) with position > {r['planes']} plane(s)")
            if nz:
                no_pos.append(f"{r['rel']}: {nz} point(s) without a position")
        rep.add("B", "B.positions", "ImageJ positions", WARN if (bad_pos or no_pos) else PASS,
                "position (1-based plane) histogram: " + ", ".join(f"{k}={v}" for k, v in sorted(pos_all.items()))
                + (f"; {len(bad_pos)} file(s) with out-of-range and {len(no_pos)} with unspecified positions (those points are never "
                   "labels)" if (bad_pos or no_pos) else ""), count=len(bad_pos) + len(no_pos), examples=bad_pos + no_pos,
                detail={"positions": dict(sorted(pos_all.items()))})
        oob = [f"{r['rel']}: {int((r['pt']['status'] == 2).sum())} point(s) outside the image" for r in with_pts if (r["pt"]["status"] == 2).any()]
        rep.add("B", "B.points_outside_image", "points outside the image", WARN if oob else PASS,
                f"{len(oob)} file(s) have marks outside the {ms[0]}x{ms[1]} image (never labels)" if oob else "all marks lie inside the image",
                count=len(oob), examples=oob)
        outg = []
        for r in with_pts:
            if r["grid"] is None:
                continue
            pt = r["pt"]
            k = int(((pt["status"] == 0) & (pt["cell"] < 0)).sum())
            if k:
                outg.append(f"{r['rel']}: {k} point(s) outside every grid cell")
        rep.add("B", "B.points_outside_grid", "points outside the grid", WARN if outg else PASS,
                f"{len(outg)} file(s) have marks outside the grid (never labels)" if outg else "all marks lie inside a grid cell",
                count=len(outg), examples=outg)
        dup = [f"{r['rel']}: {int((r['pt']['status'] == 1).sum())} duplicate(s) within {DUP_TOL_PX:g} px" for r in with_pts
               if (r["pt"]["status"] == 1).any()]
        rep.add("B", "B.points_duplicate", "duplicate points", WARN if dup else PASS,
                f"{len(dup)} file(s) have marks within {DUP_TOL_PX:g} px of another mark on the same plane (dropped as duplicates)"
                if dup else "no duplicate marks", count=len(dup), examples=dup)
        near = [f"{r['rel']}: {r['pt']['n_close_pairs']} pair(s)" for r in with_pts if r["pt"]["n_close_pairs"]]
        rep.add("B", "B.points_near", "near-duplicate points", INFO if near else PASS,
                f"{len(near)} file(s) have mark pairs closer than {NEAR_PX:g} px (double clicks or touching polonies; kept)" if near
                else f"no mark pairs closer than {NEAR_PX:g} px", count=len(near), examples=near)
    # ---- byte-identical duplicates
    if dups:
        groups = [f"keep {v[0]}; alias {', '.join(v[1:])}" for _s, v in sorted(dups.items(), key=lambda kv: kv[1][0])]
        n_alias = sum(len(v) - 1 for v in dups.values())
        rep.add("B", "B.duplicates", "byte-identical duplicates", WARN,
                f"{len(dups)} group(s), {n_alias} redundant copy(ies): the first path in sorted order is KEPT, the others are recorded "
                "as aliases (one source, never two crops sets)", count=n_alias, examples=groups,
                detail={"groups": {k[:12]: v for k, v in dups.items()}})
    else:
        rep.add("B", "B.duplicates", "byte-identical duplicates", PASS, "none")
    # ---- names
    names = [r["rel"] for r in recs]
    nonascii = [r for r in names if any(ord(c) > 127 for c in r)]
    nfd = [r for r in names if unicodedata.normalize("NFC", r) != r]
    spaces = [r for r in names if " " in r]
    odd_chars = [r for r in names if re.search(r'[<>:"|?*\x00-\x1f]|[ .]$|\\', r)]
    long_names = [f"{r} ({len(r)})" for r in names if len(Path(r).name) > 120 or len(r) > 200]
    folded: dict[str, list] = defaultdict(list)
    for r in names:
        folded[unicodedata.normalize("NFC", r).casefold()].append(r)
    coll = [f"{' | '.join(v)}" for v in folded.values() if len(v) > 1]
    base: dict[str, list] = defaultdict(list)
    for r in names:
        base[Path(r).name].append(r)
    bcoll = [" | ".join(v) for v in base.values() if len(v) > 1]
    for fid, title, items, msg, lvl in (
            ("B.names_unicode", "non-ASCII names", nonascii, "name(s) with non-ASCII characters (identity maps must match them exactly)", INFO),
            ("B.names_nfd", "decomposed unicode names", nfd, "name(s) in NFD form (macOS): an NFC identity map will not match", WARN),
            ("B.names_spaces", "names with spaces", spaces, "name(s) with spaces (quote them in shell commands)", INFO),
            ("B.names_chars", "names with reserved characters", odd_chars, "name(s) with characters that are illegal on Windows / trailing dot or space", WARN),
            ("B.names_long", "long names", long_names, "long name(s) (Windows MAX_PATH hints)", WARN),
            ("B.names_case", "case-colliding names", coll, "name(s) that differ only by case / unicode normalisation", WARN),
            ("B.names_basename", "basename collisions", bcoll, "base name(s) used in several directories (use relative paths in the mapping files)", WARN)):
        rep.add("B", fid, title, lvl if items else PASS, f"{len(items)} {msg}" if items else "none", count=len(items), examples=items)
    # ---- scanner info
    infos = {r["rel"]: r["info"] for r in good if r["info"]}
    if infos:
        def key(i):
            return (i.get("pmt"), i.get("laser"), i.get("focus"), i.get("averaging"), i.get("label"))
        cl = Counter(key(i) for i in infos.values())
        mk = _modal(cl)
        odd = [f"{k}: PMT={key(v)[0]} laser={key(v)[1]} focus={key(v)[2]} label={key(v)[4]}" for k, v in infos.items() if key(v) != mk]
        rep.add("B", "B.scanner", "scanner settings clusters", WARN if len(cl) > 1 else PASS,
                f"{len(cl)} cluster(s) of (PMT, laser power, focus, averaging, label): "
                + "; ".join(f"PMT={k[0]} laser={k[1]} focus={k[2]} avg={k[3]} '{k[4]}' x{v}" for k, v in cl.most_common(6))
                + ("; files with non-modal settings are listed (different scan conditions may need their own rule)" if len(cl) > 1 else ""),
                count=len(odd), examples=odd, detail={"clusters": [{"pmt": k[0], "laser": k[1], "focus": k[2], "averaging": k[3],
                                                                    "label": k[4], "files": v} for k, v in cl.most_common()]})
        dates = sorted({i["datetime"][:10] for i in infos.values() if i.get("datetime")})
        mism = []
        for rel, i in infos.items():
            m = re.match(r"(\d{2})-(\d{2})-(\d{4})", Path(rel).name)
            if m and i.get("datetime") and f"{m.group(3)}-{m.group(2)}-{m.group(1)}" != i["datetime"][:10]:
                mism.append(f"{rel}: name says {m.group(1)}-{m.group(2)}-{m.group(3)}, scanner says {i['datetime'][:10]}")
        rep.add("B", "B.scan_dates", "scan dates", INFO if (dates or mism) else PASS,
                f"{', '.join(dates[:6])}" + (f"; {len(mism)} file(s) whose name date differs from the scanner date "
                                                            "(no scan session id exists: scan_session stays empty unless supplied)" if mism else ""),
                count=len(mism), examples=mism)
        labels = Counter(i.get("label") for i in infos.values())
        no_label = [r["rel"] for r in good if r["is_imagej"] and not (r["info"] or {}).get("label")]
        lvl = WARN if (len([k for k in labels if k]) > 1 or no_label) else PASS
        rep.add("B", "B.label_hint", "Info label consistency ('635 [W1]')", lvl,
                "ImageDescription label of page 0: " + ", ".join(f"'{k}' x{v}" for k, v in labels.most_common())
                + f"; {len(no_label)} ImageJ file(s) without a label. INFORMATION ONLY: the channel map decides roles",
                count=len(no_label) + (len(infos) - labels.most_common(1)[0][1] if len(labels) > 1 else 0),
                examples=no_label + [f"{k}: {v['label']}" for k, v in infos.items() if len(labels) > 1 and v.get("label") != _modal(labels)])
    else:
        rep.add("B", "B.scanner", "scanner settings clusters", INFO, "no ImageJ Info metadata found")
    return facts


# ====================================================================================================== D: independent derivation
def _cell_name(idx: int, ncol: int) -> str:
    r, c = divmod(int(idx), ncol)
    return f"r{r}c{c}"


def scan_reasons(rec: dict, mode: str, ack_no_counters: bool) -> list:
    """The reason codes a correct importer must give a file (mirrors the documented rejection rules, written from the
    documentation, not from legacy_bridge)."""
    if rec["error_kind"] in ("unreadable", "zero_byte"):
        return ["tiff_unreadable"]
    if not rec["layout_ok"]:
        return ["unsupported_layout"]
    if rec["dtype"] not in SUPPORTED_DTYPES:
        return ["unsupported_dtype"]
    if rec["parse_errors"]:
        return ["corrupt_metadata"]
    if rec["n_point_blobs"] > 1:
        return ["multiple_roi_blobs"]
    if rec["n_grids"] > 1:
        return ["multiple_grids"]
    out = []
    if not rec["points"] and mode == "marked-only":
        out.append("missing_roi")
    if rec["n_grids"] == 0:
        out.append("missing_grid")
    if out:
        return out
    if rec["points"] and rec["counters"]["n_with"] == 0 and not ack_no_counters:
        return ["no_counters"]
    return []


def manifest_entries_independent(man, rels_of: dict, by_base: dict) -> dict:
    """{rel: {page: {key: stated_count_or_None}}} from an examined-fields manifest (matching by relative path / alias, then by a
    unique base name)."""
    out: dict = {}
    for fe in man.files:
        f = fe.file.replace("\\", "/")
        rel = rels_of.get(f)
        if rel is None and "/" not in f:
            c = by_base.get(Path(f).name, set())
            rel = next(iter(c)) if len(c) == 1 else None
        if rel is None:
            continue
        tgt = out.setdefault(rel, {})
        for pg, ents in fe.pages.items():
            d = tgt.setdefault(int(pg), {})
            for e in ents:
                d[e.key()] = e.stated_count
    return out


def derive(rec: dict, pages: dict, *, mode: str, entries: Optional[dict], ack_cp1: bool, min_crop_px: int = MIN_CROP_PX) -> dict:
    """Independent re-derivation of everything the bridge decides for ONE file (see LEGACY_FINETUNING.md section 1).

    ``pages`` = {page: {"role", "train"}}; ``entries`` = {page: {key: stated count | None}} from the manifest (or None)."""
    pt = rec["pt"]
    h, w = rec["h"], rec["w"]
    xs, ys = rec["grid"]
    ncol, nrow = len(xs) - 1, len(ys) - 1
    ncell = ncol * nrow
    n = len(pt["cx"])
    ok = pt["status"] == 0
    # counter -> majority cell (first encountered wins a tie)
    cnt: dict = {}
    for k in range(n):
        if ok[k] and pt["counter"][k] >= 0 and pt["cell"][k] >= 0:
            cnt.setdefault(int(pt["counter"][k]), {})
            d = cnt[int(pt["counter"][k])]
            d[int(pt["cell"][k])] = d.get(int(pt["cell"][k]), 0) + 1
    counter_cell: dict = {}
    for c, d in cnt.items():
        best, bn = None, -1
        for cell, m in d.items():
            if m > bn:
                best, bn = cell, m
        counter_cell[c] = best
    cell_counters: dict = defaultdict(list)
    for c, cell in counter_cell.items():
        cell_counters[cell].append(c)
    train_pages = sorted(p for p, v in pages.items() if v["train"])

    def resolve(ent: dict) -> tuple:
        res, clash = {}, set()
        for page, d in ent.items():
            for key, stated in d.items():
                if key[0] == "counter":
                    cell = counter_cell.get(key[1])
                elif key[0] == "cell":
                    cell = key[1] * ncol + key[2] if (0 <= key[1] < nrow and 0 <= key[2] < ncol) else None
                else:
                    cell = counter_cell.get(key[1] - 1) if ack_cp1 else None
                if cell is None:
                    continue
                if (page, cell) in res:
                    clash.add((page, cell))
                res[(page, cell)] = stated
        return res, clash
    ents, clash = resolve(entries or {})
    tuples: dict = {}
    unplaced: Counter = Counter()          # usable marks without a usable plane, per cell
    for k in range(n):
        if ok[k] and pt["cell"][k] >= 0 and not (0 <= pt["page"][k] < rec["planes"]):
            unplaced[int(pt["cell"][k])] += 1
    for page in train_pages:
        own, stray, extra = defaultdict(list), defaultdict(list), defaultdict(list)
        for k in range(n):
            if pt["page"][k] != page or not ok[k]:
                continue
            c, ctr = int(pt["cell"][k]), int(pt["counter"][k])
            owner = counter_cell.get(ctr) if ctr >= 0 else None
            if c >= 0:
                if ctr < 0 or owner == c:
                    own[c].append(k)
                else:
                    stray[c].append(k)
            if owner is not None and owner != c:
                extra[owner].append(k)
        for cell in range(ncell):
            o, s, e = own.get(cell, []), stray.get(cell, []), extra.get(cell, [])
            if cell in unplaced:
                tuples[(page, cell)] = {"status": "conflict", "n": unplaced[cell], "labels": [], "stray": 0, "straddle_only": False}
                continue
            has = (page, cell) in ents
            stated = ents.get((page, cell))
            if (page, cell) in clash:
                tuples[(page, cell)] = {"status": "conflict", "n": len(o) + len(e), "labels": [], "stray": 0, "straddle_only": False}
                continue

            def status_for(nm: int) -> str:
                if mode == "marked-only":
                    if nm == 0:
                        return "unknown"
                    return "conflict" if (stated is not None and stated != nm) else "positive_marked"
                if not has:
                    return "unknown"
                if nm > 0:
                    return "conflict" if (stated is not None and stated != nm) else "positive_marked"
                if stated == 0:
                    return "verified_zero"
                return "unknown" if stated is None else "conflict"
            n_attr = len(o) + len(e) if o else 0
            st = status_for(n_attr)
            if st == "verified_zero" and (o or s or e):
                n_attr = len(o) + len(s) + len(e)
                st = status_for(n_attr)
            straddle = bool(not o and s and st == "unknown")
            examined = st in ("positive_marked", "verified_zero")
            tuples[(page, cell)] = {"status": st, "n": n_attr, "labels": sorted(o + s) if examined else [],
                                    "stray": len(s) if examined else 0, "straddle_only": straddle}
    # ---- crops (box = pixel centres inside the half-open canonical rectangle)
    crops: dict = {}
    skipped: dict = {}
    label_state: dict = {}
    for (page, cell), t in sorted(tuples.items()):
        if t["status"] not in ("positive_marked", "verified_zero"):
            continue
        r, c = divmod(cell, ncol)
        rect = (xs[c] - 0.5, ys[r] - 0.5, xs[c + 1] - 0.5, ys[r + 1] - 0.5)
        bx0, by0, bx1, by1 = (math.ceil(v) for v in rect)
        why = None
        if bx0 < 0 or by0 < 0 or bx1 > w or by1 > h:
            why = "leaves_image"
        elif bx1 - bx0 < min_crop_px or by1 - by0 < min_crop_px:
            why = "too_small"
        if why:
            skipped[(page, cell)] = why
            for k in t["labels"]:
                label_state[k] = "crop_not_exported"
            continue
        cw, ch = bx1 - bx0, by1 - by0
        pos = []
        for k in t["labels"]:
            rx, ry = float(pt["cx"][k]) - bx0, float(pt["cy"][k]) - by0
            if -0.5 <= rx < cw - 0.5 and -0.5 <= ry < ch - 0.5:
                pos.append((ry, rx))
                label_state[k] = "label"
            else:
                label_state[k] = "outside_crop_footprint"
        crops[(page, cell)] = {"box": (bx0, by0, bx1, by1), "points": sorted(pos)}
    # ---- per point status / reason
    rows = []
    dropped: Counter = Counter()
    for k in range(n):
        pg, cl, ps = int(pt["page"][k]), int(pt["cell"][k]), int(pt["status"][k])
        if ps == 2:
            reason = "out_of_bounds"
        elif ps == 1:
            reason = "duplicate"
        elif cl < 0:
            reason = "outside_grid"
        elif pg < 0:
            reason = "plane_out_of_range" if int(pt["position"][k]) > rec["planes"] else "plane_unspecified"
        elif pg not in train_pages:
            reason = "page_not_trained" if pg in pages else "page_unmapped"
        elif k in label_state:
            reason = None if label_state[k] == "label" else label_state[k]
        else:
            t = tuples.get((pg, cl))
            reason = ("cell_not_examined:straddling_other_counter" if (t and t["straddle_only"])
                      else f"cell_not_examined:{t['status'] if t else 'unknown'}")
        rows.append(reason)
        if reason:
            dropped[reason] += 1
    return {"counter_cell": counter_cell, "cell_counters": {c: sorted(v) for c, v in cell_counters.items()}, "tuples": tuples,
            "crops": crops, "skipped": skipped, "rows": rows, "dropped": dropped, "ncol": ncol, "nrow": nrow, "train_pages": train_pages}


def compare_with_plan(rel: str, rec: dict, mine: dict, bplan, bscan) -> list:
    """Disagreements between the independent derivation (``mine``) and the bridge's plan for one file. Each is a dict with the
    exact tuple/point."""
    out: list = []

    def bad(kind, key, bridge, independent):
        out.append({"file": rel, "kind": kind, "key": key, "bridge": bridge, "independent": independent})
    pt = rec["pt"]
    n = len(pt["cx"])
    if bscan.n_points != n:
        bad("point_count", "all", bscan.n_points, n)
        return out
    if (bscan.width, bscan.height, bscan.n_planes) != (rec["w"], rec["h"], rec["planes"]):
        bad("image_geometry", "WxHxplanes", [bscan.width, bscan.height, bscan.n_planes], [rec["w"], rec["h"], rec["planes"]])
    if [round(v, 3) for v in bscan.xs] != [round(v, 3) for v in rec["grid"][0]] or [round(v, 3) for v in bscan.ys] != [round(v, 3) for v in rec["grid"][1]]:
        bad("grid_lines", "xs/ys", [list(bscan.xs), list(bscan.ys)], [list(rec["grid"][0]), list(rec["grid"][1])])
    # ---- per point scan arrays
    for k in range(n):
        got = (round(float(bscan.x[k]), 6), round(float(bscan.y[k]), 6), int(bscan.page[k]), int(bscan.counter[k]),
               int(bscan.pstatus[k]), int(bscan.cell[k]))
        exp = (round(float(pt["cx"][k]), 6), round(float(pt["cy"][k]), 6), int(pt["page"][k]), int(pt["counter"][k]),
               int(pt["status"][k]), int(pt["cell"][k]))
        if got != exp:
            names = ("x", "y", "page", "counter", "status", "cell")
            diff = [f"{nm}: bridge {a} vs independent {b}" for nm, a, b in zip(names, got, exp) if a != b]
            bad("point", f"point {k} (ImageJ x={float(rec['raw_points']['x'][k]):g}, y={float(rec['raw_points']['y'][k]):g}, "
                         f"counter={exp[3]}, position={int(pt['position'][k])})", dict(zip(names, got)), "; ".join(diff))
    # ---- counter -> cell
    bcc = {int(c): int(v) for c, v in bscan.counter_cell.items()}
    if bcc != {int(c): int(v) for c, v in mine["counter_cell"].items()}:
        for c in sorted(set(bcc) | set(mine["counter_cell"])):
            if bcc.get(c) != mine["counter_cell"].get(c):
                bad("counter_cell", f"counter {c}", bcc.get(c), mine["counter_cell"].get(c))
    # ---- tuples
    bt = bplan.tuples
    if set(bt) != set(mine["tuples"]):
        bad("tuple_set", "(page, cell) keys", sorted(set(bt) - set(mine["tuples"]))[:5], sorted(set(mine["tuples"]) - set(bt))[:5])
    ncol = mine["ncol"]
    for key, m in sorted(mine["tuples"].items()):
        b = bt.get(key)
        if b is None:
            continue
        page, cell = key
        label = f"page {page} (ImageJ position {page + 1}), cell {_cell_name(cell, ncol)} (index {cell})"
        bst = b.exam.status
        bstraddle = (b.exam.reason == "only straddling points from another counter")
        if bst != m["status"] or bstraddle != m["straddle_only"]:
            bad("tuple_status", label, f"{bst}{' (straddle-only)' if bstraddle else ''}, {b.n_marks} mark(s), reason: {b.exam.reason}",
                f"{m['status']}{' (straddle-only)' if m['straddle_only'] else ''}, {m['n']} mark(s)")
            continue
        if m["status"] in ("positive_marked", "verified_zero"):
            if int(b.n_marks) != int(m["n"]):
                bad("tuple_marks", label, int(b.n_marks), int(m["n"]))
            bl = sorted((round(float(bscan.x[i]), 6), round(float(bscan.y[i]), 6)) for i in b.labels)
            ml = sorted((round(float(pt["cx"][i]), 6), round(float(pt["cy"][i]), 6)) for i in m["labels"])
            if bl != ml:
                bad("tuple_labels", label, bl, ml)
            if int(b.n_stray) != int(m["stray"]):
                bad("tuple_stray", label, int(b.n_stray), int(m["stray"]))
        elif int(b.n_marks) != int(m["n"]) and m["status"] != "unknown":
            bad("tuple_marks", label, int(b.n_marks), int(m["n"]))
    # ---- point rows (status / reason) and dropped totals
    rows = bplan.ext.rows
    if len(rows) != n:
        bad("rows_count", "points.csv rows", len(rows), n)
    else:
        for k in range(n):
            brow = rows[k]
            breason = None if brow["status"] == "confirmed" else str(brow["basis"]).replace("dropped:", "", 1)
            if breason != mine["rows"][k]:
                bad("point_row", f"point {k} (page {int(pt['page'][k])}, canonical x={float(pt['cx'][k]):.3f}, y={float(pt['cy'][k]):.3f})",
                    breason or "label", mine["rows"][k] or "label")
    if dict(bplan.dropped) != dict(mine["dropped"]):
        bad("dropped_by_reason", "file totals", dict(sorted(bplan.dropped.items())), dict(sorted(mine["dropped"].items())))
    # ---- crops (only when a channel map made them)
    cands = {(c.page, int(_ID_RC(c.region.id, ncol))): c for c in bplan.ext.cands}
    if bplan.ext.cands or any(True for _ in mine["crops"]) and bplan.ext.cands is not None and bplan.ext.cands != []:
        if set(cands) != set(mine["crops"]):
            bad("crop_set", "(page, cell)", sorted(set(cands) - set(mine["crops"]))[:5], sorted(set(mine["crops"]) - set(cands))[:5])
        for key, c in cands.items():
            m = mine["crops"].get(key)
            if m is None:
                continue
            if tuple(c.box_file) != tuple(m["box"]):
                bad("crop_box", f"page {key[0]} cell {_cell_name(key[1], ncol)}", list(c.box_file), list(m["box"]))
            bp = sorted((round(float(a), 6), round(float(b), 6)) for a, b in c.points)
            mp = sorted((round(a, 6), round(b, 6)) for a, b in m["points"])
            if bp != mp:
                bad("crop_points", f"page {key[0]} cell {_cell_name(key[1], ncol)}", bp, mp)
        bsk = {(pg, cl): why for pg, cl, why in bplan.skipped}
        if set(bsk) != set(mine["skipped"]):
            bad("crop_skipped", "(page, cell)", sorted(bsk), sorted(mine["skipped"]))
    return out


def _ID_RC(region_id: str, ncol: int) -> int:
    m = re.match(r"r(\d+)c(\d+)$", region_id)
    return int(m.group(1)) * ncol + int(m.group(2)) if m else -1


def bright_check(path: Path, page: int, coords: np.ndarray, rng: np.random.Generator, extent: tuple, max_marks: int = 300) -> Optional[dict]:
    """Median distance (px, capped at radius + 1) of marks to the nearest local intensity maximum of one plane, and the same
    statistic for uniformly random control positions."""
    import tifffile
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    try:
        with tifffile.TiffFile(str(path)) as tif:
            plane = np.asarray(tif.pages[page].asarray(), dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None
    sm = ndi.gaussian_filter(plane, 1.5)
    mx = ndi.maximum_filter(sm, size=7)
    med = float(np.median(sm))
    mad = float(np.median(np.abs(sm - med))) * 1.4826 + 1e-6
    peaks = np.argwhere((sm == mx) & (sm > med + 3.0 * mad))
    if len(peaks) < 5 or len(coords) == 0:
        return None
    tree = cKDTree(peaks[:, ::-1].astype(float))        # (x, y)
    if len(coords) > max_marks:
        coords = coords[rng.choice(len(coords), max_marks, replace=False)]
    d, _ = tree.query(coords, distance_upper_bound=BRIGHT_RADIUS + 1)
    d = np.where(np.isfinite(d), d, BRIGHT_RADIUS + 1)
    x0, y0, x1, y1 = extent
    ctrl = np.column_stack([rng.uniform(x0, x1, len(coords)), rng.uniform(y0, y1, len(coords))])
    dc, _ = tree.query(ctrl, distance_upper_bound=BRIGHT_RADIUS + 1)
    dc = np.where(np.isfinite(dc), dc, BRIGHT_RADIUS + 1)
    return {"n_marks": int(len(coords)), "median_mark_px": float(np.median(d)), "median_random_px": float(np.median(dc)),
            "within_3px_marks": float((d <= 3).mean()), "within_3px_random": float((dc <= 3).mean()), "n_peaks": int(len(peaks))}


def pick_sample(rels: list, spec: str, odd: set) -> list:
    if str(spec).lower() == "all":
        return sorted(rels)
    try:
        k = max(0, int(spec))
    except ValueError:
        raise MLError(f"--sample must be an integer or 'all' (got {spec!r})")
    ranked = sorted(rels, key=lambda r: hashlib.sha256(r.encode("utf-8")).hexdigest())[:k]
    return sorted(set(ranked) | (set(rels) & set(odd)))


def oddities(recs: list, facts: dict) -> set:
    """Files that deserve an independent look whatever the random sample is."""
    odd: set = set()
    off = set(facts.get("grid_off_modal", []))
    sizes = Counter((r["h"], r["w"]) for r in recs if r["layout_ok"])
    ms = _modal(sizes)
    for r in recs:
        rel = r["rel"]
        if rel in off or r["error_kind"] or not r["layout_ok"] or r["parse_errors"] or r["n_point_blobs"] > 1 or r["n_grids"] > 1:
            odd.add(rel)
            continue
        if (r["h"], r["w"]) != ms or r["dtype"] not in SUPPORTED_DTYPES or not r["points"]:
            odd.add(rel)
            continue
        pt, cs = r["pt"], r["counters"]
        if pt is not None and ((pt["status"] != 0).any() or ((pt["status"] == 0) & (pt["cell"] < 0)).any() or cs["out_of_order"]
                               or cs["split_runs"] or cs["absent"] or cs["n_without"] or (pt["position"] == 0).any()
                               or (pt["position"] > r["planes"]).any()):
            odd.add(rel)
    return odd


# ====================================================================================================== orchestration
def run_preflight(p: PreflightParams) -> dict:
    """Run every check; returns the report dict (schema ``polony-preflight``). Writes nothing."""
    from . import legacy_bridge as lb
    from ..fields import examined as exm
    t_start = time.time()
    timing: dict = {}
    rep = Report(p.max_examples)
    root = Path(p.input)
    if not root.exists():
        raise MLError(f"--input {root} does not exist")
    logging.getLogger("tifffile").setLevel(logging.CRITICAL)
    workers = p.n_workers()
    opts_kw = dict(mode=p.mode, acknowledge_no_counters=p.acknowledge_no_counters, allow_unconfirmed_identity=p.allow_unconfirmed_identity,
                   acknowledge_field_equals_counter_plus_one=p.acknowledge_field_equals_counter_plus_one,
                   acknowledge_lab_csv_field_numbering=p.acknowledge_lab_csv_field_numbering, seed=p.seed, fractions=p.fractions,
                   level=p.level, valid_mask=p.valid_mask, downscale=p.downscale, normaliser=p.normaliser, batch_size=p.batch_size, epochs=p.epochs)

    # ---------------------------------------------------------------- walk + B (headers)
    t0 = time.time()
    p.say(f"[preflight] walking {root}")
    walk = walk_input(root)
    base_root = root if root.is_dir() else root.parent
    files = []
    for rel, path in walk["tiffs"]:
        outside = Path(path).is_symlink() and (not _is_within(path, base_root) or not Path(os.path.realpath(path)).is_file())
        files.append((rel, path, outside))
    env = check_environment_static(rep, p, walk)
    timing["walk_s"] = round(time.time() - t0, 3)

    t0 = time.time()
    p.say(f"[preflight] B: header scan of {len(files)} TIFF(s) with {workers} worker(s) (no pixel decode)")
    recs: list = []

    def work(t):
        return inspect_tiff(t[0], t[1], outside_symlink=t[2])
    with ThreadPoolExecutor(max_workers=workers) as exr:
        for i, r in enumerate(exr.map(work, files), 1):
            recs.append(r)
            if i % 100 == 0:
                p.say(f"[preflight]   header scan {i}/{len(files)} file(s)")
    recs.sort(key=lambda r: r["rel"])
    timing["headers_s"] = round(time.time() - t0, 3)

    t0 = time.time()
    p.say("[preflight] B: byte-identical duplicates (hashing only files that share a size)")
    need_sha = _needs_sha(p)
    dups, shas = find_duplicates([(r["rel"], dict((f[0], f[1]) for f in files)[r["rel"]]) for r in recs if not r["outside_symlink"]],
                                 workers, hash_all=need_sha)
    timing["hash_s"] = round(time.time() - t0, 3)
    facts = check_input(rep, p, walk, recs, dups, shas)
    alias_of = {a: v[0] for v in dups.values() for a in v[1:]}
    kept_recs = [r for r in recs if r["rel"] not in alias_of]

    # ---------------------------------------------------------------- C (mapping files)
    t0 = time.time()
    p.say("[preflight] C: mapping files")
    cmres = check_mappings(rep, p, kept_recs, dups, shas, alias_of, opts_kw)
    timing["mapping_s"] = round(time.time() - t0, 3)

    # ---------------------------------------------------------------- plan (the bridge's own dry-run machinery)
    t0 = time.time()
    p.say("[preflight] building the bridge's dry-run plan (scan of every file, eligibility, estimates) ...")
    plan, plan_err = None, None
    try:
        plan = lb.build_plan(root, lb.BridgeOptions(**opts_kw), channel_map=p.channel_map, identity_map=p.identity_map,
                             examined_fields=p.examined_fields, lab_csv=[Path(c) for c in p.lab_csv], lab_series=p.lab_series or None,
                             dry_run=True, data_dir=Path(p.data_dir), say=lambda m: p.say(f"[preflight]   bridge: {m}"))
    except MLError as exc:
        plan_err = f"REFUSED by the bridge: {exc}"
    except Exception as exc:  # noqa: BLE001
        plan_err = f"the bridge CRASHED ({type(exc).__name__}: {exc})"
    timing["plan_s"] = round(time.time() - t0, 3)
    if plan_err:
        rep.add("D", "D.plan", "bridge dry-run plan", FAIL, f"the plan could not be built: {plan_err}", count=1,
                detail={"error": plan_err})
    summary = plan.summary if plan else None

    # ---------------------------------------------------------------- D (independent cross-check)
    t0 = time.time()
    cross = {"ran": False}
    if plan is not None:
        cross = cross_check(rep, p, plan, recs, kept_recs, facts, walk, opts_kw)
    timing["cross_check_s"] = round(time.time() - t0, 3)

    # ---------------------------------------------------------------- A resources + E
    res = check_environment_resources(rep, p, summary)
    policy = policy_summary(rep, p, plan, summary, plan_err)
    verdict, reasons = rep.verdict()
    timing["total_s"] = round(time.time() - t_start, 3)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(rss) * (1 if sys.platform == "darwin" else 1024)
    except Exception:  # noqa: BLE001
        peak = None
    exit_code = EXIT_NOT_READY if verdict == V_NOT else EXIT_OK
    doc = {
        "schema": SCHEMA, "version": VERSION, "verdict": verdict, "exit_code": exit_code, "reasons": reasons,
        "input": str(root), "data_dir": str(p.data_dir), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "options": {"mode": p.mode, "sample": p.sample, "workers": workers, "downscale": p.downscale, "normaliser": p.normaliser,
                    "batch_size": p.batch_size, "epochs": p.epochs, "channel_map": _s(p.channel_map), "identity_map": _s(p.identity_map),
                    "examined_fields": _s(p.examined_fields), "lab_csv": [str(c) for c in p.lab_csv]},
        "counts": rep.counts(), "n_files": {"tiffs": len(recs), "unique": len(kept_recs), "aliases": len(alias_of)},
        "findings": [f.as_dict() for f in rep.findings],
        "checks": {c: [f.as_dict() for f in rep.findings if f.check == c] for c in "ABCDE"},
        "cross_check": cross, "policy": policy, "environment": env, "resources": res,
        "timing": timing, "peak_rss_bytes": peak,
        "next_commands": next_commands(p, verdict, rep),
    }
    return _jsonable(doc)


def _s(x) -> Optional[str]:
    return None if x is None else str(x)


def _needs_sha(p: PreflightParams) -> bool:
    """Do mapping files need real sha256 values (rule match.sha256, manifest sha256)?"""
    try:
        if p.channel_map:
            d = json.loads(Path(p.channel_map).read_text(encoding="utf-8-sig"))
            if any("sha256" in (r.get("match") or {}) for r in (d.get("rules") or []) if isinstance(r, dict)):
                return True
        if p.examined_fields and Path(p.examined_fields).suffix.lower() != ".csv":
            d = json.loads(Path(p.examined_fields).read_text(encoding="utf-8-sig"))
            if any(isinstance(f, dict) and f.get("sha256") for f in (d.get("files") or [])):
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


# ---------------------------------------------------------------------------------------------------- C
def check_mappings(rep: Report, p: PreflightParams, kept: list, dups: dict, shas: dict, alias_of: dict, opts_kw: dict) -> dict:
    """Channel map, identity map, examined-fields manifest and lab CSV through the bridge's OWN loaders."""
    from . import legacy_bridge as lb
    out: dict = {}
    aliases: dict = defaultdict(list)
    for a, k in alias_of.items():
        aliases[k].append(a)
    stubs = []
    for i, r in enumerate(kept):
        sc = lb.SourceScan(source_id=f"stub{i:05d}", rel=r["rel"], path=Path(p.input) / r["rel"], sha256=shas.get(r["rel"], ""),
                           size=r["size"], mtime_ns=0)
        sc.alias_paths = sorted(aliases.get(r["rel"], []))
        sc.n_planes = r["planes"]
        stubs.append(sc)
    all_rels = {r["rel"] for r in kept} | set(alias_of)
    by_base: dict = defaultdict(set)
    for r in all_rels:
        by_base[Path(r).name].add(r)
    mode_needs_manifest = p.mode == "examined-fields-manifest"
    # ---------------- channel map
    cm = None
    if p.channel_map is None:
        rep.add("C", "C.channel_map", "channel map", FAIL,
                "no --channel-map: which page shows which channel is never inferred; a real import is refused without one "
                "(write a template: import-legacy --input DIR --write-templates OUT)", count=1)
    else:
        try:
            cm = lb.load_channel_map(p.channel_map)
            if not cm.confirmed_on.strip():
                rep.add("C", "C.channel_map_date", "channel map confirmed_on", WARN, "confirmed_on is empty (confirmed_by and "
                        "source_of_truth are filled in)", count=1)
            rep.add("C", "C.channel_map", "channel map", PASS,
                    f"{Path(p.channel_map).name}: confirmed by '{cm.confirmed_by}' on '{cm.confirmed_on}' ({cm.source_of_truth}); "
                    f"{len(cm.defaults)} default(s), {len(cm.rules)} rule(s), {len(cm.files)} file entr(y/ies)",
                    detail={"describe": cm.describe()})
        except MLError as exc:
            msg = str(exc)
            unconfirmed = "confirmed_by' is empty" in msg or "source_of_truth' is empty" in msg
            rep.add("C", "C.channel_map", "channel map UNCONFIRMED" if unconfirmed else "channel map invalid", FAIL, msg, count=1)
    if cm is not None:
        no_map, conflicts, bad_page, no_train, origins = [], [], [], [], Counter()
        for sc in stubs:
            sc.identity = None
        # identity first: rules may match on assay / specimen / slide
        _attach_identity(p, stubs, lb)
        used_rules: Counter = Counter()
        for sc, r in zip(stubs, kept):
            try:
                got = lb.resolve_mapping(cm, sc)
            except MLError as exc:
                conflicts.append(f"{sc.rel}: {exc}")
                continue
            if got is None:
                no_map.append(sc.rel)
                continue
            pages, origin = got
            origins[origin] += 1
            if r["layout_ok"] and r["error_kind"] is None:
                bad = [pg for pg in pages if pg >= r["planes"]]
                if bad:
                    bad_page.append(f"{sc.rel}: maps page(s) {bad} but has {r['planes']} plane(s) ({origin})")
                elif not any(v["train"] for v in pages.values()):
                    no_train.append(sc.rel)
        for rid, match, _pg in cm.rules:
            used_rules[rid] = sum(1 for sc in stubs if lb._rule_matches(match, sc))
        stale = sorted(f for f in cm.files if f not in all_rels)
        rep.add("C", "C.map_unmapped", "files without a mapping", FAIL if no_map else PASS,
                f"{len(no_map)} file(s) are covered by no default, rule or file entry (rejected: channel_mapping_unresolved)"
                if no_map else "every file has a mapping", count=len(no_map), examples=no_map)
        rep.add("C", "C.map_conflicts", "conflicting rules", FAIL if conflicts else PASS,
                f"{len(conflicts)} file(s) match two rules that assign different pages / roles (the import is REFUSED)" if conflicts
                else "no file matches conflicting rules", count=len(conflicts), examples=conflicts)
        rep.add("C", "C.map_pages", "mapped pages exist", FAIL if bad_page else PASS,
                f"{len(bad_page)} file(s) are mapped on a page beyond their planes (rejected)" if bad_page else "all mapped pages exist",
                count=len(bad_page), examples=bad_page)
        if no_train:
            rep.add("C", "C.map_train", "no trained page", WARN, f"{len(no_train)} file(s) have no page with train: true (rejected)",
                    count=len(no_train), examples=no_train)
        rep.add("C", "C.map_stale", "mapping entries for non-existing files", FAIL if stale else PASS,
                f"{len(stale)} 'files' entr(y/ies) of the channel map name no TIFF below --input (the override is silently unused): "
                "check spelling, relative path and case" if stale else "every 'files' entry exists",
                count=len(stale), examples=stale)
        dead = [rid for rid, n in used_rules.items() if n == 0]
        rep.add("C", "C.map_dead_rules", "rules matching no file", FAIL if dead else PASS,
                f"{len(dead)} rule(s) match no file (a BLOCKER: the override would be silently unused): {', '.join(dead)}" if dead else "every rule matches at least one file",
                count=len(dead), examples=dead)
        out["origins"] = dict(origins)
    # ---------------- identity map
    if p.identity_map is None:
        rep.add("C", "C.identity_map", "identity map", WARN,
                "no --identity-map: identities are INFERRED from file names and a real import refuses to split on them "
                "(unless --allow-unconfirmed-identity)", count=1)
    else:
        try:
            table = lb.load_identity_map(p.identity_map, level=p.level)
            imap, idrep = lb.apply_identity_map(table, stubs)
            rep.add("C", "C.identity_map", "identity map", PASS, f"{Path(p.identity_map).name}: {idrep['rows']} row(s), "
                    f"{idrep['matched']} matched to files", detail={"rows": idrep["rows"], "matched": idrep["matched"]})
            for fid, title, key, lvl, msg in (
                    ("C.identity_unconfirmed", "unconfirmed identity rows", "unconfirmed_rows", WARN, "row(s) have confirmed: false (their files stay INFERRED)"),
                    ("C.identity_missing", "identity rows for non-existing files", "missing_on_disk", WARN, "row(s) name no TIFF below --input"),
                    ("C.identity_ambiguous", "ambiguous identity basenames", "ambiguous_basename", FAIL, "row(s) use a base name shared by several files (use the relative path)"),
                    ("C.identity_unlisted", "files missing from the identity map", "unlisted_files", WARN, "TIFF(s) are not in the map (identity inferred, unconfirmed)")):
                items = idrep.get(key) or []
                rep.add("C", fid, title, lvl if items else PASS, f"{len(items)} {msg}" if items else "none", count=len(items), examples=items)
            empty = [r["file"] for r in table.rows if r["confirmed"] and not (r["specimen_id"] and r["slide_id"])]
            if empty:
                rep.add("C", "C.identity_partial", "identity rows without specimen or slide", WARN,
                        f"{len(empty)} confirmed row(s) lack a specimen_id or a slide_id (replicates may not group)", count=len(empty), examples=empty)
        except MLError as exc:
            rep.add("C", "C.identity_map", "identity map invalid", FAIL, str(exc), count=1)
    # ---------------- basename collisions
    coll = {k: sorted(v) for k, v in by_base.items() if len(v) > 1}
    rep.add("C", "C.basename", "basename collisions", WARN if coll else PASS,
            f"{len(coll)} base name(s) are used by several files; mapping / manifest entries must use relative paths" if coll else "none",
            count=len(coll), examples=[" | ".join(v) for v in coll.values()])
    # ---------------- examined fields / lab csv
    if mode_needs_manifest and not (p.examined_fields or p.lab_csv):
        rep.add("C", "C.examined", "examined-fields manifest", FAIL,
                "--mode examined-fields-manifest needs --examined-fields FILE (or --lab-csv): nothing is inferred", count=1)
    if p.examined_fields:
        try:
            man = lb.load_examined(p.examined_fields)
            try:
                ents, problems = lb.manifest_entries_for(man, [s for s in stubs if s.sha256 or not _needs_sha(p)])
                rep.add("C", "C.examined", "examined-fields manifest", FAIL if problems else PASS,
                        f"{Path(p.examined_fields).name}: {len(man.files)} file entr(y/ies), source '{man.source_kind}'; "
                        + (f"{len(problems)} entr(y/ies) name no TIFF" if problems else "all file entries resolve"),
                        count=len(problems), examples=problems)
            except MLError as exc:
                rep.add("C", "C.examined", "examined-fields manifest unusable", FAIL, str(exc), count=1)
        except MLError as exc:
            rep.add("C", "C.examined", "examined-fields manifest invalid", FAIL, str(exc), count=1)
    if p.lab_csv:
        try:
            _attach_identity(p, stubs, lb)
            lopts = lb.BridgeOptions(**opts_kw)
            _m, info = lb.lab_csv_manifest([Path(c) for c in p.lab_csv], p.lab_series or None, stubs, lopts)
            bad = [f"ambiguous: {x}" for x in info["ambiguous_rows"]] + [f"unmatched: {x}" for x in info["unmatched_rows"]]
            rep.add("C", "C.lab_csv", "lab CSV", WARN if bad else PASS,
                    f"{info['rows']} CSV row(s), {info['applied_rows']} matched to a slide" + (f"; {len(bad)} unusable" if bad else ""),
                    count=len(bad), examples=bad)
        except MLError as exc:
            rep.add("C", "C.lab_csv", "lab CSV invalid", FAIL, str(exc), count=1)
    return out


def _attach_identity(p: PreflightParams, stubs: list, lb) -> None:
    imap = {}
    if p.identity_map:
        try:
            imap, _rep = lb.apply_identity_map(lb.load_identity_map(p.identity_map, level=p.level), stubs)
        except MLError:
            imap = {}
    for sc in stubs:
        sc.identity = lb._identity_of(sc, imap)


# ---------------------------------------------------------------------------------------------------- D
def cross_check(rep: Report, p: PreflightParams, plan, recs: list, kept_recs: list, facts: dict, walk: dict, opts_kw: dict) -> dict:
    from . import legacy_bridge as lb
    from ..fields import examined as exm
    by_rel = {r["rel"]: r for r in recs}
    plans = {pl.scan.rel: pl for pl in plan.plans}
    scans = {s.rel: s for s in plan.scan.sources}
    summary = plan.summary
    dis: list = []
    # ---- discovery agrees with the bridge
    mine_all = sorted(r["rel"] for r in recs if not r["outside_symlink"])
    bridge_all = sorted([s.rel for s in plan.scan.sources] + [a for s in plan.scan.sources for a in s.alias_paths])
    if mine_all != bridge_all:
        only_m = sorted(set(mine_all) - set(bridge_all))
        only_b = sorted(set(bridge_all) - set(mine_all))
        dis.append({"file": "(discovery)", "kind": "discovery", "key": "TIFF list", "bridge": only_b[:10], "independent": only_m[:10]})
    sk_b = sorted(r for r, w in getattr(plan.scan, "skipped_links", []) if "directory" not in w)
    sk_m = sorted(r["rel"] for r in recs if r["outside_symlink"])
    if sk_b != sk_m:
        dis.append({"file": "(symlinks)", "kind": "skipped_links", "key": "symlinks skipped", "bridge": sk_b[:10], "independent": sk_m[:10]})
    # ---- acceptance agrees (all files: cheap)
    odd = oddities(kept_recs, facts)
    sample = pick_sample([r["rel"] for r in kept_recs], p.sample, odd)
    map_codes = {"channel_mapping_unresolved", "channel_mapping_page_out_of_range", "no_mapped_pages"}
    n_acc_checked = 0
    for r in kept_recs:
        sc = scans.get(r["rel"])
        if sc is None or r["error_kind"] in ("symlink_outside", "truncated"):
            continue
        mine = scan_reasons(r, p.mode, p.acknowledge_no_counters)
        theirs = [c for c, _m in sc.reasons if c not in map_codes]
        n_acc_checked += 1
        if (mine[:1] or [None]) != (theirs[:1] or [None]):
            dis.append({"file": r["rel"], "kind": "acceptance", "key": "scan reason", "bridge": theirs[:2] or "accepted",
                        "independent": mine[:2] or "accepted"})
    # ---- mapping pages (from the plan; mapping itself is checked in C)
    per_file = (summary.get("channel_mappings") or {}).get("per_file") or {}
    man = None
    entries_by_rel: dict = {}
    if p.examined_fields or p.lab_csv:
        # the manifest entries as the bridge understood them: re-read the manifest ourselves
        try:
            if p.examined_fields:
                man = exm.load_manifest(p.examined_fields)
            else:
                man, _i = lb.lab_csv_manifest([Path(c) for c in p.lab_csv], p.lab_series or None,
                                              [s for s in plan.scan.sources if not s.rejected], lb.BridgeOptions(**opts_kw))
            rels_of = {r: s.rel for s in plan.scan.sources for r in [s.rel, *s.alias_paths]}
            bb: dict = defaultdict(set)
            for r, k in rels_of.items():
                bb[Path(r).name].add(k)
            entries_by_rel = manifest_entries_independent(man, rels_of, bb)
        except Exception as exc:  # noqa: BLE001
            dis.append({"file": "(manifest)", "kind": "manifest", "key": "re-read", "bridge": "ok", "independent": f"{type(exc).__name__}: {exc}"})
    n_files = n_tuples = n_points = n_labels = 0
    sampled_ok: list = []
    p.say(f"[preflight] D: independent re-derivation for {len(sample)} file(s) ({'all' if str(p.sample).lower() == 'all' else p.sample} + {len(odd)} odd)")
    for rel in sample:
        r = by_rel.get(rel)
        sc = scans.get(rel)
        pl = plans.get(rel)
        if r is None or sc is None or sc.rejected or pl is None or r["pt"] is None or r["grid"] is None:
            continue
        pm = per_file.get(rel, {}).get("pages")
        pages = ({int(k): v for k, v in pm.items()} if pm else {k: {"role": f"page{k}", "train": True} for k in range(r["planes"])})
        mine = derive(r, pages, mode=p.mode, entries=entries_by_rel.get(rel), ack_cp1=p.acknowledge_field_equals_counter_plus_one)
        dis += compare_with_plan(rel, r, mine, pl, sc)
        n_files += 1
        n_tuples += len(mine["tuples"])
        n_points += len(mine["rows"])
        n_labels += sum(1 for x in mine["rows"] if x is None)
        sampled_ok.append((r, mine))
    # ---- plan-level totals vs the sum of the independent per-file ones (all sampled files)
    if str(p.sample).lower() == "all" and sampled_ok:
        mt = Counter()
        for r, mine in sampled_ok:
            mt.update(t["status"] for t in mine["tuples"].values())
        f = summary["fields"]
        theirs = {"positive_marked": f["positive_marked_fields"], "verified_zero": f["verified_zero_fields"],
                  "unknown": f["unknown_fields_excluded"], "conflict": f["conflict_fields_excluded"]}
        mine_t = {k: int(mt.get(k, 0)) for k in theirs}
        if theirs != mine_t:
            dis.append({"file": "(all files)", "kind": "summary_totals", "key": "tuples by status", "bridge": theirs, "independent": mine_t})
        ann = summary["annotations"]
        if ann["labels_exported"] != n_labels:
            dis.append({"file": "(all files)", "kind": "summary_totals", "key": "labels exported", "bridge": ann["labels_exported"],
                        "independent": n_labels})
    # ---- bright local maxima (information)
    bright = None
    if p.pixels and sampled_ok:
        rng = np.random.default_rng(0)
        per = []
        for r, mine in sampled_ok[:max(1, min(len(sampled_ok), 40))]:
            pt = r["pt"]
            idx = [k for k, x in enumerate(mine["rows"]) if x is None and int(pt["page"][k]) == 0]
            if len(idx) < 5:
                continue
            xs, ys = r["grid"]
            res = bright_check(Path(p.input) / r["rel"], 0, np.column_stack([pt["cx"][idx], pt["cy"][idx]]), rng,
                               (xs[0] - 0.5, ys[0] - 0.5, xs[-1] - 0.5, ys[-1] - 0.5))
            if res:
                per.append(dict(res, file=r["rel"]))
        if per:
            med_mark = float(np.median([x["median_mark_px"] for x in per]))
            med_rand = float(np.median([x["median_random_px"] for x in per]))
            worst = [f"{x['file']}: median {x['median_mark_px']:.1f} px (random {x['median_random_px']:.1f})" for x in per
                     if x["median_mark_px"] > BRIGHT_WARN_PX]
            bright = {"files": len(per), "median_mark_px": med_mark, "median_random_px": med_rand, "per_file": per[:200]}
            lvl = WARN if med_mark > BRIGHT_WARN_PX else PASS
            rep.add("D", "D.bright_maxima", "marks vs local intensity maxima (page 0)", lvl,
                    f"median distance from a label to the nearest local maximum within {BRIGHT_RADIUS} px: {med_mark:.1f} px over "
                    f"{len(per)} file(s) (uniformly random positions: {med_rand:.1f} px). "
                    + (f"> {BRIGHT_WARN_PX:g} px suggests a coordinate-frame or page error. " if lvl == WARN else "")
                    + ("Files whose own median exceeds it are listed (information only)." if worst else "Information only."),
                    count=len(worst), examples=worst, detail={"median_mark_px": med_mark, "median_random_px": med_rand})
        else:
            rep.add("D", "D.bright_maxima", "marks vs local intensity maxima (page 0)", INFO, "not enough labelled marks on page 0 in the sample")
    lohi = lohi_check(rep, p, plans, sampled_ok, per_file, dis) if sampled_ok else None
    rep.add("D", "D.cross_check", "independent cross-check vs the bridge", FAIL if dis else PASS,
            (f"{len(dis)} DISAGREEMENT(S) between the independent re-derivation and the bridge's plan" if dis else
             f"independent tifffile + roifile re-derivation agrees with the bridge on {n_files} file(s): {n_tuples} (page, cell) "
             f"tuple(s), {n_points} point(s), {n_labels} label(s), grid, counters, crop boxes; scan acceptance agrees on {n_acc_checked} file(s)"),
            count=len(dis), examples=[_dis_str(d) for d in dis], detail={"disagreements": dis[:200]})
    return {"ran": True, "sample_spec": p.sample, "sampled_files": len(sample), "oddity_files": len(odd), "files_compared": n_files,
            "tuples_compared": n_tuples, "points_compared": n_points, "labels_compared": n_labels, "acceptance_checked": n_acc_checked,
            "disagreements": dis[:200], "n_disagreements": len(dis), "bright": bright, "lohi": lohi, "sample": sample[:300]}


def lohi_check(rep: Report, p: PreflightParams, plans: dict, sampled_ok: list, per_file: dict, dis: list, cap: int = 12) -> Optional[dict]:
    """Recompute the page lo/hi of the sampled files independently (valid mask per ``--valid-mask``: the app's automatic chamber
    over all planes with rim 40 -> counting region -> valid mask, falling back to the union of all grid cells; percentiles
    1 / 99.8 over the valid UNSATURATED pixels) and compare with what the bridge's own ``prepare`` hook computes."""
    import tifffile
    from polony_app import schemas
    from polony_app.detection.masks import build_valid_mask
    from polony_app.imaging import chamber as chm
    done, rows, fallbacks = 0, [], []
    for r, mine in sampled_ok:
        if not p.pixels or done >= cap:
            break
        pl = plans.get(r["rel"])
        if pl is None or not pl.ext.cands or pl.ext.prepare is None:
            continue
        try:
            with tifffile.TiffFile(str(Path(p.input) / r["rel"])) as tif:
                arr = np.asarray(tif.series[0].asarray())
        except Exception as exc:  # noqa: BLE001
            dis.append({"file": r["rel"], "kind": "lohi", "key": "pixels", "bridge": "n/a", "independent": f"unreadable: {exc}"})
            continue
        planes = [arr] if arr.ndim == 2 else [arr[i] for i in range(arr.shape[0])]
        h, w = planes[0].shape
        xs, ys = r["grid"]
        boxes = [(math.ceil(xs[c] - 0.5), math.ceil(ys[rr] - 0.5), math.ceil(xs[c + 1] - 0.5), math.ceil(ys[rr + 1] - 0.5))
                 for rr in range(len(ys) - 1) for c in range(len(xs) - 1)]

        def union(bx):
            m = np.zeros((h, w), bool)
            for x0, y0, x1, y1 in bx:
                m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
            return m
        used, why, mask = p.valid_mask, None, None
        if p.valid_mask == "chamber":
            try:
                res = chm.auto_chamber(list(planes), 40)
                reg = schemas.Region(id="chamber", kind=schemas.RegionKind.counting, name="chamber",
                                     polygon=[(float(a), float(b)) for a, b in res["polygon"]])
                mask = build_valid_mask((h, w), [reg])
                if not mask.any():
                    mask, why = None, "empty chamber"
            except Exception as exc:  # noqa: BLE001
                why = f"{type(exc).__name__}: {exc}"
            if mask is None:
                used, mask = "grid-cells", union(boxes)
                fallbacks.append(f"{r['rel']}: no automatic chamber ({why}); lo/hi over the union of all grid cells")
        elif p.valid_mask == "grid-cells":
            mask = union(boxes)
        pages = sorted({c.page for c in pl.ext.cands})
        want = {}
        for pg in pages:
            m = mask
            if p.valid_mask == "eligible-fields":
                m = union([boxes[cell] for (page, cell), t in mine["tuples"].items()
                           if page == pg and t["status"] in ("positive_marked", "verified_zero")])
            a = planes[pg]
            sat = np.iinfo(a.dtype).max
            use = m & (a < sat)
            if int(use.sum()) < 1000:
                continue
            lo, hi = np.percentile(a[use].astype(np.float64), [1.0, 99.8])
            want[pg] = (float(lo), float(hi), used if p.valid_mask != "eligible-fields" else "eligible-fields")
        try:
            pl.ext.prepare()
            norm = {c.page: c.norm for c in pl.ext.cands}
        except Exception as exc:  # noqa: BLE001
            dis.append({"file": r["rel"], "kind": "lohi", "key": "bridge prepare", "bridge": f"{type(exc).__name__}: {exc}", "independent": "ok"})
            continue
        finally:
            try:
                pl.ext.release()
            except Exception:  # noqa: BLE001
                pass
        done += 1
        for pg, (lo, hi, u) in want.items():
            b = norm.get(pg)
            if not b:
                continue
            rows.append({"file": r["rel"], "page": pg, "lo": lo, "hi": hi, "bridge_lo": b["lo"], "bridge_hi": b["hi"],
                         "mask_used": u, "bridge_mask_used": b.get("valid_mask_used")})
            if abs(b["lo"] - lo) > 1e-6 * max(1.0, abs(lo)) or abs(b["hi"] - hi) > 1e-6 * max(1.0, abs(hi)) or \
                    (b.get("valid_mask_used") and b["valid_mask_used"] != u):
                dis.append({"file": r["rel"], "kind": "lohi", "key": f"page {pg} lo/hi ({p.valid_mask})",
                            "bridge": [b["lo"], b["hi"], b.get("valid_mask_used")], "independent": [lo, hi, u]})
    if rows or fallbacks:
        rep.add("D", "D.lohi", "page lo/hi over the valid mask", WARN if fallbacks else PASS,
                f"--valid-mask {p.valid_mask}: lo/hi of {len(rows)} (file, page) recomputed independently (1 / 99.8 percentiles, saturated "
                f"excluded) over {done} file(s)" + ("; mismatches are listed under the cross-check" if any(d['kind'] == 'lohi' for d in dis) else
                                                    "; they equal the bridge's values")
                + (f"; {len(fallbacks)} file(s) have no automatic chamber (fallback to all grid cells)" if fallbacks else ""),
                count=len(fallbacks), examples=fallbacks, detail={"rows": rows[:100]})
    elif p.pixels:
        rep.add("D", "D.lohi", "page lo/hi over the valid mask", INFO, "no sampled file with a crop to recompute lo/hi for")
    return {"files": done, "rows": rows[:200], "fallbacks": fallbacks}



def _dis_str(d: dict) -> str:
    return f"{d['file']} [{d['kind']}] {d['key']}: bridge={d['bridge']!r} independent={d['independent']!r}"


# ---------------------------------------------------------------------------------------------------- E
def policy_summary(rep: Report, p: PreflightParams, plan, summary: Optional[dict], plan_err: Optional[str]) -> dict:
    if summary is None:
        rep.add("E", "E.policy", "policy summary", FAIL, f"no plan: {plan_err}", count=1)
        return {}
    f, a, so, t = summary["fields"], summary["annotations"], summary["sources"], summary["tiffs"]
    pol = {"tiffs": t, "tuples": {k: f[k] for k in ("tuples_total", "eligible_examined_fields", "positive_marked_fields",
                                                      "verified_zero_fields", "unknown_fields_excluded", "conflict_fields_excluded",
                                                      "excluded_unproven", "reasons", "by_page")},
           "annotations": a, "positives_per_page": summary["positives_per_page"], "crops": summary["crops"],
           "splits": summary["splits"], "estimates": summary["estimates"], "identity": summary["identity"],
           "sources": so, "blockers": list(plan.blockers)}
    rep.add("E", "E.tuples", "field/page tuples", PASS if f["eligible_examined_fields"] else FAIL,
            f"{f['tuples_total']} tuple(s) in {t['accepted']} accepted file(s): {f['eligible_examined_fields']} examined and eligible "
            f"(positive_marked {f['positive_marked_fields']}, verified_zero {f['verified_zero_fields']}); "
            f"{f['excluded_unproven']} EXCLUDED because examination could not be proven (unknown {f['unknown_fields_excluded']}, "
            f"conflict {f['conflict_fields_excluded']})", count=f["excluded_unproven"], detail={"reasons": f["reasons"], "by_page": f["by_page"]},
            examples=[f"{n} x {r}" for r, n in sorted(f["reasons"].items(), key=lambda kv: -kv[1])])
    if f["conflict_fields_excluded"]:
        rep.add("E", "E.conflicts", "conflicting field counts", WARN, f"{f['conflict_fields_excluded']} tuple(s) have a stated count that "
                "disagrees with the marks (excluded)", count=f["conflict_fields_excluded"])
    rep.add("E", "E.labels", "labels", PASS if a["labels_exported"] else FAIL,
            f"{a['points_total']} ImageJ point(s) -> {a['labels_exported']} label(s); {a['dropped_total']} dropped: "
            + (", ".join(f"{k}={v}" for k, v in sorted(a["dropped_by_reason"].items())) or "none")
            + f"; {a['straddling_labels_in_examined_cells']} straddling label(s) kept in examined cells",
            detail={"dropped_by_reason": a["dropped_by_reason"]})
    pp = ", ".join(f"page {x['page']}/{x['role'] or 'unmapped'}: {x['positives']} positive(s) in {x['crops']} crop(s)" for x in summary["positives_per_page"])
    rep.add("E", "E.positives", "positives per page", INFO, pp or "none", detail={"positives_per_page": summary["positives_per_page"]})
    sp = summary["splits"]
    if sp:
        small = [k for k in ("train", "val", "test") if sp["by_split"].get(k, {}).get("groups", 0) < 2]
        rep.add("E", "E.splits", "projected splits", WARN if small else PASS,
                f"{sp['groups']} independent group(s): " + "; ".join(f"{k} {v['crops']} crop(s), {v['positives']} positive(s), "
                                                                     f"{v['groups']} group(s)" for k, v in sp["by_split"].items())
                + (f"; split(s) with fewer than 2 groups: {', '.join(small)} (thin evidence)" if small else ""),
                detail={"splits": sp}, count=len(small))
    else:
        rep.add("E", "E.splits", "projected splits", INFO, "not available (no exportable crop)")
    est = summary["estimates"]
    tr = est.get("training")
    if tr:
        rep.add("E", "E.estimates", "projected cost", INFO,
                f"native snapshot ~{est['native_disk']}; training at downscale {tr['downscale']}, batch {tr['batch_size']}, {tr['epochs']} "
                f"epoch(s): {tr['steps_per_epoch']} step(s)/epoch, {tr['optimizer_steps']} optimiser step(s); validation "
                f"{tr['validation_batches']} batch(es) per pass x {tr['validation_passes']} = {tr['validation_batches_total']} + "
                f"{tr['sanity_check_batches']} sanity + {tr['final_threshold_batches']} for the final threshold search",
                detail={"training": tr})
    for b in plan.blockers:
        rep.add("E", "E.blocker", "blocker", FAIL, b, count=1)
    if p.allow_unconfirmed_identity and summary["identity"]["unconfirmed"]:
        rep.add("E", "E.identity_unconfirmed", "unconfirmed identity accepted", WARN,
                f"--allow-unconfirmed-identity: {summary['identity']['unconfirmed']} file(s) with crops are split on identities inferred "
                "from names (leakage between replicates cannot be excluded)", count=summary["identity"]["unconfirmed"])
    unres = summary["unresolved_manifest_entries"]
    if unres:
        rep.add("E", "E.unresolved", "manifest entries that cannot be resolved", FAIL,
                f"{len(unres)} examined-fields entr(y/ies) cannot be located unambiguously and will NOT be used (never guessed)",
                count=len(unres), examples=[json.dumps(u, sort_keys=True) for u in unres])
    rej = t["rejected_by_reason"]
    if rej:
        rep.add("E", "E.rejected", "rejected files", WARN, f"{t['rejected']} file(s) are rejected by the importer: "
                + ", ".join(f"{k}={v}" for k, v in sorted(rej.items())), count=t["rejected"],
                examples=[f"{k}: {','.join(v)}" for k, v in t["rejected_files"].items()])
    return pol


def next_commands(p: PreflightParams, verdict: str, rep: Report) -> list:
    q = shlex.quote
    dd = f"--data-dir {q(str(p.data_dir))} "
    base = f"python -m polony_app.ml {dd}import-legacy --input {q(str(p.input))}"
    maps = ""
    if p.channel_map:
        maps += f" --channel-map {q(str(p.channel_map))}"
    if p.identity_map:
        maps += f" --identity-map {q(str(p.identity_map))}"
    if p.examined_fields:
        maps += f" --examined-fields {q(str(p.examined_fields))}"
        if p.mode != "marked-only":
            maps += f" --mode {p.mode}"
    for c in p.lab_csv:
        maps += f" --lab-csv {q(str(c))}"
    for s in p.lab_series:
        maps += f" --lab-series {q(str(s))}"
    for flag, on in (("--acknowledge-no-counters", p.acknowledge_no_counters), ("--allow-unconfirmed-identity", p.allow_unconfirmed_identity),
                     ("--acknowledge-field-equals-counter-plus-one", p.acknowledge_field_equals_counter_plus_one),
                     ("--acknowledge-lab-csv-field-numbering", p.acknowledge_lab_csv_field_numbering)):
        if on:
            maps += f" {flag}"
    opts = (f" --valid-mask {p.valid_mask}" if p.valid_mask != "chamber" else "") + f" --downscale {p.downscale} --normaliser {p.normaliser} --batch-size {p.batch_size} --epochs {p.epochs}"
    if p.fractions:
        opts += f" --fractions {p.fractions['train']},{p.fractions['val']},{p.fractions['test']}"
    rerun = f"python -m polony_app.ml {dd}preflight --input {q(str(p.input))}{maps}{opts}"
    if verdict == V_NOT:
        cmds = []
        if not p.channel_map or not p.identity_map:
            cmds.append(f"{base} --write-templates OUT_DIR      # templates for channels.json / identities.json (nothing is confirmed)")
        cmds.append("fix every FAIL above (mapping files, unreadable / truncated files, disagreements), then re-run:")
        cmds.append(rerun + " --sample all")
        return cmds
    return [rerun + " --sample all      # optional: full independent cross-check",
            f"{base}{maps}{opts} --dry-run",
            f"{base}{maps}{opts} --workers 4      # re-running it with identical inputs prints 'already imported as <version>' and writes nothing",
            f"python -m polony_app.ml {dd}verify <snapshot>",
            f"python -m polony_app.ml {dd}split --snapshot <snapshot>            # inspect; then --lock-test (permanent)",
            "python -m polony_app.ml gpu-smoke --device rocm",
            f"python -m polony_app.ml {dd}train --snapshot <snapshot> --base general --device rocm --batch-size {p.batch_size} --epochs 1 --dry-run"]


# ====================================================================================================== text output
_MARK = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", INFO: "INFO"}
_TITLES = {"A": "A  ENVIRONMENT", "B": "B  INPUT SANITY (headers only)", "C": "C  MAPPING FILES", "D": "D  INDEPENDENT CROSS-CHECK",
           "E": "E  POLICY SUMMARY"}


def format_text(doc: dict, *, show_pass: bool = True) -> list:
    L: list = []
    ad = L.append
    ad("=" * 78)
    ad(f"PREFLIGHT  input: {doc['input']}   (read-only: nothing was written to it)")
    n = doc["n_files"]
    ad(f"{n['tiffs']} TIFF(s), {n['unique']} unique, {n['aliases']} byte-identical alias(es);  findings: "
       + "  ".join(f"{k} {v}" for k, v in doc["counts"].items()))
    ad("=" * 78)
    for c in "ABCDE":
        fs = doc["checks"].get(c) or []
        if not fs:
            continue
        ad("")
        ad(_TITLES[c])
        for f in fs:
            if f["level"] == PASS and not show_pass:
                continue
            ad(f"  [{_MARK[f['level']]}] {f['id']}  {f['title']}: {f['summary']}")
            for e in f["examples"]:
                ad(f"         - {e}")
            more = f["detail"].get("examples_total")
            if more and more > len(f["examples"]):
                ad(f"         ... {more - len(f['examples'])} more (see --json)")
    t = doc["timing"]
    ad("")
    ad(f"timing: total {t['total_s']} s (headers {t['headers_s']}, hashing {t['hash_s']}, bridge plan {t['plan_s']}, "
       f"cross-check {t['cross_check_s']});  peak RSS {_fmt(doc['peak_rss_bytes']) if doc.get('peak_rss_bytes') else 'n/a'}")
    ad("")
    ad(f"VERDICT: {doc['verdict']}")
    for r in doc["reasons"][:20]:
        ad(f"  - {r}")
    if len(doc["reasons"]) > 20:
        ad(f"  - ... {len(doc['reasons']) - 20} more")
    ad(f"exit code {doc['exit_code']}  (0 = no FAIL, 1 = NOT READY, 2 = refused)")
    ad("")
    ad("next commands:")
    for c in doc["next_commands"]:
        ad(f"  {c}")
    return L
