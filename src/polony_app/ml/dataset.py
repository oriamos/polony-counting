"""Immutable, content-addressed dataset snapshots built from reviewed analyses.

``build_snapshot`` reads analyses through the store (read-only), classifies every point
(confirmed / rejected / human_added / unresolved / unreviewed / raw proposals), cuts crops from the ORIGINAL pages
(no 8-bit conversion, no rescaling) for rectangular, fully reviewed regions only, splits them by independent group
and writes ``<data_dir>/datasets/<version>/``. ``version`` is a hash of the manifest content (which lists the sha256
of every file), so identical inputs give an identical version and byte-identical files; an existing version is
never overwritten.

Positives of a crop = objects a human confirmed or added. Rejected / unresolved / unreviewed points are NEVER
positives. A crop whose area contains unresolved points (or overlaps an exclusion region) is skipped: unlabelled
real spots would teach the model that spots are background.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tifffile

from polony_app.detection.masks import points_in_polygon
from polony_app.imaging.io import read_plane
from polony_app.review.engine import mapping_confirmed, region_fingerprint
from polony_app.review.geometry import polygons_overlap
from polony_app.schemas import (
    Analysis,
    AnalysisStatus,
    ImageKind,
    ObjectClass,
    Origin,
    Region,
    RegionKind,
)

from .common import (
    DEFAULT_NORMALISER,
    DEFAULT_PERCENTILES,
    LO_HI_POLICY,
    MLError,
    app_version,
    atomic_write_text,
    check_downscale,
    check_normaliser,
    datasets_dir,
    fnum,
    hash_obj,
    page_bounds,
    pixel_box,
    pretty_json,
    rect_of,
    sha256_bytes,
    staging_lock_path,
    try_file_lock,
    sha256_file,
    utc_now,
)
from .manifest import CropRecord, SnapshotManifest, SourceAnalysis
from .splits import (
    SourceInfo,
    assign_groups,
    lock_fingerprint,
    load_lock,
)

ROLES = ("red", "green")
_ID_KEYS = ("reviewer", "reviewer_id", "user", "username", "operator", "annotator", "author", "by")
POSITIVE = ("confirmed", "human_added")
POINT_COLUMNS = ["point_id", "analysis_id", "object_id", "page", "channel_role", "x", "y", "x_page", "y_page", "status",
                 "basis", "object_class", "origin", "region_id", "consensus_status", "engine", "score", "score_kind",
                 "valid", "split", "crop_id"]
Say = Callable[[str], None]


@dataclass
class BuildOptions:
    seed: int = 0
    fractions: Optional[dict] = None
    level: str = "auto"
    allow_unconfirmed_identity: bool = False
    allow_bbox: bool = False
    strict_reviewed: bool = False        # unreviewed objects inside an approved region are NOT accepted as positives
    roles: tuple = ROLES
    identity_map: Optional[dict] = None
    require_finalized: bool = False
    min_crop_px: int = 32
    created_at: Optional[str] = None
    include_raw: bool = True
    # What the network will see. Exported crops stay at NATIVE resolution (provenance); the downscale and the normaliser
    # are recorded and applied on the fly by `train` / `eval-model` (block mean, page-level lo/hi).
    downscale: int = 1
    normaliser: str = DEFAULT_NORMALISER
    percentiles: tuple = DEFAULT_PERCENTILES


@dataclass
class BuildResult:
    version: str
    path: Optional[Path]
    manifest: SnapshotManifest
    dry_run: bool = False


# ====================================================================== point classification
def reviewers_from(log, events) -> list[str]:
    found: set[str] = set()
    for item in list(log or []):
        p = getattr(item, "payload", None) or {}
        for k in _ID_KEYS:
            v = p.get(k)
            if isinstance(v, str) and v.strip():
                found.add(v.strip())
    for ev in events or []:
        if isinstance(ev, dict):
            for k in _ID_KEYS:
                v = ev.get(k) or (ev.get("detail") or {}).get(k) if isinstance(ev.get("detail"), dict) else ev.get(k)
                if isinstance(v, str) and v.strip():
                    found.add(v.strip())
    return sorted(found)


def fully_reviewed_regions(a: Analysis) -> list[Region]:
    """Counting regions whose approval is still valid (nothing changed since the human approved them)."""
    out = []
    for r in a.regions:
        if r.kind != RegionKind.counting or not r.approved:
            continue
        try:
            ok = r.approval_fingerprint == region_fingerprint(a, r)
        except Exception:  # pragma: no cover
            ok = False
        if ok:
            out.append(r)
    return out


def _inside(x: float, y: float, polys: list[np.ndarray]) -> bool:
    xa, ya = np.array([x]), np.array([y])
    return any(bool(points_in_polygon(xa, ya, p)[0]) for p in polys)


def _channels_for(o) -> list[str]:
    if o.cls == ObjectClass.red:
        return ["red"]
    if o.cls == ObjectClass.green:
        return ["green"]
    if o.cls == ObjectClass.yellow:
        return ["red", "green"]
    return []


def classify_objects(a: Analysis, user_deleted, reviewed: list[Region], *, strict: bool = False) -> list[dict]:
    """One row per (object, channel). status: confirmed | human_added | rejected | unresolved | unreviewed.

    * user-deleted object            -> rejected   (soft deletions that are only bookkeeping, e.g. the parts of a
                                                    merge, are superseded and not listed)
    * ``uncertain`` object           -> unresolved
    * manual add                     -> human_added
    * object the reviewer confirmed / moved / reclassified -> confirmed (basis object_confirmed / edited)
    * untouched proposal inside a region the human approved -> confirmed (basis region_approved), because approving
      a region attests that the remaining proposals are the final set; ``strict`` turns this off (-> unreviewed)
    * everything else                -> unreviewed (never a positive)
    """
    ud = set(user_deleted or [])
    rpolys = [np.asarray(r.polygon, float) for r in reviewed]
    xpolys = [np.asarray(r.polygon, float) for r in a.regions if r.kind == RegionKind.exclusion and len(r.polygon) >= 3]
    rows: list[dict] = []
    for o in a.objects:
        if o.deleted and o.id not in ud:
            continue
        evs = {"red": o.red, "green": o.green}
        if o.deleted or o.cls == ObjectClass.uncertain:
            chans = [c for c in ROLES if evs[c].detected] or _channels_for(o) or [None]
        else:
            chans = _channels_for(o)
        for ch in chans:
            ev = evs.get(ch) if ch else None
            moved = any(" moved from " in h for h in (o.history or []))
            if ev is not None and ev.detected and ev.x is not None and ev.y is not None and not moved:
                x, y = float(ev.x), float(ev.y)
            else:
                x, y = float(o.x), float(o.y)
            if o.deleted:
                status, basis = "rejected", "user_deleted"
            elif o.cls == ObjectClass.uncertain:
                status, basis = "unresolved", "uncertain_class"
            elif o.origin == Origin.manual:
                status, basis = "human_added", "manual_add"
            elif o.reviewed:
                status, basis = "confirmed", ("edited" if o.origin == Origin.edited else "object_confirmed")
            elif rpolys and _inside(x, y, rpolys) and not _inside(x, y, xpolys):
                status, basis = ("unreviewed", "region_approved_strict") if strict else ("confirmed", "region_approved")
            else:
                status, basis = "unreviewed", "not_reviewed"
            rows.append({"object_id": o.id, "role": ch, "x": x, "y": y, "status": status, "basis": basis,
                         "cls": o.cls.value, "origin": o.origin.value, "region_id": o.region_id,
                         "consensus_status": (ev.consensus_status.value if ev is not None and ev.consensus_status else None)})
    return rows


# ====================================================================== extraction of one analysis
@dataclass
class _CropCand:
    region: Region
    role: str
    page: int
    source_file: str
    rect_ref: tuple
    box_file: tuple
    bbox_only: bool
    points: list           # positives (rows), already in crop coordinates (y, x)
    n_rejected: int
    n_edge: int
    crop: Optional[np.ndarray]
    dy_dx: tuple
    source_sha: str
    norm: Optional[dict] = None     # page lo/hi record (filled after the region loop, or by ``_Extract.prepare``)
    # --- bridge sources (``ml import-legacy``): crops can be LAZY so that tens of thousands of crops never sit in memory
    shape_hw: tuple = ()            # (h, w) of the crop; derived from ``crop`` when it is given
    dtype: str = ""
    provenance: dict = field(default_factory=dict)
    loader: Optional[Callable[[], np.ndarray]] = None

    def pixels(self) -> np.ndarray:
        if self.crop is not None:
            return self.crop
        if self.loader is None:
            raise MLError("internal error: a crop candidate has neither pixels nor a loader")
        return self.loader()

    def size_bytes(self) -> int:
        if self.crop is not None:
            return int(self.crop.nbytes)
        h, w = self.shape_hw
        return int(h) * int(w) * int(np.dtype(self.dtype).itemsize)


@dataclass
class _Extract:
    src: SourceAnalysis
    info: Optional[SourceInfo]
    rows: list
    regions: list
    cands: list
    warnings: list
    skipped_crops: list
    # Optional hooks of bridge sources. ``prepare`` runs right before this source's crops are written (it may read pages,
    # fill ``src.page_normalization`` / ``cand.norm`` and append to ``late_warnings``); ``release`` frees what it loaded.
    prepare: Optional[Callable[[], None]] = None
    release: Optional[Callable[[], None]] = None
    finalize: Optional[Callable[[], None]] = None      # runs once per source after ALL sources are written (summary bookkeeping)
    late_warnings: list = field(default_factory=list)


def _shift_for(a: Analysis, role: str) -> tuple[int, int]:
    return (int(a.alignment_dy), int(a.alignment_dx)) if role == "green" else (0, 0)


def _engine_runs(store, aid: str) -> tuple[list, list]:
    """(runs as dicts, warnings). Degrades gracefully when the store does not offer engine runs yet."""
    fn = getattr(store, "get_engine_runs", None)
    if fn is None:
        return [], ["the store has no engine-run API (get_engine_runs): engine/model versions of the source runs and "
                    "raw engine proposals are NOT recorded in this snapshot"]
    try:
        runs = fn(aid)
    except Exception as exc:  # pragma: no cover
        return [], [f"engine runs could not be read ({type(exc).__name__}: {exc})"]
    keep = ("run_id", "engine", "engine_version", "model_name", "model_version", "model_checksum", "page", "channel_role",
            "status", "device")
    out = []
    for r in runs:
        d = r.model_dump(mode="json") if hasattr(r, "model_dump") else dict(r)
        out.append({k: d.get(k) for k in keep})
    return out, []


def _raw_rows(store, aid: str, runs: list) -> tuple[list, list]:
    fn = getattr(store, "get_raw_detections", None)
    if fn is None:
        return [], []
    try:
        dets = fn(aid)
    except Exception as exc:  # pragma: no cover
        return [], [f"raw detections could not be read ({type(exc).__name__}: {exc})"]
    out = []
    for d in dets:
        d = d.model_dump(mode="json") if hasattr(d, "model_dump") else dict(d)
        out.append(d)
    return out, []


def extract_analysis(store, aid: str, opts: BuildOptions) -> _Extract:
    warns: list[str] = []
    skipped: list[dict] = []
    with store.view(aid) as st:
        a: Analysis = st.analysis.model_copy(deep=True)
        user_deleted = list(getattr(st, "user_deleted", []) or [])
        log = list(getattr(st, "log", []) or [])
        events = list(getattr(st, "events", []) or [])
    src = SourceAnalysis(analysis_id=aid, name=a.name, status=a.status.value)
    reviewers = reviewers_from(log, events)
    if reviewers:
        src.reviewers, src.reviewer_identity_source = reviewers, "action_log"
    else:
        src.reviewers, src.reviewer_identity_source = ["unknown"], "unknown"
        warns.append(f"analysis {aid}: no reviewer identity in the action log; recorded as 'unknown' "
                     "(annotator count cannot be established)")
    src.provenance_engine = a.provenance.engine
    reviewed = fully_reviewed_regions(a)
    counting = [r for r in a.regions if r.kind == RegionKind.counting]
    src.n_regions_total, src.n_regions_fully_reviewed = len(counting), len(reviewed)
    src.review_status = ("finalized" if a.status == AnalysisStatus.finalized
                         else "region_approved" if reviewed else "in_review")
    runs, w = _engine_runs(store, aid)
    src.engine_runs = runs
    warns += [f"analysis {aid}: {m}" for m in w]
    base = _Extract(src, None, [], [], [], warns, skipped)

    if a.image is None:
        src.skipped = "analysis has no image"
        return base
    img = a.image
    src.image_kind, src.dtype, src.shape_hw = img.kind.value, img.dtype, [img.height, img.width]
    if a.status != AnalysisStatus.finalized and not reviewed:
        src.skipped = "neither finalized nor any region approved (fully reviewed); nothing to learn from"
    elif opts.require_finalized and a.status != AnalysisStatus.finalized:
        src.skipped = "not finalized (--require-finalized)"
    elif img.kind == ImageKind.rgb_composite:
        src.skipped = "RGB composite: no raw intensities (never used for training)"
    if src.skipped:
        warns.append(f"analysis {aid}: skipped - {src.skipped}")
        return base

    # --- verify the original bytes against the recorded hashes
    odir = Path(store.originals_dir(aid))
    shas: dict[str, str] = {}
    for fname in sorted({c.source_file for c in img.channels}):
        p = odir / fname
        if not p.is_file():
            src.skipped = f"original file '{fname}' is missing on this computer"
            warns.append(f"analysis {aid}: skipped - {src.skipped}")
            return base
        shas[fname] = sha256_file(p)
        rec = img.sha256.get(fname)
        if rec and rec != shas[fname]:
            src.skipped = f"original file '{fname}' no longer matches its recorded sha256"
            warns.append(f"analysis {aid}: skipped - {src.skipped}")
            return base
    src.tiff_sha256 = shas
    src.alignment_dy_dx = [int(a.alignment_dy), int(a.alignment_dx)]

    # --- channel mapping: only when confirmed
    mapping: Optional[dict] = None
    if mapping_confirmed(a):
        mapping = {c.role.value: {"file": c.source_file, "page": int(c.page)} for c in img.channels
                   if c.role is not None and c.role.value in ROLES}
        src.channel_mapping = mapping
    else:
        warns.append(f"analysis {aid}: channel mapping is NOT confirmed; page/colour is unknown, so no crops are "
                     "exported from it (points are listed without a page)")

    rows = classify_objects(a, user_deleted, reviewed, strict=opts.strict_reviewed)
    for r in rows:
        r["page"] = mapping[r["role"]]["page"] if (mapping and r["role"] in mapping) else None
    base.rows = rows
    unresolved_n = sum(1 for r in rows if r["status"] == "unresolved")
    if unresolved_n:
        warns.append(f"analysis {aid}: {unresolved_n} unresolved point(s) (uncertain class); they are never positives, and "
                     "any crop containing one is skipped")

    # --- raw engine proposals (kept in points.csv, never positives)
    if opts.include_raw:
        raws, w2 = _raw_rows(store, aid, runs)
        warns += [f"analysis {aid}: {m}" for m in w2]
        base.rows += [{"object_id": d["id"], "role": d.get("channel_role"), "page": d.get("page"), "x": d["x"], "y": d["y"],
                       "status": "raw_proposal", "basis": "engine_raw", "cls": "", "origin": "", "region_id": d.get("region_id"),
                       "consensus_status": None, "engine": d.get("engine"), "score": d.get("score"),
                       "score_kind": d.get("score_kind"), "valid": d.get("valid", True)} for d in raws]

    # --- regions and crops
    excl = [r for r in a.regions if r.kind == RegionKind.exclusion]
    reviewed_ids = {r.id for r in reviewed}
    loaded: dict[tuple, np.ndarray] = {}
    for r in a.regions:
        rec = {"analysis_id": aid, "region_id": r.id, "name": r.name, "kind": r.kind.value, "field_index": r.field_index,
               "polygon": [[float(x), float(y)] for x, y in r.polygon], "approved": r.approved,
               "fully_reviewed": r.id in reviewed_ids, "rectangular": rect_of(r.polygon) is not None, "crops": {}}
        base.regions.append(rec)
        if r.kind != RegionKind.counting:
            rec["note"] = "exclusion region (crops overlapping it are skipped)"
            continue
        if r.id not in reviewed_ids:
            rec["note"] = "not fully reviewed (not approved, or the approval is outdated)"
            continue
        rect = rect_of(r.polygon)
        bbox_only = False
        if rect is None:
            if not opts.allow_bbox:
                msg = f"region '{r.name}' is not a rectangle; skipped (use --allow-bbox to cut its bounding box)"
                warns.append(f"analysis {aid}: {msg}")
                rec["note"] = msg
                skipped.append({"analysis_id": aid, "region_id": r.id, "reason": "non_rectangular"})
                continue
            xs = [p[0] for p in r.polygon]
            ys = [p[1] for p in r.polygon]
            rect, bbox_only = (min(xs), min(ys), max(xs), max(ys)), True
            warns.append(f"analysis {aid}: region '{r.name}' is not a rectangle; its BOUNDING BOX is used (--allow-bbox): "
                         "spots between the polygon and the box were not reviewed and are unlabelled")
        if mapping is None:
            skipped.append({"analysis_id": aid, "region_id": r.id, "reason": "channel_mapping_unconfirmed"})
            rec["note"] = "channel mapping unconfirmed"
            continue
        if any(polygons_overlap(r.polygon, e.polygon) for e in excl):
            msg = f"region '{r.name}' overlaps an exclusion region; crop skipped (the excluded area was not labelled)"
            warns.append(f"analysis {aid}: {msg}")
            rec["note"] = msg
            skipped.append({"analysis_id": aid, "region_id": r.id, "reason": "overlaps_exclusion"})
            continue
        poly = np.array([[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]], float)
        for role in opts.roles:
            if role not in mapping:
                continue
            page, fname = mapping[role]["page"], mapping[role]["file"]
            key = f"{r.id}:{role}"
            dy, dx = _shift_for(a, role)
            bx0, by0, bx1, by1 = pixel_box(rect)
            fx0, fy0, fx1, fy1 = bx0 - dx, by0 - dy, bx1 - dx, by1 - dy
            reason = None
            if bx1 - bx0 < opts.min_crop_px or by1 - by0 < opts.min_crop_px:
                reason = f"crop smaller than {opts.min_crop_px}px"
            elif fx0 < 0 or fy0 < 0 or fx1 > img.width or fy1 > img.height:
                reason = "crop box leaves the stored image (region outside the image or the aligned strip)"
            inbox = [p for p in rows if p["role"] == role and p["status"] != "raw_proposal"
                     and bool(points_in_polygon(np.array([p["x"]]), np.array([p["y"]]), poly)[0])]
            if reason is None:
                bad = [p for p in inbox if p["status"] == "unresolved"]
                if bad:
                    reason = f"{len(bad)} unresolved point(s) inside; unlabelled positives would poison training"
                unrev = [p for p in inbox if p["status"] == "unreviewed"]
                if reason is None and unrev:
                    reason = f"{len(unrev)} unreviewed point(s) inside (strict review mode)"
            if reason:
                msg = f"region '{r.name}' page {page}: crop skipped - {reason}"
                warns.append(f"analysis {aid}: {msg}")
                rec["crops"][key] = {"skipped": reason}
                skipped.append({"analysis_id": aid, "region_id": r.id, "page": page, "reason": reason})
                continue
            if (fname, page) not in loaded:
                loaded[(fname, page)] = read_plane(odir / fname, page)
            plane = loaded[(fname, page)]
            crop = np.ascontiguousarray(plane[fy0:fy1, fx0:fx1])
            h, w = crop.shape
            pos, edge, rej = [], 0, 0
            for p in inbox:
                cx, cy = p["x"] - dx - fx0, p["y"] - dy - fy0
                if p["status"] == "rejected":
                    rej += 1
                elif p["status"] in POSITIVE:
                    if -0.5 <= cx < w - 0.5 and -0.5 <= cy < h - 0.5:
                        pos.append((cy, cx, p))
                    else:
                        edge += 1
            if edge:
                warns.append(f"analysis {aid}: region '{r.name}' page {page}: {edge} positive(s) lie outside the pixel "
                             "footprint of the exported crop and were dropped from its label file")
            pos.sort(key=lambda t: (t[0], t[1], t[2]["object_id"]))
            base.cands.append(_CropCand(r, role, page, fname, rect, (fx0, fy0, fx1, fy1), bbox_only,
                                        [(a_, b_) for a_, b_, _ in pos], rej, edge, crop, (dy, dx), shas[fname]))
    if base.cands:
        _page_normalisation(a, src, base, loaded, opts, mapping or {})
    src.n_crops = len(base.cands)
    base.info = SourceInfo(
        analysis_id=aid, name=a.name, sha256=hash_obj(shas), file_names=sorted(shas), width=img.width, height=img.height,
        dtype=img.dtype, bit_depth=img.bit_depth, n_planes=len(img.channels), slide_meta_id=a.slide.slide_id,
        file_sha256s=sorted(shas.values()))
    return base


def _stored_valid_mask(a: Analysis, role: str, shape_hw: tuple) -> np.ndarray:
    """The detection valid mask of the app (counting union minus exclusions; green also loses the strip that the
    alignment shift does not cover), expressed in the STORED page's pixel frame."""
    from polony_app.detection.masks import build_valid_mask
    from polony_app.imaging.registration import apply_shift, shifted_valid_region
    valid = build_valid_mask(shape_hw, a.regions)
    dy, dx = _shift_for(a, role)
    if (dy, dx) == (0, 0):
        return valid
    valid &= shifted_valid_region(shape_hw, dy, dx)
    return apply_shift(valid.astype(np.uint8), -dy, -dx).astype(bool)


def _page_normalisation(a: Analysis, src: SourceAnalysis, ex: "_Extract", loaded: dict, opts: BuildOptions,
                        mapping: dict) -> None:
    """lo/hi ONCE per page (valid, unsaturated pixels of the whole page), stored per channel role in the manifest and
    attached to every crop cand of that page (``cand.norm``)."""
    done: dict = {}
    for c in ex.cands:
        if c.role in done:
            continue
        plane = loaded[(c.source_file, c.page)]
        valid = _stored_valid_mask(a, c.role, plane.shape)
        b = page_bounds(plane, valid, opts.percentiles)
        b.update({"file": c.source_file, "page": int(c.page), "mode": opts.normaliser, "policy": LO_HI_POLICY,
                  "valid_mask": "counting regions minus exclusions of the analysis"
                                + (", minus the strip not covered by the alignment shift" if _shift_for(a, c.role) != (0, 0) else "")})
        if b["fallback"]:
            ex.warnings.append(f"analysis {src.analysis_id} page {c.page} ({c.role}): {b['fallback']}")
        done[c.role] = b
    src.page_normalization = done
    for c in ex.cands:
        c.norm = done[c.role]


# ====================================================================== writers
def _csv_text(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def labels_csv_text(yx: list) -> str:
    """The exact text of a label CSV (header y,x); its sha256 is the expected ``csv_sha256`` of a planned crop."""
    return _csv_text(["y", "x"], [[repr(float(y)), repr(float(x))] for y, x in yx])


def write_crop_files(dirpath: Path, crop_id: str, image: np.ndarray, yx: list) -> tuple[str, str]:
    """Write ``<id>.tif`` (original dtype, single page, uncompressed) and ``<id>.csv`` (columns y,x)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    tif = dirpath / f"{crop_id}.tif"
    tifffile.imwrite(str(tif), image, photometric="minisblack", metadata=None, software=False)
    txt = labels_csv_text(yx)
    (dirpath / f"{crop_id}.csv").write_text(txt, encoding="utf-8", newline="")
    return sha256_file(tif), sha256_file(dirpath / f"{crop_id}.csv")


def read_labels_csv(path: str | Path) -> np.ndarray:
    """(N, 2) float array of (y, x) from a label CSV (header y,x)."""
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        head = next(r, None)
        if head is None or [h.strip() for h in head] != ["y", "x"]:
            raise MLError(f"{path}: label CSV must have the header 'y,x' (got {head})")
        vals = [(float(a), float(b)) for a, b in r]
    return np.array(vals, float).reshape(-1, 2)


def safe_id(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in s)


def test_content_hash(crops: list[CropRecord]) -> str:
    return hash_obj(sorted([c.crop_id, c.tif_sha256, c.csv_sha256] for c in crops if c.split == "test"))


# ====================================================================== annotation completeness
UNMARKED_RADIUS_PX = 5.0
BRIGHT_QUANTILE = 0.25


def unmarked_proposals(raw: list, marks: list, positives: list, radius: float = UNMARKED_RADIUS_PX) -> dict:
    """How many raw engine proposals the annotator left unmarked. ``raw`` = [(x, y, score|None)], ``marks`` = every point
    the annotator touched or confirmed (positives and rejected) as [(x, y)], ``positives`` = the positive ones.
    A proposal is *unmarked* when no mark lies within ``radius`` px. It is *bright* when its score is at least the
    lower quartile of the scores of the proposals the annotator DID confirm (scores are heatmap values, not
    probabilities: the threshold only says 'comparable to what was confirmed'). A cue that the labels are incomplete
    (precision against them is a lower bound), not proof: some of these are real polonies nobody marked."""
    def near(p, pts):
        if not pts:
            return False
        a = np.asarray(pts, float).reshape(-1, 2)
        return bool((np.hypot(a[:, 0] - p[0], a[:, 1] - p[1]) <= radius).any())

    conf_scores = [sc for x, y, sc in raw if sc is not None and near((x, y), positives)]
    thr = float(np.quantile(conf_scores, BRIGHT_QUANTILE)) if len(conf_scores) >= 5 else None
    unm = [(x, y, sc) for x, y, sc in raw if not near((x, y), marks)]
    return {"n_raw_proposals": len(raw), "n_unmarked": len(unm),
            "n_unmarked_bright": (sum(1 for _, _, sc in unm if sc is not None and sc >= thr) if thr is not None else None),
            "bright_score_threshold": thr, "radius_px": radius, "n_confirmed_proposals": len(conf_scores)}


# ====================================================================== snapshot
def snapshot_path(data_dir: str | Path, version: str) -> Path:
    # a snapshot is addressed by its published name only: never an unpublished '.staging-*' directory or a path
    if not version or version.startswith(".") or "/" in version or "\\" in version:
        raise MLError(f"invalid snapshot name {version!r} (snapshots are named ds-<hash>; unpublished staging directories "
                      "are not snapshots)")
    return datasets_dir(data_dir) / version


def load_manifest(data_dir: str | Path, version: str) -> SnapshotManifest:
    p = snapshot_path(data_dir, version) / "manifest.json"
    if not p.is_file():
        have = list_snapshots(data_dir)
        raise MLError(f"dataset snapshot '{version}' not found under {datasets_dir(data_dir)}"
                      + (f" (available: {', '.join(have)})" if have else " (none built yet)"))
    return SnapshotManifest.model_validate(json.loads(p.read_text(encoding="utf-8")))


def list_snapshots(data_dir: str | Path) -> list[str]:
    d = datasets_dir(data_dir)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "manifest.json").is_file())


def rel_path_problem(root: Path, rel) -> Optional[str]:
    """Why ``rel`` is not a plain relative path that stays inside ``root`` (None = fine): absolute paths, '..' components,
    backslashes, NUL, and symlinks that resolve outside the snapshot directory are all refused."""
    if not isinstance(rel, str) or not rel:
        return f"path {rel!r} is not a non-empty string"
    if "\x00" in rel or "\\" in rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        return f"path {rel!r} is absolute or contains a forbidden character: it escapes the snapshot directory"
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return f"path {rel!r} contains an empty / '.' / '..' component: it escapes the snapshot directory"
    try:
        real_root = Path(os.path.realpath(root))
        (Path(os.path.realpath(Path(root) / rel))).relative_to(real_root)
    except ValueError:
        return f"path {rel!r} resolves outside the snapshot directory (symlink escape)"
    except OSError as exc:
        return f"path {rel!r} cannot be resolved ({exc})"
    return None


def verify_snapshot(data_dir: str | Path, version: str) -> list[str]:
    """Problems found when re-hashing a snapshot (empty list = intact). Every path the manifest names (``files`` keys and each
    crop's ``path_tif`` / ``path_csv``) must stay inside the snapshot directory and every crop file must be listed in ``files``."""
    m = load_manifest(data_dir, version)
    root = snapshot_path(data_dir, version)
    bad = []
    for rel, sha in m.files.items():
        why = rel_path_problem(root, rel)
        if why:
            bad.append(why)
            continue
        p = root / rel
        if not p.is_file():
            bad.append(f"missing {rel}")
        elif sha256_file(p) != sha:
            bad.append(f"modified {rel}")
    for c in m.crops:
        for rel in (c.path_tif, c.path_csv):
            why = rel_path_problem(root, rel)
            if why:
                bad.append(f"crop {c.crop_id}: {why}")
            elif rel not in m.files:
                bad.append(f"crop {c.crop_id}: {rel} is not listed (hashed) in the manifest files")
    if _version_of(m) != m.version:
        bad.append("manifest content does not match its version")
    return bad


def _version_of(m: SnapshotManifest) -> str:
    return "ds-" + hash_obj(m.hash_payload())[:16]

def build_snapshot(store, data_dir: str | Path, analysis_ids: list[str], opts: Optional[BuildOptions] = None, *,
                   say: Say = print, dry_run: bool = False) -> BuildResult:
    opts = opts or BuildOptions()
    opts.downscale = check_downscale(opts.downscale)
    opts.normaliser = check_normaliser(opts.normaliser)
    ids = sorted(set(analysis_ids))
    if not ids:
        raise MLError("no analysis selected (pass --analyses ID [ID ...])")
    data_dir = Path(data_dir)
    lock = load_lock(data_dir)
    say(f"selected {len(ids)} analysis(es); test lock: " + (f"present (snapshot {lock['snapshot']})" if lock else "none"))
    warnings: list[str] = []
    exts: list[_Extract] = []
    for aid in ids:
        if not store.exists(aid):
            raise MLError(f"analysis '{aid}' not found in the store")
        say(f"reading analysis {aid} ...")
        e = extract_analysis(store, aid, opts)
        say(f"  {e.src.review_status or 'skipped'}: {len(e.cands)} crop candidate(s), {len(e.rows)} point row(s)"
            + (f" [skipped: {e.src.skipped}]" if e.src.skipped else ""))
        exts.append(e)
        warnings += e.warnings
    return assemble_snapshot(data_dir, exts, opts, say=say, dry_run=dry_run, warnings=warnings, lock=lock)


# ====================================================================== assembly (shared by every source of crops)
def _estimate_free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return int(shutil.disk_usage(p).free)


def fmt_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TiB"  # pragma: no cover


def assemble_snapshot(data_dir: str | Path, exts: list, opts: BuildOptions, *, say: Say = print, dry_run: bool = False,
                      warnings: Optional[list[str]] = None, lock: Optional[dict] = None,
                      extra_options: Optional[dict] = None, extra_assumptions: Optional[list[str]] = None,
                      import_summary: Optional[dict] = None, resume_key: Optional[str] = None, workers: int = 1,
                      materialize_on_dry_run: bool = True, progress_every: int = 2000,
                      min_free_factor: float = 1.1) -> BuildResult:
    """Group, split, write and seal a snapshot from already-extracted sources (``_Extract``).

    Both ``build`` (reviewed analyses) and ``import-legacy`` (historical ImageJ annotations) end here, so they share the
    group split, the locked-test carry-over, the manifest, the content-addressed version and the atomic publication.

    * ``crops`` of a source may be lazy (``loader``); sources are written one at a time (``workers`` > 1 writes several
      sources concurrently; ids are assigned beforehand and results are merged in source order, so the snapshot is identical).
    * ``resume_key``: the staging directory is named after it and every finished source leaves a verified marker, so an
      interrupted run restarts where it stopped (same key = identical inputs and options; nothing is ever half-published).
    * ``dry_run`` with ``materialize_on_dry_run=False`` writes NOTHING and does not compute the version (it needs the crop
      hashes); the manifest is returned with the planned crops. The default keeps the old behaviour (write to a staging
      directory, compute the version, delete it).
    """
    opts.downscale = check_downscale(opts.downscale)
    opts.normaliser = check_normaliser(opts.normaliser)
    data_dir = Path(data_dir)
    if lock is None:
        lock = load_lock(data_dir)
    warnings = list(warnings or [])
    with_crops = [e for e in exts if e.cands and e.info]
    live_infos = [e.info for e in with_crops]
    ga = None
    if live_infos:
        ga = assign_groups(live_infos, seed=opts.seed, fractions=opts.fractions, level=opts.level,
                           allow_unconfirmed=opts.allow_unconfirmed_identity, identity_map=opts.identity_map, lock=lock)
        warnings += ga.warnings
    elif not lock:
        raise MLError("none of the selected analyses yields a usable crop (see the warnings above): "
                      "need fully reviewed (approved) rectangular regions and a confirmed channel mapping "
                      "(historical imports: eligible fields and a confirmed page mapping)")

    # ---- deterministic crop ids and the storage estimate (before anything is written)
    cid_of: dict[int, str] = {}
    used_ids: set[str] = set()
    est_bytes = 0
    for e in with_crops:
        aid = e.src.analysis_id
        if aid in ga.locked_analyses:
            continue
        for c in e.cands:
            cid = safe_id(f"{aid}_p{c.page}_{c.region.id}_{c.role}")
            if cid in used_ids:
                cid = f"{cid}_{len(used_ids)}"
            used_ids.add(cid)
            cid_of[id(c)] = cid
            est_bytes += c.size_bytes() + 400
    free = _estimate_free_bytes(datasets_dir(data_dir))
    if est_bytes:
        say(f"estimated native crop storage: {fmt_bytes(est_bytes)} (free on the target volume: {fmt_bytes(free)})")
    will_write = not (dry_run and not materialize_on_dry_run)
    if will_write and est_bytes * min_free_factor > free:
        raise MLError(f"not enough free disk space: the snapshot needs about {fmt_bytes(est_bytes)} (+10 % margin) but only "
                      f"{fmt_bytes(free)} are free under {datasets_dir(data_dir)}. Nothing was written. Free space, or point "
                      "--data-dir at a larger volume.")

    stage: Optional[Path]
    resume = bool(resume_key) and will_write
    lock_stack = contextlib.ExitStack()      # holds the lock of the staging directory until this function ends
    if not will_write:
        stage = None
    else:
        stage = None
        if resume:
            cand = datasets_dir(data_dir) / f".staging-{safe_id(str(resume_key))}"
            if lock_stack.enter_context(try_file_lock(staging_lock_path(data_dir, cand.name))):
                stage = cand
            else:
                say(f"another import of the identical plan is running right now (lock {staging_lock_path(data_dir, cand.name)}): "
                    "this run writes a private, non-resumable copy so that neither run can disturb the other; the first one to "
                    "publish wins, the other ends with a clear message")
                resume = False
        if stage is None:
            stage = datasets_dir(data_dir) / f".staging-{uuid.uuid4().hex[:10]}"
            lock_stack.enter_context(try_file_lock(staging_lock_path(data_dir, stage.name)))
    published = False
    try:
        if stage is not None:
            reclaim_staging(data_dir, keep=stage.name, say=say, superseded=False)
            stage.mkdir(parents=True, exist_ok=True)
        crops: list[CropRecord] = []

        def make_rec(e, c, split, tsha, csha) -> CropRecord:
            aid = e.src.analysis_id
            cid = cid_of[id(c)]
            rel = f"spotiflow/{split}"
            shape = list(c.crop.shape) if c.crop is not None else [int(c.shape_hw[0]), int(c.shape_hw[1])]
            dt = str(c.crop.dtype) if c.crop is not None else str(np.dtype(c.dtype))
            return CropRecord(
                crop_id=cid, analysis_id=aid, region_id=c.region.id, region_name=c.region.name,
                field_index=c.region.field_index, page=c.page, channel_role=c.role, source_file=c.source_file,
                source_sha256=c.source_sha, box_xyxy=list(c.box_file),
                box_reference_frame=list(pixel_box(c.rect_ref)), alignment_dy_dx=list(c.dy_dx),
                shape_hw=shape, dtype=dt, n_positive=len(c.points),
                n_rejected=c.n_rejected, n_edge_dropped=c.n_edge, split=split, group_id=ga.group_of[aid],
                tif_sha256=tsha, csv_sha256=csha, bbox_of_polygon=c.bbox_only,
                norm_lo=(c.norm or {}).get("lo"), norm_hi=(c.norm or {}).get("hi"), normaliser=opts.normaliser,
                downscale=opts.downscale, path_tif=f"{rel}/{cid_of[id(c)]}.tif", path_csv=f"{rel}/{cid_of[id(c)]}.csv",
                provenance=dict(c.provenance))

        def marker_path(e) -> Optional[Path]:
            return (stage / ".done" / f"{safe_id(e.src.analysis_id)}.json") if (resume and stage is not None) else None

        def reuse_marker(e, split) -> Optional[list[CropRecord]]:
            """A finished source of an interrupted run is reused ONLY if everything the marker claims is re-derived from the
            current plan: same resume key (inputs, options, code), the expected label CSV bytes (recomputed from the planned
            labels), every planned record field (box, labels, page, role, source sha, split, group, normaliser, provenance),
            the sha256 of both files on disk, and a readable single-page TIFF of the planned shape and dtype. Hashes written
            in the marker alone are never trusted."""
            mp = marker_path(e)
            if mp is None or not mp.is_file():
                return None
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
                if m.get("resume_key") != resume_key:
                    return None
                recs = [CropRecord.model_validate(x) for x in m["crops"]]
                if [r.crop_id for r in recs] != [cid_of[id(c)] for c in e.cands] or any(r.split != split for r in recs):
                    return None
                norms = m.get("page_normalization", {})
                for r, c in zip(recs, e.cands):
                    expected_csv = sha256_bytes(labels_csv_text(c.points).encode("utf-8"))
                    if r.csv_sha256 != expected_csv or sha256_file(stage / r.path_csv) != expected_csv:
                        return None
                    if sha256_file(stage / r.path_tif) != r.tif_sha256:
                        return None
                    nb = norms.get(c.role) or {}
                    if r.norm_lo != nb.get("lo") or r.norm_hi != nb.get("hi"):
                        return None
                    want = make_rec(e, c, split, r.tif_sha256, r.csv_sha256).model_copy(
                        update={"norm_lo": r.norm_lo, "norm_hi": r.norm_hi})
                    if want.model_dump(mode="json") != r.model_dump(mode="json"):
                        return None
                    with tifffile.TiffFile(str(stage / r.path_tif)) as tf:
                        sr = tf.series[0]
                        if len(tf.pages) != 1 or tuple(sr.shape) != tuple(want.shape_hw) or str(np.dtype(sr.dtype)) != want.dtype:
                            return None
                e.src.page_normalization = norms
                e.late_warnings += list(m.get("late_warnings", []))
                return recs
            except Exception:  # a damaged marker is never trusted: the source is simply written again
                return None

        def write_source(e) -> list[CropRecord]:
            aid = e.src.analysis_id
            split = ga.split_of[aid]
            if aid in ga.locked_analyses:
                return []
            if stage is None:                              # dry run without materialisation: planned records only
                if e.prepare:
                    pass                                    # lo/hi need the pixels; the plan does not
                return [make_rec(e, c, split, "", "") for c in e.cands]
            got = reuse_marker(e, split)
            if got is not None:
                return got
            n_late = len(e.late_warnings)
            if e.prepare:
                e.prepare()
            recs = []
            try:
                for c in e.cands:
                    tsha, csha = write_crop_files(stage / f"spotiflow/{split}", cid_of[id(c)], c.pixels(), c.points)
                    recs.append(make_rec(e, c, split, tsha, csha))
            finally:
                if e.release:
                    e.release()
            mp = marker_path(e)
            if mp is not None:
                atomic_write_text(mp, pretty_json({"resume_key": resume_key, "crops": [r.model_dump(mode="json") for r in recs],
                                                   "page_normalization": e.src.page_normalization,
                                                   "late_warnings": e.late_warnings[n_late:]}))
            return recs

        targets = with_crops
        total_crops = sum(len(e.cands) for e in targets if e.src.analysis_id not in ga.locked_analyses)
        if will_write and total_crops:
            say(f"writing {total_crops} crop(s) from {len(targets)} source(s)" + (f" with {workers} workers" if workers > 1 else "")
                + (f" (resumable: {stage.name})" if resume else "") + " ...")
        done_crops = 0
        next_report = progress_every
        if workers > 1 and len(targets) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = pool.map(write_source, targets)
                for e, recs in zip(targets, results):
                    crops += recs
                    done_crops += len(recs)
                    if will_write and done_crops >= next_report:
                        say(f"  {done_crops}/{total_crops} crops written")
                        next_report = done_crops + progress_every
        else:
            for e in targets:
                recs = write_source(e)
                crops += recs
                done_crops += len(recs)
                if will_write and done_crops >= next_report:
                    say(f"  {done_crops}/{total_crops} crops written")
                    next_report = done_crops + progress_every
        for e in exts:
            if e.finalize is not None:
                e.finalize()
        for e in exts:
            warnings += [w for w in e.late_warnings if w not in warnings]
        for e in with_crops:
            aid = e.src.analysis_id
            e.src.group_id, e.src.split = ga.group_of[aid], ga.split_of[aid]
            e.src.identity = ga.identity_of[aid]
            if aid in ga.locked_analyses:
                e.src.skipped = "belongs to the locked test set: not exported again (locked test labels never change)"
                e.src.n_crops = 0
        n_by_analysis: dict[str, int] = {}
        for c in crops:
            n_by_analysis[c.analysis_id] = n_by_analysis.get(c.analysis_id, 0) + 1
        for e in exts:  # crops per analysis for the manifest
            e.src.n_crops = n_by_analysis.get(e.src.analysis_id, 0)

        # ---- carry the locked test crops over verbatim
        test_lock_info = None
        if lock:
            lm = load_manifest(data_dir, lock["snapshot"])
            problems = verify_snapshot(data_dir, lock["snapshot"])
            if problems:
                raise MLError(f"the locked test snapshot {lock['snapshot']} is damaged ({'; '.join(problems[:3])}); "
                              "refusing to build on top of a corrupted test set")
            frozen = [c for c in lm.crops if c.split == "test"]
            for c in frozen:
                if stage is not None:
                    for rel in (c.path_tif, c.path_csv):
                        dst = stage / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(snapshot_path(data_dir, lock["snapshot"]) / rel, dst)
                fc = c.model_copy(update={"frozen_from": lock["snapshot"]})
                if fc.crop_id in used_ids:
                    raise MLError(f"crop id clash with a locked test crop: {fc.crop_id}")
                crops.append(fc)
            test_lock_info = {"lock_sha256": lock.get("lock_sha256"), "snapshot": lock["snapshot"],
                              "groups": lock.get("groups", []), "n_frozen_test_crops": len(frozen),
                              "test_content_sha256": lock.get("test_content_sha256")}
            say(f"carried {len(frozen)} locked test crop(s) over from snapshot {lock['snapshot']} (labels unchanged)")
            odd = sorted({(c.downscale, c.normaliser or "legacy") for c in frozen} - {(opts.downscale, opts.normaliser)})
            if odd:
                warnings.append("the carried-over locked test crops were built for a different downscale / normaliser "
                                f"{odd} than this snapshot ({opts.downscale}, {opts.normaliser}); each crop keeps its own "
                                "recorded values, and a model is always evaluated at ITS OWN configuration")

        crops.sort(key=lambda c: (c.split, c.crop_id))
        n_pos_total = sum(c.n_positive for c in crops)
        if not crops:
            raise MLError("no crop could be exported (see the warnings); refusing to create an empty snapshot")

        # ---- points.csv / regions.jsonl
        order = {"confirmed": 0, "human_added": 1, "rejected": 2, "unresolved": 3, "unreviewed": 4, "raw_proposal": 5,
                 "dropped": 6}
        live_crops = [c for c in crops if not c.frozen_from]
        # indexes (a snapshot of ~50,000 crops must not cost sources x crops / regions x crops comparisons)
        crops_of: dict[str, list] = {}
        crops_of_region: dict[tuple, list] = {}
        for c in live_crops:
            crops_of.setdefault(c.analysis_id, []).append(c)
            crops_of_region.setdefault((c.analysis_id, c.region_id), []).append(c)
        prow: list[list] = []
        counts: dict[str, int] = {}
        for e in exts:
            aid = e.src.analysis_id
            sp = e.src.split or ""
            mine = crops_of.get(aid, [])
            keyed = sorted(e.rows, key=lambda r: (r["role"] or "", order[r["status"]], r["y"], r["x"], str(r["object_id"])))
            for n, r in enumerate(keyed, 1):
                counts[r["status"]] = counts.get(r["status"], 0) + 1
                cid = ""
                if r["status"] not in ("raw_proposal", "dropped"):
                    for c in mine:
                        x0, y0, x1, y1 = c.box_reference_frame
                        if r.get("footprint_cell"):      # bridge rows: the cell was assigned by pixel footprint already
                            if c.channel_role == r["role"] and c.region_id == r.get("region_id"):
                                cid = c.crop_id
                                break
                        elif c.channel_role == r["role"] and x0 <= r["x"] < x1 and y0 <= r["y"] < y1:
                            cid = c.crop_id
                            break
                dy, dx = _shift_for_rows(e, r["role"])
                prow.append([f"{aid}:{n}", aid, r["object_id"], "" if r["page"] is None else r["page"],
                             r["role"] or "", fnum(r["x"]), fnum(r["y"]), fnum(r["x"] - dx), fnum(r["y"] - dy),
                             r["status"], r["basis"], r["cls"], r["origin"], r.get("region_id") or "",
                             r.get("consensus_status") or "", r.get("engine") or "", fnum(r.get("score")),
                             r.get("score_kind") or "", "" if r.get("valid") is None else int(bool(r.get("valid"))),
                             sp, cid])
        points_text = _csv_text(POINT_COLUMNS, prow)
        rlines = []
        for e in exts:
            for rec in e.regions:
                role_crops = dict(rec["crops"])
                for c in crops_of_region.get((e.src.analysis_id, rec["region_id"]), ()):
                    role_crops[f"{c.region_id}:{c.channel_role}"] = {"crop_id": c.crop_id, "split": c.split}
                rlines.append(json.dumps(dict(rec, crops=role_crops), sort_keys=True, ensure_ascii=False))
        regions_text = "\n".join(rlines) + ("\n" if rlines else "")
        if stage is not None:
            atomic_write_text(stage / "points.csv", points_text)
            atomic_write_text(stage / "regions.jsonl", regions_text)
        files = {"points.csv": sha256_bytes(points_text.encode("utf-8")),
                 "regions.jsonl": sha256_bytes(regions_text.encode("utf-8"))}
        for c in crops:
            files[c.path_tif] = c.tif_sha256
            files[c.path_csv] = c.csv_sha256

        by_split: dict[str, dict] = {}
        for c in crops:
            d = by_split.setdefault(c.split, {"crops": 0, "positives": 0, "groups": set()})
            d["crops"] += 1
            d["positives"] += c.n_positive
            d["groups"].add(c.group_id)
        label_counts = {"points_by_status": dict(sorted(counts.items())), "positives_total": n_pos_total,
                        "by_split": {k: {"crops": v["crops"], "positives": v["positives"], "groups": len(v["groups"])}
                                     for k, v in sorted(by_split.items())}}
        label_counts["precision_label"] = "precision is measured against THIS annotator's marks (unmarked polonies count as false positives)"
        if opts.include_raw:
            unm = {"match_radius_px": UNMARKED_RADIUS_PX}
            tot = {"n_raw_proposals": 0, "n_unmarked": 0, "n_unmarked_bright": 0, "n_confirmed_proposals": 0}
            have_raw = False
            for e in exts:
                mine = crops_of.get(e.src.analysis_id, [])
                for role in {c.channel_role for c in mine}:
                    boxes = [c.box_reference_frame for c in mine if c.channel_role == role]
                    inb = lambda r: any(b[0] <= r["x"] < b[2] and b[1] <= r["y"] < b[3] for b in boxes)  # noqa: E731
                    raw = [(r["x"], r["y"], r.get("score")) for r in e.rows if r["status"] == "raw_proposal"
                           and r["role"] == role and r.get("valid", True) and inb(r)]
                    if not raw:
                        continue
                    have_raw = True
                    pos = [(r["x"], r["y"]) for r in e.rows if r["status"] in POSITIVE and r["role"] == role and inb(r)]
                    mk = pos + [(r["x"], r["y"]) for r in e.rows if r["status"] == "rejected" and r["role"] == role and inb(r)]
                    u = unmarked_proposals(raw, mk, pos)
                    for k in ("n_raw_proposals", "n_unmarked", "n_confirmed_proposals"):
                        tot[k] += u[k]
                    tot["n_unmarked_bright"] += u["n_unmarked_bright"] or 0
            label_counts["unmarked_raw_proposals"] = dict(unm, **tot) if have_raw else None
            if have_raw:
                label_counts["unmarked_raw_proposals"]["note"] = (
                    "raw proposals inside exported crops with no mark of this annotator within the radius; 'bright' = score >= the "
                    "lower quartile of the confirmed proposals (heatmap value, not a probability). Precision against these labels "
                    "is a lower bound.")
        for need in ("train", "val"):
            if not by_split.get(need):
                warnings.append(f"the snapshot has NO {need} crops (too few independent groups or reviewed regions); "
                                "training will be refused until it does")
        if not by_split.get("test"):
            warnings.append("the snapshot has no test crops: nothing can be evaluated or promoted until a test set exists "
                            "and is locked (ml split --lock-test)")
        warnings = _dedupe(warnings)

        assumptions = list(ga.assumptions) if ga else []
        assumptions += ["page index = TIFF page; colour only from a user-confirmed channel mapping (null otherwise)",
                        "a region approved by the reviewer is taken as an attestation that its remaining proposals are the "
                        "final set (use --strict-reviewed to require object-level confirmation)"
                        if not opts.strict_reviewed else "strict review: only objects the reviewer touched are positives"]
        if any(r.src.reviewer_identity_source == "unknown" for r in exts):
            assumptions.append("reviewer identity unknown for at least one analysis (not in the action log)")
        assumptions += list(extra_assumptions or [])
        options = {"seed": opts.seed, "fractions": (ga.fractions if ga else None), "level": opts.level,
                   "allow_unconfirmed_identity": opts.allow_unconfirmed_identity, "allow_bbox": opts.allow_bbox,
                   "strict_reviewed": opts.strict_reviewed, "roles": list(opts.roles), "min_crop_px": opts.min_crop_px,
                   "include_raw": opts.include_raw, "downscale": opts.downscale, "normaliser": opts.normaliser,
                   "percentiles": [float(x) for x in opts.percentiles], "lo_hi_policy": LO_HI_POLICY}
        options.update(extra_options or {})
        manifest = SnapshotManifest(
            version="", created_at=opts.created_at or utc_now(), app_version=app_version(),
            options=options,
            sources=[e.src for e in exts], crops=crops,
            skipped_crops=sorted((s for e in exts for s in e.skipped_crops),
                                 key=lambda s: (s["analysis_id"], s["region_id"], str(s.get("page")))),
            regions_summary={"regions": sum(len(e.regions) for e in exts),
                             "fully_reviewed": sum(e.src.n_regions_fully_reviewed for e in exts),
                             "crops": len(crops)},
            label_counts=label_counts,
            splits=({"seed": ga.seed, "level": ga.level, "fractions": ga.fractions, "group_to_split": ga.group_to_split,
                     "analysis_to_split": dict(sorted(ga.split_of.items())), "analysis_to_group": dict(sorted(ga.group_of.items())),
                     "locked_analyses": ga.locked_analyses, "unconfirmed_identity": ga.unconfirmed} if ga else {}),
            test_lock=test_lock_info, files=dict(sorted(files.items())), warnings=warnings,
            unresolved_assumptions=_dedupe(assumptions), import_summary=dict(import_summary or {}))
        if not will_write:
            say(f"[dry-run] nothing written: {len(crops)} planned crop(s), {n_pos_total} positive(s); the snapshot version is "
                "computed from the crop hashes at build time")
            return BuildResult("", None, manifest, dry_run=True)
        version = _version_of(manifest)
        manifest.version = version
        final = snapshot_path(data_dir, version)
        if final.exists():
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)      # nothing to resume: the identical snapshot is published
            raise MLError(f"snapshot {version} already exists at {final}: an identical snapshot was built before. "
                          "Snapshots are immutable and are never overwritten.")
        if dry_run:
            say(f"[dry-run] would create snapshot {version}: {len(crops)} crop(s), {n_pos_total} positive(s)")
            return BuildResult(version, None, manifest, dry_run=True)
        atomic_write_text(stage / "manifest.json", pretty_json(manifest.model_dump(mode="json")))
        if resume:
            shutil.rmtree(stage / ".done", ignore_errors=True)
        try:
            os.rename(stage, final)
        except OSError as exc:
            if final.exists():            # a concurrent identical import published first (rename onto a non-empty directory)
                shutil.rmtree(stage, ignore_errors=True)
                raise MLError(f"snapshot {version} already exists at {final}: an identical snapshot was published by another run "
                              "while this one was writing. Snapshots are immutable and are never overwritten; nothing was changed "
                              "and this run's staging copy was removed.") from exc
            raise
        published = True
        _read_only(final)
    finally:
        if stage is not None and stage.exists() and not published and not (resume and not dry_run):
            shutil.rmtree(stage, ignore_errors=True)
        lock_stack.close()
        if stage is not None and not stage.exists():
            _drop_lock_file(data_dir, stage.name)
    if published:
        try:
            reclaim_staging(data_dir, keep=stage.name, say=say, superseded=True)
        except OSError:                   # housekeeping only: never turns a published snapshot into an error
            pass
    say(f"created snapshot {version} at {final}")
    for sp in ("train", "val", "test"):
        d = label_counts["by_split"].get(sp, {"crops": 0, "positives": 0, "groups": 0})
        say(f"  {sp}: {d['crops']} crop(s), {d['positives']} positive(s), {d['groups']} group(s)")
    say(f"  points: {label_counts['points_by_status']}")
    for w in manifest.warnings:
        say(f"  WARNING: {w}")
    return BuildResult(version, final, manifest)


STALE_STAGING_DAYS = 7


def _stage_age_days(path: Path) -> float:
    newest = path.stat().st_mtime
    done = path / ".done"
    try:
        if done.is_dir():
            newest = max([newest, done.stat().st_mtime] + [q.stat().st_mtime for q in done.iterdir()])
    except OSError:
        pass
    return max(0.0, (time.time() - newest) / 86400.0)


def _drop_lock_file(data_dir: str | Path, name: str) -> None:
    """Delete the (now unneeded) lock file of a staging directory, and the ``.locks`` directory when it became empty."""
    lp = staging_lock_path(data_dir, name)
    try:
        lp.unlink()
    except OSError:
        pass
    try:
        lp.parent.rmdir()
    except OSError:
        pass


def remove_staging_if_free(data_dir: str | Path, name: str) -> bool:
    """Delete ``datasets/<name>`` (a ``.staging-*`` directory) unless a live run holds its lock. True when it was removed."""
    d = datasets_dir(data_dir) / name
    if not name.startswith(".staging-") or not d.is_dir():
        return False
    lp = staging_lock_path(data_dir, name)
    with try_file_lock(lp) as got:
        if not got:
            return False
        shutil.rmtree(d, ignore_errors=True)
    _drop_lock_file(data_dir, name)
    return True


def reclaim_staging(data_dir: str | Path, *, keep: str = "", say: Optional[Callable] = None, superseded: bool = False,
                    max_age_days: float = STALE_STAGING_DAYS) -> list[str]:
    """Remove leftover ``.staging-*`` directories nobody is using. NEVER a directory whose lock is held by a live run.

    * always: older than ``max_age_days`` (default 7) and with a free lock;
    * ``superseded=True`` (after a successful publish): every other ``.staging-*`` whose lock is free - it belongs to inputs
      that were superseded (a crashed run whose inputs or options changed afterwards has another resume key and could never be
      resumed by this plan);
    plus lock files of directories that no longer exist. Returns the removed names."""
    d = datasets_dir(data_dir)
    removed: list[str] = []
    if not d.is_dir():
        return removed
    for q in sorted(d.iterdir()):
        if not q.name.startswith(".staging-") or q.name == keep or not q.is_dir():
            continue
        try:
            old = _stage_age_days(q) > max_age_days
        except OSError:
            continue
        if (superseded or old) and remove_staging_if_free(data_dir, q.name):
            removed.append(q.name)
    locks = Path(data_dir) / ".locks"
    if locks.is_dir():
        for lf in sorted(locks.glob("staging-*.lock")):
            name = "." + lf.name[:-len(".lock")]
            if name != keep and not (d / name).exists():
                with try_file_lock(lf) as got:
                    if got:
                        _drop_lock_file(data_dir, name)
    if removed and say is not None:
        say(f"removed {len(removed)} stale staging director{'y' if len(removed) == 1 else 'ies'} of superseded or abandoned imports "
            f"({', '.join(removed[:3])}{', ...' if len(removed) > 3 else ''})")
    return removed


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _shift_for_rows(e: _Extract, role: Optional[str]) -> tuple[int, int]:
    dy, dx = e.src.alignment_dy_dx
    return (dy, dx) if role == "green" else (0, 0)


def _read_only(root: Path) -> None:
    """Best effort immutability: files and directories lose their write bits."""
    try:
        for p in sorted(root.rglob("*"), reverse=True):
            os.chmod(p, 0o555 if p.is_dir() else 0o444)
        os.chmod(root, 0o555)
    except OSError:  # pragma: no cover (e.g. Windows)
        pass


def load_crop(data_dir: str | Path, version: str, crop: CropRecord) -> tuple[np.ndarray, np.ndarray]:
    root = snapshot_path(data_dir, version)
    return tifffile.imread(str(root / crop.path_tif)), read_labels_csv(root / crop.path_csv)


# ====================================================================== split report / test lock
def split_report(data_dir: str | Path, version: str) -> dict:
    m = load_manifest(data_dir, version)
    groups: dict[str, dict] = {}
    for s in m.sources:
        if s.group_id:
            g = groups.setdefault(s.group_id, {"split": s.split, "analyses": [], "crops": 0, "positives": 0})
            g["analyses"].append(s.analysis_id)
    for c in m.crops:
        g = groups.setdefault(c.group_id, {"split": c.split, "analyses": [], "crops": 0, "positives": 0})
        g["crops"] += 1
        g["positives"] += c.n_positive
    return {"version": m.version, "seed": m.splits.get("seed"), "level": m.splits.get("level"),
            "fractions": m.splits.get("fractions"), "groups": dict(sorted(groups.items())),
            "warnings": [w for w in m.warnings], "unconfirmed_identity": m.splits.get("unconfirmed_identity", False),
            "locked": m.test_lock}


ACK_ASSUMPTIONS = ("acknowledge_no_counters", "acknowledge_field_equals_counter_plus_one", "acknowledge_lab_csv_field_numbering")


def lock_test_set(data_dir: str | Path, version: str, *, replace: bool = False, reason: str = "",
                  allow_unconfirmed_identity: bool = False, allow_acknowledged_assumptions: bool = False) -> dict:
    """Freeze the test groups of snapshot ``version`` (``test_lock.json``). The overrides the snapshot was built with are
    recorded in the lock. A snapshot built with an ``--acknowledge-*`` assumption (counter-less marks, field N = counter N-1,
    lab-CSV numbering) is REFUSED unless ``allow_acknowledged_assumptions`` is given: a permanent test set must not rest on an
    unverified reading of the data."""
    from .splits import SourceInfo, _record, group_key, write_lock

    m = load_manifest(data_dir, version)
    problems = verify_snapshot(data_dir, version)
    if problems:
        raise MLError(f"snapshot {version} is damaged ({'; '.join(problems[:3])}); cannot lock it")
    test_crops = [c for c in m.crops if c.split == "test"]
    if not test_crops:
        raise MLError(f"snapshot {version} has no test crops; nothing to lock (build with enough independent groups)")
    if m.splits.get("unconfirmed_identity") and not allow_unconfirmed_identity:
        raise MLError("the test groups of this snapshot rest on UNCONFIRMED (file-name-inferred) identities; confirm "
                      "them (--identity-map) and rebuild, or pass --allow-unconfirmed-identity to lock anyway")
    used = (m.import_summary or {}).get("options") or {}
    acks = [k for k in ACK_ASSUMPTIONS if used.get(k)]
    if acks and not allow_acknowledged_assumptions:
        raise MLError(f"snapshot {version} was built with {', '.join('--' + k.replace('_', '-') for k in acks)}: the test set would "
                      "rest on an assumption the operator acknowledged but nobody verified. Rebuild without it, or pass "
                      "--allow-acknowledged-assumptions-in-test-lock to lock anyway (recorded in the lock).")
    overrides = {"snapshot_options": {k: bool(used.get(k)) for k in (*ACK_ASSUMPTIONS, "allow_unconfirmed_identity")},
                 "allow_unconfirmed_identity_flag": bool(allow_unconfirmed_identity),
                 "allow_acknowledged_assumptions_flag": bool(allow_acknowledged_assumptions),
                 "snapshot_built_with_allow_unconfirmed_identity": bool((m.options or {}).get("allow_unconfirmed_identity"))}
    if m.test_lock and not replace:
        raise MLError(f"snapshot {version} was itself built on a locked test set ({m.test_lock.get('snapshot')}); "
                      "to re-lock use --replace-lock --reason ...")
    imgs = []
    for s in m.sources:
        if s.split != "test" or not s.group_id:
            continue
        info = SourceInfo(analysis_id=s.analysis_id, name=s.name, sha256=hash_obj(s.tiff_sha256),
                          file_names=sorted(s.tiff_sha256), file_sha256s=sorted(s.tiff_sha256.values()))
        rec = _record(info, s.identity or {"source": "unknown", "confirmed": False}, None)
        imgs.append({"analysis_id": s.analysis_id, "name": s.name, "sha256": rec.sha256,
                     "file_sha256s": sorted(s.tiff_sha256.values()), "group": s.group_id,
                     "identity": {"specimen_id": rec.specimen_id, "slide_id": rec.slide_id, "assay": rec.assay},
                     "keys": {"specimen": group_key(rec, "specimen"), "slide": group_key(rec, "slide"),
                              "source_image": group_key(rec, "source_image")}})
    groups = sorted({c.group_id for c in test_crops})
    return write_lock(data_dir, snapshot=version, test_content_sha256=test_content_hash(m.crops), groups=groups, images=imgs,
                      level=m.splits.get("level") or "auto", seed=int(m.splits.get("seed") or 0), replace=replace, reason=reason,
                      overrides=overrides)
