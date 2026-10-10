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

import csv
import io
import json
import os
import shutil
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
    MLError,
    app_version,
    atomic_write_text,
    datasets_dir,
    fnum,
    hash_obj,
    pixel_box,
    pretty_json,
    rect_of,
    sha256_bytes,
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
            if ev is not None and ev.detected and ev.x is not None and ev.y is not None:
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
    crop: np.ndarray
    dy_dx: tuple
    source_sha: str


@dataclass
class _Extract:
    src: SourceAnalysis
    info: Optional[SourceInfo]
    rows: list
    regions: list
    cands: list
    warnings: list
    skipped_crops: list


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
    src.n_crops = len(base.cands)
    base.info = SourceInfo(
        analysis_id=aid, name=a.name, sha256=hash_obj(shas), file_names=sorted(shas), width=img.width, height=img.height,
        dtype=img.dtype, bit_depth=img.bit_depth, n_planes=len(img.channels), slide_meta_id=a.slide.slide_id)
    return base


# ====================================================================== writers
def _csv_text(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def write_crop_files(dirpath: Path, crop_id: str, image: np.ndarray, yx: list) -> tuple[str, str]:
    """Write ``<id>.tif`` (original dtype, single page, uncompressed) and ``<id>.csv`` (columns y,x)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    tif = dirpath / f"{crop_id}.tif"
    tifffile.imwrite(str(tif), image, photometric="minisblack", metadata=None, software=False)
    txt = _csv_text(["y", "x"], [[repr(float(y)), repr(float(x))] for y, x in yx])
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


# ====================================================================== snapshot
def snapshot_path(data_dir: str | Path, version: str) -> Path:
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


def verify_snapshot(data_dir: str | Path, version: str) -> list[str]:
    """Problems found when re-hashing a snapshot (empty list = intact)."""
    m = load_manifest(data_dir, version)
    root = snapshot_path(data_dir, version)
    bad = []
    for rel, sha in m.files.items():
        p = root / rel
        if not p.is_file():
            bad.append(f"missing {rel}")
        elif sha256_file(p) != sha:
            bad.append(f"modified {rel}")
    if _version_of(m) != m.version:
        bad.append("manifest content does not match its version")
    return bad


def _version_of(m: SnapshotManifest) -> str:
    return "ds-" + hash_obj(m.hash_payload())[:16]


def build_snapshot(store, data_dir: str | Path, analysis_ids: list[str], opts: Optional[BuildOptions] = None, *,
                   say: Say = print, dry_run: bool = False) -> BuildResult:
    opts = opts or BuildOptions()
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

    with_crops = [e for e in exts if e.cands and e.info]
    live_infos = [e.info for e in with_crops]
    ga = None
    if live_infos:
        ga = assign_groups(live_infos, seed=opts.seed, fractions=opts.fractions, level=opts.level,
                           allow_unconfirmed=opts.allow_unconfirmed_identity, identity_map=opts.identity_map, lock=lock)
        warnings += ga.warnings
    elif not lock:
        raise MLError("none of the selected analyses yields a usable crop (see the warnings above): "
                      "need fully reviewed (approved) rectangular regions and a confirmed channel mapping")

    stage = datasets_dir(data_dir) / f".staging-{uuid.uuid4().hex[:10]}"
    stage.mkdir(parents=True, exist_ok=True)
    try:
        crops: list[CropRecord] = []
        used_ids: set[str] = set()
        for e in with_crops:
            aid = e.src.analysis_id
            split = ga.split_of[aid]
            e.src.group_id, e.src.split = ga.group_of[aid], split
            e.src.identity = ga.identity_of[aid]
            if aid in ga.locked_analyses:
                e.src.skipped = "belongs to the locked test set: not exported again (locked test labels never change)"
                e.src.n_crops = 0
                continue
            for c in e.cands:
                cid = safe_id(f"{aid}_p{c.page}_{c.region.id}_{c.role}")
                if cid in used_ids:
                    cid = f"{cid}_{len(used_ids)}"
                used_ids.add(cid)
                rel = f"spotiflow/{split}"
                tsha, csha = write_crop_files(stage / rel, cid, c.crop, c.points)
                dyx = list(c.dy_dx)
                rec = CropRecord(
                    crop_id=cid, analysis_id=aid, region_id=c.region.id, region_name=c.region.name,
                    field_index=c.region.field_index, page=c.page, channel_role=c.role, source_file=c.source_file,
                    source_sha256=c.source_sha, box_xyxy=list(c.box_file),
                    box_reference_frame=list(pixel_box(c.rect_ref)), alignment_dy_dx=dyx,
                    shape_hw=list(c.crop.shape), dtype=str(c.crop.dtype), n_positive=len(c.points),
                    n_rejected=c.n_rejected, n_edge_dropped=c.n_edge, split=split, group_id=ga.group_of[aid],
                    tif_sha256=tsha, csv_sha256=csha, bbox_of_polygon=c.bbox_only,
                    path_tif=f"{rel}/{cid}.tif", path_csv=f"{rel}/{cid}.csv")
                crops.append(rec)
        for e in exts:  # crops per analysis for the manifest
            e.src.n_crops = sum(1 for c in crops if c.analysis_id == e.src.analysis_id)

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

        crops.sort(key=lambda c: (c.split, c.crop_id))
        n_pos_total = sum(c.n_positive for c in crops)
        if not crops:
            raise MLError("no crop could be exported (see the warnings); refusing to create an empty snapshot")

        # ---- points.csv / regions.jsonl
        order = {"confirmed": 0, "human_added": 1, "rejected": 2, "unresolved": 3, "unreviewed": 4, "raw_proposal": 5}
        live_crops = [c for c in crops if not c.frozen_from]
        prow: list[list] = []
        counts: dict[str, int] = {}
        for e in exts:
            aid = e.src.analysis_id
            sp = e.src.split or ""
            mine = [c for c in live_crops if c.analysis_id == aid]
            keyed = sorted(e.rows, key=lambda r: (r["role"] or "", order[r["status"]], r["y"], r["x"], str(r["object_id"])))
            for n, r in enumerate(keyed, 1):
                counts[r["status"]] = counts.get(r["status"], 0) + 1
                cid = ""
                if r["status"] != "raw_proposal":
                    for c in mine:
                        x0, y0, x1, y1 = c.box_reference_frame
                        if c.channel_role == r["role"] and x0 <= r["x"] < x1 and y0 <= r["y"] < y1:
                            cid = c.crop_id
                            break
                dy, dx = _shift_for_rows(e, r["role"])
                prow.append([f"{aid}:{n}", aid, r["object_id"], "" if r["page"] is None else r["page"],
                             r["role"] or "", fnum(r["x"]), fnum(r["y"]), fnum(r["x"] - dx), fnum(r["y"] - dy),
                             r["status"], r["basis"], r["cls"], r["origin"], r.get("region_id") or "",
                             r.get("consensus_status") or "", r.get("engine") or "", fnum(r.get("score")),
                             r.get("score_kind") or "", "" if r.get("valid") is None else int(bool(r.get("valid"))),
                             sp, cid])
        (stage / "points.csv").write_text(_csv_text(POINT_COLUMNS, prow), encoding="utf-8", newline="")
        rlines = []
        for e in exts:
            for rec in e.regions:
                role_crops = dict(rec["crops"])
                for c in live_crops:
                    if c.analysis_id == e.src.analysis_id and c.region_id == rec["region_id"]:
                        role_crops[f"{c.region_id}:{c.channel_role}"] = {"crop_id": c.crop_id, "split": c.split}
                rlines.append(json.dumps(dict(rec, crops=role_crops), sort_keys=True, ensure_ascii=False))
        (stage / "regions.jsonl").write_text("\n".join(rlines) + ("\n" if rlines else ""), encoding="utf-8", newline="")

        files = {"points.csv": sha256_file(stage / "points.csv"), "regions.jsonl": sha256_file(stage / "regions.jsonl")}
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
        manifest = SnapshotManifest(
            version="", created_at=opts.created_at or utc_now(), app_version=app_version(),
            options={"seed": opts.seed, "fractions": (ga.fractions if ga else None), "level": opts.level,
                     "allow_unconfirmed_identity": opts.allow_unconfirmed_identity, "allow_bbox": opts.allow_bbox,
                     "strict_reviewed": opts.strict_reviewed, "roles": list(opts.roles), "min_crop_px": opts.min_crop_px,
                     "include_raw": opts.include_raw},
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
            unresolved_assumptions=_dedupe(assumptions))
        version = _version_of(manifest)
        manifest.version = version
        final = snapshot_path(data_dir, version)
        if final.exists():
            raise MLError(f"snapshot {version} already exists at {final}: an identical snapshot was built before. "
                          "Snapshots are immutable and are never overwritten.")
        if dry_run:
            say(f"[dry-run] would create snapshot {version}: {len(crops)} crop(s), {n_pos_total} positive(s)")
            return BuildResult(version, None, manifest, dry_run=True)
        atomic_write_text(stage / "manifest.json", pretty_json(manifest.model_dump(mode="json")))
        os.rename(stage, final)
        _read_only(final)
        stage = None  # type: ignore[assignment]
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    say(f"created snapshot {version} at {final}")
    for sp in ("train", "val", "test"):
        d = label_counts["by_split"].get(sp, {"crops": 0, "positives": 0, "groups": 0})
        say(f"  {sp}: {d['crops']} crop(s), {d['positives']} positive(s), {d['groups']} group(s)")
    say(f"  points: {label_counts['points_by_status']}")
    for w in manifest.warnings:
        say(f"  WARNING: {w}")
    return BuildResult(version, final, manifest)


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


def lock_test_set(data_dir: str | Path, version: str, *, replace: bool = False, reason: str = "",
                  allow_unconfirmed_identity: bool = False) -> dict:
    """Freeze the test groups of snapshot ``version`` (``test_lock.json``)."""
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
    if m.test_lock and not replace:
        raise MLError(f"snapshot {version} was itself built on a locked test set ({m.test_lock.get('snapshot')}); "
                      "to re-lock use --replace-lock --reason ...")
    imgs = []
    for s in m.sources:
        if s.split != "test" or not s.group_id:
            continue
        info = SourceInfo(analysis_id=s.analysis_id, name=s.name, sha256=hash_obj(s.tiff_sha256),
                          file_names=sorted(s.tiff_sha256))
        rec = _record(info, s.identity or {"source": "unknown", "confirmed": False}, None)
        imgs.append({"analysis_id": s.analysis_id, "name": s.name, "sha256": rec.sha256, "group": s.group_id,
                     "keys": {"specimen": group_key(rec, "specimen"), "slide": group_key(rec, "slide"),
                              "source_image": group_key(rec, "source_image")}})
    groups = sorted({c.group_id for c in test_crops})
    return write_lock(data_dir, snapshot=version, test_content_sha256=test_content_hash(m.crops), groups=groups, images=imgs,
                      level=m.splits.get("level") or "auto", seed=int(m.splits.get("seed") or 0), replace=replace, reason=reason)
