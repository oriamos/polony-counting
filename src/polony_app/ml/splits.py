"""Leakage-safe grouping of source analyses into train / val / test, plus the locked test set.

Reuses :mod:`polony_app.legacy_import.splits` (group by specimen -> slide -> source image; byte-identical
files tie groups together; identities inferred from file names must be confirmed or explicitly allowed).
The unit that is split is the whole analysis (a slide scan): every crop of an analysis inherits its split, so
crops of one slide can never end up on both sides of a split.

Locked test set (``<data_dir>/datasets/test_lock.json``): created by ``ml split --lock-test`` from one snapshot.
Later builds (i) put every analysis whose specimen / slide / file bytes match a locked image into ``test`` and
never export it from the live analysis (the locked test crops are copied verbatim, so later corrections cannot change
test labels), (ii) never train on them (train refuses too).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from polony_app.legacy_import.identity import parse_identity
from polony_app.legacy_import.schema import Dataset, ImageKind, ImageRecord
from polony_app.legacy_import.splits import (
    DEFAULT_FRACTIONS,
    IdentityNotConfirmed,
    SplitError,
    assign_splits,
    group_key,
)

from .common import MLError, atomic_write_text, hash_obj, pretty_json, lock_file_path, utc_now

LOCK_SCHEMA = "polony-ml-test-lock"
_SLIDE_META = re.compile(r"^(?P<sample>.+)_(?P<rep>[A-Za-z])$")


@dataclass
class SourceInfo:
    analysis_id: str
    name: str
    sha256: str                       # combined hash of the stored file(s)
    file_names: list[str]
    width: int = 0
    height: int = 0
    dtype: str = "uint16"
    bit_depth: int = 16
    n_planes: int = 1
    slide_meta_id: Optional[str] = None
    # sha256 of the plain bytes of each source file (independent of the file NAME): lets a renamed byte-identical copy of
    # a locked test file be recognised (``sha256`` above combines names and hashes)
    file_sha256s: list[str] = field(default_factory=list)


@dataclass
class GroupAssignment:
    group_of: dict[str, str]          # analysis_id -> group key
    split_of: dict[str, str]          # analysis_id -> train|val|test
    identity_of: dict[str, dict]
    warnings: list[str]
    assumptions: list[str]
    unconfirmed: bool
    seed: int
    level: str
    fractions: dict[str, float]
    locked_analyses: list[str] = field(default_factory=list)   # analyses forced into test by the lock
    group_to_split: dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------ identity
def resolve_identity(src: SourceInfo, identity_map: Optional[dict] = None) -> dict:
    """Identity of one analysis. Priority: explicit map (confirmed) > slide id typed in the app (confirmed, the
    specimen is derived from the lab convention ``<sample>_<replicate letter>``) > file-name inference (UNCONFIRMED)."""
    imap = (identity_map or {}).get(src.analysis_id)
    if imap:
        return {"source": "identity_map", "confirmed": True, "assay": imap.get("assay"),
                "specimen_id": imap.get("specimen_id") or imap.get("specimen"),
                "replicate": imap.get("replicate"), "slide_id": imap.get("slide_id") or imap.get("slide"),
                "scan_session": imap.get("scan_session"), "note": "supplied by --identity-map"}
    if src.slide_meta_id:
        m = _SLIDE_META.match(src.slide_meta_id)
        return {"source": "slide_meta", "confirmed": True, "assay": None,
                "specimen_id": m.group("sample") if m else None, "replicate": m.group("rep") if m else None,
                "slide_id": src.slide_meta_id,
                "note": "slide id entered in the app; specimen derived with the lab convention <sample>_<replicate>"
                        if m else "slide id entered in the app; no specimen could be derived from it"}
    stems = [Path(n).stem for n in ([src.name] + src.file_names) if n]
    for stem in stems:
        ident = parse_identity(stem)
        if ident.source != "unknown":
            return {"source": "filename_pattern", "confirmed": False, "assay": ident.assay,
                    "specimen_id": ident.specimen_id, "replicate": ident.replicate, "slide_id": ident.slide_id,
                    "note": f"inferred from the name '{stem}'"}
    return {"source": "unknown", "confirmed": False, "assay": None, "specimen_id": None, "replicate": None,
            "slide_id": None, "note": "no identity could be inferred; isolated by file hash"}


def _record(src: SourceInfo, ident: dict, reserved: Optional[str]) -> ImageRecord:
    return ImageRecord(
        image_id=f"an_{src.analysis_id}", file_name=src.name or src.analysis_id, rel_path=src.analysis_id,
        sha256=src.sha256, size_bytes=0, kind=ImageKind.stack, width=src.width, height=src.height,
        n_planes=src.n_planes, dtype=src.dtype, bit_depth=src.bit_depth,
        assay=ident.get("assay"), specimen_id=ident.get("specimen_id"), replicate=ident.get("replicate"),
        slide_id=ident.get("slide_id"), scan_session=ident.get("scan_session"),
        identity_source=ident["source"], identity_confirmed=bool(ident["confirmed"]),
        reserved_split=reserved, reserved_reason="locked test set" if reserved else None, has_annotations=True)


# ------------------------------------------------------------------ lock
def load_lock(data_dir: str | Path) -> Optional[dict]:
    p = lock_file_path(data_dir)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MLError(f"{p} is unreadable ({exc}); refusing to continue without knowing which groups are locked")
    if d.get("schema") != LOCK_SCHEMA:
        raise MLError(f"{p} is not a test lock file")
    return d


def lock_fingerprint(lock: dict) -> str:
    return hash_obj({k: lock.get(k) for k in ("snapshot", "test_content_sha256", "groups", "images", "level")})


def _lock_keys(lock: dict) -> tuple[set[str], set[str], set[str], set[str]]:
    shas = {i["sha256"] for i in lock.get("images", [])}
    spec = {i["keys"]["specimen"] for i in lock.get("images", []) if i.get("keys", {}).get("specimen")}
    slide = {i["keys"]["slide"] for i in lock.get("images", []) if i.get("keys", {}).get("slide")}
    groups = set(lock.get("groups", []))
    return shas, spec, slide, groups


def id_fold(v) -> Optional[str]:
    """Identity comparison form: stripped, inner whitespace collapsed, case-folded; None for blank."""
    if v is None:
        return None
    t = " ".join(str(v).split()).casefold()
    return t or None


def _key_id(key: Optional[str]) -> Optional[str]:
    """The id part of a group key ``specimen:{assay}|{id}`` / ``slide:{assay}|{id}`` (the assay is dropped)."""
    if not key or "|" not in key:
        return None
    return key.split("|", 1)[1]


def lock_identity_sets(lock: Optional[dict]) -> tuple[set[str], set[str]]:
    """(folded specimen ids, folded slide ids) of the locked test images. The ASSAY is deliberately ignored: a new slide of a
    locked specimen with a blank / different assay, other case or stray spaces is still that specimen (a collision only costs a
    more conservative split). Locks written before ``identity`` was stored are read from their ``keys`` strings."""
    spec: set[str] = set()
    slide: set[str] = set()
    for i in (lock or {}).get("images", []):
        ident = i.get("identity") or {}
        for src, dst in ((ident.get("specimen_id") or _key_id((i.get("keys") or {}).get("specimen")), spec),
                         (ident.get("slide_id") or _key_id((i.get("keys") or {}).get("slide")), slide)):
            f = id_fold(src)
            if f:
                dst.add(f)
    return spec, slide


def _lock_file_shas(lock: dict) -> set[str]:
    return {s for i in lock.get("images", []) for s in (i.get("file_sha256s") or [])}


def locked_reason(rec: ImageRecord, lock: Optional[dict], file_sha256s=()) -> Optional[str]:
    """Why this image belongs to the locked test set (None = not locked). ``file_sha256s`` = plain sha256 of the source
    file(s): a byte-identical copy under another name is locked too (locks written before this field existed only
    carry the name-dependent ``sha256``). Specimen / slide identity is compared case-folded and WITHOUT the assay."""
    if not lock:
        return None
    shas, _spec, _slide, _groups = _lock_keys(lock)
    if rec.sha256 in shas or (set(file_sha256s) & _lock_file_shas(lock)):
        return "identical file bytes as a locked test image"
    spec, slide = lock_identity_sets(lock)
    f = id_fold(rec.specimen_id)
    if f and f in spec:
        return f"same specimen as a locked test image ({rec.specimen_id!r}; compared case-insensitively, assay ignored)"
    f = id_fold(rec.slide_id)
    if f and f in slide:
        return f"same slide as a locked test image ({rec.slide_id!r}; compared case-insensitively, assay ignored)"
    return None


def lock_identity_conflicts(manifest, lock: Optional[dict]) -> list[str]:
    """Problems when a snapshot's TRAIN / VAL crops come from a source whose specimen or slide identity (case-folded, assay
    ignored) matches the locked test set. ``ml train`` must refuse when this list is not empty (called from
    ``train.plan_training`` next to the existing group / sha / frozen-crop checks). ``manifest`` = SnapshotManifest or dict."""
    if not lock:
        return []
    d = manifest.model_dump(mode="json") if hasattr(manifest, "model_dump") else dict(manifest)
    spec, slide = lock_identity_sets(lock)
    by_id = {s.get("analysis_id"): s for s in d.get("sources", [])}
    bad: list[str] = []
    seen: set = set()
    for c in d.get("crops", []):
        if c.get("split") not in ("train", "val") or c.get("frozen_from"):
            continue
        src = by_id.get(c.get("analysis_id")) or {}
        ident = src.get("identity") or {}
        why = []
        if id_fold(ident.get("specimen_id")) in spec:
            why.append(f"specimen {ident.get('specimen_id')!r}")
        if id_fold(ident.get("slide_id")) in slide:
            why.append(f"slide {ident.get('slide_id')!r}")
        if why and c.get("analysis_id") not in seen:
            seen.add(c.get("analysis_id"))
            bad.append(f"source {src.get('name') or c.get('analysis_id')} has {c.get('split')} crops but shares its "
                       f"{' and '.join(why)} with the locked test set (case-insensitive, assay ignored): leakage")
    return bad


def write_lock(data_dir: str | Path, *, snapshot: str, test_content_sha256: str, groups: list[str],
               images: list[dict], level: str, seed: int, replace: bool = False, reason: str = "",
               overrides: Optional[dict] = None) -> dict:
    existing = load_lock(data_dir)
    if existing and not replace:
        raise MLError(f"a test lock already exists (snapshot {existing.get('snapshot')}); the locked test set is "
                      "permanent. To replace it deliberately pass --replace-lock together with --reason.")
    if existing and not reason.strip():
        raise MLError("replacing the test lock requires --reason (it is recorded)")
    lock = {"schema": LOCK_SCHEMA, "version": 1, "created_at": utc_now(), "snapshot": snapshot,
            "test_content_sha256": test_content_sha256, "level": level, "seed": seed,
            "groups": sorted(groups), "images": sorted(images, key=lambda i: i["analysis_id"]),
            "overrides": dict(overrides or {}), "history": (existing or {}).get("history", [])}
    if existing:
        lock["history"].append({"replaced_at": utc_now(), "reason": reason, "previous_snapshot": existing.get("snapshot"),
                                "previous_lock_sha256": lock_fingerprint(existing)})
    lock["lock_sha256"] = lock_fingerprint(lock)
    atomic_write_text(lock_file_path(data_dir), pretty_json(lock))
    return lock


# ------------------------------------------------------------------ assignment
def assign_groups(sources: list[SourceInfo], *, seed: int = 0, fractions: Optional[dict[str, float]] = None,
                  level: str = "auto", allow_unconfirmed: bool = False, identity_map: Optional[dict] = None,
                  lock: Optional[dict] = None) -> GroupAssignment:
    if not sources:
        raise MLError("no source analysis produced any crop; nothing to split")
    fr = dict(fractions or DEFAULT_FRACTIONS)
    for k in ("train", "val", "test"):
        fr.setdefault(k, 0.0)
    warnings: list[str] = []
    assumptions: list[str] = []
    if lock:
        if level != "auto" and lock.get("level") not in (None, level):
            raise MLError(f"the locked test set was made with group level '{lock.get('level')}', not '{level}'")
        if fr.get("test", 0) > 0:
            warnings.append("a locked test set exists: no NEW test groups are created; the test fraction was moved to "
                            "train/val (the locked test crops are carried over unchanged)")
        tv = fr["train"] + fr["val"]
        fr = {"train": fr["train"] / tv if tv else 0.85, "val": fr["val"] / tv if tv else 0.15, "test": 0.0}
    idents = {s.analysis_id: resolve_identity(s, identity_map) for s in sources}
    locked: dict[str, str] = {}
    recs = []
    for s in sources:
        r0 = _record(s, idents[s.analysis_id], None)
        why = locked_reason(r0, lock, s.file_sha256s)
        if why:
            locked[s.analysis_id] = why
            warnings.append(f"analysis {s.analysis_id} ({s.name}) is in the LOCKED test set ({why}); it goes to test only "
                            "and its live corrections are NOT exported (locked test labels never change)")
        recs.append(_record(s, idents[s.analysis_id], "test" if why else None))
    ds = Dataset(images=recs)
    try:
        sa = assign_splits(ds, seed=seed, fractions=fr, level=level, allow_unconfirmed=allow_unconfirmed)
    except IdentityNotConfirmed as exc:
        raise MLError(f"{exc}") from exc
    except SplitError as exc:
        raise MLError(f"split refused: {exc}") from exc
    warnings += sa.warnings
    for aid, idn in idents.items():
        if idn["source"] == "slide_meta":
            assumptions.append(f"analysis {aid}: specimen derived from slide id '{idn['slide_id']}' using the lab convention "
                               "<sample>_<replicate letter>; confirm that replicates of one sample must stay together")
        if not idn["confirmed"]:
            assumptions.append(f"analysis {aid}: identity {idn['note']} (UNCONFIRMED)")
    n_groups = len(set(sa.image_to_group.values()))
    if n_groups < 3:
        warnings.append(f"only {n_groups} independent group(s): train / validation / test cannot all be independent "
                        "(an empty split is reported, never filled from the same slide)")
    return GroupAssignment(
        group_of={s.analysis_id: sa.image_to_group[f"an_{s.analysis_id}"] for s in sources},
        split_of={s.analysis_id: sa.image_to_split[f"an_{s.analysis_id}"] for s in sources},
        identity_of=idents, warnings=warnings, assumptions=sorted(set(assumptions)),
        unconfirmed=sa.unconfirmed_identity, seed=seed, level=sa.level, fractions=fr,
        locked_analyses=sorted(locked), group_to_split=dict(sorted(sa.group_to_split.items())))


def lock_images_from(sources: list[SourceInfo], ga: GroupAssignment, split: str = "test") -> list[dict]:
    """Per locked image: the keys later builds compare against."""
    out = []
    for s in sources:
        if ga.split_of.get(s.analysis_id) != split:
            continue
        rec = _record(s, ga.identity_of[s.analysis_id], None)
        out.append({"analysis_id": s.analysis_id, "name": s.name, "sha256": s.sha256, "file_sha256s": sorted(s.file_sha256s),
                    "group": ga.group_of[s.analysis_id],
                    "identity": {"specimen_id": rec.specimen_id, "slide_id": rec.slide_id, "assay": rec.assay},
                    "keys": {"specimen": group_key(rec, "specimen"), "slide": group_key(rec, "slide"),
                             "source_image": group_key(rec, "source_image")}})
    return out
