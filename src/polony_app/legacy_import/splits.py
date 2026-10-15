"""Leakage-safe, deterministic train/validation/test assignment.

The unit that is split is a *group* of images, never a patch or a point.  Patches inherit
the split of their source image (:func:`split_of_patch`).  Groups are chosen from the
strongest identity available for each image:

1. ``specimen``      biological sample (e.g. both slides A and B of sample AM)
2. ``slide``         one microscope slide
3. ``scan_session``  scanner date + settings
4. ``source_image``  the file itself (byte hash)

Identity recovered from file names is *inferred*: unless ``allow_unconfirmed`` is given
(or the image records are marked ``identity_confirmed``), :func:`assign_splits` raises
:class:`IdentityNotConfirmed` listing what must be confirmed.  Images reserved to a split
(e.g. the historical test image) are pinned there together with their whole group.
"""
from __future__ import annotations

import hashlib
import math
import json
from dataclasses import dataclass, field
from typing import Optional

from .schema import Dataset, ImageKind, ImageRecord

LEVELS = ("specimen", "slide", "scan_session", "source_image")
DEFAULT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}


class SplitError(ValueError):
    pass


class IdentityNotConfirmed(SplitError):
    """The identities needed for leakage-safe grouping are inferred only; confirm them first."""


@dataclass
class SplitAssignment:
    seed: int
    level: str
    fractions: dict[str, float]
    image_to_split: dict[str, str]
    group_to_split: dict[str, str]
    image_to_group: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    unconfirmed_identity: bool = False

    def to_json(self) -> dict:
        return {
            "seed": self.seed, "level": self.level, "fractions": self.fractions,
            "unconfirmed_identity": self.unconfirmed_identity, "warnings": self.warnings,
            "group_to_split": dict(sorted(self.group_to_split.items())),
            "image_to_group": dict(sorted(self.image_to_group.items())),
            "image_to_split": dict(sorted(self.image_to_split.items())),
        }

    def dumps(self) -> str:
        return json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n"


def group_key(im: ImageRecord, level: str) -> Optional[str]:
    if level == "specimen":
        return f"specimen:{im.assay}|{im.specimen_id}" if im.specimen_id else None
    if level == "slide":
        return f"slide:{im.assay}|{im.slide_id}" if im.slide_id else None
    if level == "scan_session":
        return f"scan_session:{im.scan_session}" if im.scan_session else None
    if level == "source_image":
        return f"source_image:{im.sha256}"
    raise SplitError(f"unknown level {level!r}; expected one of {LEVELS}")


def _resolve_composites(ds: Dataset) -> dict[str, ImageRecord]:
    """image_id -> record whose identity governs it (an RGB composite follows its raw stack)."""
    by_id = {i.image_id: i for i in ds.images}
    return {i.image_id: by_id.get(i.composite_of, i) if i.composite_of else i for i in ds.images}


def _order_key(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def assign_splits(
    ds: Dataset, *, seed: int = 0, fractions: Optional[dict[str, float]] = None,
    level: str = "auto", allow_unconfirmed: bool = False,
    confirmed_slides: Optional[set[str]] = None,
) -> SplitAssignment:
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    if (any(not math.isfinite(v) or v < 0 for v in fractions.values())
            or not math.isfinite(sum(fractions.values())) or abs(sum(fractions.values()) - 1.0) > 1e-9):
        raise SplitError(f"fractions must be non-negative and sum to 1, got {fractions}")
    if level != "auto" and level not in LEVELS:
        raise SplitError(f"level must be 'auto' or one of {LEVELS}")
    if not ds.images:
        raise SplitError("dataset has no images")
    gov = _resolve_composites(ds)
    warnings: list[str] = []

    # ---- choose a key per image
    keys: dict[str, str] = {}
    fallbacks: list[str] = []
    slide_only: list[str] = []
    for im in ds.images:
        g = gov[im.image_id]
        if level == "auto":
            key = next((k for lv in LEVELS if (k := group_key(g, lv)) is not None), None)
            if not g.specimen_id:
                (slide_only if g.slide_id else fallbacks).append(im.file_name)
        else:
            key = group_key(g, level)
            if key is None:
                raise SplitError(f"{im.file_name}: no '{level}' identity; use a weaker level or supply identity")
        keys[im.image_id] = key  # type: ignore[assignment]
    if slide_only:
        warnings.append(
            "no specimen identity (only a slide id) for: " + ", ".join(sorted(set(slide_only))) +
            "; each is grouped by its slide alone, so replicate slides of one specimen can land in different splits")
    if fallbacks:
        warnings.append(
            "no specimen/slide identity could be reconstructed for: " + ", ".join(sorted(set(fallbacks))) +
            "; each is isolated by its source-file hash, so a hidden relation to another image cannot be excluded")

    # ---- byte-identical files tie their groups together (union-find on the group keys)
    parent: dict[str, str] = {k: k for k in set(keys.values())}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    first_by_sha: dict[str, str] = {}
    for im in ds.images:
        k = keys[im.image_id]
        other = first_by_sha.setdefault(im.sha256, k)
        if find(other) != find(k):
            a_, b_ = sorted((find(other), find(k)))
            parent[b_] = a_
            warnings.append(f"byte-identical images in groups '{a_}' and '{b_}' were merged into one group")
    keys = {iid: find(k) for iid, k in keys.items()}

    # ---- identity confirmation gate
    confirmed_slides = confirmed_slides or set()
    unconfirmed = sorted({i.file_name for i in ds.images
                          if not (gov[i.image_id].identity_confirmed or gov[i.image_id].slide_id in confirmed_slides)})
    if unconfirmed and not allow_unconfirmed:
        raise IdentityNotConfirmed(
            "identity of these images is inferred from file names only and has not been confirmed "
            f"({', '.join(unconfirmed)}). Confirm specimen/slide identity with the researcher, set "
            "identity_confirmed, or pass allow_unconfirmed=True (--allow-unconfirmed-identity) to proceed "
            "with a recorded warning.")
    if unconfirmed:
        warnings.append("splits were made on UNCONFIRMED (file-name-inferred) identities; leakage across "
                        "mislabelled samples cannot be excluded")

    # ---- pins from reserved splits (whole group follows)
    groups: dict[str, list[ImageRecord]] = {}
    for im in ds.images:
        groups.setdefault(keys[im.image_id], []).append(im)
    pinned: dict[str, str] = {}
    for key, members in groups.items():
        res = {m.reserved_split for m in members if m.reserved_split}
        if len(res) > 1:
            raise SplitError(f"group {key} contains images reserved to different splits: {sorted(res)}")
        if res:
            pinned[key] = res.pop()
    for key, sp in pinned.items():
        if sp not in fractions:
            raise SplitError(f"group {key} is reserved to unknown split '{sp}' (known: {sorted(fractions)})")

    # ---- deterministic greedy allocation by image count
    total = len(ds.images)
    target = {s: f * total for s, f in fractions.items()}
    have = {s: 0 for s in fractions}
    group_to_split: dict[str, str] = {}
    for key, sp in pinned.items():
        group_to_split[key] = sp
        have[sp] += len(groups[key])
    order = sorted((k for k in groups if k not in pinned), key=lambda k: _order_key(seed, k))
    split_names = list(fractions)
    for key in order:
        sp = max(split_names, key=lambda s: (target[s] - have[s], -split_names.index(s)))
        group_to_split[key] = sp
        have[sp] += len(groups[key])
    empty = [s for s, f in fractions.items() if f > 0 and have[s] == 0]
    if empty:
        warnings.append(f"split(s) {empty} received no images: too few independent groups ({len(groups)}) "
                        "for the requested fractions")
    result = SplitAssignment(
        seed=seed, level=level, fractions=fractions,
        image_to_split={i.image_id: group_to_split[keys[i.image_id]] for i in ds.images},
        group_to_split=group_to_split, image_to_group=dict(keys), warnings=warnings,
        unconfirmed_identity=bool(unconfirmed))
    if level in ("slide", "scan_session", "source_image"):
        for lv in ("specimen",) + (("slide",) if level != "slide" else ()):
            straddle = _straddling(ds, result, lv)
            if straddle:
                warnings.append(f"grouping by '{level}' lets {lv} groups straddle splits "
                                f"(possible leakage between replicates): {'; '.join(straddle[:5])}")
    problems = verify_no_leakage(ds, result)
    if problems:   # defensive: construction should make this impossible
        raise SplitError("leakage detected after assignment: " + "; ".join(problems))
    return result


_STRICT = {
    "auto": ("specimen", "slide", "source_image"),
    "specimen": ("specimen", "slide", "source_image"),
    "slide": ("slide", "source_image"),
    "scan_session": ("scan_session", "source_image"),
    "source_image": ("source_image",),
}


def _straddling(ds: Dataset, a: SplitAssignment, level: str) -> list[str]:
    gov = _resolve_composites(ds)
    seen: dict[str, set[str]] = {}
    for im in ds.images:
        k = group_key(gov[im.image_id], level)
        if k is not None:
            seen.setdefault(k, set()).add(a.image_to_split[im.image_id])
    return [f"{k} spans {sorted(v)}" for k, v in sorted(seen.items()) if len(v) > 1]


def verify_no_leakage(ds: Dataset, a: SplitAssignment) -> list[str]:
    """Violations at every identity level the chosen grouping promises to keep apart, plus
    byte-identical images in two splits and reserved images outside their reserved split.
    (Weaker, e.g. scan-session, overlaps are expected and not violations.)"""
    problems: list[str] = []
    for lv in _STRICT[a.level]:
        problems += _straddling(ds, a, lv)
    by_sha: dict[str, set[str]] = {}
    for im in ds.images:
        by_sha.setdefault(im.sha256, set()).add(a.image_to_split[im.image_id])
    problems += [f"identical file {sha[:12]} in {sorted(v)}" for sha, v in by_sha.items() if len(v) > 1]
    for im in ds.images:
        if im.reserved_split and a.image_to_split[im.image_id] != im.reserved_split:
            problems.append(f"{im.file_name} is reserved to {im.reserved_split} but assigned {a.image_to_split[im.image_id]}")
    return problems


def split_of_patch(image_id: str, a: SplitAssignment) -> str:
    """Patches (tiles, crops, augmentations) ALWAYS take the split of their source image."""
    try:
        return a.image_to_split[image_id]
    except KeyError:
        raise SplitError(f"unknown source image {image_id}") from None


def assert_patch_assignment(patches: list[tuple[str, str, str]], a: SplitAssignment) -> None:
    """``patches``: (patch_id, source_image_id, split_used).  Raises if any patch is not in its image's split."""
    bad = [pid for pid, iid, sp in patches if split_of_patch(iid, a) != sp]
    if bad:
        raise SplitError(f"{len(bad)} patch(es) are not in the split of their source image, e.g. {bad[:3]}")


def training_images(ds: Dataset, a: SplitAssignment) -> list[ImageRecord]:
    """Raw, annotated images in the train split (composites and reserved/test images never)."""
    return [i for i in ds.images
            if a.image_to_split[i.image_id] == "train" and i.kind != ImageKind.rgb_composite
            and i.has_annotations and i.reserved_split in (None, "train")]
