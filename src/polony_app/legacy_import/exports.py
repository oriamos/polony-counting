"""Adapters from the canonical dataset to the places the labels are needed.

* :func:`reference_points` / :func:`counted_field_polygons` / :func:`score_detections` --
  evaluation of Big-FISH (or any detector) restricted to the fields the human counted;
* :func:`export_spotiflow` -- ``y,x`` CSVs (+ valid-field polygons) for Spotiflow fine-tuning;
* :func:`legacy_squares` -- the counted grid cells in the legacy ``grid_to_squares`` box
  convention, for benchmarking the old model;
* :func:`to_review_seed` -- regions / objects that seed a human-review analysis.

Why the restriction to counted fields matters: the human only marked polonies inside the
fields they counted.  Polonies elsewhere are *unlabelled, not absent*; scoring or training
on whole images would count them as false positives / background.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ..detection.masks import points_in_polygon
from ..schemas import DetectedObject, ObjectClass, Origin, Region, RegionKind
from .schema import Dataset, FieldStatus, ImageKind
from .splits import SplitAssignment, SplitError


# ---------------------------------------------------------------- evaluation
def reference_points(ds: Dataset, image_id: str, page: int) -> np.ndarray:
    """(N, 2) array of (x, y) canonical coordinates of usable human points on ``page``."""
    pts = ds.points_of(image_id, page=page)
    return np.array([[p.x, p.y] for p in pts], dtype=np.float64).reshape(-1, 2)


def counted_field_polygons(ds: Dataset, image_id: str, page: int) -> list[np.ndarray]:
    """Polygons (canonical coords) of the grid cells with at least one usable point on ``page``."""
    out = []
    for f in ds.fields_of(image_id):
        if f.n_points_by_page.get(str(page), 0) > 0:
            out.append(np.asarray(f.polygon, dtype=np.float64))
    return out


def _inside_any(xy: np.ndarray, polys: Sequence[np.ndarray]) -> np.ndarray:
    m = np.zeros(len(xy), bool)
    for poly in polys:
        m |= points_in_polygon(xy[:, 0], xy[:, 1], poly)
    return m


def score_detections(ds: Dataset, image_id: str, page: int, detections_xy: np.ndarray,
                     tolerance_px: float) -> dict:
    """One-to-one matching (Hungarian, distance <= tolerance) of detections to the human
    points, **only inside counted fields**.  Returns counts and the unmatched sets.

    ``detections_xy``: (M, 2) canonical (x, y).  No claim is made about the human marks
    being exact: they are clicks (observed median click-to-peak offset ~6 px on test_img).
    """
    ref = reference_points(ds, image_id, page)
    polys = counted_field_polygons(ds, image_id, page)
    det = np.asarray(detections_xy, dtype=np.float64).reshape(-1, 2)
    det_in = det[_inside_any(det, polys)] if len(det) and polys else det[:0]
    tp = 0
    if len(ref) and len(det_in):
        d = cdist(ref, det_in)
        d[d > tolerance_px] = 1e9
        r, c = linear_sum_assignment(d)
        tp = int(sum(1 for i, j in zip(r, c) if d[i, j] < 1e9))
    fn = len(ref) - tp
    fp = len(det_in) - tp
    return {
        "image_id": image_id, "page": page, "tolerance_px": tolerance_px,
        "n_counted_fields": len(polys), "n_reference": len(ref),
        "n_detections_in_counted_fields": int(len(det_in)),
        "n_detections_ignored_outside_counted_fields": int(len(det) - len(det_in)),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


# ---------------------------------------------------------------- old-model benchmark
def legacy_squares(ds: Dataset, image_id: str, page: Optional[int] = None) -> list[dict]:
    """Counted cells as the legacy ``grid_to_squares`` would box them: top-left (x, y) and
    side ``size`` in whole pixels, obtained by truncating the ImageJ line positions.

    ``n_points`` counts usable human points on ``page`` (all planes when None).
    Cells with no points are included with n_points = 0 but flagged ``counted=False``
    because the ROI cannot tell "not counted" from "counted, zero".  No extra skip rules
    (e.g. squares with exactly one point) are applied here.
    """
    out = []
    for f in ds.fields_of(image_id):
        poly = np.asarray(f.polygon)
        x0, y0 = poly[:, 0].min() + 0.5, poly[:, 1].min() + 0.5       # back to the ImageJ frame
        x1 = poly[:, 0].max() + 0.5
        n = (sum(f.n_points_by_page.values()) if page is None else f.n_points_by_page.get(str(page), 0))
        out.append({"field_id": f.field_id, "x": int(x0), "y": int(y0), "size": int(x1) - int(x0),
                    "counters": list(f.counters), "n_points": int(n),
                    "counted": f.status == FieldStatus.has_points})
    return out


# ---------------------------------------------------------------- Spotiflow
def export_spotiflow(
    ds: Dataset, out_dir: Path, assignment: SplitAssignment, *, pages: Sequence[int],
    splits: Sequence[str] = ("train", "val"), source_root: Optional[Path] = None,
    write_planes: bool = False,
) -> dict:
    """Write per-(image, plane) ``y,x`` CSVs for the requested splits.

    * ``pages`` is mandatory and explicit: the plane a point was placed on is NOT proof that
      it shows that plane's signal (see the weak-plane-signal warning), and which plane is
      which marker is unconfirmed.
    * the ``test`` split is refused unless named explicitly in ``splits``; images reserved
      to ``test`` are never written to another split.
    * a ``.fields.json`` next to each CSV lists the counted-field polygons: train only
      inside them (everything else is unlabelled, not background).
    * pixel data is not copied unless ``write_planes`` (then the plane is written
      unchanged, same dtype, from ``source_root``).
    """
    if not pages:
        raise SplitError("pages must be given explicitly (e.g. pages=[0])")
    out_dir = Path(out_dir)
    entries = []
    for im in ds.images:
        sp = assignment.image_to_split[im.image_id]
        if sp not in splits or im.kind == ImageKind.rgb_composite or not im.has_annotations:
            continue
        if im.reserved_split and im.reserved_split != sp:
            raise SplitError(f"{im.file_name} is reserved to '{im.reserved_split}' but assigned '{sp}'")
        for pg in pages:
            pts = ds.points_of(im.image_id, page=pg)
            if not pts:
                continue
            d = out_dir / sp
            d.mkdir(parents=True, exist_ok=True)
            stem = f"{im.image_id}__p{pg}"
            with open(d / f"{stem}.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["y", "x"])
                for p in sorted(pts, key=lambda q: (q.y, q.x)):
                    w.writerow([repr(p.y), repr(p.x)])
            polys = [f.polygon for f in ds.fields_of(im.image_id) if f.n_points_by_page.get(str(pg), 0) > 0]
            (d / f"{stem}.fields.json").write_text(json.dumps({"counted_fields": polys}, sort_keys=True))
            if write_planes:
                if source_root is None:
                    raise SplitError("source_root is required with write_planes")
                arr = tifffile.imread(str(Path(source_root) / im.rel_path), series=0)
                plane = arr[pg] if arr.ndim == 3 else arr
                tifffile.imwrite(str(d / f"{stem}.tif"), plane)       # unchanged dtype
            entries.append({"split": sp, "image_id": im.image_id, "page": pg, "n_points": len(pts),
                            "source": im.rel_path, "sha256": im.sha256, "file": f"{sp}/{stem}.csv"})
    manifest = {"dataset_id": ds.manifest.dataset_id, "coordinate_convention": "pixel centre = integer, y=row, x=column",
                "columns": ["y", "x"], "pages": list(pages), "splits": list(splits),
                "split_seed": assignment.seed, "split_level": assignment.level,
                "unconfirmed_identity": assignment.unconfirmed_identity, "entries": entries}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


# ---------------------------------------------------------------- human review seed
def to_review_seed(ds: Dataset, image_id: str, page: Optional[int] = None, *, ts: str = "") -> tuple[list[Region], list[DetectedObject]]:
    """Counted grid cells as ``counting`` regions and human points as MANUAL, UNCLASSIFIED
    objects (``uncertain``): the colour of a historical mark is not known, so none is assigned.

    ``Region.field_index`` stays None: the CSV field number is not provably the ImageJ
    counter (see the reconciliation table in the audit)."""
    regions = [
        Region(id=f.field_id, field_index=None, kind=RegionKind.counting,
               name=f"historical field r{f.grid_row}c{f.grid_col}", polygon=[tuple(q) for q in f.polygon])
        for f in ds.fields_of(image_id) if f.status == FieldStatus.has_points
    ]
    objs = []
    for p in ds.points_of(image_id, page=page):
        objs.append(DetectedObject(
            id=p.label_id, x=p.x, y=p.y, cls=ObjectClass.uncertain, origin=Origin.manual,
            region_id=p.field_id,
            history=[f"{ts} imported historical mark {p.label_id} (ImageJ counter={p.counter}, "
                     f"position={p.position}) from {p.source_file}; class unknown".strip()]))
    return regions, objs
