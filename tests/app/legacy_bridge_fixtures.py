"""Synthetic ImageJ-annotated TIFF corpora + mapping files for the ``ml import-legacy`` tests (no real data)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from legacy_fixtures import GRID_XS, GRID_YS, grid_overlay, point_roi, textured, write_stack

H, W = 120, 160
# 3 x 3 grid: columns [10,60,110,160), rows [10,45,80,115); cell (r, c) is 50 px wide, 35 px high
CW, CH = 50.0, 35.0
T0 = "2026-01-01T00:00:00Z"


def cell_xy(r: int, c: int, fx: float = 0.5, fy: float = 0.5) -> tuple[float, float]:
    """ImageJ-frame (sub-pixel) coordinates of a point at fractions (fx, fy) inside cell (r, c)."""
    return (GRID_XS[c] + fx * CW, GRID_YS[r] + fy * CH)


def marks_in(r: int, c: int, n: int, counter: Optional[int], position: int = 1, seed: int = 0) -> list[tuple]:
    """n well separated marks inside cell (r, c): (x, y, counter, position)."""
    out = []
    for k in range(n):
        fx = 0.15 + 0.7 * ((k * 0.37 + seed * 0.11) % 1.0)
        fy = 0.15 + 0.7 * ((k * 0.61 + seed * 0.23 + 0.1) % 1.0)
        # keep points at least 3 px apart (duplicates are flagged within 1 px)
        x, y = cell_xy(r, c, fx, fy)
        out.append((round(x * 2) / 2 + 0.17 * (k % 3), round(y * 2) / 2 + 0.17 * (k % 2), counter, position))
    return out


def point_roi_no_counters(points: Sequence[tuple[float, float]], position: int = 1) -> bytes:
    """A multi-point ROI WITHOUT ImageJ counters (the plane is then the ROI-level position)."""
    from roifile import ROI_TYPE, ImagejRoi
    roi = ImagejRoi.frompoints(np.array(points, dtype=np.float32))
    roi.roitype = ROI_TYPE.POINT
    roi.position = position
    return roi.tobytes()


def make_tiff(path: Path, *, marks: Sequence[tuple] = (), planes: int = 2, h: int = H, w: int = W, dtype=np.uint16, seed: int = 1,
              grid: bool = True, grid_xs=GRID_XS, grid_ys=GRID_YS, roi: Optional[bytes] = None, with_counters: bool = True,
              with_positions: bool = True, extra_overlays: Sequence[bytes] = (), subpixel: bool = True,
              info: Optional[str] = None) -> Path:
    """Write an ImageJ stack with (optionally) a counting-grid overlay and a multi-point ROI. ``marks`` =
    [(x, y, counter, position)] in the ImageJ frame (position is 1-based)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hi = 60000 if np.dtype(dtype) == np.uint16 else 250
    pl = [textured(h, w, seed * 10 + k, dtype=dtype, hi=hi) for k in range(planes)]
    if roi is None and marks:
        pts = [(m[0], m[1]) for m in marks]
        ctr = [m[2] for m in marks] if with_counters and all(m[2] is not None for m in marks) else None
        pos = [m[3] for m in marks] if with_positions else None
        roi = point_roi(pts, ctr, pos, subpixel=subpixel)
    ovs = ([grid_overlay(grid_xs, grid_ys, w, h)] if grid else []) + list(extra_overlays)
    return write_stack(path, pl, roi=roi, overlays=ovs or None, info=info)


def standard_corpus(root: Path, specimens: Sequence[str] = ("AM", "AS")) -> dict:
    """Two slides (A, B) per specimen, each with both pages marked in a few cells:

    * page 0 (position 1): cells (0,0) counter 0 -> 4 marks, (0,1) counter 1 -> 3 marks, (1,0) counter 2 -> 2 marks
    * page 1 (position 2): cell (0,0) counter 0 -> 2 marks (a DIFFERENT page of the same field)
    """
    out = {}
    seed = 0
    for sp in specimens:
        for rep in ("A", "B"):
            seed += 1
            n = f"{sp}-{rep}"
            rel = f"29-09-2026_T7_AB_{n}.tif"
            marks = (marks_in(0, 0, 4, 0, 1, seed) + marks_in(0, 1, 3, 1, 1, seed) + marks_in(1, 0, 2, 2, 1, seed)
                     + marks_in(0, 0, 2, 0, 2, seed + 7))
            make_tiff(root / rel, marks=marks, seed=seed)
            out[n] = rel
    return out


def write_channel_map(path: Path, *, defaults: Optional[dict] = None, rules: Sequence[dict] = (), files: Optional[dict] = None,
                      confirmed_by: str = "E. Researcher") -> Path:
    """``defaults``/``files`` pages are {page: role} or {page: (role, train)}."""
    def pages(d):
        return {str(p): ({"role": v[0], "train": v[1]} if isinstance(v, tuple) else {"role": v, "train": True})
                for p, v in d.items()}
    doc = {"schema": "polony-channel-map", "version": 1, "confirmed_by": confirmed_by, "confirmed_on": "2026-10-13",
           "source_of_truth": "lab notebook p. 12"}
    if defaults:
        doc["defaults"] = {"pages": pages(defaults)}
    if rules:
        doc["rules"] = [{"id": r["id"], "match": r["match"], "pages": pages(r["pages"])} for r in rules]
    if files:
        doc["files"] = {k: {"pages": pages(v)} for k, v in files.items()}
    Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return Path(path)


def write_identity_map(path: Path, corpus: dict, *, confirmed: bool = True) -> Path:
    """Specimen = first token before '-' of the corpus key (AM-A, AM-B -> AM), slide = the key."""
    rows = [{"file": rel, "specimen_id": key.split("-")[0], "slide_id": key.replace("-", "_"), "assay": "T7_AB",
             "replicate": key.split("-")[1], "scan_session": "2026-09-30", "confirmed": confirmed} for key, rel in corpus.items()]
    Path(path).write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    return Path(path)


def write_examined(path: Path, files: dict, *, evidence: str = "unit-test sidecar: researcher counted these fields",
                   source_kind: str = "explicit_manifest") -> Path:
    """files = {rel: {page: [entry dict, ...]}}"""
    doc = {"schema": "polony-examined-fields", "version": 1, "evidence": evidence, "source_kind": source_kind,
           "examined_by": "E. Researcher", "recorded_on": "2026-10-13",
           "files": [{"file": rel, "pages": {str(p): e for p, e in pg.items()}} for rel, pg in files.items()]}
    Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return Path(path)


RED_GREEN = {0: "red", 1: "green"}
