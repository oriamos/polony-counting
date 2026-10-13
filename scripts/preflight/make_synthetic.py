#!/usr/bin/env python3
"""Deterministic SYNTHETIC corpus of ImageJ-annotated two-page TIFFs + an INDEPENDENT ORACLE.

The files mimic the structure of the lab's historical scans (2 pages uint16, ImageJ stack with one POINT ROI blob carrying
counters and positions, one composite line-grid overlay, clipped scanner background) at a reduced pixel size, plus injected
pathologies with exact counts. Everything the oracle says is computed ONLY from the generator's own parameters (the points it
placed, the grid it drew, the counters it assigned) with the rules written in docs/app/LEGACY_FINETUNING.md - never from the
bridge code (``polony_app.ml.legacy_bridge``). The only app code the oracle may call is the *inference* mask helper
(``imaging.chamber`` / ``detection.masks``) for the page normalisation.

    python scripts/preflight/make_synthetic.py --out DIR --n 1500 --cell-px 64 --seed 1 [--workers 4] [--real-size]

writes DIR/input (the TIFF tree), DIR/oracle.json, DIR/mapping/*.json (confirmed mapping files + deliberate ERROR variants).
Nothing is ever written into the repository.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

GENERATOR_VERSION = 3
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DATES = ["15-09-2026", "22-09-2026", "29-09-2026", "06-10-2026", "13-10-2026"]
INFO = "ImageDescription: 635 [W1]\nDateTime: 2026/09/30 17:{mm:02d}:11\nPMT: 550\nLaserPower: 1.97\nFocus: 50\n"

# status of a point in the oracle
OK, OOB, DUP = "ok", "out_of_bounds", "duplicate"


@dataclass
class Spec:
    n: int = 1500
    seed: int = 20261013
    cell_px: int = 64
    cell_frac: float = 0.23
    rows: int = 4
    cols: int = 7
    real_size: bool = False
    workers: int = 1
    chamber_fraction: float = 0.85

    @property
    def step(self) -> float:
        return self.cell_px + self.cell_frac

    @property
    def origin0(self) -> int:
        return int(round(self.step / 2))

    @property
    def hw(self) -> tuple:
        if self.real_size:
            return (1460, 2438)
        o = self.origin0
        return (int(math.ceil(o + self.rows * self.step + 0.23 * o)), int(math.ceil(o + self.cols * self.step + 0.42 * o)))


# ====================================================================================== geometry (pure functions)
@dataclass
class Grid:
    xs: list
    ys: list
    bx: list           # pixel-box edges (ceil(edge - 0.5)): cell column c owns pixel columns [bx[c], bx[c+1])
    by: list

    @property
    def cols(self):
        return len(self.xs) - 1

    @property
    def rows(self):
        return len(self.ys) - 1

    def box(self, cell: int) -> list:
        r, c = divmod(cell, self.cols)
        return [self.bx[c], self.by[r], self.bx[c + 1], self.by[r + 1]]

    def cell_of_pixel(self, px: int, py: int) -> Optional[int]:
        if not (self.bx[0] <= px < self.bx[-1] and self.by[0] <= py < self.by[-1]):
            return None
        c = bisect.bisect_right(self.bx, px) - 1
        r = bisect.bisect_right(self.by, py) - 1
        return r * self.cols + c


def make_grid(spec: Spec, ox: float, oy: float) -> Grid:
    xs = [round(ox + i * spec.step, 3) for i in range(spec.cols + 1)]
    ys = [round(oy + i * spec.step, 3) for i in range(spec.rows + 1)]
    return Grid(xs, ys, [math.ceil(v - 0.5) for v in xs], [math.ceil(v - 0.5) for v in ys])


def q64(v: float) -> float:
    return round(v * 64.0) / 64.0


def avoid_half(v: float) -> float:
    """Canonical coordinates are never exactly k + 0.5 (the pixel-rounding rule is then unambiguous)."""
    return v + 1.0 / 64 if abs((v % 1.0) - 0.5) < 1e-9 else v


# ====================================================================================== ROI / TIFF writers
def point_roi_bytes(xy, counters, positions, subpixel: bool) -> bytes:
    from roifile import ROI_OPTIONS, ROI_TYPE, ImagejRoi
    arr = np.array(xy, dtype=np.float32).reshape(-1, 2)
    roi = ImagejRoi.frompoints(arr if subpixel else arr.astype(np.int32).astype(np.float32))
    roi.roitype = ROI_TYPE.POINT
    roi.name = ''                                    # roifile would invent a time-based uuid name (non-deterministic bytes)
    if not subpixel:
        roi.options = roi.options & ~ROI_OPTIONS.SUB_PIXEL_RESOLUTION
        roi.subpixel_coordinates = None
    if counters is not None:
        roi.counters = np.array(counters, dtype=np.uint8)
    roi.counter_positions = np.array(positions, dtype=np.uint32)
    return roi.tobytes()


def grid_overlay_bytes(grid: Grid, w: int, h: int) -> bytes:
    from roifile import ROI_TYPE, ImagejRoi
    segs: list = []
    for x in grid.xs:
        segs += [0, x, 0, 1, x, h]
    for y in grid.ys:
        segs += [0, 0, y, 1, w, y]
    roi = ImagejRoi(roitype=ROI_TYPE.RECT, left=0, top=0, right=w, bottom=h)
    roi.multi_coordinates = np.array(segs, dtype=">f4")
    roi.shape_roi_size = len(segs)
    return roi.tobytes()


def write_stack(path: Path, planes, roi: Optional[bytes], overlays: list, info: Optional[str]) -> None:
    import tifffile
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.stack(planes) if len(planes) > 1 else planes[0]
    meta: dict = {}
    if roi is not None:
        meta["ROI"] = roi
    if overlays:
        meta["Overlays"] = overlays if len(overlays) > 1 else overlays[0]
    if info:
        meta["Info"] = info
    tifffile.imwrite(str(path), data, imagej=True, metadata=meta)


# ====================================================================================== pixel rendering
def render_pages(rng, spec: Spec, h: int, w: int, blobs: list, chamber: bool, dtype) -> list:
    """Two pages. Page 1 carries the clipped scanner background OUTSIDE a chamber-like frame (when ``chamber``), like the real
    scans (~15-18 % clipped), page 0 only a few hot spots. ``blobs`` = [(page, cx, cy, amplitude)]."""
    from scipy import ndimage as ndi
    yy, xx = np.mgrid[0:h, 0:w]
    pages = []
    frame_w = max(12.0, 0.045 * h) if not spec.real_size else 66.0
    for page in (0, 1):
        low = rng.normal(size=(max(2, h // 24), max(2, w // 24)))
        smooth = ndi.zoom(low, (h / low.shape[0], w / low.shape[1]), order=1)[:h, :w]
        smooth = (smooth - smooth.min()) / max(1e-9, np.ptp(smooth))
        img = 3000.0 + 9000.0 * smooth + rng.normal(0, 250.0, size=(h, w))
        for (pg, cx, cy, amp) in blobs:
            if pg != page:
                continue
            x0, x1 = int(max(0, math.floor(cx) - 8)), int(min(w, math.ceil(cx) + 9))
            y0, y1 = int(max(0, math.floor(cy) - 8)), int(min(h, math.ceil(cy) + 9))
            if x0 < x1 and y0 < y1:
                img[y0:y1, x0:x1] += amp * np.exp(-((xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2) / (2 * 1.5 ** 2))
        if page == 1 and chamber:
            jitter = ndi.gaussian_filter(rng.normal(size=(h, w)), 12)
            jitter = jitter / max(1e-9, np.abs(jitter).max()) * 3.0
            edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)).astype(float)
            img[edge < frame_w + jitter] = 65535.0
        else:                                    # a few saturated hot spots only (well below the 2 % chamber threshold)
            for _ in range(3):
                cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
                img[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3] = 65535.0
        img = np.clip(img, 0, 65535).astype(np.uint16)
        pages.append(img)
    if np.dtype(dtype) == np.uint8:
        pages = [(p >> 8).astype(np.uint8) for p in pages]
    elif np.dtype(dtype) == np.float32:
        pages = [p.astype(np.float32) for p in pages]
    return pages


# ====================================================================================== one file
FLAGS_P = dict(has_p1=0.55, straddle=0.25, dup=0.2, near_dup=0.2, boundary=0.3, noplane=0.06, oob=0.06, outgrid=0.2,
               shifted=0.06, int_roi=0.03)


def _far(pts_xy: list, x: float, y: float, d: float = 3.5) -> bool:
    return all((x - a) ** 2 + (y - b) ** 2 >= d * d for a, b in pts_xy)


def gen_content(rng, spec: Spec, kind: str) -> dict:
    """Everything that defines one normal file (grid, points with their ground truth), as plain python data."""
    h, w = spec.hw
    R, C = spec.rows, spec.cols
    cells = R * C
    fl = {k: bool(rng.random() < p) for k, p in FLAGS_P.items()}
    if kind == "no_eligible":
        fl.update(has_p1=False, straddle=False, boundary=False, noplane=False, dup=False, near_dup=False)
    o = spec.origin0
    ox, oy = float(o), float(o)
    if fl["shifted"]:                                 # like FS-B_D whose grid starts at (3, 123) instead of (158, 158)
        ox, oy = float(max(1, round(0.0095 * spec.step))), float(round(0.389 * spec.step))
    grid = make_grid(spec, ox, oy)
    subpixel = not fl["int_roi"]
    big = spec.cell_px >= 150
    lo_n, hi_n = (4, 27) if big else (2, 8)
    # --- which (page, cell) carry marks
    marked0 = [c for c in range(cells) if rng.random() < 0.55]
    while len(marked0) < 2:
        marked0 = sorted(set(marked0) | {int(rng.integers(0, cells))})
    marked = {0: sorted(marked0), 1: []}
    if fl["has_p1"]:
        marked[1] = sorted(int(c) for c in rng.choice(cells, size=int(rng.integers(1, 5)), replace=False))
    if kind == "no_eligible":                           # marks only on page 1 (not trained by default): accepted, zero crops
        marked = {0: [], 1: sorted(int(c) for c in rng.choice(cells, size=2, replace=False))}
    # --- counters: raster order, sometimes shuffled pairs / shifted / with skipped numbers (absent counters)
    perm = list(range(cells))
    if rng.random() < 0.3:
        for _ in range(int(rng.integers(1, 4))):
            a, b = rng.choice(cells, size=2, replace=False)
            perm[a], perm[b] = perm[b], perm[a]
    skip = set(int(v) for v in rng.choice(cells, size=int(rng.integers(1, 3)), replace=False)) if rng.random() < 0.25 else set()
    off = int(rng.choice([0, 0, 0, 1]))
    nums = [v for v in range(cells + len(skip) + off + 2) if v not in skip][off:off + cells]
    ctr_of = {c: nums[perm[c]] for c in range(cells)}
    pts: list = []                                    # dicts, in generation order

    def canon_from_px(cx, cy):
        return (cx, cy)

    def add(page, cx, cy, ctr_cell, kind_, pos=None, status=OK):
        cx, cy = avoid_half(q64(cx)), avoid_half(q64(cy))
        if not subpixel:
            cx, cy = float(math.floor(cx + 0.5)), float(math.floor(cy + 0.5))
        d = dict(page=page, cx=cx, cy=cy, home=ctr_cell, kind=kind_, status=status, pos=(page + 1 if pos is None else pos))
        pts.append(d)
        return d

    def xy_on(page):
        return [(p["cx"], p["cy"]) for p in pts if p["page"] == page and p["kind"] not in ("oob",)]

    def sample_in(cell, margin=2.0):
        r, c = divmod(cell, C)
        return (rng.uniform(grid.bx[c] - 0.5 + margin, grid.bx[c + 1] - 0.5 - margin),
                rng.uniform(grid.by[r] - 0.5 + margin, grid.by[r + 1] - 0.5 - margin))

    # --- normal marks
    for page in (0, 1):
        for cell in marked[page]:
            n = int(rng.integers(lo_n, hi_n + 1))
            for _ in range(n):
                for _try in range(60):
                    x, y = sample_in(cell)
                    if _far(xy_on(page), x, y):
                        add(page, x, y, cell, "normal")
                        break
    base = [p for p in pts if p["kind"] == "normal"]
    # --- boundary points: within 0.4 px of a pixel-box edge of their home cell (either side); rounding decides the cell
    if fl["boundary"] and subpixel:
        for _ in range(int(rng.integers(1, 4))):
            page = 0 if marked[0] else 1
            if not marked[page]:
                continue
            cell = int(rng.choice(marked[page]))
            r, c = divmod(cell, C)
            u = float(rng.uniform(1 / 16, 0.4)) * float(rng.choice([-1, 1]))
            if rng.random() < 0.5:
                x, y = grid.bx[int(rng.choice([c, c + 1]))] - 0.5 + u, rng.uniform(grid.by[r] + 2, grid.by[r + 1] - 2)
            else:
                x, y = rng.uniform(grid.bx[c] + 2, grid.bx[c + 1] - 2), grid.by[int(rng.choice([r, r + 1]))] - 0.5 + u
            if _far(xy_on(page), x, y) and not any(p["home"] == cell and p["kind"] == "boundary" for p in pts):
                add(page, x, y, cell, "boundary")
    # --- straddling clicks: a point of cell A's counter physically inside the neighbour B
    if fl["straddle"]:
        used = set()
        for _ in range(int(rng.integers(1, 3))):
            page = 0 if marked[0] else 1
            cand = [c for c in marked[page] if c not in used]
            if not cand:
                break
            a = int(rng.choice(cand))
            r, c = divmod(a, C)
            nbrs = [(r, c + 1)] * (c < C - 1) + [(r, c - 1)] * (c > 0) + [(r + 1, c)] * (r < R - 1) + [(r - 1, c)] * (r > 0)
            rb, cb = nbrs[int(rng.integers(0, len(nbrs)))]
            b = rb * C + cb
            for _try in range(60):
                x, y = sample_in(b, margin=2.0)
                near_edge = (cb != c and abs(x - (grid.bx[max(c, cb)] - 0.5)) < 6) or (rb != r and abs(y - (grid.by[max(r, rb)] - 0.5)) < 6)
                if near_edge and _far(xy_on(page), x, y):
                    add(page, x, y, a, "straddle")
                    used.add(a)
                    break
    # --- unspecified plane (position 0 / beyond the planes) inside a marked cell
    if fl["noplane"]:
        page = 0 if marked[0] else 1
        cell = int(rng.choice(marked[page]))
        for _try in range(60):
            x, y = sample_in(cell)
            if _far(xy_on(page), x, y):
                d = add(page, x, y, cell, "noplane", pos=int(rng.choice([0, 3])))
                d["page"] = None
                break
    # --- near duplicates (2.5 px apart: both are labels) and duplicates (within 1 px of an earlier point: flagged)
    for _ in range(int(rng.integers(1, 3)) if fl["near_dup"] and base else 0):
        o_ = base[int(rng.integers(0, len(base)))]
        for ang in rng.uniform(0, 2 * math.pi, size=12):
            x, y = o_["cx"] + 2.5 * math.cos(ang), o_["cy"] + 2.5 * math.sin(ang)
            r, c = divmod(o_["home"], C)
            inside = grid.bx[c] + 1 < x < grid.bx[c + 1] - 2 and grid.by[r] + 1 < y < grid.by[r + 1] - 2
            others = [(p["cx"], p["cy"]) for p in pts if p["page"] == o_["page"] and p is not o_]
            if inside and _far(others, x, y) and math.hypot(x - o_["cx"], y - o_["cy"]) > 1.5:
                add(o_["page"], x, y, o_["home"], "near_dup")
                break
    dups = []
    for _ in range(int(rng.integers(1, 3)) if fl["dup"] and base else 0):
        o_ = base[int(rng.integers(0, len(base)))]
        if any(d["of"] is o_ for d in dups):
            continue
        d = dict(page=o_["page"], cx=avoid_half(q64(o_["cx"] + 0.25)), cy=avoid_half(q64(o_["cy"] - 0.25)), home=o_["home"], kind="dup",
                 status=DUP, pos=o_["pos"], of=o_)
        if not subpixel:
            d["cx"], d["cy"] = o_["cx"], o_["cy"]
        dups.append(d)
    # --- outside the grid / out of bounds (counter of a marked cell)
    all_marked = sorted(set(marked[0]) | set(marked[1]))
    if fl["outgrid"] and all_marked:
        for _ in range(int(rng.integers(1, 4))):
            cell = int(rng.choice(all_marked))
            page = int(rng.integers(0, 2))
            spots = []
            if grid.bx[0] - 0.5 > 4:
                spots.append((rng.uniform(1.5, grid.bx[0] - 3.0), rng.uniform(1.5, h - 2)))
            if h - (grid.by[-1] - 0.5) > 4:
                spots.append((rng.uniform(1.5, w - 2), rng.uniform(grid.by[-1] + 1.5, h - 1.5)))
            if w - (grid.bx[-1] - 0.5) > 4:
                spots.append((rng.uniform(grid.bx[-1] + 1.5, w - 1.5), rng.uniform(1.5, h - 2)))
            if spots:
                x, y = spots[int(rng.integers(0, len(spots)))]
                add(page, x, y, cell, "outgrid")
    if fl["oob"] and all_marked:
        for _ in range(int(rng.integers(1, 3))):
            cell = int(rng.choice(all_marked))
            if rng.random() < 0.5:
                x, y = w + rng.uniform(0.6, 6.0), rng.uniform(2, h - 2)
            else:
                x, y = rng.uniform(2, w - 2), h + rng.uniform(0.6, 6.0)
            d = add(int(rng.integers(0, 2)), x, y, cell, "oob", status=OOB)
    # --- ROI order: one block per home cell (counter order, or a random block permutation = out-of-order counters)
    blocks = defaultdict(list)
    for p in pts:
        blocks[p["home"]].append(p)
    order = sorted(blocks, key=lambda c: ctr_of[c])
    if rng.random() < 0.3:
        order = [order[i] for i in rng.permutation(len(order))]
    seq: list = []
    for c in order:
        for p in sorted(blocks[c], key=lambda p: (p["page"] if p["page"] is not None else 0)):
            seq.append(p)
            for d in dups:
                if d["of"] is p:
                    seq.append(d)
    out = []
    pos_of = {id(p): i for i, p in enumerate(seq)}
    for i, p in enumerate(seq):
        cx, cy = p["cx"], p["cy"]
        ijx, ijy = (cx + 0.5, cy + 0.5) if subpixel else (cx + 0.5, cy + 0.5)     # centre in the ImageJ frame
        stored = (ijx, ijy) if subpixel else (int(cx), int(cy))
        in_bounds = 0.0 <= ijx < w and 0.0 <= ijy < h
        px, py = math.floor(cx + 0.5), math.floor(cy + 0.5)
        geo = grid.cell_of_pixel(px, py) if in_bounds else None
        status = p["status"]
        if status == OK and not in_bounds:
            status = OOB
        out.append(dict(i=i, x=stored[0], y=stored[1], cx=cx, cy=cy, ctr=ctr_of[p["home"]], pos=int(p["pos"]), page=p["page"],
                        kind=p["kind"], home=int(p["home"]), geo=(None if status == OOB else geo), status=status,
                        dup_of=(pos_of[id(p["of"])] if p["kind"] == "dup" else None)))
    return dict(h=h, w=w, grid=dict(xs=grid.xs, ys=grid.ys, bx=grid.bx, by=grid.by), subpixel=subpixel, flags=fl,
                ctr_of={str(k): int(v) for k, v in ctr_of.items()}, marked={str(k): v for k, v in marked.items()}, points=out)


def self_check(rec: dict) -> None:
    """Invariants the oracle relies on (guards against generator bugs, not against product bugs)."""
    pts = rec["points"]
    by_page = defaultdict(list)
    for p in pts:
        if p["status"] == OK:
            by_page[p["page"]].append(p)
    for page, g in by_page.items():
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                d = math.hypot(g[a]["cx"] - g[b]["cx"], g[a]["cy"] - g[b]["cy"])
                assert d > 1.0, f"unintended duplicate pair in {rec['rel']}"
    cnt = defaultdict(Counter)
    for p in pts:
        if p["status"] == OK and p["geo"] is not None:
            cnt[p["ctr"]][p["geo"]] += 1
    for ctr, cc in cnt.items():
        ranked = cc.most_common()
        assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], f"counter {ctr} has no strict majority cell in {rec['rel']}"
        home = {p["home"] for p in pts if p["ctr"] == ctr}
        assert ranked[0][0] in home


# ====================================================================================== plan
PATHOLOGY_RATES = [          # (kind, rate, reject class under the default channel map or None=accepted)
    ("copy", 0.020, None), ("missing_roi", 0.010, "missing_roi"), ("missing_grid", 0.006, "missing_grid"),
    ("corrupt_roi", 0.006, "corrupt_metadata"), ("single_page", 0.010, "channel_mapping_page_out_of_range"),
    ("rgb", 0.007, "unsupported_layout"), ("multi_roi", 0.005, "multiple_roi_blobs"), ("uint8", 0.012, None),
    ("no_counters", 0.005, "no_counters"), ("float32", 0.004, "unsupported_dtype"), ("unreadable", 0.004, "tiff_unreadable"),
    ("no_eligible", 0.005, None), ("rescan", 0.030, None)]
REJECT = {k: c for k, _r, c in PATHOLOGY_RATES}


def plan_files(spec: Spec) -> list:
    rng = np.random.default_rng([spec.seed, 7])
    n = spec.n
    budget = int(0.45 * n)
    counts: dict = {}
    for kind, rate, _c in PATHOLOGY_RATES:
        k = max(1, int(round(rate * n))) if n >= 12 else 0
        if k > budget:
            k = max(0, budget)
        budget -= k
        counts[kind] = k
    n_path = sum(counts.values())
    n_normal = n - n_path
    assert n_normal >= 4, "N too small"
    # specimens: 1-3 replicates (mean ~2); pathology files become extra replicates of existing specimens
    specimens, left, k = [], n_normal, 0
    while left > 0:
        reps = min(left, int(rng.choice([1, 2, 2, 2, 3])))
        a, b = k // 26 % 26, k % 26
        sample = f"{LETTERS[a]}{LETTERS[b]}{k}"
        assay = "T7_FS" if rng.random() < 0.2 else "T7_AB"
        specimens.append(dict(sample=sample, assay=assay, reps=reps, next=0, date=DATES[int(rng.integers(0, len(DATES)))]))
        left -= reps
        k += 1
    items = []

    def take(spec_i, kind):
        s = specimens[spec_i]
        rep = LETTERS[s["next"]]
        s["next"] += 1
        return dict(kind=kind, sample=s["sample"], assay=s["assay"], rep=rep, date=s["date"])

    for si, s in enumerate(specimens):
        for _ in range(s["reps"]):
            items.append(take(si, "good"))
    for kind in ("uint8", "no_eligible", "missing_roi", "missing_grid", "corrupt_roi", "single_page", "rgb", "multi_roi", "no_counters",
                 "float32", "unreadable"):
        for _ in range(counts[kind]):
            items.append(take(int(rng.integers(0, len(specimens))), kind))
    for i, it in enumerate(items):
        it["idx"] = i
    first_unreadable = next((it for it in items if it["kind"] == "unreadable"), None)
    if first_unreadable:
        first_unreadable["empty"] = True
    # --- names / directories
    for it in items:
        stem = f"{it['date']}_{it['assay']}_{it['sample']}-{it['rep']}"
        ext = ".tif"
        r = rng.random()
        if r < 0.03:
            ext = ".tiff"
        elif r < 0.05:
            ext = ".TIF"
        d = it["date"]
        r = rng.random()
        if r < 0.03:
            d = "Série été (lot 2)"
        elif r < 0.06:
            d = "lab scans 2"
        elif r < 0.09:
            d = f"{it['date']}/{it['assay']}"
        if it["kind"] == "rgb":
            stem += "_(635_532)"
        elif rng.random() < 0.02:
            stem += "_D"
        elif rng.random() < 0.02:
            stem += " café"
        it["rel"] = f"{d}/{stem}{ext}"
    seen = set()
    for it in items:                                    # unique relative paths
        while it["rel"].lower() in seen:
            it["rel"] = it["rel"].replace(".", "_x.", 1) if "_x." not in it["rel"] else it["rel"] + ".tif"
        seen.add(it["rel"].lower())
    good = [it for it in items if it["kind"] in ("good", "uint8")]
    # rescans: same slide re-scanned (own pixels), possibly under the SAME base name in another directory
    for j in range(counts["rescan"]):
        src = good[int(rng.integers(0, len(good)))]
        stem = Path(src["rel"]).stem
        it = dict(kind="rescan", sample=src["sample"], assay=src["assay"], rep=src["rep"], date=src["date"], idx=len(items),
                  rel=(f"rescans/{Path(src['rel']).name}" if j % 2 == 0 else f"rescans/{stem}_rescan{Path(src['rel']).suffix}"))
        if it["rel"].lower() in seen:
            it["rel"] = f"rescans/{j}_{Path(src['rel']).name}"
        seen.add(it["rel"].lower())
        items.append(it)
    # copies: byte-identical, other names / directories (some sort BEFORE the original so the copy becomes the kept path)
    for j in range(counts["copy"]):
        src = good[int(rng.integers(0, len(good)))]
        nm = Path(src["rel"]).name
        variants = [f"{Path(src['rel']).parent}/{Path(nm).stem} (1){Path(nm).suffix}", f"backup/{nm}", f"0-first/Copy of {nm}", f"other_dir/{nm}"]
        rel = variants[j % len(variants)]
        while rel.lower() in seen:
            rel = "dup" + rel
        seen.add(rel.lower())
        items.append(dict(kind="copy", of=src["rel"], rel=rel, idx=len(items), sample=src["sample"], assay=src["assay"], rep=src["rep"],
                          date=src["date"]))
    return items


# ====================================================================================== generate one item
def generate_item(args) -> dict:
    spec_d, it, root = args
    spec = Spec(**spec_d)
    path = Path(root) / it["rel"]
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = it["kind"]
    ident = dict(specimen_id=it["sample"], slide_id=f"{it['sample']}_{it['rep']}", assay=it["assay"], replicate=it["rep"],
                 scan_session=it["date"], confirmed=True)
    rec = dict(rel=it["rel"], kind=kind, discovered=True, identity=ident, reject=REJECT.get(kind) if kind != "copy" else None,
               points=[], grid=None, h=0, w=0, n_pages=2, dtype="uint16")
    if kind == "copy":
        shutil.copyfile(Path(root) / it["of"], path)
        rec.update(of=it["of"])
        return rec
    rng = np.random.default_rng([spec.seed, 1000 + it["idx"]])
    if kind == "unreadable":
        data = b"" if it.get("empty") else bytes(rng.integers(0, 256, size=400, dtype=np.uint8))   # only ONE empty file (empty files are all byte-identical)
        path.write_bytes(data)
        rec.update(n_pages=0, dtype="")
        return rec
    h, w = spec.hw
    if kind == "rgb":
        import tifffile
        rgb = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        rgb[..., 2] = np.where(rng.random((h, w)) < 0.18, 255, 0)
        tifffile.imwrite(str(path), rgb, photometric="rgb", compression="lzw")
        rec.update(n_pages=1, dtype="uint8", h=h, w=w)
        return rec
    content = gen_content(rng, spec, kind)
    rec.update(content)
    rec["points_total_written"] = len(content["points"])
    self_check(rec)
    pts = content["points"]
    grid = Grid(**content["grid"])
    dtype = {"uint8": np.uint8, "float32": np.float32}.get(kind, np.uint16)
    chamber = bool(rng.random() < spec.chamber_fraction)
    rec["chamber_expected"] = chamber
    # polonies: a blob at every mark (+ jitter, clicks are ~5 px off in the real data), plus unmarked blobs
    blobs = []
    for p in pts:
        if p["page"] is not None and p["status"] == OK:
            blobs.append((p["page"], p["cx"] + rng.normal(0, 1.5), p["cy"] + rng.normal(0, 1.5), float(rng.uniform(6000, 30000))))
    for _ in range(int(rng.integers(5, 5 + 2 * spec.rows * spec.cols))):
        blobs.append((int(rng.random() < 0.3), rng.uniform(0, content["w"]), rng.uniform(0, content["h"]), float(rng.uniform(4000, 20000))))
    pages = render_pages(rng, spec, content["h"], content["w"], blobs, chamber, dtype)
    mm = int(it["idx"] % 59)
    info = INFO.format(mm=mm)
    subpixel = content["subpixel"]
    roi = point_roi_bytes([(p["x"], p["y"]) for p in pts], None if kind == "no_counters" else [p["ctr"] for p in pts],
                          [p["pos"] for p in pts], subpixel)
    ov = [grid_overlay_bytes(grid, content["w"], content["h"])]
    if kind == "missing_roi":
        roi = None
    elif kind == "missing_grid":
        ov = []
    elif kind == "corrupt_roi":
        roi = b"NOTAROI" + bytes(rng.integers(0, 256, size=48, dtype=np.uint8))
    elif kind == "multi_roi":
        second = point_roi_bytes([(pts[0]["x"] + 1.0, pts[0]["y"] + 1.0)], [0], [1], True)
        ov = ov + [second]
    elif kind == "single_page":
        pages = pages[:1]
        rec["n_pages"] = 1
    if kind == "no_eligible":                           # accepted file whose marks are all out of the grid
        pass
    write_stack(path, pages, roi, ov, info)
    rec["dtype"] = str(np.dtype(dtype))
    return rec


# ====================================================================================== channel / identity / examined files
def resolve_pages(ch: dict, rec: dict) -> tuple:
    """Independent resolver of the channel-map precedence (files > rules > defaults). -> (origin, {page: {role, train}})."""
    def pg(block):
        return {int(k): dict(v) for k, v in block["pages"].items()}
    if rec["rel"] in (ch.get("files") or {}):
        return "file", pg(ch["files"][rec["rel"]])
    ident = rec["identity"]
    import fnmatch
    for r in ch.get("rules", []):
        m = r["match"]
        ok = True
        for k, v in m.items():
            if k == "glob":
                ok &= fnmatch.fnmatch(rec["rel"], v) or ("/" not in v and fnmatch.fnmatch(Path(rec["rel"]).name, v))
            elif k in ("assay", "specimen_id", "slide_id"):
                ok &= ident.get(k) == v
            else:
                ok = False
        if ok:
            return f"rule:{r['id']}", pg(r)
    return "default", pg(ch["defaults"])


def channel_doc(files: dict, rng) -> dict:
    page = lambda role, train: {"role": role, "train": train}          # noqa: E731
    default = {"0": page("red", True), "1": page("green", False)}
    good = [r for r in files.values() if r["kind"] == "good" and r["points"]]
    doc = {"schema": "polony-channel-map", "version": 1, "confirmed_by": "Rehearsal Operator", "confirmed_on": "2026-10-13",
           "source_of_truth": "synthetic rehearsal: generator parameters", "defaults": {"pages": default}, "rules": [], "files": {}}
    if any(r["identity"]["assay"] == "T7_FS" for r in good):
        doc["rules"].append({"id": "fs-scans", "match": {"assay": "T7_FS"}, "pages": dict(default)})
    ab = sorted({r["identity"]["specimen_id"] for r in good if r["identity"]["assay"] == "T7_AB"})
    if ab:
        doc["rules"].append({"id": "p0-only", "match": {"specimen_id": ab[len(ab) // 3]}, "pages": {"0": page("red", True)}})
    copied = {r["of"] for r in files.values() if r["kind"] == "copy"}
    cand = sorted(r["rel"] for r in good if r["identity"]["assay"] == "T7_AB" and r["rel"] not in copied
                  and r["identity"]["specimen_id"] != (ab[len(ab) // 3] if ab else ""))
    if cand:
        doc["files"][cand[len(cand) // 2]] = {"pages": {"0": page("red", True), "1": page("green", True)}}
    return doc


def identity_rows(files: dict) -> list:
    rows = []
    for rel in sorted(files):
        i = files[rel]["identity"]
        rows.append({"file": rel, "specimen_id": i["specimen_id"], "slide_id": i["slide_id"], "assay": i["assay"],
                     "replicate": i["replicate"], "scan_session": i["scan_session"], "confirmed": True})
    return rows


def write_json(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def error_variants(out: Path, files: dict, ch: dict, rows: list) -> dict:
    """Deliberately broken mapping files. -> {name: {"kind": ..., "path": ..., "expect": {...}}}"""
    E = out / "errors"
    V: dict = {}
    clone = lambda d: json.loads(json.dumps(d))          # noqa: E731

    def chan(name, mut, any_text, dry_exit=2, real_exit=2):
        d = clone(ch)
        mut(d)
        V[name] = dict(kind="channels", path=str(write_json(E / f"channels.{name}.json", d)), dry_exit=dry_exit, real_exit=real_exit, any_text=any_text)

    chan("missing_confirmed_by", lambda d: d.update(confirmed_by=""), ["confirmed_by"])
    chan("empty_source_of_truth", lambda d: d.update(source_of_truth=" "), ["source_of_truth"])
    chan("bad_schema", lambda d: d.update(schema="something-else"), ["schema"])
    chan("unknown_key", lambda d: d.update(confirmd_by="typo"), ["unknown", "confirmd_by", "not allowed", "unexpected"])
    chan("page_without_train", lambda d: d["defaults"]["pages"]["1"].pop("train"), ["train"])
    chan("duplicate_role", lambda d: d["defaults"]["pages"]["1"].update(role="red"), ["role", "unique"])
    chan("no_role", lambda d: d["defaults"]["pages"]["1"].update(role=""), ["role"])
    chan("rule_matches_no_file", lambda d: d["rules"].append({"id": "ghost", "match": {"assay": "NO_SUCH_ASSAY"}, "pages": clone(d["defaults"]["pages"])}),
         ["ghost", "matches no", "no file"], dry_exit=[0, 2], real_exit=2)
    chan("file_entry_not_found", lambda d: d["files"].update({"no/such/file.tif": {"pages": clone(d["defaults"]["pages"])}}),
         ["no/such/file.tif", "not found", "does not exist", "no file"], dry_exit=[0, 2], real_exit=2)
    tmp = clone(ch)
    other = {"id": "other", "match": {"glob": "*.tif"}, "pages": {"0": {"role": "green", "train": True}, "1": {"role": "red", "train": False}}}
    tmp["rules"] = [{"id": "r1", "match": {"glob": "*.tif"}, "pages": clone(tmp["defaults"]["pages"])}, other]
    tmp["files"] = {}
    V["conflicting_rules"] = dict(kind="channels", path=str(write_json(E / "channels.conflicting_rules.json", tmp)), dry_exit=2, real_exit=2,
                                  any_text=["r1", "other", "disjoint", "both match"])
    p = E / "channels.invalid_json.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"schema": "polony-channel-map", ', encoding="utf-8")
    V["invalid_json"] = dict(kind="channels", path=str(p), dry_exit=2, real_exit=2, any_text=["JSON"])

    def ident(name, rows_, any_text, dry_exit=2, real_exit=2, level=None):
        V["id_" + name] = dict(kind="identity", path=str(write_json(E / f"identities.{name}.json", {"rows": rows_})), dry_exit=dry_exit,
                               real_exit=real_exit, any_text=any_text, level=level)

    r0 = clone(rows)
    for r in r0[::7]:
        r["confirmed"] = False
    ident("unconfirmed_rows", r0, ["UNCONFIRMED", "unconfirmed"], dry_exit=0, real_exit=2)
    r1 = clone(rows)
    r1.append(dict(r1[0], specimen_id="ZZ-OTHER"))
    ident("conflicting_rows", r1, ["two rows", "different identities"])
    r2 = clone(rows)
    r2[0].pop("confirmed")
    ident("confirmed_missing", r2, ["confirmed"])
    r3 = clone(rows)
    r3[1]["specimen_id"] = None
    ident("no_specimen", r3, ["specimen", "--level slide"])
    copied = {r["of"] for r in files.values() if r["kind"] == "copy"} | {r["rel"] for r in files.values() if r["kind"] == "copy"}
    ui = [i for i, r in enumerate(rows) if files[r["file"]]["kind"] in ("good", "uint8") and r["file"] not in copied]
    r4 = clone(rows)
    r4[ui[0]]["specimen_id"] = rows[ui[1]]["specimen_id"].lower()
    r4[ui[0]]["assay"] = "T7_OTHER"
    # architect notice (3): refused; the landed code keeps the two assays as separate groups and prints a note. Either is accepted here,
    # the difference is reported in the final report.
    ident("assay_conflict", r4, ["assay", "specimen"])
    r5 = clone(rows)
    r5[0]["confirmed"] = "maybe"
    ident("confirmed_not_bool", r5, ["confirmed"])
    p = E / "identities.invalid.json"
    p.write_text("{not json", encoding="utf-8")
    V["id_invalid_json"] = dict(kind="identity", path=str(p), dry_exit=2, real_exit=2, any_text=["identity map", "cannot be read"], level=None)
    return V


# ====================================================================================== the oracle's rules (docs -> code)
def _classify_core(rec: dict, trained_pages, mode: str, entries: dict):
    """Per (page, cell) status from the generator's ground truth. ``entries`` = {(page, cell): {"count": int|None, "clash": bool}}."""
    pts = [p for p in rec["points"] if p["status"] == OK]
    ncell = (len(rec["grid"]["xs"]) - 1) * (len(rec["grid"]["ys"]) - 1)
    home_of = {}                       # counter -> its cell (strict majority of the in-grid points, asserted by self_check)
    cnt = defaultdict(Counter)
    for p in pts:
        if p["geo"] is not None:
            cnt[p["ctr"]][p["geo"]] += 1
    for k, c in cnt.items():
        home_of[k] = c.most_common(1)[0][0]
    noplane_cells = {p["geo"] for p in pts if p["page"] is None and p["geo"] is not None}
    out = {}
    for page in trained_pages:
        own, stray, extra = defaultdict(list), defaultdict(list), defaultdict(list)
        for p in pts:
            if p["page"] != page:
                continue
            owner = home_of.get(p["ctr"])
            if p["geo"] is not None:
                (own if owner == p["geo"] else stray)[p["geo"]].append(p)
            if owner is not None and owner != p["geo"]:
                extra[owner].append(p)
        for cell in range(ncell):
            o, s, e = own.get(cell, []), stray.get(cell, []), extra.get(cell, [])
            ent = entries.get((page, cell))
            has = (page, cell) in entries
            t = dict(status=None, reason="", n_marks=0, labels=[], n_stray=0, source=None)
            if cell in noplane_cells:
                t.update(status="conflict", reason="unspecified_plane")
            elif has and ent is None or (has and ent.get("clash")):
                t.update(status="conflict", reason="manifest_clash", n_marks=len(o) + len(e))
            else:
                stated = ent["count"] if ent else None
                n = len(o) + len(e) if o else 0
                if mode == "marked-only":
                    if n == 0:
                        t.update(status="unknown", reason="straddle" if (not o and s) else "no_marks")
                    elif stated is not None and stated != n:
                        t.update(status="conflict", reason="stated_mismatch", n_marks=n)
                    else:
                        t.update(status="positive_marked", n_marks=n)
                else:
                    if not has:
                        t.update(status="unknown", reason="not_listed", n_marks=n)
                    elif n > 0:
                        t.update(status="conflict" if (stated is not None and stated != n) else "positive_marked", n_marks=n,
                                 reason="stated_mismatch" if (stated is not None and stated != n) else "")
                    elif stated == 0:
                        t.update(status="verified_zero")
                    elif stated is None:
                        t.update(status="unknown", reason="no_count")
                    else:
                        t.update(status="conflict", reason="stated_mismatch")
                    if t["status"] == "verified_zero" and (o or s or e):
                        t.update(status="conflict", reason="stated_mismatch", n_marks=len(o) + len(s) + len(e))
                    elif not o and s and t["status"] == "unknown":
                        t.update(reason="straddle")
                if mode == "marked-only" and t["status"] == "unknown" and False:
                    pass
            if t["status"] in ("positive_marked", "verified_zero"):
                idx = sorted(id(p) for p in o + s)
                t["labels"] = [p["i"] for p in sorted(o + s, key=lambda p: (p["cy"], p["cx"], p["i"]))]
                t["n_stray"] = len(s)
            out[(page, cell)] = t
    return out, home_of


def resolve_entries_for(rec: dict, doc_entries: dict, home_of: dict) -> tuple:
    """Manifest page entries of ONE file -> ({(page, cell): entry|None}, n_unresolved). Keys: counter | row+col | field (never
    resolved without the explicit acknowledgement, which the rehearsal never passes)."""
    ncols = len(rec["grid"]["xs"]) - 1
    nrows = len(rec["grid"]["ys"]) - 1
    res: dict = {}
    unresolved = 0
    for page, ents in doc_entries.items():
        for e in ents:
            cell = None
            if e.get("counter") is not None:
                cell = home_of.get(e["counter"])
            elif e.get("row") is not None:
                if 0 <= e["row"] < nrows and 0 <= e["col"] < ncols:
                    cell = e["row"] * ncols + e["col"]
            if cell is None:
                unresolved += 1
                continue
            cnt = e.get("count")
            if cnt is None and e.get("counted_zero"):
                cnt = 0
            key = (int(page), cell)
            res[key] = dict(count=cnt, clash=key in res)
            if key in res and res[key]["clash"]:
                res[key] = dict(count=cnt, clash=True)
    return res, unresolved


def derive(rec: dict, pages_map: dict, mode: str = "marked-only", manifest_pages: Optional[dict] = None) -> dict:
    """Oracle expectations of ONE accepted file for a channel mapping (``pages_map`` = {page: {role, train}}):

    ``tuples``  {(page, cell): status/reason/n_marks/labels(point ids)} for every trained page and every grid cell;
    ``crops``   {(page, cell): {box, labels [(y, x) in the crop frame, native], n_positive, status, n_marks, n_stray}};
    ``dropped`` {point id: reason}   (reason strings as listed in docs: out_of_bounds, duplicate, outside_grid, plane_unspecified,
                page_unmapped, page_not_trained, cell_not_examined:<unknown|straddling_other_counter|conflict>);
    ``counters`` {cell: [counters]}  (cells whose counter has points on the grid)."""
    trained = sorted(p for p, v in pages_map.items() if v["train"])
    ents, unresolved = {}, 0
    if rec["kind"] == "missing_roi":                 # the file carries no marks at all (accepted only in manifest mode)
        rec = dict(rec, points=[])
    pts = rec["points"]
    _, home_of = _classify_core(rec, [], mode, {})
    if manifest_pages:
        ents, unresolved = resolve_entries_for(rec, manifest_pages, home_of)
    tuples, home_of = _classify_core(rec, trained, mode, ents)
    grid = Grid(**rec["grid"])
    crops = {}
    byid = {p["i"]: p for p in pts}
    for (page, cell), t in tuples.items():
        if t["status"] in ("positive_marked", "verified_zero"):
            x0, y0, x1, y1 = grid.box(cell)
            labs = sorted(((byid[i]["cy"] - y0, byid[i]["cx"] - x0) for i in t["labels"]))
            crops[(page, cell)] = dict(box=[x0, y0, x1, y1], labels=labs, n_positive=len(labs), status=t["status"], n_marks=t["n_marks"],
                                       n_stray=t["n_stray"], role=pages_map[page]["role"])
    label_ids = {i for t in tuples.values() for i in t["labels"] if t["status"] in ("positive_marked", "verified_zero")}
    dropped = {}
    for p in pts:
        if p["i"] in label_ids:
            continue
        if p["status"] == OOB:
            r = "out_of_bounds"
        elif p["status"] == DUP:
            r = "duplicate"
        elif p["geo"] is None:
            r = "outside_grid"
        elif p["page"] is None:
            r = "plane_unspecified" if p["pos"] <= 0 else "plane_out_of_range"
        elif p["page"] not in pages_map:
            r = "page_unmapped"
        elif not pages_map[p["page"]]["train"]:
            r = "page_not_trained"
        else:
            t = tuples[(p["page"], p["geo"])]
            r = "cell_not_examined:" + ("straddling_other_counter" if t["reason"] == "straddle" else t["status"])
        dropped[p["i"]] = r
    counters = defaultdict(list)
    for k, c in home_of.items():
        counters[c].append(k)
    return dict(tuples=tuples, crops=crops, dropped=dropped, counters={c: sorted(v) for c, v in counters.items()}, unresolved=unresolved,
                labels_total=len(label_ids), points_total=len(pts))


def dedupe_groups(files: dict) -> tuple:
    """Byte-identical files: kept path = first in sorted relative-path order (documented). -> (kept: {sha: rel}, alias: {kept: [aliases]})."""
    by = defaultdict(list)
    for rel in files:
        if files[rel]["discovered"]:
            by[files[rel]["sha256"]].append(rel)
    kept, alias = {}, {}
    for sha, rels in by.items():
        rels.sort()
        kept[sha] = rels[0]
        if len(rels) > 1:
            alias[rels[0]] = rels[1:]
    return kept, alias


def expected_status(files: dict, ch: dict, mode: str = "marked-only") -> dict:
    """Per kept file: accepted / rejected + reason class under channel map ``ch``. Aliases are not listed."""
    kept, alias = dedupe_groups(files)
    out = {}
    for sha, rel in kept.items():
        r = files[rel]
        reason = r["reject"]
        if reason == "missing_roi" and mode != "marked-only":
            reason = None                       # without ROI the file is only rejected in marked-only mode (nothing to mark)
        if reason is None and r["kind"] not in ("rgb", "unreadable"):
            _o, pages = resolve_pages(ch, r)
            if any(p >= r["n_pages"] for p in pages):
                reason = "channel_mapping_page_out_of_range"
        out[rel] = dict(status="rejected" if reason else "accepted", reason=reason)
    return out


def make_examined_manifest(files: dict, ch: dict, seed: int, fraction: float = 0.3) -> tuple:
    """examined-fields manifest with verified zeros and DELIBERATE conflicts (+ unresolved / not-counted entries).
    -> (doc, plan_notes)"""
    rng = np.random.default_rng([seed, 99])
    kept, _ = dedupe_groups(files)
    exp = expected_status(files, ch)
    entries = []
    notes = Counter()
    for rel in sorted(kept.values()):
        if exp[rel]["status"] != "accepted" or rng.random() > fraction:
            continue
        rec = files[rel]
        _o, pm = resolve_pages(ch, rec)
        d = derive(rec, pm, "marked-only")
        ncols = len(rec["grid"]["xs"]) - 1
        pages = defaultdict(list)
        used = set()
        for (page, cell), t in sorted(d["tuples"].items()):
            if page != 0 or rng.random() > 0.35:
                continue
            r, c = divmod(cell, ncols)
            if t["status"] == "unknown" and t["reason"] == "no_marks":
                kind = rng.choice(["zero", "zero", "zero", "nocount"])
                if kind == "zero":
                    pages["0"].append({"row": r, "col": c, "count": 0})
                    notes["verified_zero"] += 1
                else:
                    pages["0"].append({"row": r, "col": c})
                    notes["listed_no_count"] += 1
            elif t["status"] == "unknown" and t["reason"] == "straddle":
                pages["0"].append({"row": r, "col": c, "count": 0})
                notes["zero_on_straddle_cell"] += 1
            elif t["status"] == "positive_marked":
                kind = rng.choice(["ok", "ok", "plus1", "zero", "ctr"])
                if kind == "ok":
                    pages["0"].append({"row": r, "col": c, "count": t["n_marks"]})
                elif kind == "plus1":
                    pages["0"].append({"row": r, "col": c, "count": t["n_marks"] + 1})
                    notes["stated_mismatch"] += 1
                elif kind == "zero":
                    pages["0"].append({"row": r, "col": c, "count": 0})
                    notes["zero_with_marks"] += 1
                else:
                    ctr = d["counters"].get(cell)
                    if ctr:
                        pages["0"].append({"counter": ctr[0], "count": t["n_marks"]})
                        notes["by_counter"] += 1
        if rng.random() < 0.1:
            pages["0"].append({"counter": 250, "count": 0})          # counter that does not exist: unresolved, never guessed
            notes["unresolved_counter"] += 1
        if rng.random() < 0.1:
            pages["0"].append({"field": 3, "count": 0})              # lab field number without the acknowledgement: unresolved
            notes["unresolved_field"] += 1
        if pages:
            entries.append({"file": rel, "pages": dict(pages)})
    doc = {"schema": "polony-examined-fields", "version": 1, "evidence": "synthetic rehearsal: generator ground truth",
           "source_kind": "explicit_manifest", "examined_by": "Rehearsal Operator", "recorded_on": "2026-10-13", "files": entries}
    return doc, dict(notes)


def examined_error_variants(out: Path, files: dict, doc: dict) -> dict:
    E = out / "errors"
    V = {}
    clone = lambda d: json.loads(json.dumps(d))          # noqa: E731
    if doc["files"]:
        d = clone(doc)
        d["files"][0]["sha256"] = "0" * 64
        V["ex_sha_mismatch"] = dict(kind="examined", path=str(write_json(E / "examined.sha_mismatch.json", d)), dry_exit=2, real_exit=2,
                                    any_text=["sha256", "different file"])
        d = clone(doc)
        d["files"][0]["pages"].setdefault("0", []).extend([{"row": 0, "col": 0, "count": 1}, {"row": 0, "col": 0, "count": 1}])
        V["ex_duplicate_entry"] = dict(kind="examined", path=str(write_json(E / "examined.duplicate_entry.json", d)), dry_exit=2, real_exit=2,
                                       any_text=["twice", "listed twice"])
        d = clone(doc)
        d["evidence"] = ""
        V["ex_no_evidence"] = dict(kind="examined", path=str(write_json(E / "examined.no_evidence.json", d)), dry_exit=2, real_exit=2,
                                   any_text=["evidence"])
    names = defaultdict(set)
    for rel, r in files.items():
        if r["discovered"] and r["kind"] == "rescan" or r["kind"] == "good":
            names[Path(rel).name].add(rel)
    amb = [n for n, s in names.items() if len(s) > 1 and len({files[x]["sha256"] for x in s}) > 1]
    if amb:
        d = {"schema": "polony-examined-fields", "version": 1, "evidence": "x", "files": [{"file": amb[0], "pages": {"0": [{"row": 0, "col": 0, "count": 0}]}}]}
        V["ex_ambiguous_basename"] = dict(kind="examined", path=str(write_json(E / "examined.ambiguous.json", d)), dry_exit=2, real_exit=2,
                                          any_text=["ambiguous", "base name"])
    return V


# ====================================================================================== extras for the post-lock import
def make_extras(spec: Spec, out_dir: Path, locked_rels: list, src_root: Path, files: dict, n_new: int = 6) -> dict:
    """A SECOND input directory (only new material) for the import after the test lock:
    new specimens; a NEW replicate of a locked specimen; a renamed byte-identical copy of a locked test file (original absent) with
    a wrong identity; a renamed copy of a non-locked file. -> oracle-style ``files`` dict plus ``expect`` per file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([spec.seed, 555])
    res = {}
    base_idx = 100000
    for j in range(n_new):
        it = dict(kind="good", sample=f"XN{j}", assay="T7_AB", rep="A", date=DATES[0], idx=base_idx + j, rel=f"extra/{DATES[0]}_T7_AB_XN{j}-A.tif")
        r = generate_item((asdict(spec), it, str(out_dir)))
        r["expect"] = "new"
        res[it["rel"]] = r
    for j, lr in enumerate(locked_rels[:3]):
        src = files[lr]
        ident = dict(src["identity"])
        it = dict(kind="good", sample=ident["specimen_id"], assay=ident["assay"], rep="Z", date=DATES[1], idx=base_idx + 50 + j,
                  rel=f"extra/{DATES[1]}_{ident['assay']}_{ident['specimen_id']}-Z.tif")
        r = generate_item((asdict(spec), it, str(out_dir)))
        r["expect"] = "locked_specimen"
        res[it["rel"]] = r
    for j, lr in enumerate(locked_rels[3:5]):
        rel = f"extra/renamed_{j}.tif"
        shutil.copyfile(Path(src_root) / lr, out_dir / rel)
        r = dict(files[lr])
        r = dict(r, rel=rel, kind="renamed_locked_copy", identity=dict(specimen_id=f"NEW{j}", slide_id=f"NEW{j}_A", assay="T7_AB", replicate="A",
                                                                         scan_session=DATES[2], confirmed=True), of=lr, expect="locked_bytes")
        r["sha256"] = file_sha256(out_dir / rel)
        res[rel] = r
    for rel, r in res.items():
        r.setdefault("sha256", file_sha256(out_dir / rel))
    return res


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ====================================================================================== top level
def non_tiff_extras(root: Path, sample_file: Path) -> list:
    """Things the importer must ignore. -> list of relative paths created (all NOT discovered)."""
    import zipfile
    made = []

    def w(rel, data=b"x"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        made.append(rel)

    w("notes.txt", b"lab notes\n")
    w("counts/T7B_15-09-2026.csv", b"slide ID,1,2,3\nAM_A,4,5,6\n")
    w("Thumbs.db", b"\x00\x01")
    w(".DS_Store", b"\x00")
    data = Path(sample_file).read_bytes()
    w(".hidden/hidden_scan.tif", data)
    w(".trash/old/deleted.tif", data)
    w("29-09-2026/._resourcefork.tif", data)
    w("29-09-2026/.dotfile.tif", data)
    zp = root / "archive.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("inside.tif", data)
    made.append("archive.zip")
    w("scan_backup.tif.zip", data)
    try:
        os.symlink("29-09-2026", root / "sym_scans")           # symlinked directories are not followed
        made.append("sym_scans")
    except OSError:
        pass
    return made


def generate(spec: Spec, out: Path) -> dict:
    out = Path(out)
    root = out / "input"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    items = plan_files(spec)
    # originals first, then copies (they need the original on disk)
    first = [it for it in items if it["kind"] != "copy"]
    second = [it for it in items if it["kind"] == "copy"]
    args = [(asdict(spec), it, str(root)) for it in first]
    if spec.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(spec.workers) as pool:
            recs = pool.map(generate_item, args, chunksize=8)
    else:
        recs = [generate_item(a) for a in args]
    recs += [generate_item((asdict(spec), it, str(root))) for it in second]
    files = {}
    for r in recs:
        r["sha256"] = file_sha256(root / r["rel"])
        r["size"] = (root / r["rel"]).stat().st_size
        if r["kind"] == "copy":                          # a copy carries the original's ground truth (it may become the kept path)
            r = dict(files[r["of"]], rel=r["rel"], kind="copy", of=r["of"], sha256=r["sha256"], size=r["size"])
        files[r["rel"]] = r
    sample = root / next(r["rel"] for r in recs if r["kind"] == "good")
    ignored = non_tiff_extras(root, sample)
    rng = np.random.default_rng([spec.seed, 3])
    ch = channel_doc(files, rng)
    for r in files.values():
        if r["kind"] not in ("unreadable",):
            r["mapping_origin"], pm = resolve_pages(ch, r)
            r["mapping_pages"] = {str(k): v for k, v in pm.items()}
    rows = identity_rows(files)
    M = out / "mapping"
    write_json(M / "channels.json", ch)
    write_json(M / "identities.json", {"rows": rows})
    exdoc, exnotes = make_examined_manifest(files, ch, spec.seed)
    write_json(M / "examined-fields.json", exdoc)
    variants = error_variants(M, files, ch, rows)
    variants.update(examined_error_variants(M, files, exdoc))
    oracle = dict(generator=dict(version=GENERATOR_VERSION, spec=asdict(spec)), files=files, ignored=ignored, examined_notes=exnotes,
                  error_variants=variants, counts=dict(Counter(r["kind"] for r in files.values())))
    (out / "oracle.json").write_text(json.dumps(oracle), encoding="utf-8")
    return oracle


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500, help="number of visible .tif/.tiff files (including pathologies and copies)")
    ap.add_argument("--cell-px", type=int, default=64, help="grid cell size in px (real scans: 316)")
    ap.add_argument("--real-size", action="store_true", help="2438x1460 images with 316.23 px cells (calibration sample)")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    cell = 316 if a.real_size else a.cell_px
    spec = Spec(n=a.n, seed=a.seed, cell_px=cell, real_size=a.real_size, workers=a.workers)
    o = generate(spec, Path(a.out))
    print(f"wrote {a.out}/input: {sum(o['counts'].values())} TIFF-like files {o['counts']}; oracle.json; mapping/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
