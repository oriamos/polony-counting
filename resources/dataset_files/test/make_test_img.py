"""Generate the synthetic example slide ``test_img.tif`` and its ground truth ``test_img.truth.json``.

The repository ships one example image that tests, the e2e flow and the science scripts read. It is
**synthetic**: every byte of ``test_img.tif`` is written by this script (seeded, deterministic), and no
laboratory acquisition is involved. It mimics the *structure* of the lab's two-page scanner stacks as
described in ``docs/app/DATA_FORMATS.md``:

* ImageJ hyperstack, 2 pages (ZYX, ``slices=2``), ``uint16``, 2450 x 1438 (H x W), resolution tag
  1000 px/cm (informational only), deflate-compressed with a horizontal predictor (readable by
  ``tifffile`` with or without ``imagecodecs``).
* Page 0 = the red marker, with an ``Info`` text carrying the scanner-style label hint
  ``ImageDescription: 635 [W1]``; page 1 = green, no hint.
* A stadium-shaped chamber with a dark wall band, a saturated (clipped) outside, an uneven inner
  background with a darker crescent and Poisson-like noise.
* Gaussian polony spots (sigma around 5 px): red-only, green-only, yellow pairs (red + green with a
  small registration offset), a few saturated spots, some close to the chamber wall, and small bright
  debris specks (some in both channels).
* ``Overlays``: one composite RECT shape of axis-aligned lines, the counting grid at
  ``158 + k * sqrt(1e5)`` px (316.23 px = 0.316 cm at 1000 px/cm), 4 x 7 cells.
* ``ROI``: one multi-point ROI (sub-pixel) of "researcher marks": clicks on a subset of the red-bearing
  spots inside the counted grid cells, placed on plane 1 (``counter_positions`` = 1 = page 0), with the
  ImageJ counter = field number (counter *i* = field *i+1*, cells visited in serpentine order). Clicks
  have a few px of placement error, faint spots are skipped, and one mark is stored out of order, as in
  real annotations.

Only generic metadata is written (no instrument identifiers, no dates, no names).

The ground truth (spots, marks, grid, counted cells, ROI bounding box and a hash of the decoded pixels)
is written to ``test_img.truth.json``; tests read it instead of hard-coding numbers. Coordinates in the
truth are *canonical* (pixel *i* is centred at *i*); ``x_imagej``/``y_imagej`` of a mark are the raw
ImageJ values (canonical + 0.5).

Usage (from the repository root)::

    .venv/bin/python resources/dataset_files/test/make_test_img.py            # rewrite both files
    .venv/bin/python resources/dataset_files/test/make_test_img.py --check    # verify the committed files
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import tifffile
from roifile import ROI_TYPE, ImagejRoi

SEED = 20261015
H, W = 2450, 1438
HERE = Path(__file__).resolve().parent
TIF_NAME = "test_img.tif"
TRUTH_NAME = "test_img.truth.json"

INFO = ("ImageDescription: 635 [W1]\n"
        "Software: polony-counting make_test_img.py (synthetic example, not an acquisition)\n")

# ---- chamber: a stadium (vertical capsule) -------------------------------------------------------
CX = 719.0                      # centre column
R_IN = 668.0                    # inner radius (chamber interior)
WALL = 34.0                     # dark wall band width
Y_TOP, Y_BOT = 14.0 + R_IN + WALL, H - 14.0 - R_IN - WALL   # ends of the straight section

# ---- grid (ImageJ edge frame, as the lab's Overlays) ---------------------------------------------
GRID_STEP = math.sqrt(1e5)      # 316.2278 px
GRID_X0 = GRID_Y0 = 158.0
N_COLS, N_ROWS = 4, 7
MIN_INSIDE_FRACTION = 0.85    # a cell is eligible if at least this much of it is chamber interior
N_FIELDS_COUNTED = 17         # the "researcher" counts fields 1..N of the eligible cells (fields beyond N are left blank)

# ---- intensities -----------------------------------------------------------------------------------
RED_BG, GREEN_BG = 1500.0, 330.0
RED_GAIN, GREEN_GAIN = 0.42, 0.30   # noise variance = gain * signal  (std ~25 / ~10 at background)
BLEED_RED_TO_GREEN = 0.0004
CLIP = 65535


def chamber_distance(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Distance from the stadium's medial segment (inside the chamber interior where < R_IN)."""
    cy = np.clip(yy, Y_TOP, Y_BOT)
    return np.hypot(xx - CX, yy - cy)


def grid_lines() -> tuple[list[float], list[float]]:
    xs = [float(np.float32(GRID_X0 + k * GRID_STEP)) for k in range(N_COLS + 1)]
    ys = [float(np.float32(GRID_Y0 + k * GRID_STEP)) for k in range(N_ROWS + 1)]
    return xs, ys


def counted_cells(xs, ys) -> list[dict]:
    """The first N_FIELDS_COUNTED eligible cells in serpentine order (row 0 left->right, row 1 back, ...).

    Eligible = at least MIN_INSIDE_FRACTION of the cell is chamber interior (cells cut by the rounded
    ends or the wall are skipped), as a researcher counting fields 1..N would do.
    """
    out = []
    for r in range(N_ROWS):
        cols = range(N_COLS) if r % 2 == 0 else range(N_COLS - 1, -1, -1)
        for c in cols:
            gx, gy = np.meshgrid(np.linspace(xs[c], xs[c + 1], 41), np.linspace(ys[r], ys[r + 1], 41))
            if float(np.mean(chamber_distance(gx, gy) < R_IN)) < MIN_INSIDE_FRACTION:
                continue
            if len(out) < N_FIELDS_COUNTED:
                out.append({"counter": len(out), "field": len(out) + 1, "row": r, "col": c,
                            "x0": xs[c], "x1": xs[c + 1], "y0": ys[r], "y1": ys[r + 1]})
    return out


def overlay_bytes(xs, ys) -> bytes:
    segs: list[float] = []
    for x in xs:
        segs += [0, x, 0, 1, x, H]
    for y in ys:
        segs += [0, 0, y, 1, W, y]
    roi = ImagejRoi(roitype=ROI_TYPE.RECT, version=228, left=0, top=0, right=W, bottom=H,
                    stroke_color=b"\xff\x00\xff\xff")
    roi.multi_coordinates = np.array(segs, dtype=">f4")
    roi.shape_roi_size = len(segs)
    return roi.tobytes()


def point_roi(xy_imagej: np.ndarray, counters, positions) -> ImagejRoi:
    roi = ImagejRoi.frompoints(np.asarray(xy_imagej, dtype=np.float32))
    roi.roitype = ROI_TYPE.POINT
    roi.version = 228
    roi.counters = np.asarray(counters, dtype=np.uint8)
    roi.counter_positions = np.asarray(positions, dtype=np.uint32)
    return roi


# ---- spot placement ------------------------------------------------------------------------------
def place_spots(rng) -> list[dict]:
    spots: list[dict] = []

    def free(x, y, d):
        return all(math.hypot(x - s["x"], y - s["y"]) >= d for s in spots)

    def sample_inside(lo_d=0.0, hi_d=R_IN - 12.0):
        while True:
            x, y = rng.uniform(40, W - 40), rng.uniform(40, H - 40)
            d = float(chamber_distance(np.array(x), np.array(y)))
            if lo_d <= d < hi_d:
                return x, y

    def add(kind, n, sigma=(5.3, 0.6), red=None, green=None, d=(0.0, R_IN - 12.0), sat=False,
            min_sep=18.0):
        k = 0
        while k < n:
            x, y = sample_inside(*d)
            if not free(x, y, min_sep):
                continue
            s = float(np.clip(rng.normal(*sigma), 1.0, 8.0))
            # log-uniform amplitudes: many modest spots, a tail of bright ones
            ar = float(np.exp(rng.uniform(*np.log(red)))) if red else 0.0
            ag = float(np.exp(rng.uniform(*np.log(green)))) if green else 0.0
            dx, dy = (rng.normal(0, 0.8, 2) if (ar and ag) else (0.0, 0.0))
            spots.append({"id": len(spots), "kind": kind, "x": round(x, 2), "y": round(y, 2),
                          "sigma": round(s, 2), "amp_red": round(ar, 1), "amp_green": round(ag, 1),
                          "green_dx": round(float(dx), 2), "green_dy": round(float(dy), 2),
                          "saturated": sat, "near_rim": bool(d[0] > 0)})
            k += 1

    add("red", 150, red=(250.0, 20000.0))                              # red-only polonies
    add("yellow", 28, red=(400.0, 15000.0), green=(150.0, 5000.0))     # both markers
    add("green", 32, green=(120.0, 5000.0))                            # green-only polonies
    add("red", 4, red=(90000.0, 160000.0), sat=True)                   # clipped bright red polonies
    add("yellow", 2, red=(90000.0, 140000.0), green=(80000.0, 120000.0), sat=True)
    add("red", 8, red=(800.0, 8000.0), d=(R_IN - 40.0, R_IN - 10.0))  # close to the chamber wall
    add("green", 3, green=(300.0, 3000.0), d=(R_IN - 40.0, R_IN - 10.0))
    add("debris", 10, sigma=(1.5, 0.2), red=(3000.0, 20000.0), min_sep=12.0)
    add("debris", 8, sigma=(1.5, 0.2), red=(3000.0, 20000.0), green=(1000.0, 8000.0), min_sep=12.0)
    for s in spots:            # a dual-channel debris speck is aligned (dust shines in both channels)
        if s["kind"] == "debris":
            s["green_dx"] = s["green_dy"] = 0.0
    return spots


def ensure_marks_possible(rng, spots, cells) -> None:
    """Every counted cell holds at least two clearly visible red spots well inside it."""
    for c in cells:
        inside = [s for s in spots if _markable(s, c)]
        while len(inside) < 2:
            x = rng.uniform(c["x0"] + 30, c["x1"] - 30)
            y = rng.uniform(c["y0"] + 30, c["y1"] - 30)
            if all(math.hypot(x - s["x"], y - s["y"]) >= 18.0 for s in spots):
                s = {"id": len(spots), "kind": "red", "x": round(x, 2), "y": round(y, 2),
                     "sigma": round(float(np.clip(rng.normal(5.3, 0.6), 3.5, 8.0)), 2),
                     "amp_red": round(float(rng.uniform(2000, 9000)), 1), "amp_green": 0.0,
                     "green_dx": 0.0, "green_dy": 0.0, "saturated": False, "near_rim": False}
                spots.append(s)
                inside.append(s)


VISIBLE_RED = 600.0      # the "researcher" skips red spots fainter than this (~24 x background noise)
MARK_MISS = 0.12         # ... and misses this fraction of the visible ones
CELL_INNER = 12.0        # ... and only marks spots this far inside the cell (no border-crossing clicks)


def _markable(s, c) -> bool:
    return (s["kind"] in ("red", "yellow") and s["amp_red"] >= VISIBLE_RED
            and c["x0"] + CELL_INNER <= s["x"] + 0.5 <= c["x1"] - CELL_INNER
            and c["y0"] + CELL_INNER <= s["y"] + 0.5 <= c["y1"] - CELL_INNER)


def make_marks(rng, spots, cells) -> list[dict]:
    marks = []
    for c in cells:
        cand = [s for s in spots if _markable(s, c)]
        keep = [s for s in cand if rng.random() >= MARK_MISS] or cand[:1]
        # researcher's order inside a cell: roughly top to bottom
        for s in sorted(keep, key=lambda s: (s["y"], s["x"])):
            err = np.clip(rng.normal(0.0, 2.5, 2), -6.0, 6.0)
            # a click lands on a pixel; ImageJ stores the pixel centre in the edge frame (i + 0.5)
            px, py = int(round(s["x"] + err[0])), int(round(s["y"] + err[1]))
            marks.append({"x_imagej": px + 0.5, "y_imagej": py + 0.5, "counter": c["counter"],
                          "position": 1, "spot_id": s["id"]})
            s["marked"] = True
    # one late click: a point of an earlier field added after the last field (stored out of order)
    k = next(i for i in range(len(marks) - 1, -1, -1) if marks[i]["counter"] == cells[-2]["counter"])
    marks.append(marks.pop(k))
    return marks


# ---- rendering -----------------------------------------------------------------------------------
def render(rng, spots) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    d = chamber_distance(xx, yy)
    inside = d < R_IN
    wall = (d >= R_IN) & (d < R_IN + WALL)

    # slow illumination gradient + a darker crescent (upper right) inside the chamber
    illum = 1.0 + 0.12 * (yy / H) - 0.08 * (xx / W) + 0.05 * np.sin(yy / 310.0 + 0.7)
    cres = np.exp(-(((xx - 1000) / 300.0) ** 2 + ((yy - 760) / 360.0) ** 2) ** 2)
    illum *= 1.0 - 0.38 * cres
    red = np.where(inside, RED_BG * illum, 0.0).astype(np.float32)
    green = np.where(inside, GREEN_BG * illum, 0.0).astype(np.float32)
    # dark wall with a brighter inner lip
    lip = np.exp(-((d - R_IN - 3.0) / 3.0) ** 2)
    red = np.where(wall, 520.0 + 2600.0 * lip, red)
    green = np.where(wall, 140.0 + 700.0 * lip, green)
    # outside the chamber: the scanner saturates
    outside = d >= R_IN + WALL
    red = np.where(outside, 64000.0, red)
    green = np.where(outside, 64500.0, green)

    sig_r = np.zeros_like(red)
    sig_g = np.zeros_like(green)
    for s in spots:
        r = int(math.ceil(4.5 * s["sigma"])) + 3
        x0, x1 = max(int(s["x"]) - r, 0), min(int(s["x"]) + r + 1, W)
        y0, y1 = max(int(s["y"]) - r, 0), min(int(s["y"]) + r + 1, H)
        lx, ly = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
        two_s2 = 2.0 * s["sigma"] ** 2
        if s["amp_red"]:
            sig_r[y0:y1, x0:x1] += s["amp_red"] * np.exp(-((lx - s["x"]) ** 2 + (ly - s["y"]) ** 2) / two_s2)
        if s["amp_green"]:
            gx, gy = s["x"] + s["green_dx"], s["y"] + s["green_dy"]
            sig_g[y0:y1, x0:x1] += s["amp_green"] * np.exp(-((lx - gx) ** 2 + (ly - gy) ** 2) / two_s2)
    red += sig_r
    green += sig_g + BLEED_RED_TO_GREEN * sig_r

    # Poisson-like noise; outside the chamber the clipped plateau carries sparse dark specks only
    red += rng.standard_normal(red.shape, dtype=np.float32) * np.sqrt(RED_GAIN * red)
    green += rng.standard_normal(green.shape, dtype=np.float32) * np.sqrt(GREEN_GAIN * green)
    for ch in (red, green):
        ch[outside] = CLIP
    specks = outside & (rng.random((H, W), dtype=np.float32) < 0.004)
    red[specks] = rng.uniform(20000, 60000, int(specks.sum()))
    green[specks] = rng.uniform(20000, 60000, int(specks.sum()))
    return np.stack([np.clip(np.rint(c), 0, CLIP).astype(np.uint16) for c in (red, green)])


def pixels_sha256(stack: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(stack, dtype="<u2").tobytes()).hexdigest()


def build(seed: int = SEED):
    rng = np.random.default_rng(seed)
    xs, ys = grid_lines()
    cells = counted_cells(xs, ys)
    spots = place_spots(rng)
    ensure_marks_possible(rng, spots, cells)
    for s in spots:
        s["marked"] = False
    marks = make_marks(rng, spots, cells)
    stack = render(rng, spots)
    xy = np.array([[m["x_imagej"], m["y_imagej"]] for m in marks])
    roi = point_roi(xy, [m["counter"] for m in marks], [m["position"] for m in marks])
    meta = {"axes": "ZYX", "unit": "cm", "Info": INFO, "ROI": roi.tobytes(), "Overlays": overlay_bytes(xs, ys)}
    for s in spots:
        s["saturated"] = bool(stack[0][int(round(s["y"])), int(round(s["x"]))] == CLIP
                              or stack[1][int(round(s["y"])), int(round(s["x"]))] == CLIP)
    truth = {
        "generator": "resources/dataset_files/test/make_test_img.py",
        "seed": seed,
        "synthetic": True,
        "shape": [2, H, W], "dtype": "uint16",
        "pages": {"0": {"role": "red", "label_hint": "635 [W1]"}, "1": {"role": "green", "label_hint": None}},
        "pixels_sha256": pixels_sha256(stack),
        "resolution_px_per_cm": 1000,
        "background": {"red": RED_BG, "green": GREEN_BG, "noise_gain_red": RED_GAIN, "noise_gain_green": GREEN_GAIN},
        "chamber": {"kind": "stadium", "cx": CX, "y_top": Y_TOP, "y_bottom": Y_BOT, "r_inner": R_IN, "wall": WALL},
        "grid": {"xs": xs, "ys": ys, "spacing": GRID_STEP, "n_cols": N_COLS, "n_rows": N_ROWS},
        "counted_cells": cells,
        "roi_bbox": {"left": int(roi.left), "top": int(roi.top), "right": int(roi.right), "bottom": int(roi.bottom)},
        "marks": marks,
        "spots": spots,
        "notes": "Coordinates are canonical (pixel i centred at i) except x_imagej/y_imagej (= canonical + 0.5). "
                 "kind: red / green / yellow polonies, debris = small bright specks (sigma ~1.5). "
                 "Marks: counter = field - 1, position 1 = page 0.",
    }
    return stack, meta, truth


def write(out_dir: Path, seed: int = SEED) -> tuple[Path, Path]:
    stack, meta, truth = build(seed)
    tif = out_dir / TIF_NAME
    tifffile.imwrite(str(tif), stack, imagej=True, metadata=meta, resolution=(1000.0, 1000.0),
                     resolutionunit="CENTIMETER",
                     compression="zlib", compressionargs={"level": 9}, predictor=True)
    js = out_dir / TRUTH_NAME
    js.write_text(json.dumps(truth, indent=1) + "\n")
    return tif, js


def check(out_dir: Path, seed: int = SEED) -> list[str]:
    """Return problems: the committed files must be exactly what this script generates."""
    stack, _, truth = build(seed)
    problems = []
    tif, js = out_dir / TIF_NAME, out_dir / TRUTH_NAME
    if json.loads(js.read_text()) != json.loads(json.dumps(truth)):
        problems.append(f"{js.name} differs from the generator output")
    if pixels_sha256(tifffile.imread(str(tif))) != truth["pixels_sha256"]:
        problems.append(f"{tif.name} pixels differ from the generator output")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=HERE, help="output directory (default: next to this script)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--check", action="store_true", help="verify the files in --out instead of writing")
    a = ap.parse_args(argv)
    if a.check:
        problems = check(a.out, a.seed)
        print("\n".join(problems) or "OK: files match the generator")
        return 1 if problems else 0
    tif, js = write(a.out, a.seed)
    print(f"wrote {tif} ({tif.stat().st_size / 1e6:.2f} MB) and {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
