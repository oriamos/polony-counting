"""Shared helpers for the science experiments (not part of the app).

Run with the project venv, e.g.
    python scripts/experiments/01_inspect_image.py
Outputs go to $EXP_OUT (default: ./exp_out).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import roifile
import tifffile
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
IMG = REPO / "resources/dataset_files/test/test_img.tif"
OUT = Path(os.environ.get("EXP_OUT", REPO / "exp_out"))
OUT.mkdir(parents=True, exist_ok=True)


def load():
    """Return (stack uint16 (2,H,W), imagej_metadata dict)."""
    with tifffile.TiffFile(IMG) as t:
        return t.asarray(), t.imagej_metadata


def manual_points(md):
    """The embedded ImageJ 'ROI' is a multi-point ROI (originally 77 points on the real scan; the
    synthetic test_img.tif carries its own marks, see test_img.truth.json) (counters
    0..16), NOT a rectangle. Returns (xy float array (N,2), counters)."""
    r = roifile.ImagejRoi.frombytes(md["ROI"])
    return np.asarray(r.subpixel_coordinates, float), np.asarray(r.counters)


def grid_lines(md):
    """The embedded 'Overlays' is a shape ROI made of straight lines (a grid).
    Returns sorted unique x positions and y positions of the grid lines."""
    o = md["Overlays"]
    o = o if isinstance(o, list) else [o]
    xs, ys = set(), set()
    for b in o:
        r = roifile.ImagejRoi.frombytes(b)
        mc = np.asarray(r.multi_coordinates, float)
        # segments of 6 floats: [op, x0, y0, op, x1, y1] (moveTo/lineTo)
        for k in range(0, len(mc) - 5, 6):
            x0, y0, x1, y1 = mc[k + 1], mc[k + 2], mc[k + 4], mc[k + 5]
            if abs(x0 - x1) < 1e-3:
                xs.add(round(x0, 2))
            elif abs(y0 - y1) < 1e-3:
                ys.add(round(y0, 2))
    return sorted(xs), sorted(ys)


def to8(a, lo=None, hi=None, gamma=1.0):
    a = a.astype(np.float32)
    if lo is None:
        lo, hi = np.percentile(a, [0.5, 99.5])
    x = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) ** gamma
    return (x * 255).astype(np.uint8)


def overlay(gray8, pts_list, path, radius=10, scale=1.0, lines=None):
    """pts_list = [(xy array (N,2) in x,y order, rgb), ...]."""
    im = Image.fromarray(gray8).convert("RGB")
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    if lines:
        xs, ys = lines
        for x in xs:
            d.line([(x * scale, 0), (x * scale, im.height)], fill=(255, 0, 255), width=1)
        for y in ys:
            d.line([(0, y * scale), (im.width, y * scale)], fill=(255, 0, 255), width=1)
    for xy, col in pts_list:
        for x, y in np.asarray(xy).reshape(-1, 2):
            r = radius * scale
            d.ellipse([x * scale - r, y * scale - r, x * scale + r, y * scale + r], outline=col, width=2)
    im.save(path)
    return path


def match(a, b, tol):
    """Greedy-free optimal 1:1 matching (Hungarian) of point sets a,b (N,2)
    within tol. Returns (pairs list[(i,j,d)], unmatched_a, unmatched_b)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    a = np.asarray(a, float).reshape(-1, 2)
    b = np.asarray(b, float).reshape(-1, 2)
    if len(a) == 0 or len(b) == 0:
        return [], list(range(len(a))), list(range(len(b)))
    D = cdist(a, b)
    big = 1e6
    C = np.where(D <= tol, D, big)
    ri, ci = linear_sum_assignment(C)
    pairs = [(i, j, D[i, j]) for i, j in zip(ri, ci) if D[i, j] <= tol]
    ma = {p[0] for p in pairs}
    mb = {p[1] for p in pairs}
    return pairs, [i for i in range(len(a)) if i not in ma], [j for j in range(len(b)) if j not in mb]
