"""Overlay PNG: page-0 (red) / page-1 (green) display + app detections + researcher points.
python -I render_overlay.py <raw.tif> <det.json> <out.png> [x0 y0 size scale]
Detections: circles (cyan=red_only, white=green_only, yellow=yellow, magenta=uncertain; thicker if flagged).
Researcher points: white cross (slice 1 / Clade B), orange cross (slice 2 / Clade A).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GRID_COLS, GRID_ROWS, GRID_STEP, GRID_X0, GRID_Y0, read_points, read_raw  # noqa: E402

p, dj, out = sys.argv[1], sys.argv[2], sys.argv[3]
arr, _ = read_raw(p)
H, W = arr.shape[1:]
if len(sys.argv) > 4:
    x0, y0, sz, k = (int(v) for v in sys.argv[4:8])
    x1, y1 = min(W, x0 + sz), min(H, y0 + sz)
else:
    x0, y0, x1, y1, k = 0, 0, W, H, 1
d = json.loads(Path(dj).read_text())
ch = []
for i in (0, 1):
    a = arr[i, y0:y1, x0:x1].astype(np.float32)
    lo, hi = 300.0, (6000.0 if i == 0 else 3000.0)
    ch.append((np.clip((np.sqrt(np.maximum(a - lo, 0)) / np.sqrt(hi - lo)), 0, 1) * 255).astype(np.uint8))
rgb = np.dstack([ch[0], ch[1], np.zeros_like(ch[0])])
im = Image.fromarray(rgb).resize(((x1 - x0) * k, (y1 - y0) * k), Image.NEAREST)
dr = ImageDraw.Draw(im)
for g in range(GRID_COLS + 1):
    x = (GRID_X0 + g * GRID_STEP - x0) * k
    dr.line([(x, 0), (x, (y1 - y0) * k)], fill=(255, 0, 255), width=1)
for g in range(GRID_ROWS + 1):
    y = (GRID_Y0 + g * GRID_STEP - y0) * k
    dr.line([(0, y), ((x1 - x0) * k, y)], fill=(255, 0, 255), width=1)
col = {"red": (0, 200, 255), "green": (255, 255, 255), "yellow": (255, 255, 0), "uncertain": (255, 0, 255)}
rad = d["radius"] * 2 * k
for o in d["objects"]:
    if not (x0 <= o["x"] < x1 and y0 <= o["y"] < y1):
        continue
    cx, cy = (o["x"] - x0) * k, (o["y"] - y0) * k
    dr.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=col[o["cls"]], width=3 if o["review_reasons"] else 1)
xy, cnt, pos = read_points(p)
for (x, y), P in zip(xy, pos):
    if not (x0 <= x < x1 and y0 <= y < y1):
        continue
    cx, cy = (x - x0) * k, (y - y0) * k
    c = (255, 255, 255) if P == 1 else (255, 128, 0)
    s = 5 * k
    dr.line([(cx - s, cy), (cx + s, cy)], fill=c, width=1)
    dr.line([(cx, cy - s), (cx, cy + s)], fill=c, width=1)
im.save(out)
print("saved", out, im.size)
