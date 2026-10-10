"""Zoomed crops of both raw pages with embedded ROI points. python -I crop_view.py <raw.tif> <x0> <y0> <size> <out.png> [scale]"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_raw, read_points  # noqa: E402
p, x0, y0, sz, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
k = int(sys.argv[6]) if len(sys.argv) > 6 else 2
arr, _ = read_raw(p)
xy, cnt, pos = read_points(p)
tiles = []
for i in (0, 1):
    a = arr[i, y0:y0 + sz, x0:x0 + sz].astype(np.float32)
    lo, hi = np.percentile(a, [1, 99.5])
    g = (np.clip((a - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
    rgb = np.dstack([g, g, g])
    for (x, y), P in zip(xy, pos):
        xi, yi = int(round(x)) - x0, int(round(y)) - y0
        if 0 <= xi < sz and 0 <= yi < sz:
            col = (255, 0, 0) if P == 1 else (0, 255, 0)
            for d in range(4, 8):
                for (u, v) in ((xi + d, yi), (xi - d, yi), (xi, yi + d), (xi, yi - d)):
                    if 0 <= u < sz and 0 <= v < sz:
                        rgb[v, u] = col
    tiles.append(rgb)
    tiles.append(np.full((sz, 4, 3), 255, np.uint8))
im = np.hstack(tiles[:-1])
Image.fromarray(im).resize((im.shape[1] * k, im.shape[0] * k), Image.NEAREST).save(out)
print("page ranges", [(float(np.percentile(arr[i, y0:y0+sz, x0:x0+sz], 1)), float(np.percentile(arr[i, y0:y0+sz, x0:x0+sz], 99.5))) for i in (0, 1)])
