"""Experiment 1: inspect test_img.tif - intensity stats, saturation, embedded
ROI/overlay geometry, overview renders with manual points + grid."""
import numpy as np

from _common import OUT, grid_lines, load, manual_points, overlay, to8

stack, md = load()
print("shape", stack.shape, stack.dtype)
for i, a in enumerate(stack):
    sat = a == 65535
    print(f"page {i}: min={a.min()} max={a.max()} p1/50/90/99={np.percentile(a,[1,50,90,99])} "
          f"saturated_frac={sat.mean():.4f}")
pts, ctr = manual_points(md)
xs, ys = grid_lines(md)
print("manual points", len(pts), "counters", np.unique(ctr), "per counter", np.bincount(ctr))
print("grid x", xs, "\ngrid y", ys, "\nspacing x", np.diff(xs), "spacing y", np.diff(ys))
# which grid cell (col,row) each point is in
ci = np.searchsorted(xs, pts[:, 0]) - 1
ri = np.searchsorted(ys, pts[:, 1]) - 1
for c in np.unique(ctr):
    sel = ctr == c
    cells = set(zip(ci[sel].tolist(), ri[sel].tolist()))
    print(f"counter {c}: n={sel.sum()} cells(col,row)={sorted(cells)}")
for i, a in enumerate(stack):
    lo, hi = np.percentile(a, [1, 99])
    overlay(to8(a, lo, min(hi, 8000), 0.5), [(pts, (0, 255, 255))],
            OUT / f"01_overview_p{i}.png", radius=12, scale=0.5, lines=(xs, ys))
    # saturation map
    overlay((a == 65535).astype(np.uint8) * 255, [(pts, (255, 0, 0))],
            OUT / f"01_saturation_p{i}.png", radius=12, scale=0.5, lines=(xs, ys))
print("wrote", OUT)
