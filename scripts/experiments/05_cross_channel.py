"""Experiment 5: cross-channel facts on test_img.tif (page 0 vs page 1).
 a) global shift between pages by phase correlation (whole image; chamber rim
    dominates) and on the chamber interior only;
 b) bleed-through estimate: at manually marked page-0 polonies, ratio of
    (centre - annulus) signal in page 1 vs page 0;
 c) page-1 detections and nearest page-0 detections (distance distribution),
    i.e. what a matching tolerance would see on this image.
"""
import warnings

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation
from importlib import import_module

from _common import OUT, overlay, to8

warnings.filterwarnings("ignore")
sw = import_module("03_bigfish_sweep")
a0, a1 = sw.stack_img[0].astype(np.float32), sw.stack_img[1].astype(np.float32)
pts = sw.pts

# a) shift (skimage convention: shift to apply to moving (page1) to register on reference (page0))
for name, m in (("whole image", None),):
    sh, err, _ = phase_cross_correlation(a0, a1, upsample_factor=10)
    print(f"phase correlation {name}: shift(dy,dx) to apply to page1 = {sh}, error={err:.3f}")
# gradient-magnitude images are less dominated by absolute intensity
g0 = ndi.gaussian_gradient_magnitude(np.log1p(a0), 2); g1 = ndi.gaussian_gradient_magnitude(np.log1p(a1), 2)
sh, err, _ = phase_cross_correlation(g0, g1, upsample_factor=10)
print(f"phase correlation on log-gradient: shift(dy,dx)={sh}, error={err:.3f}")
# interior-only: crop the central part of the chamber (no rim) - polony/dust content only
y0, y1, x0, x1 = 500, 1900, 300, 1150
sh, err, _ = phase_cross_correlation(g0[y0:y1, x0:x1], g1[y0:y1, x0:x1], upsample_factor=10)
print(f"phase correlation interior crop (log-gradient): shift(dy,dx)={sh}, error={err:.3f}")

# b) bleed-through at red polonies
yy, xx = np.mgrid[-20:21, -20:21]; rr = np.hypot(yy, xx)
cen, ann = rr <= 4, (rr >= 14) & (rr <= 20)
r0, r1 = [], []
for x, y in pts:
    xi, yi = int(round(x)), int(round(y))
    # refine on page 0
    w = ndi.gaussian_filter(a0[yi - 12:yi + 13, xi - 12:xi + 13], 2)
    dy, dx = np.unravel_index(np.argmax(w), w.shape); yi, xi = yi - 12 + dy, xi - 12 + dx
    p0 = a0[yi - 20:yi + 21, xi - 20:xi + 21]; p1 = a1[yi - 20:yi + 21, xi - 20:xi + 21]
    r0.append(p0[cen].mean() - np.median(p0[ann])); r1.append(p1[cen].mean() - np.median(p1[ann]))
r0, r1 = np.array(r0), np.array(r1)
ratio = r1 / np.maximum(r0, 1)
print(f"page0 net signal at marks: median={np.median(r0):.0f} (p10 {np.percentile(r0,10):.0f}, p90 {np.percentile(r0,90):.0f})")
print(f"page1 net signal at same: median={np.median(r1):.1f} (p10 {np.percentile(r1,10):.1f}, p90 {np.percentile(r1,90):.1f}); "
      f"page1 annulus noise std ~{np.median([a1[int(y)-20:int(y)+21, int(x)-20:int(x)+21][ann].std() for x, y in pts]):.1f}")
print(f"ratio page1/page0: median={np.median(ratio):.4f} p90={np.percentile(ratio,90):.4f}; "
      f"slope (least squares through 0)={np.sum(r0*r1)/np.sum(r0*r0):.4f}")

# c) detections in each page inside counted cells & chamber
mask = sw.counted
img0 = sw.bg_correct(sw.stack_img[0], "gauss", 50); img1 = sw.bg_correct(sw.stack_img[1], "gauss", 50)
s0, t0, _, _ = sw.detect(img0, mask, 5, 5)
s1, t1, _, _ = sw.detect(img1, mask, 5, 5)
print(f"page0 auto thr={t0} n={0 if s0 is None else len(s0)}; page1 auto thr={t1} n={0 if s1 is None else len(s1)}")
if s0 is not None and s1 is not None and len(s1):
    d, _ = cKDTree(s0).query(s1)
    print("page1 -> nearest page0 detection distance px: pctl10/50/90 =", np.round(np.percentile(d, [10, 50, 90]), 1),
          " n<=5:", int((d <= 5).sum()), " n<=10:", int((d <= 10).sum()), " n<=15:", int((d <= 15).sum()))
    overlay(to8(sw.stack_img[1], 0, 1500, 0.5), [(s1[:, ::-1], (0, 255, 0)), (s0[:, ::-1], (255, 0, 0))],
            OUT / "05_page1_detections.png", radius=9, scale=0.5)

# d) what are the co-localised (would-be 'yellow') pairs? compare with manual marks + size
if s0 is not None and s1 is not None and len(s1):
    d, j = cKDTree(s0).query(s1)
    co = d <= 5
    dm, _ = cKDTree(pts[:, ::-1]).query(s1[co])
    print(f"co-localised page1/page0 pairs (<=5px): {int(co.sum())}; of these near a manual mark (<=15px): {int((dm<=15).sum())}")
    # apparent size in page 0: second-moment sigma of background-subtracted 21x21 patch
    for (y, x) in s1[co][:25]:
        p = img0[y - 10:y + 11, x - 10:x + 11]; q = img1[y - 10:y + 11, x - 10:x + 11]
        yy2, xx2 = np.mgrid[-10:11, -10:11]
        w = np.clip(p - np.median(p), 0, None); s = np.sqrt((w * (yy2**2 + xx2**2)).sum() / max(w.sum(), 1) / 2)
        print(f"  pair at x={x},y={y}: page0 peak={p.max():.0f} page1 peak={q.max():.0f} page0 moment-sigma={s:.1f}px")
