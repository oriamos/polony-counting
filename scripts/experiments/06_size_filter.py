"""Experiment 6: are the unmatched detections compact bright debris ('dust')?
Measure a scale-selection ratio per detection: scale-normalised LoG at sigma=2
vs sigma=5 (s^2 * |LoG|). Debris (sigma ~1-3 px) peaks at small scale;
polonies (sigma ~5 px) at large scale. Then: detection at auto*scale with and
without a 'compact object' filter, scored against the manual marks (TOL px).
Manual marks are a reference, not validated truth."""
import warnings

import numpy as np
from scipy import ndimage as ndi
from importlib import import_module

from _common import OUT, match, overlay, to8

warnings.filterwarnings("ignore")
sw = import_module("03_bigfish_sweep")
a = sw.stack_img[0]
img = sw.bg_correct(a, "gauss", 50)
L = {s: -(s * s) * ndi.gaussian_laplace(img, s) for s in (1.5, 2, 3, 5, 7)}
mask = sw.counted
SIG, MD = 5.0, 5.0
sp, thr_auto, log_m, lmax_m = sw.detect(img, mask, SIG, MD)
print(f"auto thr={thr_auto}")


def ratio_at(yx):
    """max over small scales / max over large scales of scale-normalised LoG,
    evaluated as max in a 5x5 window around the detection."""
    out = []
    for y, x in yx:
        sl = (slice(max(y - 2, 0), y + 3), slice(max(x - 2, 0), x + 3))
        small = max(L[1.5][sl].max(), L[2][sl].max())
        large = max(L[5][sl].max(), L[7][sl].max())
        out.append(small / max(large, 1e-6))
    return np.array(out)


r_manual = ratio_at(np.round(sw.pts[:, ::-1]).astype(int))
print("scale ratio at manual marks: pctl 10/50/90 =", np.round(np.percentile(r_manual, [10, 50, 90]), 2))
for scale in (1.0, 0.5, 0.3):
    sp, thr, _, _ = sw.detect(img, mask, SIG, MD, thr=thr_auto * scale)
    r = ratio_at(sp)
    pairs, ua, ub = match(sp[:, ::-1], sw.pts, sw.TOL)
    m = np.zeros(len(sp), bool); m[[p[0] for p in pairs]] = True
    print(f"thr={thr:.1f} (auto x{scale}): n={len(sp)} tp={len(pairs)} R={len(pairs)/77:.2f} P={len(pairs)/len(sp):.2f}; "
          f"scale-ratio matched median={np.median(r[m]):.2f}, unmatched median={np.median(r[~m]):.2f}")
    for rc in (1.0, 0.8, 0.6):
        keep = r < rc
        p2, _, _ = match(sp[keep][:, ::-1], sw.pts, sw.TOL)
        P = len(p2) / max(keep.sum(), 1); R = len(p2) / 77
        print(f"    + reject compact (ratio>={rc}): n={keep.sum()} tp={len(p2)} R={R:.2f} P={P:.2f} F1={2*P*R/max(P+R,1e-9):.2f}")
    if scale == 0.5:
        keep = r < 0.8
        overlay(to8(a, 300, 6000, 0.5), [(sp[keep][:, ::-1], (255, 0, 0)), (sp[~keep][:, ::-1], (255, 255, 0)),
                                        (sw.pts, (0, 255, 255))], OUT / "06_size_filter.png", radius=10, scale=0.5)
        # zoomed crop of the densest-marked area
        y0, x0 = 1450, 900
        c = lambda s: s[(s[:, 0] >= y0) & (s[:, 0] < y0 + 400) & (s[:, 1] >= x0) & (s[:, 1] < x0 + 400)] - [y0, x0]
        pm = sw.pts[(sw.pts[:, 1] >= y0) & (sw.pts[:, 1] < y0 + 400) & (sw.pts[:, 0] >= x0) & (sw.pts[:, 0] < x0 + 400)] - [x0, y0]
        overlay(to8(a[y0:y0 + 400, x0:x0 + 400], 300, 6000, 0.5),
                [(c(sp[keep])[:, ::-1], (255, 0, 0)), (c(sp[~keep])[:, ::-1], (255, 255, 0)), (pm, (0, 255, 255))],
                OUT / "06_size_filter_crop.png", radius=12, scale=2)

# nearest-neighbour spacing of kept detections (for the matching tolerance)
from scipy.spatial import cKDTree
sp, thr, _, _ = sw.detect(img, mask, SIG, MD, thr=thr_auto * 0.3)
keep = ratio_at(sp) < 1.0
k = sp[keep]
d, _ = cKDTree(k).query(k, k=2)
print("NN distance between kept detections px: min/p5/p25/median =", np.round([d[:, 1].min(), *np.percentile(d[:, 1], [5, 25, 50])], 1))
dm, _ = cKDTree(sw.pts).query(sw.pts, k=2)
print("NN distance between manual marks px: min/p5/p25/median =", np.round([dm[:, 1].min(), *np.percentile(dm[:, 1], [5, 25, 50])], 1))
