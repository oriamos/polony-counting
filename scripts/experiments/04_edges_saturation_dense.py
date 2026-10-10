"""Experiment 4: failure modes on the real image.
 a) unmasked whole-image detection: where do detections go (outside chamber,
    saturated area, rim) vs. inside the counted grid cells;
 b) saturation plateaus: LoG response on 65535 plateaus;
 c) decompose_dense on top of the best plain detection: how many spots are added
    and where (are bright single polonies split?).
Overlays written to $EXP_OUT."""
import warnings

import numpy as np
import bigfish.detection as det
from scipy import ndimage as ndi

from _common import OUT, load, manual_points, match, overlay, to8
from importlib import import_module

sw = import_module("03_bigfish_sweep")  # reuses masks/helpers (runs no sweep on import)
warnings.filterwarnings("ignore")

SIGMA, MD, BG = float(__import__("os").environ.get("SIGMA", 5)), float(__import__("os").environ.get("MD", 5)), 50
a = sw.stack_img[0]
pts = sw.pts
sat = a == 65535
# chamber interior: large dark connected component of a heavily smoothed, non-saturated image
sm = ndi.gaussian_filter((~sat).astype(np.float32), 8) > 0.99
lab, n = ndi.label(sm)
sizes = ndi.sum(sm, lab, range(1, n + 1))
chamber = lab == (1 + int(np.argmax(sizes)))
chamber = ndi.binary_fill_holes(chamber)
rim_band = chamber & ~ndi.binary_erosion(chamber, iterations=60)
print(f"saturated frac={sat.mean():.3f}; chamber frac={chamber.mean():.3f}; inside-chamber saturated px={int((sat & chamber).sum())}")

for bgname, img in (("gauss50", sw.bg_correct(a, "gauss", BG)),
                    ("gauss50_masked", sw.bg_correct(a, "gauss_masked", BG, ~sat)),
                    ("none", a.astype(np.float32))):
    full = np.ones_like(sat)
    sp, thr, _, _ = sw.detect(img, full, SIGMA, MD)
    if sp is None:
        print(bgname, "auto threshold failed on full image"); continue
    yx = sp
    in_ch = chamber[yx[:, 0], yx[:, 1]]
    in_rim = rim_band[yx[:, 0], yx[:, 1]]
    in_cnt = sw.counted[yx[:, 0], yx[:, 1]]
    on_sat = ndi.binary_dilation(sat, iterations=3)[yx[:, 0], yx[:, 1]]
    print(f"[{bgname}] UNMASKED whole image: auto thr={thr:.1f} n={len(yx)} | outside chamber={int((~in_ch).sum())} "
          f"rim band(60px)={int(in_rim.sum())} within 3px of saturated={int(on_sat.sum())} in counted cells={int(in_cnt.sum())}")
    sp2, thr2, _, _ = sw.detect(img, sw.counted & chamber, SIGMA, MD)
    print(f"[{bgname}] MASKED (counted cells & chamber): auto thr={thr2} n={0 if sp2 is None else len(sp2)}")
    if bgname == "gauss50":
        overlay(to8(a, 300, 6000, 0.5), [(yx[:, ::-1], (255, 0, 0)), (pts, (0, 255, 255))],
                OUT / "04_unmasked_gauss50.png", radius=9, scale=0.5)

# b) saturation plateau LoG behaviour
from bigfish import stack as bstack
log = bstack.log_filter(a.astype(np.float32), SIGMA)
lmax = det.local_maximum_detection(log, MD)
sat_core = ndi.binary_erosion(sat, iterations=int(3 * SIGMA))
sat_edge = ndi.binary_dilation(sat, iterations=int(2 * SIGMA)) & ~ndi.binary_erosion(sat, iterations=int(2 * SIGMA))
print(f"raw LoG: median response in saturated core={np.median(log[sat_core]):.1f}, "
      f"p99 at saturation edges={np.percentile(log[sat_edge],99):.1f}, p99 inside chamber={np.percentile(log[chamber & ~sat_edge],99):.1f}; "
      f"local maxima >0 in sat edge band={int((lmax & sat_edge & (log>0)).sum())}")

# c) dense decomposition
img = sw.bg_correct(a, "gauss", BG)
mask = sw.counted & chamber
sp, thr, _, _ = sw.detect(img, mask, SIGMA, MD)
pairs, ua, ub = match(sp[:, ::-1], pts, sw.TOL)
print(f"plain: n={len(sp)} tp={len(pairs)}")
# big-fish 0.6.2 decompose_dense crashes on float images when a dense region is found
# (np.iinfo(float32) in _gaussian_mixture_2d) -> it needs uint8/uint16 input.
try:
    det.decompose_dense(img, sp, voxel_size=1, spot_radius=SIGMA)
    print("decompose_dense accepted float32")
except ValueError as e:
    print("decompose_dense on float32 raised ValueError:", e)
img_u16 = np.clip(np.rint(img), 0, 65535).astype(np.uint16)
for beta in (1, 2):
    sp_d, regions, ref = det.decompose_dense(img_u16, sp, voxel_size=1, spot_radius=SIGMA,
                                              kernel_size=None, alpha=0.5, beta=beta, gamma=5)
    sp_d = sp_d[mask[np.clip(sp_d[:, 0], 0, a.shape[0]-1), np.clip(sp_d[:, 1], 0, a.shape[1]-1)]]
    pd, _, _ = match(sp_d[:, ::-1], pts, sw.TOL)
    added = len(sp_d) - len(sp)
    print(f"decompose_dense beta={beta}: n={len(sp_d)} (+{added}) dense regions={len(regions)} "
          f"spots-in-regions={int(regions[:, 2].sum()) if len(regions) else 0} tp={len(pd)}; ref spot max={ref.max():.0f}")
    if beta == 1 and len(regions):
        overlay(to8(a, 300, 6000, 0.5), [(sp_d[:, ::-1], (255, 0, 0)), (sp[:, ::-1], (0, 255, 0)),
                                        (regions[:, 1::-1].astype(float), (255, 255, 0))],
                OUT / "04_dense_overlay.png", radius=7, scale=0.5)
        # crop around the biggest region
        k = int(np.argmax(regions[:, 2])); cy, cx = int(regions[k, 0]), int(regions[k, 1])
        y0, x0 = max(cy - 60, 0), max(cx - 60, 0)
        crop = a[y0:y0 + 120, x0:x0 + 120]
        sel = lambda s: s[(s[:, 0] >= y0) & (s[:, 0] < y0 + 120) & (s[:, 1] >= x0) & (s[:, 1] < x0 + 120)] - [y0, x0]
        overlay(to8(crop), [(sel(sp_d)[:, ::-1], (255, 0, 0)), (sel(sp)[:, ::-1], (0, 255, 0))],
                OUT / "04_dense_crop.png", radius=3, scale=4)
        print(f"  largest dense region at (x={cx},y={cy}): {int(regions[k,2])} spots, area {int(regions[k,3])} px")
