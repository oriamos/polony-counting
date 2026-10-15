"""Experiment 3: Big-FISH on the real test image (page 0 = '635 [W1]', page 1).

Reference = the points embedded in the TIFF ImageJ multi-point ROI (77 on the original real scan) (one
researcher's manual marks, one counter per grid cell). This is NOT validated
ground truth; it is the only human annotation available. We restrict scoring to
the union of the 17 grid cells that were counted.

For each background method x LoG sigma x minimum distance:
  * Big-FISH auto threshold (masked, decomposed API) -> n detections, matches.
  * oracle sweep over thresholds -> best F1 vs manual points (separability only).
"""
import time
import warnings

import numpy as np
import bigfish.detection as det
import bigfish.stack as stack
from scipy import ndimage as ndi
from skimage.morphology import disk, white_tophat

from _common import OUT, grid_lines, load, manual_points, match

warnings.filterwarnings("ignore")
stack_img, md = load()
pts, ctr = manual_points(md)
xs, ys = grid_lines(md)
H, W = stack_img.shape[1:]
ci = np.searchsorted(xs, pts[:, 0]) - 1
ri = np.searchsorted(ys, pts[:, 1]) - 1
cells = sorted(set(zip(ci.tolist(), ri.tolist())))
counted = np.zeros((H, W), bool)
for c, r in cells:
    counted[int(round(ys[r])):int(round(ys[r + 1])), int(round(xs[c])):int(round(xs[c + 1]))] = True
print("counted cells:", len(cells), "area px:", counted.sum())
import os
TOL = float(os.environ.get("TOL", 15.0))  # manual clicks are often ~10 px off-centre (see notes)


def bg_correct(a, method, p, valid=None):
    a = a.astype(np.float32)
    if method == "none":
        return a
    if method == "gauss":
        return np.clip(a - ndi.gaussian_filter(a, p), 0, None)
    if method == "gauss_masked":  # normalised convolution ignoring saturated / invalid pixels
        w = valid.astype(np.float32)
        num = ndi.gaussian_filter(a * w, p)
        den = ndi.gaussian_filter(w, p)
        bg = num / np.maximum(den, 1e-3)
        return np.clip(a - bg, 0, None) * valid
    if method == "tophat":
        return white_tophat(a, disk(int(p)))
    raise ValueError(method)


def detect(img, mask, sigma, mdist, thr=None):
    log = stack.log_filter(img, float(sigma))
    lmax = det.local_maximum_detection(log, float(mdist))
    log_m = np.where(mask, log, 0).astype(log.dtype)
    lmax_m = lmax & mask
    if thr is None:
        try:
            thr = det.automated_threshold_setting(log_m, lmax_m)
        except ValueError:
            thr = None
    if thr is None:
        return None, None, log_m, lmax_m
    sp, _ = det.spots_thresholding(log_m, lmax_m, float(thr), remove_duplicate=True)
    return sp, float(thr), log_m, lmax_m


def oracle(log_m, lmax_m):
    yx = np.argwhere(lmax_m & (log_m > 0))
    v = log_m[lmax_m & (log_m > 0)]
    best = (0.0, None, 0.0, 0.0, 0)
    for t in np.quantile(v, np.linspace(0.5, 0.9995, 300)):
        sel = yx[v > t]
        if len(sel) > 2000:
            continue
        pairs, ua, ub = match(sel[:, ::-1], pts, TOL)
        tp = len(pairs)
        prec = tp / max(len(sel), 1); rec = tp / len(pts)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best[0]:
            best = (f1, float(t), prec, rec, len(sel))
    return best


if __name__ == "__main__":
    ch = 0
    a = stack_img[ch]
    valid_nonsat = a < 65535
    configs = [("none", 0), ("gauss", 15), ("gauss", 25), ("gauss", 50), ("gauss", 100),
               ("gauss_masked", 50), ("tophat", 10), ("tophat", 20)]
    rows = []
    for meth, p in configs:
        t0 = time.time()
        img = bg_correct(a, meth, p, valid_nonsat)
        for sigma in (3, 4, 5, 6, 7, 8):
            for mfac in (1.0, 2.0):
                mdist = sigma * mfac
                sp, thr, log_m, lmax_m = detect(img, counted, sigma, mdist)
                if sp is None:
                    n = tp = 0; rec = prec = 0.0
                else:
                    pairs, ua, ub = match(sp[:, ::-1], pts, TOL)
                    n, tp = len(sp), len(pairs)
                    rec, prec = tp / len(pts), tp / max(n, 1)
                f1o, to, po, ro, no = oracle(log_m, lmax_m)
                rows.append((meth, p, sigma, mdist, thr, n, tp, rec, prec, f1o, to, po, ro, no))
                print(f"{meth:12s} p={p:3d} sigma={sigma} md={mdist:4.1f} | auto thr={thr if thr is None else round(thr,2)} "
                      f"n={n} tp={tp} recall={rec:.2f} prec={prec:.2f} | oracle F1={f1o:.2f} thr={to and round(to,2)} "
                      f"P={po:.2f} R={ro:.2f} n={no}", flush=True)
        print(f"  ({time.time()-t0:.1f}s)")
    np.save(OUT / "03_rows.npy", np.array(rows, dtype=object), allow_pickle=True)
