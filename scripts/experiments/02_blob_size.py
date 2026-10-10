"""Experiment 2: measure blob size of manually marked polonies (page 0 and 1).
For each manual point: refine to local max of a lightly smoothed image within
6 px, then radial profile -> FWHM and Gaussian sigma (2D fit via moments of the
background-subtracted 41x41 patch). Also cross-channel peak offsets."""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

from _common import OUT, load, manual_points, to8
from PIL import Image

stack, md = load()
pts, ctr = manual_points(md)
H = 25


def g2(xy, A, x0, y0, s, b):
    x, y = xy
    return (A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * s * s)) + b).ravel()


res = {0: [], 1: []}
peaks = {0: [], 1: []}
for ch in (0, 1):
    img = stack[ch].astype(np.float64)
    sm = ndi.gaussian_filter(img, 1.5)
    for (x, y) in pts:
        xi, yi = int(round(x)), int(round(y))
        win = sm[yi - 6:yi + 7, xi - 6:xi + 7]
        dy, dx = np.unravel_index(np.argmax(win), win.shape)
        cy, cx = yi - 6 + dy, xi - 6 + dx
        patch = img[cy - H:cy + H + 1, cx - H:cx + H + 1]
        yy, xx = np.mgrid[-H:H + 1, -H:H + 1]
        bg = np.median(np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]]))
        sat = bool((patch[H - 2:H + 3, H - 2:H + 3] == 65535).any())
        try:
            p, _ = curve_fit(g2, (xx, yy), patch.ravel(),
                             p0=(patch[H, H] - bg, 0, 0, 4, bg), maxfev=4000)
            A, x0, y0, s, b = p
            ok = A > 0 and 0.5 < abs(s) < 20 and abs(x0) < 6 and abs(y0) < 6
        except Exception:
            ok = False
        if ok:
            res[ch].append((abs(s), A, b, sat, A / max(np.std(patch[0]), 1)))
            peaks[ch].append((cx + x0, cy + y0))
        else:
            res[ch].append((np.nan, np.nan, bg, sat, np.nan))
            peaks[ch].append((np.nan, np.nan))
for ch in (0, 1):
    r = np.array([[a[0], a[1], a[2], a[3], a[4]] for a in res[ch]], float)
    good = ~np.isnan(r[:, 0])
    s = r[good, 0]
    print(f"page {ch}: fitted {good.sum()}/{len(r)}; sigma px median={np.median(s):.2f} "
          f"IQR=({np.percentile(s,25):.2f},{np.percentile(s,75):.2f}) min={s.min():.2f} max={s.max():.2f}; "
          f"FWHM median={2.355*np.median(s):.1f}px; amplitude median={np.nanmedian(r[:,1]):.0f} "
          f"p10={np.nanpercentile(r[:,1],10):.0f} p90={np.nanpercentile(r[:,1],90):.0f}; "
          f"local bg median={np.median(r[:,2]):.0f}; saturated-core={int(r[:,3].sum())}; "
          f"amp/edge-std median={np.nanmedian(r[:,4]):.1f}")
p0 = np.array(peaks[0]); p1 = np.array(peaks[1])
d = p1 - p0
ok = ~np.isnan(d).any(1)
r1 = np.array([a[1] for a in res[1]], float)
strong1 = ok & (r1 > 200)
print("page1 fits with amplitude>200 at manual points:", strong1.sum())
if strong1.sum():
    print("  dx,dy (p1-p0) median:", np.median(d[strong1], 0), " per-point:", np.round(d[strong1], 2).tolist())
# montage of 24 patches, both pages, each auto-scaled
tiles = []
for i in range(min(24, len(pts))):
    row = []
    for ch in (0, 1):
        x, y = pts[i]
        xi, yi = int(round(x)), int(round(y))
        p = stack[ch][yi - H:yi + H + 1, xi - H:xi + H + 1]
        t = Image.fromarray(to8(p)).resize((153, 153), Image.NEAREST)
        row.append(np.asarray(t))
    tiles.append(np.hstack([row[0], np.full((153, 3), 255, np.uint8), row[1]]))
rows = [np.hstack([tiles[k] if k < len(tiles) else np.zeros_like(tiles[0]) for k in range(j, j + 4)])
        for j in range(0, 24, 4)]
Image.fromarray(np.vstack(rows)).save(OUT / "02_patch_montage.png")
np.save(OUT / "02_sigmas_p0.npy", np.array([a[0] for a in res[0]]))
