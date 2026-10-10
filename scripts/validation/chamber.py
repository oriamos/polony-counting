"""Chamber (dark stadium-shaped hybridisation area) mask for the real slides.

The scanner background OUTSIDE the chamber is very bright: page 1 is clipped at 65535 there (~18 % of
every frame), page 0 is at 20-30k. Inside, both pages sit at ~500-900 counts. So:
  outside = smoothed fraction of page-1 pixels >= 60000 (8x binned) > 0.5
  chamber = largest connected component of NOT outside, holes filled
  counting region = chamber eroded by RIM_PX (rim band excluded)
Usage (CLI): python -I chamber.py <out_dir> <raw.tif> [...]  -> rim profile + per-cell stats JSON, mask npz
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GRID_COLS, GRID_ROWS, GRID_STEP, GRID_X0, GRID_Y0, cell_of, read_points, read_raw  # noqa: E402

RIM_PX = 40
BIN = 8


def chamber_mask(arr):
    p1 = arr[1]
    h, w = p1.shape
    hb, wb = h // BIN, w // BIN
    sat = (p1[: hb * BIN, : wb * BIN] >= 60000).reshape(hb, BIN, wb, BIN).mean(axis=(1, 3))
    outside = ndi.uniform_filter(sat, 3) > 0.5
    inside = ~outside
    lab, n = ndi.label(inside)
    if n == 0:
        return np.zeros_like(p1, bool)
    sizes = ndi.sum(inside, lab, range(1, n + 1))
    m = ndi.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))
    # upsample to full res and refine the boundary at full resolution
    full = np.zeros((h, w), bool)
    full[: hb * BIN, : wb * BIN] = np.kron(m, np.ones((BIN, BIN), bool))
    if hb * BIN < h:
        full[hb * BIN:, :] = full[hb * BIN - 1: hb * BIN, :]
    if wb * BIN < w:
        full[:, wb * BIN:] = full[:, wb * BIN - 1: wb * BIN]
    # refine: within 2*BIN of the coarse edge, pixel is inside iff page-1 below 60000 after a 5px median
    band = ndi.binary_dilation(full, iterations=2 * BIN) & ~ndi.binary_erosion(full, iterations=2 * BIN)
    sm = ndi.median_filter(p1, size=5)
    full = (full & ~band) | (band & (sm < 60000))
    lab, n = ndi.label(full)
    if n > 1:
        sizes = ndi.sum(full, lab, range(1, n + 1))
        full = lab == 1 + int(np.argmax(sizes))
    return ndi.binary_fill_holes(full)


def counting_region(mask, rim_px=RIM_PX):
    return ndi.binary_erosion(mask, structure=np.ones((3, 3), bool), iterations=rim_px) if rim_px > 0 else mask


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for p in sys.argv[2:]:
        slide = Path(p).stem.split("_T7_AB_")[1]
        arr, _ = read_raw(p)
        m = chamber_mask(arr)
        dist = ndi.distance_transform_edt(m)
        prof = {}
        for d0 in list(range(0, 100, 5)) + [100, 150, 200]:
            sel = (dist > d0) & (dist <= d0 + 5)
            prof[d0] = [float(np.median(arr[i][sel])) for i in (0, 1)] + [float((arr[0][sel] == 65535).mean())]
        cr = counting_region(m)
        ys, xs = np.nonzero(m)
        xy, cnt, pos = read_points(p)
        inside_pts = m[np.clip(xy[:, 1].round().astype(int), 0, m.shape[0] - 1), np.clip(xy[:, 0].round().astype(int), 0, m.shape[1] - 1)] if len(xy) else []
        dpts = dist[np.clip(xy[:, 1].round().astype(int), 0, m.shape[0] - 1), np.clip(xy[:, 0].round().astype(int), 0, m.shape[1] - 1)] if len(xy) else np.zeros(0)
        cells = {}
        rr, cc = cell_of(xy[:, 0], xy[:, 1]) if len(xy) else ([], [])
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                y0, x0 = int(round(GRID_Y0 + r * GRID_STEP)), int(round(GRID_X0 + c * GRID_STEP))
                y1, x1 = int(round(GRID_Y0 + (r + 1) * GRID_STEP)), int(round(GRID_X0 + (c + 1) * GRID_STEP))
                sub = m[y0:y1, x0:x1]
                subc = cr[y0:y1, x0:x1]
                bg = [float(np.median(arr[i, y0:y1, x0:x1][subc])) if subc.any() else None for i in (0, 1)]
                npts = int(((np.asarray(rr) == r) & (np.asarray(cc) == c) & (pos == 1)).sum()) if len(xy) else 0
                cells[f"r{r}c{c}"] = dict(chamber_cov=round(float(sub.mean()), 3), count_cov=round(float(subc.mean()), 3),
                                          bg_p0=bg[0], bg_p1=bg[1], roi_pts_p0=npts)
        res[slide] = dict(area=int(m.sum()), frac=float(m.mean()), bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                          count_area=int(cr.sum()), rim_profile=prof,
                          roi_inside=int(np.sum(inside_pts)), roi_n=int(len(xy)),
                          roi_dist_min=float(dpts.min()) if len(dpts) else None,
                          roi_dist_p5=float(np.percentile(dpts, 5)) if len(dpts) else None,
                          roi_within_rim=int((dpts <= RIM_PX).sum()) if len(dpts) else 0, cells=cells)
        np.savez_compressed(out / f"chamber_{slide}.npz", mask=m, count=cr)
        print(slide, res[slide]["frac"], res[slide]["bbox"], "roi in", res[slide]["roi_inside"], "/", len(xy),
              "min dist", res[slide]["roi_dist_min"], "within rim", res[slide]["roi_within_rim"], flush=True)
    (out / "chamber.json").write_text(json.dumps(res, indent=1))
