"""STAGE 1 of docs/app/REAL_DATA_VALIDATION.md: per-file facts, embedded ROI vs researcher sheet,
RGB-composite channel identity, chamber geometry. No detection.

Usage: python -I stage1_facts.py <tiff_dir> <Counts.xlsx> <out_dir>
Writes <out_dir>/stage1.json, overview PNGs and chamber masks (<out_dir>/chamber_<slide>.npz).
NOTE: the Otsu chamber mask in this script FAILS on these slides (it selects the clipped outside);
the chamber numbers in the doc come from chamber.py, which supersedes it.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (GRID_COLS, GRID_ROWS, GRID_STEP, GRID_X0, GRID_Y0, cell_of,  # noqa: E402
                    counts_sheets, read_points)

tdir, xlsx, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
sheets = counts_sheets(xlsx)
SHEET_FOR = {"B": "Clade B", "A": "Clade A"}


def parse_info(s):
    d = {}
    for line in (s or "").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def block_mean(a, f):
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def chamber_mask(img2):
    """Robust chamber mask: the chamber is the large bright-background region (the 'dark oval' is the
    region OUTSIDE the scanned/hybridised area? -> decided from data below). Strategy: heavy Gaussian of
    log(sum of pages), Otsu on the smoothed image, keep the largest connected component, fill holes,
    then erode by a rim margin."""
    from skimage.filters import threshold_otsu
    s = np.log1p(img2.astype(np.float32).sum(0))
    sm = ndi.gaussian_filter(block_mean(s, 4), 6)
    th = threshold_otsu(sm)
    m = sm > th
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    m = ndi.binary_fill_holes(m)
    m = np.kron(m, np.ones((4, 4), bool))
    full = np.zeros(img2.shape[1:], bool)
    full[: m.shape[0], : m.shape[1]] = m
    return full, float(th)


res = {"raw": {}, "rgb": {}}
raws = sorted(p for p in tdir.glob("*.tif") if "(" not in p.name)
for p in raws:
    m = re.search(r"_T7_AB_([A-Z]+)-([A-Z])", p.name)
    sample, letter = m.group(1), m.group(2)
    slide = p.stem.split("_T7_AB_")[1]
    with tifffile.TiffFile(str(p)) as t:
        arr = t.asarray()
        md = t.imagej_metadata or {}
        dt = [pg.tags["DateTime"].value if "DateTime" in pg.tags else None for pg in t.pages]
    info = parse_info(md.get("Info"))
    pages = []
    for i in range(arr.shape[0]):
        a = arr[i]
        pages.append(dict(min=int(a.min()), max=int(a.max()), p1=float(np.percentile(a, 1)),
                          p50=float(np.median(a)), p99=float(np.percentile(a, 99)),
                          p999=float(np.percentile(a, 99.9)),
                          sat_frac=float((a == 65535).mean()), sat_px=int((a == 65535).sum())))
    # pixel correlation between the two pages (log, 4x binned)
    l0, l1 = (np.log1p(block_mean(arr[i].astype(np.float32), 4)).ravel() for i in (0, 1))
    page_corr = float(np.corrcoef(l0, l1)[0, 1])
    # embedded ROI points
    xy, cnt, pos = read_points(p)
    rr, cc = cell_of(xy[:, 0], xy[:, 1]) if len(xy) else (np.zeros(0, int), np.zeros(0, int))
    counters = {}
    for k in np.unique(cnt):
        sel = cnt == k
        cells = [f"r{a}c{b}" for a, b in zip(rr[sel], cc[sel])]
        u, n = np.unique(cells, return_counts=True)
        counters[int(k)] = dict(n=int(sel.sum()), cells=dict(zip(u.tolist(), n.tolist())),
                                cx=float(xy[sel, 0].mean()), cy=float(xy[sel, 1].mean()))
    sheet_b = sheets.get(SHEET_FOR.get(letter, ""), {}).get((sample, letter))
    sheet_vals = {sh: sheets[sh].get((sample, letter)) for sh in sheets}
    # sanity: counters (sorted, excluding empty) vs sheet series
    counter_ns = [counters[k]["n"] for k in sorted(counters)]
    match = {sh: (v is not None and [x for x in v if x is not None] == counter_ns) for sh, v in sheet_vals.items()}
    # chamber
    cm, th = chamber_mask(arr)
    ys, xs = np.nonzero(cm)
    ch = dict(area_px=int(cm.sum()), frac=float(cm.mean()), bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
              otsu_log=th)
    # per-cell chamber coverage
    cov = np.zeros((GRID_ROWS, GRID_COLS))
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y0, x0 = int(GRID_Y0 + r * GRID_STEP), int(GRID_X0 + c * GRID_STEP)
            cov[r, c] = cm[y0:int(y0 + GRID_STEP), x0:int(x0 + GRID_STEP)].mean()
    ch["cell_coverage"] = np.round(cov, 3).tolist()
    # chamber vs outside median per page
    ch["median_in"] = [float(np.median(arr[i][cm])) for i in (0, 1)]
    ch["median_out"] = [float(np.median(arr[i][~cm])) if (~cm).any() else None for i in (0, 1)]
    np.savez_compressed(out / f"chamber_{slide}.npz", mask=cm)
    res["raw"][slide] = dict(file=p.name, shape=list(arr.shape), dtype=str(arr.dtype), ij_keys=list(md.keys()),
                             ij_minmax=[md.get("min"), md.get("max")], info=info, page_datetime=dt,
                             labels=md.get("Labels"), pages=pages, page_log_corr=page_corr,
                             roi_n=int(len(xy)), roi_counter_positions=np.unique(pos).tolist(),
                             counters=counters, sheet_vals=sheet_vals, counters_equal_sheet=match,
                             chamber=ch)
    # overview PNG: page0 -> red, page1 -> green (display only), grid + ROI points
    disp = []
    for i in (0, 1):
        a = np.log1p(arr[i].astype(np.float32))
        lo, hi = np.percentile(a, [1, 99.8])
        disp.append(np.clip((a - lo) / (hi - lo), 0, 1))
    rgb = np.dstack([disp[0], disp[1], np.zeros_like(disp[0])])
    rgb = (rgb * 255).astype(np.uint8)
    edge = cm ^ ndi.binary_erosion(cm, iterations=3)
    rgb[edge] = (0, 128, 255)
    for k in range(GRID_COLS + 1):
        x = int(round(GRID_X0 + k * GRID_STEP))
        rgb[:, max(0, x - 1):x + 1] = (255, 0, 255)
    for k in range(GRID_ROWS + 1):
        y = int(round(GRID_Y0 + k * GRID_STEP))
        rgb[max(0, y - 1):y + 1, :] = (255, 0, 255)
    for (x, y) in xy:
        xi, yi = int(round(x)), int(round(y))
        rgb[max(0, yi - 6):yi + 7, max(0, xi - 1):xi + 2] = (255, 255, 255)
        rgb[max(0, yi - 1):yi + 2, max(0, xi - 6):xi + 7] = (255, 255, 255)
    Image.fromarray(rgb).resize((730, 1219)).save(out / f"overview_{slide}.png")
    print(slide, "done", flush=True)

# RGB composites: which raw page is in which plane?
for p in sorted(tdir.glob("*_(635_532).tif")):
    slide = p.name.split("_T7_AB_")[1].split("_(")[0]
    im = np.asarray(Image.open(p))
    with tifffile.TiffFile(str(p)) as t:
        tags = {k: str(t.pages[0].tags[k].value) for k in ("DateTime", "HostComputer", "Artist", "XPosition", "YPosition", "XResolution")
                if k in t.pages[0].tags}
    entry = dict(file=p.name, shape=list(im.shape), dtype=str(im.dtype), tags=tags,
                 plane_stats=[dict(min=int(im[..., j].min()), max=int(im[..., j].max()), sat255=float((im[..., j] == 255).mean()),
                                   mean=float(im[..., j].mean())) for j in range(3)])
    rawp = tdir / f"29-09-2026_T7_AB_{slide}.tif"
    if rawp.exists():
        arr = tifffile.imread(str(rawp))
        corr = {}
        shifts = {}
        for j, cname in enumerate("RGB"):
            pl = block_mean(im[..., j].astype(np.float32), 4).ravel()
            if pl.std() == 0:
                corr[cname] = None
                continue
            for i in (0, 1):
                rp = block_mean(arr[i].astype(np.float32), 4)
                corr[f"{cname}~page{i}_log"] = float(np.corrcoef(pl, np.log1p(rp).ravel())[0, 1])
                corr[f"{cname}~page{i}_lin"] = float(np.corrcoef(pl, rp.ravel())[0, 1])
                # spearman
                from scipy.stats import spearmanr
                corr[f"{cname}~page{i}_spearman"] = float(spearmanr(pl, rp.ravel()).correlation)
        # alignment: phase correlation of the brightest-plane pairing at full res (central crop)
        from skimage.registration import phase_cross_correlation
        for j, cname in enumerate("RG"):
            for i in (0, 1):
                a = im[600:1800, 200:1260, j].astype(np.float32)
                b = np.log1p(arr[i][600:1800, 200:1260].astype(np.float32))
                if a.std() == 0:
                    continue
                sh, err, _ = phase_cross_correlation(b, a, upsample_factor=4, normalization=None)
                shifts[f"{cname}~page{i}"] = [float(sh[0]), float(sh[1]), float(err)]
        # transfer function: how does the 8-bit plane map to raw intensity (pixelwise fit at full res)
        tf = {}
        for j, cname in enumerate("RG"):
            best = max((0, 1), key=lambda i: corr.get(f"{cname}~page{i}_spearman") or -9)
            v = im[..., j].ravel().astype(np.float32)
            r = arr[best].ravel().astype(np.float32)
            sel = (v > 2) & (v < 253)
            if sel.sum() > 1000:
                for nm, x in (("lin", r[sel]), ("log", np.log(r[sel])), ("sqrt", np.sqrt(r[sel]))):
                    A = np.vstack([x, np.ones_like(x)]).T
                    coef, resid, *_ = np.linalg.lstsq(A, v[sel], rcond=None)
                    pred = A @ coef
                    tf[f"{cname}<-page{best}:{nm}"] = dict(r2=float(1 - ((v[sel] - pred) ** 2).sum() / ((v[sel] - v[sel].mean()) ** 2).sum()),
                                                           coef=[float(c) for c in coef])
                # raw value at which the plane saturates
                tf[f"{cname}<-page{best}:raw_at_255_p5"] = float(np.percentile(r[v == 255], 5)) if (v == 255).any() else None
                tf[f"{cname}<-page{best}:raw_at_0_p95"] = float(np.percentile(r[v == 0], 95)) if (v == 0).any() else None
        entry.update(corr=corr, shifts=shifts, transfer=tf)
    res["rgb"][slide] = entry
    print("rgb", slide, flush=True)

(out / "stage1.json").write_text(json.dumps(res, indent=1, default=str))
print("wrote", out / "stage1.json")
