"""STAGES 2-4 analysis: detections (run_app.py cache) vs researcher (embedded ROI points + sheet).

Usage: python -I analyze.py <det_dir> <Counts.xlsx> <out_json> <raw.tif> [...]
Prints markdown tables; writes a JSON with all numbers.
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GRID_COLS, GRID_ROWS, cell_of, counts_sheets, read_points  # noqa: E402

MATCH_PX = 15.0
det_dir, xlsx, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sheets = counts_sheets(xlsx)


def match(a, b, tol=MATCH_PX):
    """one-to-one matching of point sets a, b (N,2),(M,2) within tol: returns n matched, pairs."""
    if len(a) == 0 or len(b) == 0:
        return 0, []
    ta, tb = cKDTree(a), cKDTree(b)
    sp = ta.sparse_distance_matrix(tb, tol, output_type="coo_matrix")
    if sp.nnz == 0:
        return 0, []
    C = np.full((len(a), len(b)), 1e6)
    C[sp.row, sp.col] = sp.data
    r, c = linear_sum_assignment(C)
    ok = C[r, c] <= tol
    return int(ok.sum()), list(zip(r[ok].tolist(), c[ok].tolist()))


def stats(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return dict(n=len(x))
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    slope0 = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else None   # det = k * sheet
    return dict(n=len(x), r=r, slope_through0=slope0, mae=float(np.abs(y - x).mean()),
                bias=float((y - x).mean()), sum_sheet=float(x.sum()), sum_det=float(y.sum()))


def field_cells(cnt, pos, xy, sheet_vals, P):
    """field number (1-based) -> (r, c), from the embedded counters on slice P.
    Tries counter k -> field k+1 ('index') and k-th non-empty counter -> k-th field ('ordinal');
    keeps the one whose per-field counts agree best with the sheet."""
    s = pos == P
    ks = sorted(np.unique(cnt[s]).tolist())
    rr, cc = cell_of(xy[s, 0], xy[s, 1])
    cell = {}
    for k in ks:
        sel = cnt[s] == k
        vals, n = np.unique(np.stack([rr[sel], cc[sel]], 1), axis=0, return_counts=True)
        cell[k] = (tuple(vals[np.argmax(n)].tolist()), int(sel.sum()))
    best = None
    for mode in ("index", "ordinal"):
        fm = {(k + 1 if mode == "index" else i + 1): cell[k] for i, k in enumerate(ks)}
        agree = sum(1 for f, (_, n) in fm.items() if f <= len(sheet_vals) and sheet_vals[f - 1] == n)
        if best is None or agree > best[0]:
            best = (agree, mode, fm)
    return best


def orderings(valid_cells):
    """Plausible fixed field orderings over the valid cells: row/column-major and snake, from each corner."""
    outs = {}
    for major, snake, fr, fc in product(("row", "col"), (False, True), (False, True), (False, True)):
        rows = list(range(GRID_ROWS))[::-1] if fr else list(range(GRID_ROWS))
        cols = list(range(GRID_COLS))[::-1] if fc else list(range(GRID_COLS))
        seq = []
        if major == "row":
            for i, r in enumerate(rows):
                cs = cols[::-1] if (snake and i % 2) else cols
                seq += [(r, c) for c in cs]
        else:
            for i, c in enumerate(cols):
                rs = rows[::-1] if (snake and i % 2) else rows
                seq += [(r, c) for r in rs]
        name = f"{major}{'-snake' if snake else ''}-from-{'bottom' if fr else 'top'}{'right' if fc else 'left'}"
        outs[name] = [x for x in seq if x in valid_cells]
    return outs


CLASSES = {
    "red_only": lambda o: o["cls"] == "red",
    "green_only": lambda o: o["cls"] == "green",
    "yellow": lambda o: o["cls"] == "yellow",
    "uncertain": lambda o: o["cls"] == "uncertain",
    "red_positive": lambda o: o["cls"] in ("red", "yellow"),
    "green_positive": lambda o: o["cls"] in ("green", "yellow"),
    "red_detected_any": lambda o: o["red"]["detected"],
    "green_detected_any": lambda o: o["green"]["detected"],
    "all": lambda o: True,
}

result = {}
for p in sys.argv[4:]:
    slide = Path(p).stem.split("_T7_AB_")[1]
    sample, letter = slide.split("-")[0], slide.split("-")[1][0]
    xy, cnt, pos = read_points(p)
    chm = np.load(det_dir.parent / "ch" / f"chamber_{slide}.npz")
    sheetB = sheets["Clade B"].get((sample, letter))
    sheetA = sheets["Clade A"].get((sample, letter))
    res_s = {}
    # chamber coverage per cell -> valid cells for orderings
    for f in sorted(det_dir.glob(f"det_{slide}_r*.json")):
        d = json.loads(f.read_text())
        rad = d["radius"]
        tsc = d.get("threshold_scale", 1.0)
        objs = [o for o in d["objects"] if not o.get("deleted")]
        oxy = np.array([[o["x"], o["y"]] for o in objs]) if objs else np.zeros((0, 2))
        rim = d["rim_px"]
        from scipy import ndimage as ndi
        cr = ndi.binary_erosion(chm["mask"], structure=np.ones((3, 3), bool), iterations=rim)
        e = dict(runtime_s=d["runtime_s"], notes=d["notes"], n=len(objs))
        e["classes"] = {k: int(sum(1 for o in objs if fn(o))) for k, fn in CLASSES.items()}
        reasons = {}
        for o in objs:
            for rr_ in o["review_reasons"]:
                reasons[rr_] = reasons.get(rr_, 0) + 1
        e["review_reasons"] = reasons
        e["flagged_any"] = int(sum(1 for o in objs if o["review_reasons"]))
        # point-level matching against embedded ROI points (slice 1 = page 0 / Clade B, slice 2 = Clade A)
        pm = {}
        # cells the researcher counted (slice-1 counters); detections outside them are not comparable
        counted = set()
        if (pos == 1).any():
            _rr, _cc = cell_of(xy[pos == 1, 0], xy[pos == 1, 1])
            for k in np.unique(cnt[pos == 1]):
                sel_ = cnt[pos == 1] == k
                vals_, n_ = np.unique(np.stack([_rr[sel_], _cc[sel_]], 1), axis=0, return_counts=True)
                counted.add(tuple(vals_[np.argmax(n_)].tolist()))
        orr0, occ0 = cell_of(oxy[:, 0], oxy[:, 1]) if len(oxy) else (np.zeros(0, int), np.zeros(0, int))
        in_counted = np.array([(a, b) in counted for a, b in zip(orr0, occ0)], bool)
        e["counted_cells"] = sorted(counted)
        e["n_in_counted_cells"] = int(in_counted.sum())
        for P, nm in ((1, "slice1_cladeB"), (2, "slice2_cladeA")):
            s = pos == P
            pts = xy[s]
            inreg = cr[np.clip(pts[:, 1].round().astype(int), 0, cr.shape[0] - 1),
                       np.clip(pts[:, 0].round().astype(int), 0, cr.shape[1] - 1)] if len(pts) else np.zeros(0, bool)
            pr_, pc_ = cell_of(pts[:, 0], pts[:, 1]) if len(pts) else (np.zeros(0, int), np.zeros(0, int))
            pts_in = pts[inreg]
            pm[nm] = {"n_points": int(s.sum()), "n_in_counting_region": int(inreg.sum())}
            for k in ("red_positive", "red_only", "green_positive", "green_only", "yellow", "all"):
                sel = np.array([CLASSES[k](o) for o in objs], bool) if objs else np.zeros(0, bool)
                sel = sel & in_counted
                nmch, pairs = match(pts_in, oxy[sel])
                ndet = int(sel.sum())
                pm[nm][k] = dict(matched=nmch, recall=nmch / len(pts_in) if len(pts_in) else None,
                                 precision=nmch / ndet if ndet else None, n_det=ndet)
            # reasons on matched vs unmatched red_positive detections (slice 1 only)
            if P == 1:
                sel = np.array([CLASSES["red_positive"](o) for o in objs], bool) & in_counted
                idx = np.nonzero(sel)[0]
                nmch, pairs = match(pts_in, oxy[sel])
                matched = set(idx[[b for _, b in pairs]].tolist())
                fp = [objs[i] for i in idx if i not in matched]
                tp = [objs[i] for i in matched]
                def rs(lst):
                    out = {}
                    for o in lst:
                        for x in (o["review_reasons"] or ["<none>"]):
                            out[x] = out.get(x, 0) + 1
                    return out
                pm[nm]["reasons_tp"] = rs(tp)
                pm[nm]["reasons_fp"] = rs(fp)
                missed = [i for i in range(len(pts_in)) if i not in {a for a, _ in pairs}]
                pm[nm]["missed_xy"] = pts_in[missed].round(1).tolist()
                pm[nm]["fp_xy"] = [[round(o["x"], 1), round(o["y"], 1), o["cls"]] for o in fp]
                pm[nm]["missed_by_any_class"] = int(len(pts_in) - match(pts_in, oxy)[0])
                pm[nm]["fp_reason_free"] = int(sum(1 for o in fp if not o["review_reasons"]))
                pm[nm]["tp_flagged"] = int(sum(1 for o in tp if o["review_reasons"]))
        e["point_match"] = pm
        # per-field comparison through the embedded field -> cell mapping
        per_field = {}
        if sheetB is not None and (pos == 1).any():
            agree, mode, fm = field_cells(cnt, pos, xy, sheetB, 1)
            e["field_map"] = dict(mode=mode, agree_fields=agree, n_fields=len([v for v in sheetB if v is not None]),
                                  cells={f: list(c[0]) for f, c in fm.items()})
            orr, occ = cell_of(oxy[:, 0], oxy[:, 1]) if len(oxy) else (np.zeros(0, int), np.zeros(0, int))
            for k, fn in CLASSES.items():
                sel = np.array([fn(o) for o in objs], bool) if objs else np.zeros(0, bool)
                xs, ys_B, ys_A = [], [], []
                rows = []
                for fnum, ((r, c), _n) in sorted(fm.items()):
                    if fnum > len(sheetB) or sheetB[fnum - 1] is None:
                        continue
                    ndet = int(((orr == r) & (occ == c) & sel).sum())
                    rows.append((fnum, r, c, sheetB[fnum - 1], sheetA[fnum - 1] if sheetA else None, ndet))
                xb = [x[3] for x in rows]
                xa = [x[4] for x in rows if x[4] is not None]
                yd = [x[5] for x in rows]
                per_field[k] = dict(vs_cladeB=stats(xb, yd), vs_cladeA=stats(xa, [x[5] for x in rows if x[4] is not None]),
                                    rows=rows)
            # fixed orderings: how many fields land in the true cell, and r vs detections
            valid_cells = set()
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    # cell counts if >= 85 % inside the chamber (empirically all counted cells are)
                    from common import cell_rect
                    x0, y0, x1, y1 = cell_rect(r, c)
                    if chm["mask"][int(y0):int(y1), int(x0):int(x1)].mean() >= 0.85:
                        valid_cells.add((r, c))
            ordres = {}
            true = {f: tuple(c[0]) for f, c in fm.items()}
            sel = np.array([CLASSES["red_positive"](o) for o in objs], bool)
            for name, seq in orderings(valid_cells).items():
                nf = len([v for v in sheetB if v is not None])
                assign = {f + 1: seq[f] for f in range(min(nf, len(seq)))}
                hits = sum(1 for f, c in assign.items() if true.get(f) == c)
                yd = [int(((orr == c[0]) & (occ == c[1]) & sel).sum()) for f, c in sorted(assign.items())]
                xb = [sheetB[f - 1] for f in sorted(assign)]
                st = stats(xb, yd)
                ordres[name] = dict(hits=hits, n=len(assign), r=st.get("r"))
            e["orderings"] = ordres
            e["valid_cells"] = sorted(valid_cells)
        e["per_field"] = per_field
        res_s[f"r{rad:g}" + ("" if tsc == 1 else f"_t{tsc:g}")] = e
    result[slide] = res_s

Path(out_json).write_text(json.dumps(result, indent=1, default=str))

# ---- printed summary
for slide, rs in result.items():
    for rk, e in rs.items():
        c = e["classes"]
        pm = e["point_match"]["slice1_cladeB"]
        print(f"| {slide} | {rk} | {e['runtime_s']:.1f} | {e['n']} | {c['red_only']} | {c['green_only']} | {c['yellow']} | {c['uncertain']} | "
              f"{e['flagged_any']} | {pm['n_points']}/{pm['n_in_counting_region']} | "
              f"{pm['red_positive']['recall']:.2f}/{pm['red_positive']['precision']:.2f} | {pm['missed_by_any_class']} |")
