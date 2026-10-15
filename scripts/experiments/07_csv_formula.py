"""Experiment 7: reverse-engineer FAGE ABUNDENCE in the researcher's CSVs.
Usage: python 07_csv_formula.py <csv> [<csv> ...]
Checks abundance == mean(non-blank fields) * (field/grid) / volume_uL * 1000
and reports blanks vs zeros, field counts, QC values."""
import csv
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    hdr = [h.strip() for h in rows[0]]
    fi = [i for i, h in enumerate(hdr) if h.lower().startswith("field ") and h.split()[-1].isdigit()]
    ix = {h: i for i, h in enumerate(hdr)}
    print(f"== {path}: {len(rows)-1} rows, {len(fi)} field columns")
    worst = 0.0
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        vals = [r[i].strip() if i < len(r) else "" for i in fi]
        nums = [float(v) for v in vals if v != ""]
        # blanks inside the run (not only trailing)?
        last = max((k for k, v in enumerate(vals) if v != ""), default=-1)
        inner_blank = any(v == "" for v in vals[:last + 1])
        vol = float(r[ix["SAMPLE VOLUM[ul]"]]); fld = float(r[ix["field [Pixel^2]"]]); grd = float(r[ix["grid [Pixel^2]"]])
        conc = float(r[ix["CONCENTRATION FACTOR"]])
        ab = float(r[ix["FAGE ABUNDENCE [fage mL-1]"]])
        mean = sum(nums) / len(nums)
        pred = mean * (fld / grd) / vol * 1000.0
        pred5 = mean * (fld / grd) / 5.0 * 1000.0
        rel = abs(pred - ab) / max(ab, 1e-9)
        worst = max(worst, rel)
        print(f"{r[0]:6s} n_fields={len(nums):2d} sum={int(sum(nums)):4d} mean={mean:7.3f} csv={ab:12.3f} "
              f"pred(2.5uL)={pred:12.3f} pred(5uL)={pred5:11.3f} QC={r[ix['QC']]} conc={conc} inner_blank={inner_blank}")
    print(f"max relative error of formula: {worst:.2e} (factor = (field/grid)/vol*1000 = {fld/grd/vol*1000:.1f})")
