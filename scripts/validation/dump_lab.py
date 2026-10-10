"""Print the researcher's per-field counts. python -I dump_lab.py Counts.xlsx T7A.csv T7B.csv"""
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import counts_sheets, t7_csv  # noqa: E402

sh = counts_sheets(sys.argv[1])
for name, rows in sh.items():
    for k, v in rows.items():
        nz = [x for x in v if x is not None]
        print(f"{name:10s} {k[0]:5s} {k[1]} n={len(nz):2d} sum={sum(nz) if nz else '-':>5} {v}")
for p in sys.argv[2:]:
    for k, v in t7_csv(p).items():
        nz = [x for x in v if x is not None]
        print(f"{p.split('/')[-1][:18]:18s} {k:8s} n={len(nz):2d} sum={sum(nz) if nz else '-'} {[None if x is None else int(x) for x in v]}")
