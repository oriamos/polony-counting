"""Characterise researcher points missed by the app and app detections not marked by the researcher.
python -I missed.py <an.json> <raw.tif> [...]  (uses the r4 entry unless RADIUS env set)"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_raw  # noqa: E402
an = json.loads(Path(sys.argv[1]).read_text())
rk = os.environ.get("RADIUS", "r4")
def feat(a, x, y, r=6, rb=(20, 30)):
    xi, yi = int(round(x)), int(round(y))
    yy, xx = np.ogrid[-rb[1]:rb[1] + 1, -rb[1]:rb[1] + 1]
    sub = a[max(0, yi - rb[1]):yi + rb[1] + 1, max(0, xi - rb[1]):xi + rb[1] + 1].astype(float)
    if sub.shape != (2 * rb[1] + 1,) * 2:
        return None
    d = np.sqrt(yy ** 2 + xx ** 2)
    peak = sub[d <= r].max(); bg = np.median(sub[(d >= rb[0])]); sd = 1.4826 * np.median(np.abs(sub[d >= rb[0]] - bg))
    # half-max radius
    hm = bg + (peak - bg) / 2
    rad = np.sqrt((sub[d <= 25] >= hm).sum() / np.pi)
    return peak, bg, (peak - bg) / max(sd, 1), rad, int((sub[d <= r] == 65535).sum())
for p in sys.argv[2:]:
    slide = Path(p).stem.split("_T7_AB_")[1]
    arr, _ = read_raw(p)
    e = an[slide][rk]["point_match"]["slice1_cladeB"]
    for nm, pts in (("missed", e["missed_xy"]), ("fp", [q[:2] for q in e["fp_xy"]])):
        F = [f for f in (feat(arr[0], x, y) for x, y in pts) if f is not None]
        if not F:
            print(slide, nm, 0); continue
        F = np.array(F)
        sat = (F[:, 4] > 0).sum(); faint = (F[:, 2] < 5).sum(); mid = ((F[:, 2] >= 5) & (F[:, 4] == 0)).sum()
        print(f"{slide} {rk} {nm:6s} n={len(F):3d} saturated_core={sat:3d} snr<5={faint:3d} snr>=5&unsat={mid:3d} "
              f"median snr={np.median(F[:,2]):.1f} median hm-radius={np.median(F[:,3]):.1f}px median peak={np.median(F[:,0]):.0f}")
    # all researcher points, for reference
