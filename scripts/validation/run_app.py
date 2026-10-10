"""STAGE 2/4: run the app's own pipeline (read-only use of src/polony_app) on real raw slides.

Usage: PYTHONPATH=src python -I run_app.py <out_dir> <rim_px> <radii comma list> <raw.tif> [...]
  (-I ignores PYTHONPATH, so the src path is taken from POLONY_SRC or ../../src relative to this file)
Channel mapping (stage 1 evidence): red = page 0 (635, R plane), green = page 1 (532, G plane).
Counting region = one polygon = chamber mask eroded by <rim_px> (chamber.py). All other parameters are
the app defaults except spot_radius_px (= the radius under test, both channels) and, if env THR_SCALE
is set, threshold_scale (sensitivity test; output file gets a _t<scale> suffix).
Caches <out_dir>/det_<slide>_r<radius>.json (skips existing), renders overlay PNGs.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("POLONY_SRC", str(HERE.parent.parent / "src")))

from common import read_points, read_raw  # noqa: E402
from chamber import chamber_mask, counting_region  # noqa: E402
from polony_app.imaging.ingest import ingest_files  # noqa: E402
from polony_app.detection.pipeline import run_detection  # noqa: E402
from polony_app.schemas import DetectionParams, DetectionRequest, Region, RegionKind  # noqa: E402


def mask_polygon(mask):
    from skimage import measure
    cs = measure.find_contours(np.pad(mask, 1).astype(np.uint8), 0.5)
    c = max(cs, key=len) - 1
    c = measure.approximate_polygon(c, tolerance=1.0)
    return [(float(x), float(y)) for y, x in c[:-1]]


def main():
    out = Path(sys.argv[1])
    rim = int(sys.argv[2])
    radii = [float(r) for r in sys.argv[3].split(",")]
    out.mkdir(parents=True, exist_ok=True)
    for p in sys.argv[4:]:
        slide = Path(p).stem.split("_T7_AB_")[1]
        arr, _ = read_raw(p)
        meta_f = out / f"ingest_{slide}.json"
        if not meta_f.exists():
            meta = ingest_files([(Path(p).name, Path(p))], out / "ingest" / slide)
            meta_f.write_text(meta.model_dump_json(indent=1))
            for f in (out / "ingest" / slide).glob("*"):
                f.unlink()  # do not keep a second copy of the researcher's file
        m = chamber_mask(arr)
        cr = counting_region(m, rim)
        poly = mask_polygon(cr)
        region = Region(id="chamber", kind=RegionKind.counting, name=f"chamber-rim{rim}", polygon=poly)
        for r in radii:
            ts = float(os.environ.get("THR_SCALE", "1"))
            tag = "" if ts == 1 else f"_t{ts:g}"
            f = out / f"det_{slide}_r{r:g}{tag}.json"
            if f.exists():
                print("cached", f.name, flush=True)
                continue
            dp = DetectionParams(spot_radius_px=r, threshold_scale=ts)
            req = DetectionRequest(red=dp, green=dp)
            t0 = time.time()
            res = run_detection(arr[0], arr[1], [region], req)
            dt = time.time() - t0
            objs = [o.model_dump(mode="json") for o in res.objects]
            f.write_text(json.dumps(dict(slide=slide, radius=r, threshold_scale=ts, rim_px=rim, runtime_s=dt, notes=res.notes,
                                         n_poly=len(poly), objects=objs)))
            print(slide, r, f"{dt:.1f}s", len(objs), flush=True)


if __name__ == "__main__":
    main()
