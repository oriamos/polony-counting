"""Run on the example slide (software smoke test, NOT accuracy evidence).

The example (resources/dataset_files/test/test_img.tif) is synthetic, written by make_test_img.py next to it with
the lab files' structure; its known spots (test_img.truth.json) let these tests also check recall / precision.
Agreement with a synthetic truth says nothing about accuracy on real slides.
"""
import json
from pathlib import Path

import pytest

from polony_app.detection.pipeline import run_detection
from polony_app.detection.masks import build_valid_mask
from polony_app.imaging.ingest import ingest_files
from polony_app.imaging.io import load_channel
from polony_app.schemas import DetectionParams, DetectionRequest, Region, RegionKind

REAL = Path(__file__).resolve().parents[2] / "resources/dataset_files/test/test_img.tif"
TOL_PX = 6.0     # = default match_tolerance_px


def _recall(truth_xy, det_xy):
    import numpy as np
    if not len(truth_xy):
        return 1.0
    det = np.asarray(det_xy, float).reshape(-1, 2)
    if not len(det):
        return 0.0
    return float(np.mean([np.hypot(*(det - t).T).min() <= TOL_PX for t in np.asarray(truth_xy, float)]))


@pytest.mark.skipif(not REAL.exists(), reason="real example image missing")
def test_real_image_masked_run(tmp_path):
    meta = ingest_files([("test_img.tif", REAL)], tmp_path)
    red, green = load_channel(meta, tmp_path, 0), load_channel(meta, tmp_path, 1)
    reg = Region(id="c1", kind=RegionKind.counting, name="chamber",
                 polygon=[(250, 300), (1250, 300), (1250, 2100), (250, 2100)])
    p = DetectionParams(spot_radius_px=5, minimum_distance_px=5)
    t = p.model_copy(update={"tile_size": 800, "tile_overlap_px": 64})
    a = run_detection(red, green, [reg], DetectionRequest(red=p, green=p), lambda f, m: None, lambda: False)
    b = run_detection(red, green, [reg], DetectionRequest(red=t, green=t), lambda f, m: None, lambda: False)
    assert len(a.objects) > 10
    valid = build_valid_mask(red.shape, [reg])
    assert all(valid[int(o.y), int(o.x)] and o.region_id == "c1" for o in a.objects)  # floor: half-open pixel cell of the centre
    # tiling does not change results on real data
    assert [(o.id, o.cls, round(o.x, 2), round(o.y, 2)) for o in a.objects] == \
           [(o.id, o.cls, round(o.x, 2), round(o.y, 2)) for o in b.objects]
    # the file's original pixels were not touched
    assert red.dtype.name == "uint16" and int(red.max()) == 65535
    # against the generator's known spots inside the region (all of them are clearly above the noise:
    # weakest red ~10x, green ~12x the background noise). Measured: recall 1.0 / precision 1.0 for both
    # channels and every yellow pair classed yellow; thresholds leave margin for detector changes.
    truth = json.loads(REAL.with_name("test_img.truth.json").read_text())
    inside = [s for s in truth["spots"] if 250 <= s["x"] < 1250 and 300 <= s["y"] < 2100]
    xy = lambda kinds, ch: [(s["x"], s["y"]) for s in inside if s["kind"] in kinds and s["amp_" + ch] > 0]
    det = lambda ch: [(getattr(o, ch).x, getattr(o, ch).y) for o in a.objects if getattr(o, ch).detected]
    assert len(xy(("red", "yellow"), "red")) > 50 and len(xy(("green", "yellow"), "green")) > 20
    assert _recall(xy(("red", "yellow"), "red"), det("red")) >= 0.9
    assert _recall(xy(("green", "yellow"), "green"), det("green")) >= 0.9
    any_spot = [(s["x"], s["y"]) for s in truth["spots"]]
    assert _recall(det("red"), any_spot) >= 0.9 and _recall(det("green"), any_spot) >= 0.9   # precision
    yel = [(o.x, o.y) for o in a.objects if o.cls.value == "yellow"]
    assert _recall(xy(("yellow",), "red"), yel) >= 0.8


@pytest.mark.skipif(not REAL.exists(), reason="real example image missing")
def test_real_image_detection_independent_of_brightness(tmp_path):
    """Automatic threshold is in noise units: halving / quartering the exposure does not change detections."""
    import numpy as np
    meta = ingest_files([("test_img.tif", REAL)], tmp_path)
    red, green = load_channel(meta, tmp_path, 0), load_channel(meta, tmp_path, 1)
    reg = Region(id="c1", kind=RegionKind.counting, name="part",
                 polygon=[(400, 500), (900, 500), (900, 1000), (400, 1000)])
    p = DetectionParams(spot_radius_px=5, minimum_distance_px=5)
    ref = run_detection(red, green, [reg], DetectionRequest(red=p, green=p))
    pts = lambda r: np.array(sorted((o.red.x, o.red.y) for o in r.objects if o.red.detected))
    assert len(pts(ref)) > 5
    for f in (0.5, 0.25):
        r2 = np.clip(np.rint(red.astype(float) * f), 0, 65535).astype(red.dtype)
        res = run_detection(r2, green, [reg], DetectionRequest(red=p, green=p))
        a, b = pts(res), pts(ref)
        # (a clipped spot loses its plateau-centroid position when the scaled image is no longer clipped)
        assert a.shape == b.shape and np.abs(a - b).max() < 3.0, f
