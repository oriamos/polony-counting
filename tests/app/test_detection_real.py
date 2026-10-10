"""Run on the one real example image (software smoke test, NOT accuracy evidence)."""
from pathlib import Path

import pytest

from polony_app.detection.pipeline import run_detection
from polony_app.detection.masks import build_valid_mask
from polony_app.imaging.ingest import ingest_files
from polony_app.imaging.io import load_channel
from polony_app.schemas import DetectionParams, DetectionRequest, Region, RegionKind

REAL = Path(__file__).resolve().parents[2] / "resources/dataset_files/test/test_img.tif"


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
    pts = lambda r: sorted((round(o.red.x), round(o.red.y)) for o in r.objects if o.red.detected)
    assert len(pts(ref)) > 5
    for f in (0.5, 0.25):
        r2 = np.clip(np.rint(red.astype(float) * f), 0, 65535).astype(red.dtype)
        res = run_detection(r2, green, [reg], DetectionRequest(red=p, green=p))
        assert pts(res) == pts(ref), f
