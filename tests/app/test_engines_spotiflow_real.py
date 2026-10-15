"""Opt-in integration test with the REAL Spotiflow 'general' model (downloads ~88 MB into a temp cache).
Run with:  POLONY_SPOTIFLOW_E2E=1 pytest tests/app/test_engines_spotiflow_real.py -s"""
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("POLONY_SPOTIFLOW_E2E") != "1",
                                reason="set POLONY_SPOTIFLOW_E2E=1 to run the real Spotiflow model (downloads 88 MB)")

try:
    from tests.app import synthetic as syn
except ImportError:  # pragma: no cover
    import synthetic as syn

REAL_TIF = Path(__file__).resolve().parents[2] / "resources" / "research_tiffs" / "29-09-2026_T7_AB_AI-A.tif"


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from polony_app.engines import SpotiflowEngine
    cache = Path(os.environ.get("POLONY_MODEL_DIR") or tmp_path_factory.mktemp("sfcache"))
    eng = SpotiflowEngine(cache_dir=cache)
    ok, why = eng.available()
    assert ok, why
    return eng


def _run(engine, img, page=0, role="red", valid=None):
    t = time.perf_counter()
    res = engine.detect_page(img, analysis_id="e2e", page=page, channel_role=role, valid_mask=valid, params={},
                             image_sha256={}, progress=lambda f, m: None, cancelled=lambda: False)
    return res, time.perf_counter() - t


def test_real_model_on_synthetic_300(engine):
    sc = syn.make_scene((300, 300), [(60, 50, 4000), (140, 80, 3500), (230, 200, 4000), (90, 240, 3800)],
                        [(60, 50, 4000)], sigma=2.0, noise=15.0)
    res, dt = _run(engine, sc.red)
    r = res.run
    assert r.status == "ok" and r.model_name == "general" and len(r.model_checksum) == 64
    assert r.preprocessing["input_dtype"] == "uint16" and r.device in ("cpu", "cuda", "mps")
    assert r.params["prob_thresh_used"] is not None
    assert res.detections and all(0.0 < d.score <= 1.0 for d in res.detections)
    found = [min(np.hypot(d.x - x, d.y - y) for d in res.detections) for x, y, _ in sc.red_spots]
    print(f"\nsynthetic 300x300: {len(res.detections)} detections in {dt:.1f} s on {r.device}; "
          f"distance to the 4 true spots: {np.round(found, 2).tolist()}; model {r.model_version} sha {r.model_checksum[:12]}")
    assert max(found) < 2.5
    # second call hits the in-process model cache and the on-disk cache (no download)
    res2, dt2 = _run(engine, sc.red)
    assert [(d.x, d.y) for d in res2.detections] == [(d.x, d.y) for d in res.detections]       # deterministic
    assert res2.run.model_checksum == r.model_checksum


@pytest.mark.skipif(not REAL_TIF.exists(), reason="local research TIFF not present")
def test_real_model_on_research_crop(engine):
    import tifffile
    with tifffile.TiffFile(REAL_TIF) as tf:
        page = tf.pages[0].asarray()
    h, w = page.shape
    y0, x0 = h // 2 - 500, w // 2 - 500
    crop = np.ascontiguousarray(page[y0:y0 + 1000, x0:x0 + 1000])
    res, dt = _run(engine, crop)
    print(f"\nresearch crop 1000x1000 {crop.dtype}: {len(res.detections)} detections ({res.run.n_valid} valid) "
          f"in {dt:.1f} s on {res.run.device}; prob_thresh {res.run.params['prob_thresh_used']}")
    assert res.run.status == "ok" and res.detections
    assert all(0 <= d.x < 1000 and -1 <= d.y < 1001 for d in res.detections)


def test_real_model_downscale4_on_25px_blobs(engine):
    """~25 px blobs (sigma 5.3): downscale 4 must find them; centre error reported (subpix is off => ~<= d/2 px)."""
    spots = [(120.3, 100.7, 4000), (300.6, 180.2, 3600), (210.4, 330.9, 4200), (420.1, 400.5, 3800),
             (90.8, 420.2, 4000), (400.0, 90.0, 3900)]
    sc = syn.make_scene((512, 512), spots, [spots[0]], sigma=5.3, noise=15.0)
    out = {}
    for d in (1, 2, 4):
        res = engine.detect_page(sc.red, analysis_id="e2e", page=0, channel_role="red", valid_mask=None,
                                 params={"downscale": d, "prob_thresh": 0.1 if d > 1 else None}, image_sha256={},
                                 progress=lambda f, m: None, cancelled=lambda: False)
        err = [min((np.hypot(q.x - x, q.y - y) for q in res.detections), default=np.inf) for x, y, _ in spots]
        out[d] = (len(res.detections), np.round(err, 2).tolist())
        print(f"\ndownscale {d}: {len(res.detections)} detections; centre error per true spot: {out[d][1]}; "
              f"preprocessing={res.run.preprocessing['downscale']} trimmed={res.run.preprocessing['trimmed_px']}")
        if d == 4:
            found = [e for e in out[d][1] if np.isfinite(e)]
            assert len(found) >= 4 and max(found) <= 4.5
            assert res.run.preprocessing["downscale"] == 4 and res.run.params["downscale"] == 4


def test_real_model_valid_unsaturated_is_independent_of_the_counting_area_origin(engine):
    """Real model, synthetic 25 px blobs on a page with a saturated rim: the detections are (nearly) identical when the
    valid area is shifted by 1-3 px (crop origin and block grid are anchored), and the normaliser ignores the rim."""
    rng = np.random.default_rng(3)
    xs, ys = np.meshgrid(np.arange(80, 920, 70), np.arange(80, 720, 70))
    spots = [(float(x + rng.uniform(-8, 8)), float(y + rng.uniform(-8, 8)), 3500.0) for x, y in zip(xs.ravel(), ys.ravel())]
    sc = syn.make_scene((800, 1000), spots, [spots[0]], sigma=5.3, noise=15.0)
    img = sc.red.copy()
    img[:, :25] = 65535                                       # saturated rim outside the counting area
    found = {}
    for sh in (0, 1, 2, 3):
        valid = np.zeros(img.shape, bool)
        valid[40 + sh:760 + sh, 60 + sh:960 + sh] = True
        res = engine.detect_page(img, analysis_id="e2e", page=0, channel_role="red", valid_mask=valid,
                                 params={"downscale": 4, "prob_thresh": 0.1, "normalizer": "valid_unsaturated"},
                                 image_sha256={}, progress=lambda f, m: None, cancelled=lambda: False)
        assert res.run.preprocessing["normalizer_stats"]["hi"] < 20000      # not set by the 65535 rim
        found[sh] = np.array([[d.x, d.y] for d in res.detections if d.valid]).reshape(-1, 2)
    for sh in (1, 2, 3):
        a, b = found[0], found[sh]
        d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
        miss = int((d.min(1) > 2.0).sum()) if len(b) else len(a)
        assert miss <= max(2, 0.03 * len(a)), (sh, miss, len(a))
