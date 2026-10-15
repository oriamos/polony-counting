import numpy as np
import pytest

from polony_app.detection.bigfish_engine import BigFishEngine
from polony_app.detection.engine import SpotEngine
from polony_app.imaging.preprocess import correct_background
from polony_app.schemas import BackgroundParams, DetectionParams

try:
    from tests.app import synthetic as syn
except ImportError:  # pragma: no cover
    import synthetic as syn

P = DetectionParams(spot_radius_px=3, minimum_distance_px=5, background=BackgroundParams(sigma_px=40))


def corrected(arr, p=P, valid=None):
    return correct_background(arr, p.background, valid)


def detect(arr, p=P, valid=None, **kw):
    e = BigFishEngine()
    measured = correct_background(arr, p.background.model_copy(update={"clip_negative": False}), valid)
    return e.detect(np.maximum(measured, 0), p, valid, raw=arr, measure=measured, **kw), e


def match_truth(spots, truth, tol=1.5):
    out = []
    for x, y, _ in truth:
        d = [np.hypot(s.x - x, s.y - y) for s in spots]
        out.append(min(d) if d else np.inf)
    return out


def test_engine_protocol_and_version():
    e = BigFishEngine()
    assert e.name == "bigfish" and e.version() == "0.6.2"
    assert isinstance(e, SpotEngine)


def test_known_coordinates_recovered_and_no_extras():
    sc = syn.default_scene()
    for arr, truth in ((sc.red, sc.red_spots), (sc.green, sc.green_spots)):
        spots, _ = detect(arr)
        assert len(spots) == len(truth)
        assert max(match_truth(spots, truth)) < 1.5
        assert all(s.snr and s.snr > 10 for s in spots)
        assert all(not s.saturated for s in spots)
        assert all(abs(s.background) < 0.2 * s.peak for s in spots)  # residual of the broad bump only


def test_auto_threshold_equals_bigfish_detect_spots_threshold():
    from bigfish.detection import detect_spots
    sc = syn.default_scene()
    img = np.maximum(corrected(sc.red), 0)
    p = DetectionParams(spot_radius_px=3, minimum_distance_px=3)
    ours = BigFishEngine().auto_threshold(img.copy(), p, None)
    _, thr = detect_spots(img, threshold=None, return_threshold=True, voxel_size=(1, 1), spot_radius=(3, 3))
    assert ours == pytest.approx(thr)


def test_threshold_scale_and_manual_threshold():
    sc = syn.default_scene()
    n1 = len(detect(sc.red)[0])
    e = BigFishEngine()
    img = corrected(sc.red)
    hi = e.detect(img, P.model_copy(update={"threshold_scale": 1000.0}), None)
    assert len(hi) == 0 and e.last_info["threshold"] == pytest.approx(1000 * e.last_info["auto_threshold"])
    man = e.detect(img, P.model_copy(update={"threshold": 1e9}), None)
    assert man == []
    assert len(e.detect(img, P.model_copy(update={"threshold": 5.0}), None)) >= n1


def test_saturation_flag_uses_raw_dtype_max():
    sc = syn.saturated_spot_scene()
    assert sc.red.max() == 65535
    spots, _ = detect(sc.red)
    by_x = {round(s.x): s for s in spots}
    assert by_x[60].saturated and not by_x[140].saturated


def test_masked_bright_edge_creates_no_spots():
    sc = syn.default_scene(edge_band=25)
    valid = np.ones(sc.shape, bool)
    valid[:, :40] = False
    spots, e = detect(sc.red, valid=valid)
    assert all(s.x >= 40 for s in spots)
    assert len(spots) == len(sc.red_spots)  # the masked rim did not add spots nor bias the threshold
    assert max(match_truth(spots, sc.red_spots)) < 1.5


def test_spots_outside_mask_dropped_and_empty_mask():
    sc = syn.default_scene()
    valid = np.ones(sc.shape, bool)
    valid[:, :100] = False  # spot at x=60 and 80 excluded
    spots, _ = detect(sc.red, valid=valid)
    assert spots and all(s.x >= 100 for s in spots)
    assert detect(sc.red, valid=np.zeros(sc.shape, bool))[0] == []


@pytest.mark.parametrize("tile,overlap", [(100, 64), (128, 20), (97, 64)])
def test_tiling_equals_untiled(tile, overlap):
    sc = syn.make_scene((300, 400), [(100, 100, 3000), (200.0, 150, 3000), (199, 99, 2500), (50, 297, 3000), (350, 3, 3000),
                                      (288, 200, 3000)], [], seed=9)
    whole, e0 = detect(sc.red)
    p = P.model_copy(update={"tile_size": tile, "tile_overlap_px": overlap})
    tiled, e1 = detect(sc.red, p)
    assert e1.last_info["n_tiles"] > 1
    assert e1.last_info["threshold"] == e0.last_info["threshold"]  # global, not per tile
    key = lambda s: (round(s.y, 3), round(s.x, 3))
    assert [key(s) for s in tiled] == [key(s) for s in whole]
    assert len(whole) == 6  # none lost or duplicated on tile boundaries


def test_tiling_with_mask_equals_untiled():
    sc = syn.default_scene(edge_band=20)
    valid = np.ones(sc.shape, bool); valid[:, :30] = False
    a, _ = detect(sc.red, valid=valid)
    b, _ = detect(sc.red, P.model_copy(update={"tile_size": 90}), valid=valid)
    assert [(round(s.x, 3), round(s.y, 3)) for s in a] == [(round(s.x, 3), round(s.y, 3)) for s in b]


def test_deterministic():
    sc = syn.default_scene()
    a, _ = detect(sc.red)
    b, _ = detect(sc.red)
    assert [(s.x, s.y, s.peak, s.snr) for s in a] == [(s.x, s.y, s.peak, s.snr) for s in b]


def test_no_spots_on_pure_noise_does_not_crash():
    img = np.random.default_rng(0).normal(0, 5, (100, 100)).astype(np.float32).clip(0)
    e = BigFishEngine()
    e.detect(img, P, None)  # may find noise maxima; must not raise
    assert "threshold" in e.last_info


def test_dense_decomposition_does_not_crash_and_warns():
    sc = syn.default_scene()
    e = BigFishEngine()
    spots = e.detect(corrected(sc.red), P.model_copy(update={"dense_decomposition": True}), None, raw=sc.red)
    assert len(spots) >= len(sc.red_spots)
    assert any("dense decomposition" in n for n in e.last_info["notes"])


def test_scale_ratio_separates_compact_from_extended():
    sc = syn.make_scene((150, 150), [(40, 75, 20000)], [], sigma=1.0, seed=2)
    big = syn.make_scene((150, 150), [(100, 75, 3000)], [], sigma=5.0, seed=2)
    p = DetectionParams(spot_radius_px=5, minimum_distance_px=5, background=BackgroundParams(sigma_px=40))
    small_spots, _ = detect(sc.red, p)
    big_spots, _ = detect(big.red, p)
    assert small_spots and small_spots[0].scale_ratio >= 1.0
    assert big_spots and big_spots[0].scale_ratio < 1.0


def test_elongation_measure():
    sc = syn.make_scene((150, 150), [(50, 75, 3000), (54.5, 75, 3000)], [(100, 75, 3000)], sigma=3.0, seed=4)
    merged, _ = detect(sc.red)
    single, _ = detect(sc.green)
    assert len(merged) == 1 and merged[0].elongation > 1.3
    assert single[0].elongation < 1.2
