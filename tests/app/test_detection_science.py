"""Scientific-review fixes: debris yellow, saturated plateaus, masking, preview==full, region rule,
alignment strip, SNR bias, ambiguity beyond tolerance. Synthetic data: software checks only."""
import numpy as np
import pytest

from polony_app.detection import pipeline
from polony_app.detection.bigfish_engine import BigFishEngine
from polony_app.detection.engine import RawSpot
from polony_app.detection.matching import associate
from polony_app.detection.pipeline import run_detection
from polony_app.imaging import preprocess
from polony_app.imaging.registration import apply_shift
from polony_app.review.counts import REASON_PRIORITY, object_priority
from polony_app.schemas import (BackgroundParams, DetectionParams, DetectionRequest, ObjectClass, Region,
                                RegionKind, ReviewReason)

try:
    from tests.app import synthetic as syn
except ImportError:  # pragma: no cover
    import synthetic as syn

NOP = lambda f, m: None
NEVER = lambda: False
BG = BackgroundParams(sigma_px=40)


def rect(i, x0, y0, x1, y1, kind=RegionKind.counting):
    return Region(id=i, kind=kind, name=i, polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def req(r=5.0, md=5.0, tol=5.0, **kw):
    p = DetectionParams(spot_radius_px=r, minimum_distance_px=md, background=BG, **kw)
    return DetectionRequest(red=p, green=p, match_tolerance_px=tol)


# ------------------------------------------------------------------ 1. debris-like yellow
def _dual(red_sigma, green_sigma, amp_r, amp_g, seed):
    pts = [(60, 60), (140, 60), (60, 140), (140, 140)]
    r = syn.make_scene((200, 200), [(x, y, amp_r) for x, y in pts], [], sigma=red_sigma, seed=seed).red
    g = syn.make_scene((200, 200), [], [(x + 0.3, y - 0.2, amp_g) for x, y in pts], sigma=green_sigma, seed=seed).green
    return r, g


def test_debris_in_both_channels_is_uncertain_not_yellow():
    red, green = _dual(1.0, 1.0, 20000, 20000, seed=21)
    res = run_detection(red, green, [rect("c", 0, 0, 200, 200)], req(), NOP, NEVER)
    assert len(res.objects) == 4
    for o in res.objects:
        assert o.cls == ObjectClass.uncertain and ReviewReason.debris_like in o.review_reasons
        assert o.red.detected and o.green.detected
        assert any("debris-like" in h and "BOTH channels" in h for h in o.history)
        assert object_priority(o) is not None  # reaches the review queue


def test_genuine_extended_dual_polonies_stay_yellow():
    red, green = _dual(5.0, 5.0, 3000, 3000, seed=22)
    res = run_detection(red, green, [rect("c", 0, 0, 200, 200)], req(), NOP, NEVER)
    assert [o.cls for o in res.objects] == [ObjectClass.yellow] * 4
    assert all(ReviewReason.debris_like not in o.review_reasons for o in res.objects)


def test_debris_in_one_channel_only_stays_yellow_but_is_queued():
    red, green = _dual(1.0, 5.0, 20000, 3000, seed=23)
    res = run_detection(red, green, [rect("c", 0, 0, 200, 200)], req(), NOP, NEVER)
    assert len(res.objects) == 4
    for o in res.objects:
        assert o.cls == ObjectClass.yellow and ReviewReason.debris_like in o.review_reasons
        assert object_priority(o) is not None


def test_every_review_reason_has_a_queue_priority():
    assert set(ReviewReason) <= set(REASON_PRIORITY)
    assert REASON_PRIORITY[ReviewReason.debris_like] < REASON_PRIORITY[ReviewReason.dense_cluster]
    assert REASON_PRIORITY[ReviewReason.debris_like] > REASON_PRIORITY[ReviewReason.saturated]


# ------------------------------------------------------------------ 3. saturated plateaus
@pytest.mark.parametrize("amp,plateau", [(1.5e5, 131), (5e5, 322), (2e6, 540), (1e7, 794), (1e8, 1152)])
def test_saturated_plateau_is_one_object(amp, plateau):
    H, W, s = 160, 160, 5.0
    yy, xx = np.mgrid[:H, :W].astype(float)
    im = 1000 + amp * np.exp(-((xx - 80.3) ** 2 + (yy - 80.4) ** 2) / (2 * s * s))
    im += np.random.default_rng(0).normal(0, 30, im.shape)
    raw = np.clip(np.rint(im), 0, 65535).astype(np.uint16)
    n_sat = int((raw >= 65535).sum())
    assert abs(n_sat - plateau) / plateau < 0.25  # the reviewer's plateau sizes
    res = run_detection(raw, np.full_like(raw, 900), [], req(threshold=30.0), NOP, NEVER)
    assert len(res.objects) == 1
    o = res.objects[0]
    assert o.red.saturated and ReviewReason.saturated in o.review_reasons
    assert abs(o.x - 80.3) < 0.6 and abs(o.y - 80.4) < 0.6  # plateau centroid
    if n_sat > 700:
        assert ReviewReason.suspicious_shape in o.review_reasons  # much larger than a spot


# ------------------------------------------------------------------ 4. masking
def _border_image():
    H, W, s = 600, 220, 5.0
    ds = np.arange(-8, 8.5, 1.0)
    ys = 20 + np.arange(len(ds)) * 32
    yy, xx = np.mgrid[:H, :W].astype(float)
    im = np.full((H, W), 1000.0)
    for d, y in zip(ds, ys):
        im += 3000 * np.exp(-((xx - (100 + d)) ** 2 + (yy - y) ** 2) / (2 * s * s))
    im += np.random.default_rng(3).normal(0, 30, im.shape)
    return np.clip(np.rint(im), 0, 65535).astype(np.uint16), ds, ys


def test_mask_border_spots_outside_not_counted_inside_exact():
    raw, ds, ys = _border_image()
    p = DetectionParams(spot_radius_px=5.0, minimum_distance_px=5.0, threshold=30)
    res = run_detection(raw, np.full_like(raw, 1000), [rect("A", 0, 0, 100, 600)],
                        DetectionRequest(red=p, green=p), NOP, NEVER)
    for d, y in zip(ds, ys):
        m = [o for o in res.objects if abs(o.y - y) < 6]
        if 100 + d >= 100:
            assert not m, f"spot centred at x={100 + d} is outside the region [0,100) but was counted"
        elif 100 + d <= 99:
            assert len(m) == 1 and abs(m[0].x - (100 + d)) <= 0.1 and abs(m[0].y - y) <= 0.1


def test_threshold_from_maxima_inside_mask_only():
    # glare outside the mask must not change the automatic threshold computed inside
    sc = syn.default_scene(edge_band=25)
    valid = np.ones(sc.shape, bool)
    valid[:, :40] = False
    clean = syn.default_scene(edge_band=0)
    p = DetectionParams(spot_radius_px=3, minimum_distance_px=5, background=BG)
    e = BigFishEngine()
    a = e.estimate_threshold(np.maximum(preprocess.correct_background(sc.red, p.background, valid), 0), p, valid, sc.red)
    b = e.estimate_threshold(np.maximum(preprocess.correct_background(clean.red, p.background, valid), 0), p, valid, clean.red)
    assert a["base"] == pytest.approx(b["base"], rel=0.25)


# ------------------------------------------------------------------ 5. preview == full
def test_preview_window_objects_equal_full_run_objects():
    sc = syn.default_scene()
    r = req(3.0, 5.0)
    regions = [rect("c", 0, 0, 400, 300)]
    pipeline.clear_threshold_cache()
    win = (90, 40, 260, 140)
    prev = run_detection(sc.red, sc.green, regions, r, NOP, NEVER, window=win)  # cold cache: threshold from full image
    full = run_detection(sc.red, sc.green, regions, r, NOP, NEVER)
    key = lambda o: (round(o.x, 4), round(o.y, 4), o.cls.value, tuple(x.value for x in o.review_reasons))
    inwin = sorted(key(o) for o in full.objects if win[0] <= o.x < win[2] and win[1] <= o.y < win[3])
    assert inwin and sorted(key(o) for o in prev.objects) == inwin
    thr = lambda res, ch: [n for n in res.notes if n.startswith(ch + ": ") and "LoG threshold" in n][0].split(";")[1]
    assert thr(prev, "red") == thr(full, "red") and thr(prev, "green") == thr(full, "green")
    assert any("identical to the full run" in n for n in prev.notes)


def test_threshold_cache_hit_and_key_sensitivity():
    sc = syn.default_scene()
    pipeline.clear_threshold_cache()
    regions = [rect("c", 0, 0, 400, 300)]
    run_detection(sc.red, sc.green, regions, req(3.0, 5.0), NOP, NEVER)
    assert len(pipeline._THR_CACHE) == 2  # red + green
    run_detection(sc.red, sc.green, regions, req(3.0, 5.0, threshold_scale=0.5), NOP, NEVER)
    assert len(pipeline._THR_CACHE) == 2  # scale does not change the automatic value
    run_detection(sc.red, sc.green, [rect("c", 0, 0, 300, 300)], req(3.0, 5.0), NOP, NEVER)
    assert len(pipeline._THR_CACHE) == 4  # other mask => other key


# ------------------------------------------------------------------ 7. one region rule in the pipeline
def test_keep_drop_uses_half_open_rule_at_region_edges():
    # region [10,100) x ... : x=99.6 is inside, x=9.6 is outside (pixel-centre rule; no rounding)
    H, W, s = 120, 200, 3.0
    yy, xx = np.mgrid[:H, :W].astype(float)
    im = np.full((H, W), 1000.0)
    for x, y in ((99.6, 40.0), (9.6, 80.0), (50.0, 60.0)):
        im += 3000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))
    raw = np.clip(np.rint(im), 0, 65535).astype(np.uint16)
    p = DetectionParams(spot_radius_px=3, minimum_distance_px=5, threshold=20)
    res = run_detection(raw, np.full_like(raw, 1000), [rect("A", 10, 0, 100, 120)],
                        DetectionRequest(red=p, green=p), NOP, NEVER)
    xs = sorted(round(o.x, 1) for o in res.objects)
    assert len(res.objects) == 2 and xs[0] == 50.0 and abs(xs[1] - 99.6) < 0.15
    assert all(o.region_id == "A" for o in res.objects)


# ------------------------------------------------------------------ 9. alignment strip
def test_empty_green_with_shift_gives_no_green_spots_in_strip():
    rng = np.random.default_rng(0)
    H = W = 400
    yy, xx = np.mgrid[:H, :W].astype(float)

    def ch(pts):
        im = 1000 + rng.normal(0, 30, (H, W))
        for x, y in pts:
            im += 2000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 50)
        return np.clip(np.rint(im), 0, 65535).astype(np.uint16)

    red = ch([(60 + 80 * i, 60 + 80 * j) for i in range(5) for j in range(5)])
    green = apply_shift(ch([]), 2, -3)
    reg = [rect("A", 0, 0, W, H)]
    res = run_detection(red, green, reg, req(), NOP, NEVER, alignment=(2, -3))
    assert not [o for o in res.objects if o.green.detected]
    assert any("green alignment shift" in n for n in res.notes)
    # a real green spot just inside the covered area is still detected with the shift applied
    g2 = apply_shift(ch([(200, 200)]), 2, -3)
    res2 = run_detection(red, g2, reg, req(), NOP, NEVER, alignment=(2, -3))
    assert any(o.green.detected for o in res2.objects)


# ------------------------------------------------------------------ 10. background and saturated pixels
def test_saturated_pixels_have_zero_weight_in_background():
    H, W = 200, 300
    rng = np.random.default_rng(5)
    im = np.full((H, W), 1000.0) + rng.normal(0, 10, (H, W))
    im[:, :180] = 1e6  # saturated slide area (60% of the image: the robust pass cannot reject it)
    raw = np.clip(np.rint(im), 0, 65535).astype(np.uint16)
    p = BackgroundParams(sigma_px=30, clip_negative=False)
    corr = preprocess.correct_background(raw, p)
    med_near = float(np.median(corr[:, 185:215]))
    assert abs(med_near) < 3.0  # next to the rim: unbiased
    biased = raw.astype(np.float32) - preprocess.gaussian_background(raw.astype(np.float32), 30.0)
    assert float(np.median(biased[:, 185:215])) < -100  # without the weight-0 rule the background is lifted


# ------------------------------------------------------------------ 11. SNR bias
def test_snr_is_unbiased_at_zero_amplitude():
    eng = BigFishEngine()
    s, sig = 5.0, 30.0
    rng = np.random.default_rng(1)
    H = W = 900
    yy, xx = np.mgrid[:H, :W].astype(float)
    pts = [(50 + i * 80.0, 50 + j * 80.0) for i in range(11) for j in range(11)]
    med = {}
    for A in (0, 150):
        im = np.zeros((H, W))
        for x, y in pts:
            im += A * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))
        im += rng.normal(0, sig, im.shape)
        im = im.astype(np.float32)
        coords = np.array([[int(y), int(x)] for x, y in pts])
        sp = eng._measure(np.maximum(im, 0), None, coords, np.zeros(len(coords), bool),
                          DetectionParams(spot_radius_px=s), None, 1.0, measure=im)
        med[A] = float(np.median([q.snr for q in sp]))
    assert abs(med[0]) < 0.4  # was ~ +1.5 with the 3x3 maximum
    assert abs(med[150] - 5.0) < 0.6


# ------------------------------------------------------------------ 12. ambiguity beyond tolerance
def _S(x, y):
    return RawSpot(x=x, y=y, peak=100.0, background=0.0, snr=10.0, saturated=False)


def test_alternative_partner_just_beyond_tolerance_is_ambiguous():
    out = associate([_S(0, 0)], [_S(5.9, 0), _S(-6.1, 0)], 6.0)
    assert len(out) == 2 and all(o.cls == ObjectClass.uncertain for o in out)
    assert all(ReviewReason.ambiguous_match in o.review_reasons and o.ambiguous_with for o in out)
    # a far alternative (> tol + margin) does not make it ambiguous
    out = associate([_S(0, 0)], [_S(2.0, 0), _S(-9.5, 0)], 6.0)
    assert sorted(o.cls.value for o in out) == ["green", "yellow"]


# ------------------------------------------------------------------ SCIENCE-FIX-2
def _pos(res):
    return sorted((round(o.x, 3), round(o.y, 3), o.cls.value) for o in res.objects)


def test_auto_threshold_is_independent_of_brightness_scale_and_offset():
    """B1: thresholds are in noise units, so exposure (x0.5, x0.25) or a constant offset does not
    change which spots are detected."""
    sc = syn.default_scene(seed=3)
    r = [rect("c", 0, 0, sc.shape[1], sc.shape[0])]
    base = run_detection(sc.red, sc.green, r, req(3.0, 5.0), NOP, NEVER)
    assert len(base.objects) >= 9
    for f, off in ((0.5, 0), (0.25, 0), (0.5, 700), (0.25, 300)):
        pipeline.clear_threshold_cache()
        red = np.clip(np.rint(sc.red.astype(float) * f + off), 0, 65535).astype(np.uint16)
        green = np.clip(np.rint(sc.green.astype(float) * f + off), 0, 65535).astype(np.uint16)
        res = run_detection(red, green, r, req(3.0, 5.0), NOP, NEVER)
        assert [(o.cls.value, round(o.x), round(o.y)) for o in res.objects] == \
               [(o.cls.value, round(o.x), round(o.y)) for o in base.objects], (f, off)


def test_noise_unit_threshold_values():
    sc = syn.default_scene(seed=3)
    p = DetectionParams(spot_radius_px=3, minimum_distance_px=5, background=BG)
    e = BigFishEngine()
    meas = preprocess.correct_background(sc.red, p.background.model_copy(update={"clip_negative": False}), None)
    a = e.estimate_threshold(np.maximum(meas, 0), p, None, sc.red, meas)
    b = e.estimate_threshold(np.maximum(meas * 0.25, 0), p, None, None, meas * 0.25)
    assert a["noise_sigma"] == pytest.approx(4 * b["noise_sigma"], rel=1e-3)
    assert a["base"] == b["base"] and a["base"] >= 5.0
    # manual threshold is in the same unit: absolute LoG threshold = value x sigma
    spots = e.detect(np.maximum(meas, 0), p.model_copy(update={"threshold": 12.0}), None, raw=sc.red, measure=meas)
    assert e.last_info["threshold"] == 12.0
    assert e.last_info["threshold_abs"] == pytest.approx(12.0 * a["noise_sigma"], rel=1e-3)
    assert all(s.threshold == pytest.approx(e.last_info["threshold_abs"]) for s in spots)
    # flat image: sigma 0 is guarded
    z = e.estimate_threshold(np.zeros((60, 60), np.float32), p, None)
    assert z["base"] is None and z["noise_sigma"] is None


def test_manual_threshold_preview_window_uses_full_image_sigma():
    sc = syn.default_scene(seed=3)
    r = [rect("c", 0, 0, sc.shape[1], sc.shape[0])]
    q = req(3.0, 5.0, threshold=15.0)
    pipeline.clear_threshold_cache()
    full = run_detection(sc.red, sc.green, r, q, NOP, NEVER)
    pipeline.clear_threshold_cache()
    win = (0, 0, 200, 150)
    prev = run_detection(sc.red, sc.green, r, q, NOP, NEVER, window=win)
    inside = [o for o in full.objects if win[0] <= o.x < win[2] and win[1] <= o.y < win[3]]
    assert _pos(prev) == _pos(type("R", (), {"objects": inside}))


def _touching(ds, amp=2e5, s=5.0):
    H, W = 200, 200 * len(ds)
    yy, xx = np.mgrid[:H, :W].astype(float)
    im = np.full((H, W), 1000.0)
    x0s = []
    for k, d in enumerate(ds):
        x0 = 100 + 200 * k - d / 2
        x0s.append(x0)
        for x in (x0, x0 + d):
            im += amp * np.exp(-((xx - x) ** 2 + (yy - 100.3) ** 2) / (2 * s * s))
    im += np.random.default_rng(0).normal(0, 30, im.shape)
    return np.clip(np.rint(im), 0, 65535).astype(np.uint16), x0s


def test_touching_saturated_polonies_are_flagged_or_resolved():
    """B3: merged touching plateaus (10-18 px apart) stay ONE object but get suspicious_shape;
    >= 20 px apart they are two objects."""
    ds = [10, 14, 18, 20, 24]
    raw, x0s = _touching(ds)
    p = DetectionParams(spot_radius_px=5.0, minimum_distance_px=5.0, threshold=30.0)
    res = run_detection(raw, np.full_like(raw, 900), [], DetectionRequest(red=p, green=p), NOP, NEVER)
    for d, x0 in zip(ds, x0s):
        m = [o for o in res.objects if x0 - 30 < o.x < x0 + d + 30]
        if d <= 18:
            assert len(m) == 1 and ReviewReason.suspicious_shape in m[0].review_reasons, d
        else:
            assert len(m) == 2, d
            assert all(ReviewReason.suspicious_shape not in o.review_reasons for o in m), d


def test_plateau_merged_spot_has_no_weak_signal_or_near_threshold():
    """B4: SNR is undefined for a clipped plateau -> weak_signal / near_threshold are skipped."""
    raw, _ = _touching([0], amp=2e5)
    p = DetectionParams(spot_radius_px=5.0, minimum_distance_px=5.0, threshold=30.0)
    res = run_detection(raw, np.full_like(raw, 900), [], DetectionRequest(red=p, green=p), NOP, NEVER)
    m = [o for o in res.objects if o.red.detected]
    assert len(m) == 1
    rs = m[0].review_reasons
    assert ReviewReason.saturated in rs
    assert ReviewReason.weak_signal not in rs and ReviewReason.near_threshold not in rs


def test_empty_channel_gets_pipeline_warning():
    """B5: a channel without spots must not silently report its noise maxima as a normal result."""
    sc = syn.make_scene((300, 400), [(100, 100, 3000), (250, 200, 3000), (320, 80, 3000)], [], sigma=3.0, seed=7)
    r = [rect("c", 0, 0, 400, 300)]
    res = run_detection(sc.red, sc.green, r, req(3.0, 5.0), NOP, NEVER)
    warn = [n for n in res.notes if n.startswith("green:") and "WARNING" in n]
    assert warn and any(("noise regime" in n) or ("debris-like" in n) for n in warn)
    # the sparse real channel still finds its spots (noise-regime elbow -> 5 noise-unit floor)
    assert sum(1 for o in res.objects if o.red.detected) >= 3
    assert not [n for n in res.notes if n.startswith("WARNING red:")]
