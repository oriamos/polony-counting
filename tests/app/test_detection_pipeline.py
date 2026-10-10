import numpy as np
import pytest

from polony_app.detection.pipeline import DetectionCancelled, run_detection
from polony_app.imaging.ingest import IngestError
from polony_app.schemas import (BackgroundParams, DetectionParams, DetectionRequest, ObjectClass, Region,
                                RegionKind, ReviewReason)

try:
    from tests.app import synthetic as syn
except ImportError:  # pragma: no cover
    import synthetic as syn

DP = DetectionParams(spot_radius_px=3, minimum_distance_px=5, background=BackgroundParams(sigma_px=40))
REQ = DetectionRequest(red=DP, green=DP, match_tolerance_px=5.0)
NOP = lambda f, m: None
NEVER = lambda: False


def rect(i, x0, y0, x1, y1, kind=RegionKind.counting):
    return Region(id=i, kind=kind, name=i, polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def run(sc, regions=(), req=REQ, **kw):
    return run_detection(sc.red, sc.green, list(regions), req, NOP, NEVER, **kw)


def test_yellow_association_end_to_end():
    sc = syn.default_scene()
    res = run(sc, [rect("c", 0, 0, 400, 300)])
    cls = [o.cls for o in res.objects]
    assert cls.count(ObjectClass.yellow) == 4 and cls.count(ObjectClass.red) == 5
    assert cls.count(ObjectClass.green) == 4 and ObjectClass.uncertain not in cls
    for o in res.objects:
        assert o.region_id == "c" and o.origin.value == "auto" and not o.reviewed
        if o.cls == ObjectClass.yellow:
            assert o.association_distance_px < 2.0
    assert [o.id for o in res.objects] == [f"o{i:06d}" for i in range(1, len(res.objects) + 1)]
    assert set(res.package_versions) >= {"bigfish", "numpy", "scipy", "skimage", "tifffile"}
    assert res.package_versions["bigfish"] == "0.6.2"
    # known coordinates (< 1.5 px) for the red-only truth
    truth = [(x, y) for x, y, _ in sc.red_spots[:5]]
    for x, y in truth:
        assert min(np.hypot(o.x - x, o.y - y) for o in res.objects) < 1.5


def test_pipeline_is_deterministic():
    sc = syn.default_scene()
    a, b = run(sc), run(sc)
    assert [o.model_dump() for o in a.objects] == [o.model_dump() for o in b.objects]


def test_region_assignment_and_out_of_mask_drop():
    sc = syn.default_scene()
    regs = [rect("left", 0, 0, 200, 300), rect("right", 200, 0, 400, 300),
            rect("x", 0, 0, 120, 300, RegionKind.exclusion)]
    res = run(sc, regs)
    assert res.objects and all(o.x >= 118 for o in res.objects)
    assert {o.region_id for o in res.objects} == {"left", "right"}
    for o in res.objects:
        assert (o.region_id == "left") == (o.x < 200)


def test_saturated_bright_edge_outside_region_does_not_appear():
    sc = syn.default_scene(edge_band=30)
    res = run(sc, [rect("c", 45, 0, 400, 300)])
    assert all(o.x > 44 for o in res.objects)
    cls = [o.cls for o in res.objects]
    assert cls.count(ObjectClass.yellow) + cls.count(ObjectClass.red) >= 8


def test_no_counting_region_warns_and_masks_big_saturated_area():
    sc = syn.default_scene()
    sc.red[:, :150] = 65535  # large saturated area
    sc.green[:, :150] = 65535
    res = run(sc)
    assert any("no counting region" in n for n in res.notes)
    assert all(o.x > 140 for o in res.objects)


def test_tiled_pipeline_equals_untiled():
    sc = syn.default_scene()
    t = DetectionParams(**{**DP.model_dump(), "tile_size": 100, "tile_overlap_px": 64})
    a = run(sc)
    b = run(sc, req=DetectionRequest(red=t, green=t, match_tolerance_px=5.0))
    assert [(o.id, o.cls, round(o.x, 3), round(o.y, 3)) for o in a.objects] == \
           [(o.id, o.cls, round(o.x, 3), round(o.y, 3)) for o in b.objects]


def test_window_preview_subset_of_full():
    sc = syn.default_scene()
    full = run(sc)
    win = run(sc, window=(90, 40, 260, 140))
    assert win.objects and all(90 <= o.x < 260 and 40 <= o.y < 140 for o in win.objects)
    pos = {(round(o.x), round(o.y)) for o in full.objects}
    assert all((round(o.x), round(o.y)) in pos for o in win.objects)
    assert any("preview window" in n for n in win.notes)
    with pytest.raises(ValueError):
        run(sc, window=(10, 10, 10, 50))


def test_progress_and_cancellation():
    sc = syn.default_scene()
    seen = []
    run_detection(sc.red, sc.green, [], REQ, lambda f, m: seen.append(f), NEVER)
    assert seen[0] == 0.0 and seen[-1] == 1.0 and seen == sorted(seen)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(DetectionCancelled):
        run_detection(sc.red, sc.green, [], REQ, NOP, cancel)
    with pytest.raises(DetectionCancelled):
        run_detection(sc.red, sc.green, [], REQ, NOP, lambda: True)


def test_shape_mismatch_raises_ingest_error():
    with pytest.raises(IngestError):
        run_detection(np.zeros((10, 10), np.uint16), np.zeros((10, 11), np.uint16), [], REQ, NOP, NEVER)


def reasons_at(res, x, y, tol=3):
    o = min(res.objects, key=lambda o: np.hypot(o.x - x, o.y - y))
    assert np.hypot(o.x - x, o.y - y) < tol
    return o, set(o.review_reasons)


def test_review_reasons_from_measurements():
    red = [(100, 100, 3000), (200, 100, 90000), (4, 150, 3000), (100, 200, 3000), (108, 200, 3000), (300, 100, 120)]
    sc = syn.make_scene((300, 400), red, [], seed=11)
    res = run(sc)
    assert reasons_at(res, 100, 100)[1] == set()
    assert ReviewReason.saturated in reasons_at(res, 200, 100)[1]
    assert ReviewReason.near_edge in reasons_at(res, 4, 150)[1]
    assert ReviewReason.dense_cluster in reasons_at(res, 100, 200)[1]
    assert ReviewReason.dense_cluster in reasons_at(res, 108, 200)[1]
    # weak_signal is a measured SNR bound: with an absurdly high bound everything is weak, with 0 nothing is
    from polony_app.detection.review_reasons import ReviewConfig
    hi = run(sc, review=ReviewConfig(weak_snr=1e9))
    # (saturated plateau-merged spots are exempt: SNR is undefined for a clipped peak)
    assert all(ReviewReason.weak_signal in o.review_reasons for o in hi.objects
               if ReviewReason.saturated not in o.review_reasons)
    assert any(ReviewReason.saturated in o.review_reasons and ReviewReason.weak_signal not in o.review_reasons
               for o in hi.objects)
    lo = run(sc, review=ReviewConfig(weak_snr=0.0))
    assert not any(ReviewReason.weak_signal in o.review_reasons for o in lo.objects)


def test_debris_like_is_flag_only():
    sc = syn.make_scene((200, 200), [(50, 100, 20000), (150, 100, 3000)], [], seed=2)
    sc2 = syn.make_scene((200, 200), [(50, 100, 20000)], [], sigma=1.0, seed=2)
    big = syn.make_scene((200, 200), [(150, 100, 3000)], [], sigma=5.0, seed=2)
    sc.red = np.maximum(sc2.red, big.red)
    p = DetectionParams(spot_radius_px=5, minimum_distance_px=5, background=BackgroundParams(sigma_px=40))
    res = run(sc, req=DetectionRequest(red=p, green=p))
    small, flags = reasons_at(res, 50, 100)
    assert ReviewReason.debris_like in flags and small.cls == ObjectClass.red  # kept, only flagged
    assert ReviewReason.debris_like not in reasons_at(res, 150, 100)[1]


def test_ambiguous_pair_in_pipeline():
    red = [(100, 100, 3000)]
    green = [(101.5, 100, 3000), (100, 102.5, 3000)]
    sc = syn.make_scene((200, 200), red, green, seed=5)
    p = DetectionParams(spot_radius_px=2, minimum_distance_px=2, background=BackgroundParams(sigma_px=40))
    res = run(sc, req=DetectionRequest(red=p, green=p, match_tolerance_px=5.0))
    assert res.objects
    for o in res.objects:
        if ReviewReason.ambiguous_match in o.review_reasons:
            assert o.cls == ObjectClass.uncertain and o.ambiguous_with
