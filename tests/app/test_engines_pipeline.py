"""run_detection with engine='bigfish' (unchanged), 'spotiflow' and 'hybrid' using a fake Spotiflow."""
import json
from pathlib import Path

import numpy as np
import pytest

from polony_app.detection.pipeline import DetectionCancelled, run_detection
from polony_app.engines import BigFishAdapter, EngineError
from polony_app.schemas import (BackgroundParams, ConsensusStatus, DetectionParams, DetectionRequest, EngineChoice,
                                EngineName, Region, RegionKind, ReviewReason)

try:
    from tests.app import synthetic as syn
    from tests.app.engines_fakes import FakeSpotiflow, local_max_points
except ImportError:  # pragma: no cover
    import synthetic as syn
    from engines_fakes import FakeSpotiflow, local_max_points

DP = DetectionParams(spot_radius_px=3, minimum_distance_px=5, background=BackgroundParams(sigma_px=40))
NOP = lambda f, m: None
NEVER = lambda: False
FULL = [Region(id="c", kind=RegionKind.counting, name="c", polygon=[(0, 0), (400, 0), (400, 300), (0, 300)])]
GOLD = Path(__file__).with_name("golden_bigfish_scene.json")


def req(engine=EngineChoice.bigfish, **kw):
    return DetectionRequest(red=DP, green=DP, match_tolerance_px=5.0, engine=engine, **kw)


def run(sc, request, regions=FULL, **kw):
    return run_detection(sc.red, sc.green, list(regions), request, NOP, NEVER, analysis_id="a1",
                         image_sha256={"f": "x"}, **kw)


# ------------------------------------------------------------------ bigfish path unchanged
def test_bigfish_objects_identical_to_stored_old_result():
    sc = syn.default_scene()
    res = run(sc, req())
    gold = json.loads(GOLD.read_text())["objects"]
    new = [json.loads(o.model_dump_json()) for o in res.objects]
    assert len(new) == len(gold) == 13
    for n, g in zip(new, gold):
        # new optional two-engine fields default to None; everything the old path produced must be identical
        for ch in ("red", "green"):
            for k in list(n[ch]):
                if k not in g[ch]:
                    assert n[ch][k] is None, k
                    del n[ch][k]
        assert n == g
    assert not any(d.consensus_status for o in res.objects for d in (o.red, o.green))


def test_bigfish_path_records_runs_and_raw_detections():
    sc = syn.default_scene()
    res = run(sc, req())
    assert [(r.engine, r.page, r.channel_role, r.status) for r in res.engine_runs] == [
        (EngineName.bigfish, 0, "red", "ok"), (EngineName.bigfish, 1, "green", "ok")]
    r0 = res.engine_runs[0]
    assert r0.engine_version == "0.6.2" and r0.preprocessing["background"]["method"] == r0.params["background"]["method"]
    assert r0.image_sha256 == {"f": "x"} and r0.n_raw == r0.n_valid == 9
    d = [x for x in res.raw_detections if x.channel_role == "red"]
    assert len(d) == 9 and d[0].id == "bf-p0-000001" and {x.score_kind for x in d} == {"bigfish_log_response_noise_units"}
    assert all(x.score > 0 and x.valid for x in d)
    assert len({x.id for x in res.raw_detections}) == len(res.raw_detections)
    assert "spotiflow" not in res.package_versions


def test_bigfish_raw_detections_flag_excluded():
    sc = syn.default_scene()
    regions = FULL + [Region(id="x", kind=RegionKind.exclusion, name="x",
                             polygon=[(0, 0), (120, 0), (120, 300), (0, 300)])]
    res = run(sc, req(), regions)
    red = [x for x in res.raw_detections if x.channel_role == "red"]
    # Big-FISH never searches inside an exclusion (mask-restricted), so its raw output there is empty; any
    # detection on the +1 px rim is flagged by the same point rule
    assert red and all(x.x > 118 for x in red)
    assert all(x.valid == (x.x >= 120) for x in red)
    assert all(o.x >= 120 for o in res.objects)


def test_pages_and_roles_follow_the_caller_not_the_order():
    sc = syn.default_scene()
    res = run(sc, req(), pages=(3, 2), roles=(None, None))
    assert [(r.page, r.channel_role) for r in res.engine_runs] == [(3, None), (2, None)]
    assert res.raw_detections[0].id.startswith("bf-p3-")


def test_bigfish_adapter_standalone_matches_pipeline_spots():
    sc = syn.default_scene()
    res = BigFishAdapter().detect_page(sc.red, analysis_id="a", page=0, channel_role="red", valid_mask=None,
                                       params=DP.model_dump(), image_sha256={})
    assert res.run.n_raw == 9 and res.run.engine_version and res.detections[0].score_kind.startswith("bigfish")
    assert len(res.extra["spots"]) == 9
    assert res.run.params["spot_radius_px"] == 3


# ------------------------------------------------------------------ hybrid
def fake_points(drop=(), add=()):
    """Fake model: finds the bright blobs of the array it receives, drops those near `drop` (x, y) and adds
    `add` (x, y). The crop is the whole image here (counting region = full image => no offset)."""
    n = {"call": 0}

    def fn(img):
        first = n["call"] == 0          # the pipeline runs red (page 0) first, then green: edits apply to red only
        n["call"] += 1
        pts, prob = local_max_points(img)
        if not first:
            return pts, prob
        keep = [i for i, (y, x) in enumerate(pts) if not any(abs(x - dx) < 3 and abs(y - dy) < 3 for dx, dy in drop)]
        pts, prob = pts[keep], prob[keep]
        if add:
            pts = np.vstack([pts, [[y, x] for x, y in add]])
            prob = np.concatenate([prob, np.full(len(add), 0.55)])
        return pts, prob
    return fn


def test_hybrid_end_to_end(monkeypatch, tmp_path):
    sc = syn.default_scene()
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=fake_points(drop=[(60, 50)], add=[(30.0, 270.0)]))
    res = run(sc, req(EngineChoice.hybrid), spotiflow_engine=fs.engine())
    objs = res.objects
    base = run(sc, req())
    # per page execution, original dtype preserved
    assert len(fs.calls) == 2 and all(c["dtype"] == np.uint16 for c in fs.calls)
    assert np.array_equal(fs.calls[0]["values"], sc.red) and np.array_equal(fs.calls[1]["values"], sc.green)
    # same objects as Big-FISH plus the Spotiflow-only one; classes of the shared ones unchanged
    assert len(objs) == len(base.objects) + 1
    bf_only = [o for o in objs if o.red.consensus_status == ConsensusStatus.bigfish_only]
    sf_only = [o for o in objs if o.red.consensus_status == ConsensusStatus.spotiflow_only]
    assert len(bf_only) == 1 and len(sf_only) == 1
    o = bf_only[0]
    assert abs(o.x - 60) < 1.5 and ReviewReason.bigfish_only in o.review_reasons
    assert o.red.bigfish_raw_id and o.red.spotiflow_raw_id is None and o.red.bigfish_score > 0
    s = sf_only[0]
    assert (s.x, s.y) == (30.0, 270.0)                         # Spotiflow position kept exactly
    assert ReviewReason.spotiflow_only in s.review_reasons and s.cls.value in ("red", "uncertain")
    assert s.red.spotiflow_score == pytest.approx(0.55) and s.red.spotiflow_raw_id and s.red.bigfish_raw_id is None
    # measured evidence at the Spotiflow coordinates on the same corrected image (flat background: weak)
    assert s.red.peak_intensity is not None and s.red.local_background is not None
    assert s.red.snr is not None and s.red.snr < 5 and ReviewReason.weak_signal in s.review_reasons
    # agreement: Big-FISH spot (position / peak / snr as in the Big-FISH run) + spotiflow fields, no extra reasons
    ag = [x for x in objs if x.red.consensus_status == ConsensusStatus.agreement]
    assert len(ag) >= 7
    for x in ag:
        assert x.red.spotiflow_x is not None and x.red.match_distance_px is not None and x.red.match_distance_px <= 4
        assert x.red.bigfish_x == x.red.x and x.red.bigfish_y == x.red.y
        assert ReviewReason.bigfish_only not in x.review_reasons and ReviewReason.spotiflow_only not in x.review_reasons
        b = min(base.objects, key=lambda q: (q.x - x.x) ** 2 + (q.y - x.y) ** 2)
        assert (b.red.x, b.red.y, b.red.peak_intensity, b.red.snr) == (x.red.x, x.red.y, x.red.peak_intensity, x.red.snr)
    # green: the fake finds all green spots
    assert all(x.green.consensus_status == ConsensusStatus.agreement for x in objs if x.green.detected)
    # runs / raw detections
    eng = {(r.engine, r.page) for r in res.engine_runs}
    assert eng == {(EngineName.bigfish, 0), (EngineName.bigfish, 1), (EngineName.spotiflow, 0), (EngineName.spotiflow, 1)}
    assert len({r.run_id for r in res.engine_runs}) == 4
    sfd = [d for d in res.raw_detections if d.engine == EngineName.spotiflow]
    bfd = [d for d in res.raw_detections if d.engine == EngineName.bigfish]
    assert {d.run_id for d in sfd} <= {r.run_id for r in res.engine_runs} and len(bfd) == 17
    assert len({(d.run_id, d.id) for d in res.raw_detections}) == len(res.raw_detections)
    assert res.package_versions["spotiflow"] and any("consensus" in n for n in res.notes)
    sm = {(c.page, c.channel_role): c for c in res.consensus}
    assert (sm[(0, "red")].bigfish_only, sm[(0, "red")].spotiflow_only) == (1, 1)
    assert sm[(1, "green")].agreement == 8


def test_consensus_tolerance_follows_the_spotiflow_downscale_unless_explicit(monkeypatch, tmp_path):
    from polony_app.schemas import SpotiflowParams
    sc = syn.default_scene()
    none = lambda a: (np.zeros((0, 2)), np.zeros(0))

    count = iter(range(100))

    def note(**kw):
        fs = FakeSpotiflow(monkeypatch, tmp_path / f"t{next(count)}", points_fn=none)
        r = run(sc, req(EngineChoice.hybrid, **kw), spotiflow_engine=fs.engine())
        return next(n for n in r.notes if "consensus (tolerance" in n)
    assert "tolerance 4 px" in note()                                                          # downscale 1: the 4 px default
    n8 = note(spotiflow=SpotiflowParams(downscale=8))
    assert "tolerance 8 px" in n8 and "raised" in n8
    assert "tolerance 3 px" in note(spotiflow=SpotiflowParams(downscale=8), consensus_tolerance_px=3.0)   # explicit wins
    assert "tolerance 4 px" in note(spotiflow=SpotiflowParams(downscale=2))


def test_hybrid_is_deterministic(monkeypatch, tmp_path):
    sc = syn.default_scene()
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=fake_points(drop=[(60, 50)], add=[(30.0, 270.0)]))
    a = run(sc, req(EngineChoice.hybrid), spotiflow_engine=fs.engine())
    fs.points_fn = fake_points(drop=[(60, 50)], add=[(30.0, 270.0)])       # fresh call counter
    b = run(sc, req(EngineChoice.hybrid), spotiflow_engine=fs.engine())
    assert [o.model_dump() for o in a.objects] == [o.model_dump() for o in b.objects]
    assert [d.model_dump(exclude={"run_id"}) for d in a.raw_detections] == [d.model_dump(exclude={"run_id"}) for d in b.raw_detections]


def test_hybrid_respects_counting_area_and_flags_raw(monkeypatch, tmp_path):
    sc = syn.default_scene()
    regions = [Region(id="c", kind=RegionKind.counting, name="c", polygon=[(0, 0), (400, 0), (400, 300), (0, 300)]),
               Region(id="x", kind=RegionKind.exclusion, name="x", polygon=[(0, 0), (120, 0), (120, 300), (0, 300)])]
    # the model input is the valid-area bbox + 32 px: x offset 120 - 32 = 88, so crop (30, 270) = full (118, 270)
    # (inside the exclusion) and crop (112, 150) = full (200, 150)
    fs = FakeSpotiflow(monkeypatch, tmp_path, points_fn=fake_points(add=[(30.0, 270.0), (112.0, 150.0)]))
    res = run(sc, req(EngineChoice.hybrid), regions, spotiflow_engine=fs.engine())
    assert all(o.x >= 120 for o in res.objects)
    sf_raw = [d for d in res.raw_detections if d.engine == EngineName.spotiflow and d.channel_role == "red"]
    inval = [d for d in sf_raw if not d.valid]
    assert inval and all(d.x < 120 and d.excluded_reason for d in inval)
    assert any(d.valid and (d.x, d.y) == (200.0, 150.0) for d in sf_raw)       # kept raw + valid
    assert any((o.x, o.y) == (200.0, 150.0) for o in res.objects)
    assert not any((o.x, o.y) == (118.0, 270.0) for o in res.objects)           # excluded one never becomes an object
    r = next(r for r in res.engine_runs if r.engine == EngineName.spotiflow and r.page == 0)
    assert r.n_raw > r.n_valid


@pytest.mark.parametrize("fail", ["load", "predict"])
def test_hybrid_spotiflow_failure_falls_back_to_bigfish(monkeypatch, tmp_path, fail):
    sc = syn.default_scene()
    kw = {"fail_load": RuntimeError("weights broken")} if fail == "load" else {"fail_predict": ValueError("kaboom")}
    fs = FakeSpotiflow(monkeypatch, tmp_path, **kw)
    base = run(sc, req())
    res = run(sc, req(EngineChoice.hybrid), spotiflow_engine=fs.engine())
    assert [o.model_dump() for o in res.objects] == [o.model_dump() for o in base.objects]
    failed = [r for r in res.engine_runs if r.status == "failed"]
    assert len(failed) == 2 and all(r.engine == EngineName.spotiflow and r.error for r in failed)
    assert ("weights broken" if fail == "load" else "kaboom") in failed[0].error
    assert any(n.startswith("WARNING") and "Spotiflow failed" in n for n in res.notes)
    assert not any(d.engine == EngineName.spotiflow for d in res.raw_detections)
    assert len([r for r in res.engine_runs if r.engine == EngineName.bigfish and r.status == "ok"]) == 2


def test_hybrid_unavailable_spotiflow_is_a_failed_run(monkeypatch, tmp_path):
    from polony_app.engines import SpotiflowEngine, spotiflow_adapter as sa
    monkeypatch.setattr(sa, "_have", lambda n: n not in ("spotiflow", "torch"))
    sc = syn.default_scene()
    res = run(sc, req(EngineChoice.hybrid), spotiflow_engine=SpotiflowEngine(tmp_path))
    fr = [r for r in res.engine_runs if r.status == "failed"]
    assert len(fr) == 2 and "not installed" in fr[0].error
    assert len(res.objects) == 13


def test_spotiflow_only_failure_raises_clear_error(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path, fail_predict=ValueError("kaboom"))
    with pytest.raises(EngineError, match="kaboom"):
        run(syn.default_scene(), req(EngineChoice.spotiflow), spotiflow_engine=fs.engine())


def test_spotiflow_only_mode(monkeypatch, tmp_path):
    sc = syn.default_scene()
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    res = run(sc, req(EngineChoice.spotiflow), spotiflow_engine=fs.engine())
    assert {r.engine for r in res.engine_runs} == {EngineName.spotiflow}
    assert len(res.objects) >= 12
    for o in res.objects:
        ev = o.red if o.red.detected else o.green
        assert ev.consensus_status is None and ev.spotiflow_x is not None and ev.bigfish_x is None
        assert ev.peak_intensity is not None and ev.snr is not None
        assert ReviewReason.spotiflow_only not in o.review_reasons and ReviewReason.bigfish_only not in o.review_reasons
    assert [o.cls.value for o in res.objects].count("yellow") == 4


def test_hybrid_cancellation(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    fs.n_tiles_fake = 5
    sc = syn.default_scene()
    with pytest.raises(DetectionCancelled):      # cancelled once the model is running: stops between tiles
        run_detection(sc.red, sc.green, FULL, req(EngineChoice.hybrid), NOP, lambda: bool(fs.calls),
                      spotiflow_engine=fs.engine())
    assert len(fs.calls) == 1
    with pytest.raises(DetectionCancelled):
        run_detection(syn.default_scene().red, syn.default_scene().green, FULL, req(EngineChoice.hybrid), NOP,
                      lambda: True, spotiflow_engine=fs.engine())


def test_progress_monotonic_hybrid(monkeypatch, tmp_path):
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    seen = []
    sc = syn.default_scene()
    run_detection(sc.red, sc.green, FULL, req(EngineChoice.hybrid), lambda f, m: seen.append(f), NEVER,
                  spotiflow_engine=fs.engine())
    assert seen[0] == 0.0 and seen[-1] == 1.0 and all(0 <= f <= 1 for f in seen)
    assert max(f for f, _ in zip(seen, seen) if f < 1) <= 0.93
    assert seen == sorted(seen)


def test_hybrid_window_preview(monkeypatch, tmp_path):
    sc = syn.default_scene()
    fs = FakeSpotiflow(monkeypatch, tmp_path)
    res = run(sc, req(EngineChoice.hybrid), window=(50, 30, 200, 140), spotiflow_engine=fs.engine())
    assert res.objects and all(50 <= o.x < 200 and 30 <= o.y < 140 for o in res.objects)
    assert any("preview window with Spotiflow" in n for n in res.notes)
    # the model input is limited to the (padded) window: never larger than the page
    assert all(c["shape"][0] <= 300 and c["shape"][1] <= 400 for c in fs.calls)


def test_spotiflow_only_measurement_matches_bigfish_measurement_code():
    from polony_app.detection.bigfish_engine import BigFishEngine
    from polony_app.detection.two_engine import measure_spots_at
    from polony_app.imaging.preprocess import correct_background
    sc = syn.default_scene()
    eng = BigFishEngine()
    measured = correct_background(sc.red, DP.background.model_copy(update={"clip_negative": False}), None)
    corrected = np.maximum(measured, 0.0)
    (bf_spot,) = [s for s in eng.detect(corrected, DP, None, raw=sc.red, measure=measured)
                  if abs(s.x - 60) < 2 and abs(s.y - 50) < 2]
    (m,) = measure_spots_at(eng, DP, corrected, measured, np.ones(sc.red.shape, bool), sc.red, [60.3], [50.2], 10.0)
    assert (m.x, m.y) == (60.3, 50.2)                       # Spotiflow position kept, not refined
    assert m.peak == pytest.approx(bf_spot.peak, rel=0.03) and m.snr > 20
    assert m.background == pytest.approx(bf_spot.background, abs=5.0) and not m.saturated
