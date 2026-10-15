"""Evaluation metrics on hand-computed cases, preliminary labelling, store integration."""
import numpy as np
import pytest

from polony_app.ml import evaluate as ev
from polony_app.ml.common import MLError
from test_ml_support import add_analysis, new_store


def item(pred, ref, page=0, field=1, slide="S", role="red"):
    return {"pred": pred, "ref": ref, "page": page, "field": field, "slide": slide, "role": role, "radius": 5.0, "model_version": "m"}


def test_matching_is_one_to_one_and_inside_tolerance():
    pairs, up, ur = ev.match_points(np.array([[0, 0], [1, 0]]), np.array([[0.5, 0]]), 5)
    assert len(pairs) == 1 and len(up) == 1 and ur == []
    pairs, up, ur = ev.match_points(np.array([[0, 0]]), np.array([[6, 0]]), 5)
    assert pairs == [] and up == [0] and ur == [0]
    assert ev.match_points(np.zeros((0, 2)), np.array([[1, 1]]), 5) == ([], [], [0])


def test_matching_maximises_matches_not_nearest_first():
    # greedy nearest-first would pair B with R1 (distance 0) and leave A and R2 apart: 1 match; the optimum has 2
    pred = np.array([[0.0, 0.0], [5.0, 0.0]])
    ref = np.array([[5.0, 0.0], [9.9, 0.0]])
    pairs, up, ur = ev.match_points(pred, ref, 5.0)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 0), (1, 1)] and up == [] and ur == []


def test_metrics_hand_computed():
    a = item([(0, 0), (10, 10), (50, 50)], [(1, 0), (10, 12)])      # tp 2 (d=1, d=2), fp 1
    b = item([], [(5, 5), (6, 6), (7, 7)], field=2)                 # fn 3
    m = ev.metrics_of([a, b], 5)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 3)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(0.4)
    assert m["f1"] == pytest.approx(0.5)
    assert m["mean_match_distance_px"] == pytest.approx(1.5)
    assert m["count_mae"] == pytest.approx(2.0)               # |3-2| = 1, |0-3| = 3
    assert m["count_bias"] == pytest.approx(-1.0)             # (+1 - 3) / 2
    assert m["slope"] == pytest.approx(-3.0)                  # OLS: ref [2,3], pred [3,0] -> sxy -1.5 / sxx 0.5
    assert m["intercept"] == pytest.approx(9.0)               # mean(pred) - slope x mean(ref) = 1.5 + 7.5
    assert m["pearson_r"] == pytest.approx(-1.0)              # one slide: within-slide r
    assert m["fp_per_field"] == pytest.approx(0.5) and m["fn_per_field"] == pytest.approx(1.5)
    assert m["n_fields"] == 2 and m["n_reference"] == 5 and m["n_predicted"] == 3


def test_undefined_values_are_none_not_zero():
    m = ev.metrics_of([item([], [])])
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None and m["mean_match_distance_px"] is None
    assert m["slope"] is None and m["intercept"] is None and m["pearson_r"] is None and m["loa_lower"] is None
    m = ev.metrics_of([item([(0, 0)], [])])
    assert m["precision"] == 0.0 and m["recall"] is None
    assert ev.metrics_of([])["count_mae"] is None


def test_perfect_slope_and_r():
    items = [item([(i, 0)] * n, [(i, 0)] * n, field=i) for i, n in enumerate([1, 2, 3, 4, 5])]
    m = ev.metrics_of(items)
    assert m["slope"] == pytest.approx(1.0) and m["pearson_r"] == pytest.approx(1.0) and m["count_mae"] == 0.0


def many_fields(n, page, per=1):
    return [item([(i, 0)] * per, [(i, 0)] * per, page=page, field=i) for i in range(n)]


def test_preliminary_labels():
    assert ev.preliminary_reasons(5, 100, 2) and "only 5 field" in ev.preliminary_reasons(5, 100, 2)[0]
    assert any("reference event" in r for r in ev.preliminary_reasons(40, 10, 2))
    assert any("single annotator" in r for r in ev.preliminary_reasons(40, 100, 1))
    assert ev.preliminary_reasons(40, 100, 2) == []
    s = ev.summarise_config(many_fields(40, 0), 5, 2)
    assert s["per_page"]["0"]["label"] == "ok"
    s = ev.summarise_config(many_fields(40, 0), 5, 1)
    assert s["per_page"]["0"]["label"] == "preliminary"
    s = ev.summarise_config(many_fields(29, 0), 5, 3)
    assert s["per_page"]["0"]["label"] == "preliminary"


def test_sparse_page_is_never_merged_into_the_dense_totals():
    items = many_fields(40, 0, per=2) + many_fields(40, 2, per=2) + many_fields(3, 1, per=2)    # page 1: 6 events
    s = ev.summarise_config(items, 5, 2)
    assert s["per_page"]["1"]["label"] == "preliminary" and s["per_page"]["0"]["label"] == "ok"
    p = s["pooled_dense_pages"]
    assert p["pages"] == ["0", "2"] and p["excluded_sparse_pages"] == ["1"] and p["n_fields"] == 80
    s1 = ev.summarise_config(many_fields(40, 0, per=2) + many_fields(3, 1, per=2), 5, 2)
    assert s1["pooled_dense_pages"] is None          # one dense page only: the page row IS the total


def test_review_minutes_caps_idle_gaps():
    class A:
        def __init__(self, ts):
            self.ts = ts
    log = [A("2026-01-01T10:00:00Z"), A("2026-01-01T10:00:30Z"), A("2026-01-01T12:00:00Z"), A("2026-01-01T12:01:00Z")]
    assert ev.review_minutes(log) == pytest.approx((30 + 120 + 60) / 60)
    assert ev.review_minutes(log[:1]) is None and ev.review_minutes([]) is None


# ---------------------------------------------------------------- with real analyses in a store
def test_evaluate_analyses_against_final_reviewed_points(tmp_path):
    store = new_store(tmp_path)
    aid = add_analysis(store, "slide", slide_id="S1_A", reviewer="alice")
    rep = ev.evaluate_analyses(store, [aid], reference="analysis")
    u = rep["configs"]["union_before_review"]
    assert u["available"]
    f = u["by_field"]["S1_A / field 1"]            # keyed on (slide, field), never on the field number alone
    # field 1 red page: union = r1_0, r1_1 (deleted by the user), r1_2; final = r1_0, r1_2 + manual add at (50,40)
    p0 = u["per_page"]["0"]
    assert (p0["tp"], p0["fp"], p0["fn"]) == (2 + 2, 1, 1)       # field 2 adds 2 kept objects
    assert f["tp"] == 3 and f["fp"] == 1 and f["fn"] == 1 and f["n_fields"] == 2   # field 1 on both pages (green: 1 tp)
    assert p0["label"] == "preliminary" and any("only 2 field" in r for r in p0["preliminary_reasons"])
    assert any("single annotator" in r for r in p0["preliminary_reasons"])
    assert rep["n_annotators"] == 1 and rep["reviewers"] == ["alice"]
    assert rep["tolerance_px"] == 10.0 and rep["tolerances_px"] == [5.0, 10.0, 15.0]
    assert not rep["configs"]["bigfish_default"]["available"]            # no stored engine run
    assert "identical to the reference" in rep["configs"]["final_human_reviewed"]["reason"]
    # acceptance: 1 of the 3 red candidates in field 1 was rejected -> acceptance < 1
    acc = rep["acceptance"]["S1_A"]
    assert acc["red:unlabelled"]["candidates"] == 5 and acc["red:unlabelled"]["accepted"] == 4
    assert acc["human_added_rate"] == pytest.approx(1 / 6)
    md = ev.report_markdown(rep)
    assert "preliminary" in md and "per TIFF page" in md and "union_before_review" in md
    assert "5, 10, 15 px" in md and "vs this annotator's marks" in md and "@15 px" in md
    assert "count agreement per slide" in md and "model score" in md


def test_evaluate_degrades_without_engine_api(tmp_path):
    store = new_store(tmp_path)
    aid = add_analysis(store, "slide", slide_id="S1_A")

    class Thin:
        def __init__(self, s):
            self._s = s

        def view(self, a):
            return self._s.view(a)

        def exists(self, a):
            return self._s.exists(a)

        def originals_dir(self, a):
            return self._s.originals_dir(a)

    rep = ev.evaluate_analyses(Thin(store), [aid], reference="analysis")
    assert "no engine-run / raw-detection API" in rep["configs"]["spotiflow_pretrained"]["reason"]
    assert rep["configs"]["union_before_review"]["available"]


def test_roi_reference_without_marks_is_skipped_with_a_message(tmp_path):
    store = new_store(tmp_path)
    aid = add_analysis(store, "slide", slide_id="S1_A")
    rep = ev.evaluate_analyses(store, [aid], reference="roi")
    assert rep["skipped"] and "reference marks" in rep["skipped"][0]["reason"] or rep["skipped"][0]["reason"]
    assert all(not c["available"] for c in rep["configs"].values())


def test_unknown_analysis_and_bad_reference(tmp_path):
    store = new_store(tmp_path)
    with pytest.raises(MLError, match="not found"):
        ev.evaluate_analyses(store, ["nope"])
    aid = add_analysis(store, "s", slide_id="S1_A")
    with pytest.raises(MLError, match="reference must be"):
        ev.evaluate_analyses(store, [aid], reference="truth")


def test_evaluate_on_snapshot_with_a_stub_model(tmp_path):
    from polony_app.ml import dataset as ds
    from test_ml_support import Recorder
    store = new_store(tmp_path)
    aid = add_analysis(store, "s", slide_id="S1_A")
    r = ds.build_snapshot(store, tmp_path / "d", [aid], ds.BuildOptions(created_at="2026-01-01T00:00:00Z"), say=Recorder())
    # label-perfect predictor: returns the labels it is "shown" via a lookup on the image content hash
    table = {}
    for c in r.manifest.crops:
        img, lab = ds.load_crop(tmp_path / "d", r.version, c)
        table[img.tobytes()] = lab
        c.split = "train"
    from polony_app.ml.common import normalise_image
    lookup = {}
    for c in r.manifest.crops:
        img, lab = ds.load_crop(tmp_path / "d", r.version, c)
        lookup[normalise_image(img, c.norm_lo, c.norm_hi).tobytes()] = lab
    perfect = lambda x: lookup[x.tobytes()]
    res = ev.evaluate_on_snapshot(tmp_path / "d", r.version, perfect, split="train")
    assert res["metrics"]["f1"] == pytest.approx(1.0) and res["metrics"]["fp"] == 0
    empty = lambda img: np.zeros((0, 2))
    res = ev.evaluate_on_snapshot(tmp_path / "d", r.version, empty, split="train")
    assert res["metrics"]["recall"] == 0.0 and res["metrics"]["precision"] is None


# ---------------------------------------------------------------- tolerances, count agreement, keys (hand-computed)
def test_default_tolerance_is_10_and_the_triple_is_always_reported():
    assert ev.DEFAULT_TOLERANCE_PX == 10.0 and ev.TOLERANCES == (5.0, 10.0, 15.0)
    # three references, predictions 4 / 8 / 12 px away along x
    it = item([(4, 0), (108, 0), (212, 0)], [(0, 0), (100, 0), (200, 0)])
    m = ev.metrics_of([it])                                      # headline = 10 px
    assert m["tolerance_px"] == 10.0 and m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1
    bt = m["by_tolerance"]
    assert set(bt) == {"5", "10", "15"}
    assert [bt[k]["tp"] for k in ("5", "10", "15")] == [1, 2, 3]
    assert bt["5"]["recall"] == pytest.approx(1 / 3) and bt["15"]["f1"] == pytest.approx(1.0)
    assert bt["10"]["mean_match_distance_px"] == pytest.approx(6.0)
    m7 = ev.metrics_of([it], 7)                                  # a custom headline is added to, not replacing, the triple
    assert set(m7["by_tolerance"]) == {"5", "7", "10", "15"} and m7["tp"] == 1


def counts(ref, pred, slide="S", page=0):
    return [item([(i, 0)] * p, [(i, 0)] * r, field=i, slide=slide, page=page) for i, (r, p) in enumerate(zip(ref, pred))]


def test_count_agreement_slope_intercept_bias_and_limits_of_agreement():
    ref = [10, 20, 30, 40]
    # pred = 0.8 ref + 3 -> [11, 19, 27, 35]: proportional bias AND an offset (a line through the origin hides the offset)
    ca = ev.count_agreement(ref, [11, 19, 27, 35])
    assert ca["slope"] == pytest.approx(0.8) and ca["intercept"] == pytest.approx(3.0)
    assert ca["bias"] == pytest.approx(-2.0)                                 # diffs [1, -1, -3, -5]
    sd = (20 / 3) ** 0.5
    assert ca["sd_diff"] == pytest.approx(sd)
    assert ca["loa_lower"] == pytest.approx(-2 - 1.96 * sd) and ca["loa_upper"] == pytest.approx(-2 + 1.96 * sd)
    assert ca["pearson_r"] == pytest.approx(1.0)
    # pred = 1.2 ref - 4 -> [8, 20, 32, 44]
    ca = ev.count_agreement(ref, [8, 20, 32, 44])
    assert ca["slope"] == pytest.approx(1.2) and ca["intercept"] == pytest.approx(-4.0)
    assert ca["bias"] == pytest.approx(1.0)                                  # diffs [-2, 0, 2, 4]
    # pred = ref + 3: slope 1, pure offset, zero spread -> degenerate limits of agreement at the bias
    ca = ev.count_agreement(ref, [13, 23, 33, 43])
    assert ca["slope"] == pytest.approx(1.0) and ca["intercept"] == pytest.approx(3.0) and ca["bias"] == pytest.approx(3.0)
    assert ca["sd_diff"] == pytest.approx(0.0) and ca["loa_lower"] == pytest.approx(3.0) == ca["loa_upper"]
    # through the origin the same data would claim slope (sum ref*pred / sum ref^2) = 1.1: the offset is invisible
    assert sum(r * p for r, p in zip(ref, [13, 23, 33, 43])) / sum(r * r for r in ref) == pytest.approx(1.1)
    m = ev.metrics_of(counts(ref, [11, 19, 27, 35]))
    assert m["slope"] == pytest.approx(0.8) and m["intercept"] == pytest.approx(3.0) and m["bias"] == pytest.approx(-2.0)
    assert m["loa_lower"] == pytest.approx(-2 - 1.96 * sd)


def test_pooled_r_across_slides_is_between_slide_and_per_slide_r_is_reported():
    # each slide: within-slide counts are unrelated to the reference (r = 0), but the slides differ in density
    a = counts([10, 11, 12, 13], [20, 22, 22, 20], slide="A")          # r(ref, pred) = 0
    b = counts([100, 101, 102, 103], [200, 202, 202, 200], slide="B")  # r = 0
    assert ev.pearson([10, 11, 12, 13], [20, 22, 22, 20]) == pytest.approx(0.0)
    m = ev.metrics_of(a + b)
    assert m["n_slides"] == 2 and m["pearson_r_pooled"] > 0.99          # the inflated, between-slide value
    assert m["pearson_r"] is None and "between-slide (not accuracy)" in m["pearson_r_label"]
    assert m["pearson_r_per_slide"] == {"A": pytest.approx(0.0), "B": pytest.approx(0.0)}
    assert m["pearson_r_median_per_slide"] == pytest.approx(0.0)
    s = ev.summarise_config(a + b, 5, 2)
    ca = s["count_agreement"]
    assert ca["median_per_slide_r"] == pytest.approx(0.0) and ca["pooled_r"] > 0.99
    assert ca["pooled_r_label"] == "between-slide (not accuracy)"
    assert set(ca["per_slide"]) == {"A / page 0", "B / page 0"}
    assert ca["per_slide"]["A / page 0"]["slope"] == pytest.approx(0.0)
    md = ev.report_markdown({"reference": "analysis", "tolerance_px": 10.0, "tolerances_px": [5, 10, 15], "n_annotators": 2,
                             "reviewers": ["x"], "analyses": ["a"], "configs": {"cfg": dict(available=True, **s)}, "acceptance": {},
                             "review_time": [], "skipped": [], "warnings": [], "definitions": ev.DEFINITIONS})
    assert "between-slide (not accuracy)" in md and "Median within-slide r" in md


def test_by_field_is_keyed_on_slide_and_field_not_the_field_number_alone():
    items = [item([(0, 0)], [(0, 0)], field=1, slide="A"), item([], [(0, 0)], field=1, slide="B"),
             item([(0, 0)], [(0, 0)], field=2, slide="A")]
    s = ev.summarise_config(items, 10, 2)
    assert set(s["by_field"]) == {"A / field 1", "A / field 2", "B / field 1"}
    assert s["by_field"]["A / field 1"]["recall"] == 1.0 and s["by_field"]["B / field 1"]["recall"] == 0.0


def test_pairs_straddling_a_field_border_are_matched_once_per_page_not_per_field():
    # prediction (49.5) falls in field 1, its reference mark (51) in field 2: 1.5 px apart, but in different fields
    f1 = dict(item([(49.5, 10)], [], field=1), unit=("a", 0, "red"))
    f2 = dict(item([], [(51, 10)], field=2), unit=("a", 0, "red"))
    s = ev.summarise_config([f1, f2], 5, 2)
    pg = s["per_page"]["0"]
    assert (pg["tp"], pg["fp"], pg["fn"]) == (1, 0, 0)                     # per-field matching would give fp 1 + fn 1
    assert s["by_field"]["S / field 2"]["tp"] == 1 and s["by_field"]["S / field 1"]["tp"] == 0   # credited to the reference's field
    # without a shared unit (separate crops / images) nothing is matched across items
    g1, g2 = item([(49.5, 10)], [], field=1), item([], [(51, 10)], field=2)
    assert ev.metrics_of([g1, g2], 5)["tp"] == 0
    # slices never re-match: a slice of one field keeps the global matching
    assert ev.metrics_of([f2], 5)["tp"] == 1


# ---------------------------------------------------------------- Spotiflow configuration identity
def _run(rid, page, *, downscale=1, thr=0.5, norm="auto", checksum="abcdef0123456789", started="2026-01-01T00:00:00Z", box=(0, 0, 160, 120)):
    from types import SimpleNamespace as NS
    return NS(run_id=rid, status="ok", engine=NS(value="spotiflow"), engine_version="0.6.5", model_name="general",
              model_version="1", model_checksum=checksum, page=page, started_at=started,
              params={"downscale": downscale, "prob_thresh_used": thr, "normalizer": norm, "crop_to_valid": True},
              preprocessing={"crop_box_xyxy": list(box)})


class RunStore:
    """store wrapper with an engine-run / raw-detection API"""
    def __init__(self, s, runs, dets):
        self._s, self._runs, self._dets = s, runs, dets

    def __getattr__(self, k):
        return getattr(self._s, k)

    def get_engine_runs(self, aid):
        return self._runs

    def get_raw_detections(self, aid, run_id=None, valid=None):
        from types import SimpleNamespace as NS
        pts = self._dets.get(run_id, []) if run_id else [d for v in self._dets.values() for d in v]
        return [NS(x=x, y=y, score=0.9, channel_role="red", page=0, valid=True) for x, y in pts]


def test_spotiflow_configurations_are_listed_separately_never_silently_picked(tmp_path):
    store = new_store(tmp_path)
    aid = add_analysis(store, "slide", slide_id="S1_A")
    good = [(20.0, 20.0), (60.0, 50.0), (50.0, 40.0), (100.0, 20.0), (120.0, 40.0)]
    runs = [_run("r1", 0, downscale=1, thr=0.5, started="2026-01-01T00:00:00Z"),
            _run("r2", 0, downscale=4, thr=0.5, started="2026-01-02T00:00:00Z"),      # the LATEST run, but another configuration
            _run("r3", 0, downscale=1, thr=0.3, started="2026-01-03T00:00:00Z"),
            _run("r4", 0, downscale=1, thr=0.5, norm="valid_unsaturated", started="2026-01-04T00:00:00Z"),
            _run("r5", 0, downscale=1, thr=0.5, checksum="ffff000011112222", started="2026-01-05T00:00:00Z")]
    dets = {"r1": good, "r2": good[:2], "r3": good + [(30.0, 30.0)], "r4": good[:3], "r5": good[:4]}
    rep = ev.evaluate_analyses(RunStore(store, runs, dets), [aid], reference="analysis", configs=("spotiflow_pretrained",))
    names = [n for n in rep["configs"] if n.startswith("spotiflow_pretrained [")]
    assert len(names) == 5, names
    assert any("downscale=4" in n for n in names) and any("thr=0.3" in n for n in names)
    assert any("normaliser=valid_unsaturated" in n for n in names) and any("model=ffff0000" in n for n in names)
    assert "spotiflow_pretrained" not in rep["configs"]                       # no merged / "latest wins" entry
    assert sorted(rep["engine_configurations_found"]["spotiflow_pretrained"]) == sorted(names)
    assert any("5 different spotiflow_pretrained configurations" in w for w in rep["warnings"])
    n_pred = {n: rep["configs"][n]["per_page"]["0"]["n_predicted"] for n in names}
    assert sorted(n_pred.values()) == [2, 3, 4, 5, 6]                         # each configuration keeps its own predictions
    one = rep["configs"][[n for n in names if "downscale=4" in n][0]]
    assert one["configuration"]["identity"]["downscale"] == 4 and one["configuration"]["crop_boxes"]
    md = ev.report_markdown(rep)
    assert "Engine configurations found" in md and "none is preferred" in md


def test_same_configuration_runs_use_the_most_recent_and_say_so(tmp_path):
    store = new_store(tmp_path)
    aid = add_analysis(store, "slide", slide_id="S1_A")
    runs = [_run("r1", 0, started="2026-01-01T00:00:00Z"), _run("r2", 0, started="2026-01-02T00:00:00Z")]
    rep = ev.evaluate_analyses(RunStore(store, runs, {"r1": [(20.0, 20.0)], "r2": [(20.0, 20.0), (60.0, 50.0)]}), [aid],
                               reference="analysis", configs=("spotiflow_pretrained",))
    names = [n for n in rep["configs"] if n.startswith("spotiflow_pretrained [")]
    assert len(names) == 1 and rep["configs"][names[0]]["per_page"]["0"]["n_predicted"] == 2
    assert any("2 runs share the configuration" in w and "r2" in w for w in rep["warnings"])


def test_unmarked_bright_proposals_are_reported_when_raw_detections_exist(tmp_path):
    store = new_store(tmp_path)
    aid = add_analysis(store, "slide", slide_id="S1_A")
    good = [(20.0, 20.0), (60.0, 50.0), (50.0, 40.0), (100.0, 20.0), (120.0, 40.0)]
    extra = [(30.0, 45.0), (70.0, 15.0)]                                       # inside field 1, nobody marked them
    rep = ev.evaluate_analyses(RunStore(store, [_run("r1", 0)], {"r1": good + extra}), [aid], reference="analysis",
                               configs=("spotiflow_pretrained",))
    c = rep["annotation_completeness"]["per_slide"]["S1_A"]
    assert c == {"n_raw_proposals": 7, "n_unmarked": 2, "n_unmarked_bright": 2, "n_confirmed_proposals": 5}
    assert "lower bound" in rep["annotation_completeness"]["note"]
    assert "Annotation completeness" in ev.report_markdown(rep)


def test_unmarked_proposals_helper_hand_computed():
    from polony_app.ml.dataset import unmarked_proposals
    raw = [(0, 0, 0.9), (10, 0, 0.8), (20, 0, 0.7), (30, 0, 0.6), (40, 0, 0.5), (100, 0, 0.55), (200, 0, 0.1)]
    pos = [(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)]
    u = unmarked_proposals(raw, pos, pos, radius=5.0)
    assert u["n_raw_proposals"] == 7 and u["n_unmarked"] == 2 and u["n_confirmed_proposals"] == 5
    assert u["bright_score_threshold"] == pytest.approx(0.6)                  # lower quartile of [0.9 .. 0.5]
    assert u["n_unmarked_bright"] == 0                                         # 0.55 < 0.6 and 0.1 < 0.6
    u2 = unmarked_proposals(raw[:5] + [(100, 0, 0.95)], pos, pos, 5.0)
    assert u2["n_unmarked"] == 1 and u2["n_unmarked_bright"] == 1
    assert unmarked_proposals([], [], [])["n_unmarked"] == 0


# ---------------------------------------------------------------- the model is evaluated at ITS configuration
def _snapshot_with_two_pages(tmp_path):
    from polony_app.ml import dataset as ds
    from test_ml_support import Recorder
    store = new_store(tmp_path)
    aid = add_analysis(store, "s", slide_id="S1_A")
    return ds, ds.build_snapshot(store, tmp_path / "d", [aid], ds.BuildOptions(created_at="2026-01-01T00:00:00Z", downscale=4),
                                 say=Recorder())


def test_evaluate_on_snapshot_predicts_at_the_models_downscale_and_maps_back(tmp_path):
    from polony_app.ml.common import prepare_crop, to_downscaled
    ds, r = _snapshot_with_two_pages(tmp_path)
    c0 = r.manifest.crops[0]
    seen = {}
    img, lab = ds.load_crop(tmp_path / "d", r.version, c0)
    expect, lab_s, _, off = prepare_crop(img, lab, lo=c0.norm_lo, hi=c0.norm_hi, downscale=4, origin_xy=c0.box_xyxy[:2])
    assert off != (0, 0) or c0.box_xyxy[0] % 4 == 0 == c0.box_xyxy[1] % 4

    def pred(x):
        seen.setdefault("x", []).append(x)
        if x.shape == expect.shape and np.allclose(x, expect):
            return lab_s                                    # a perfect model in the downscaled frame
        return np.zeros((0, 2))
    res = ev.evaluate_on_snapshot(tmp_path / "d", r.version, pred, split=c0.split, downscale=4)
    assert all(x.dtype == np.float32 and x.shape[0] <= 120 // 4 for x in seen["x"])       # never the native image
    assert res["config"] == {"downscale": 4, "normaliser": "valid_unsaturated"}
    assert res["metrics"]["tp"] == len(lab) and res["metrics"]["fp"] == 0 and res["tolerance_px"] == 10.0
    assert set(res["metrics"]["by_tolerance"]) == {"5", "10", "15"}
    # at native resolution (wrong for this model) the same predictor sees a different input and finds nothing
    res1 = ev.evaluate_on_snapshot(tmp_path / "d", r.version, pred, split=c0.split, downscale=1)
    assert res1["metrics"]["tp"] == 0
