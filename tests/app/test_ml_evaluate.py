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
    assert m["slope"] == pytest.approx(6 / 13)                # (2*3 + 3*0) / (2^2 + 3^2)
    assert m["pearson_r"] == pytest.approx(-1.0)              # ref [2,3] vs pred [3,0]
    assert m["fp_per_field"] == pytest.approx(0.5) and m["fn_per_field"] == pytest.approx(1.5)
    assert m["n_fields"] == 2 and m["n_reference"] == 5 and m["n_predicted"] == 3


def test_undefined_values_are_none_not_zero():
    m = ev.metrics_of([item([], [])])
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None and m["mean_match_distance_px"] is None
    assert m["slope"] is None and m["pearson_r"] is None
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
    f = u["by_field"]["1"]
    # field 1 red page: union = r1_0, r1_1 (deleted by the user), r1_2; final = r1_0, r1_2 + manual add at (50,40)
    p0 = u["per_page"]["0"]
    assert (p0["tp"], p0["fp"], p0["fn"]) == (2 + 2, 1, 1)       # field 2 adds 2 kept objects
    assert f["tp"] == 3 and f["fp"] == 1 and f["fn"] == 1 and f["n_fields"] == 2   # field 1 on both pages (green: 1 tp)
    assert p0["label"] == "preliminary" and any("only 2 field" in r for r in p0["preliminary_reasons"])
    assert any("single annotator" in r for r in p0["preliminary_reasons"])
    assert rep["n_annotators"] == 1 and rep["reviewers"] == ["alice"]
    assert not rep["configs"]["bigfish_default"]["available"]            # no stored engine run
    assert "identical to the reference" in rep["configs"]["final_human_reviewed"]["reason"]
    # acceptance: 1 of the 3 red candidates in field 1 was rejected -> acceptance < 1
    acc = rep["acceptance"]["S1_A"]
    assert acc["red:unlabelled"]["candidates"] == 5 and acc["red:unlabelled"]["accepted"] == 4
    assert acc["human_added_rate"] == pytest.approx(1 / 6)
    md = ev.report_markdown(rep)
    assert "preliminary" in md and "per TIFF page" in md and "union_before_review" in md


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
    perfect = lambda img: table[img.tobytes()]
    res = ev.evaluate_on_snapshot(tmp_path / "d", r.version, perfect, split="train")
    assert res["metrics"]["f1"] == pytest.approx(1.0) and res["metrics"]["fp"] == 0
    empty = lambda img: np.zeros((0, 2))
    res = ev.evaluate_on_snapshot(tmp_path / "d", r.version, empty, split="train")
    assert res["metrics"]["recall"] == 0.0 and res["metrics"]["precision"] is None
