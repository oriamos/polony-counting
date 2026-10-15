"""Pure review-engine tests (no FastAPI)."""
import pytest

from fakes import counting, exclusion, make_state as _make_state, obj, rect
from polony_app.review import engine
from polony_app.review.counts import REASON_PRIORITY, compute_counts, review_queue
from polony_app.review.state import ReviewError
from polony_app.schemas import ObjectClass, Origin, ReportingMode, ReviewReason


def make_state(*a, **k):
    """fakes.make_state + what a real detection run records (request + input signature)."""
    st = _make_state(*a, **k)
    if st.analysis.provenance.detection_ran_at is not None:
        st.analysis.provenance.detection_request = st.analysis.detection_request.model_copy(deep=True)
        st.detection_sig = engine.input_signature(st.analysis)
    return st


def live(st):
    return [o for o in st.analysis.objects if not o.deleted]


def act(st, t, **p):
    return engine.apply_action(st, t, p)


def inspect_and_approve(st, rid="c1"):
    act(st, "region_inspected", region_id=rid, inspected=True)
    act(st, "approve_region", region_id=rid)


def test_add_delete_undo_redo_roundtrip():
    st = make_state([obj("o1", 20, 20)], [counting()])
    act(st, "add", x=30, y=30, cls="green")
    assert len(live(st)) == 2
    new = [o for o in st.analysis.objects if o.origin == Origin.manual][0]
    assert new.region_id == "c1" and new.reviewed
    act(st, "delete", object_ids=["o1"])
    assert [o.id for o in live(st)] == [new.id]
    assert st.analysis.objects[0].deleted  # soft delete, still present
    engine.undo(st)
    engine.undo(st)
    assert [o.id for o in st.analysis.objects] == ["o1"]
    assert st.analysis.objects[0].model_dump() == obj("o1", 20, 20).model_copy(update={"region_id": "c1"}).model_dump()
    assert (st.analysis.undo_depth, st.analysis.redo_depth) == (0, 2)
    engine.redo(st)
    engine.redo(st)
    assert len(live(st)) == 1 and live(st)[0].id == new.id


def test_new_action_clears_redo():
    st = make_state([obj("o1", 20, 20)], [counting()])
    act(st, "delete", object_ids=["o1"])
    engine.undo(st)
    assert st.analysis.redo_depth == 1
    act(st, "confirm", object_ids=["o1"])
    assert st.analysis.redo_depth == 0
    with pytest.raises(ReviewError) as e:
        engine.redo(st)
    assert e.value.code == "nothing_to_redo"


def test_move_reclassify_set_origin_and_history():
    st = make_state([obj("o1", 20, 20)], [counting()])
    act(st, "move", object_id="o1", x=25, y=26)
    o = st.analysis.objects[0]
    assert (o.x, o.y, o.origin, o.reviewed) == (25, 26, Origin.edited, True)
    act(st, "reclassify", object_ids=["o1"], cls="yellow")
    assert o.cls == ObjectClass.yellow
    assert len(o.history) == 2 and "moved from" in o.history[0] and "red -> yellow" in o.history[1]
    engine.undo(st)
    assert st.analysis.objects[0].cls == ObjectClass.red


def test_move_updates_region_id_and_refuses_outside():
    st = make_state([obj("o1", 20, 20)], [counting("c1", rect(5, 5, 60, 60)), counting("c2", rect(70, 5, 150, 60))])
    assert st.analysis.objects[0].region_id == "c1"
    act(st, "move", object_id="o1", x=100, y=20)
    assert st.analysis.objects[0].region_id == "c2"
    with pytest.raises(ReviewError) as e:
        act(st, "move", object_id="o1", x=65, y=20)  # between regions
    assert e.value.code == "outside_counting_region"
    with pytest.raises(ReviewError) as e:
        act(st, "add", x=999, y=1, cls="red")
    assert e.value.code == "out_of_image"


def test_merge_and_split():
    st = make_state([obj("r", 20, 20, ObjectClass.red), obj("g", 22, 20, ObjectClass.green)], [counting()])
    act(st, "merge", object_ids=["r", "g"])
    lv = live(st)
    assert len(lv) == 1 and lv[0].cls == ObjectClass.yellow and (lv[0].x, lv[0].y) == (21, 20)
    assert lv[0].red.detected and lv[0].green.detected and lv[0].origin == Origin.edited
    with pytest.raises(ReviewError):
        act(st, "merge", object_ids=["r"])  # ids must be >=2
    mid = lv[0].id
    act(st, "split", object_id=mid, positions=[[20, 20], [24, 20]], classes=["red", "green"])
    lv = live(st)
    assert sorted(o.cls.value for o in lv) == ["green", "red"]
    engine.undo(st)
    assert [o.id for o in live(st)] == [mid]


def test_counts_both_modes_and_uncertain_excluded():
    objs = [obj("a", 10, 10, ObjectClass.red), obj("b", 12, 10, ObjectClass.red), obj("c", 14, 10, ObjectClass.green),
            obj("d", 16, 10, ObjectClass.yellow), obj("e", 18, 10, ObjectClass.uncertain), obj("f", 20, 10, ObjectClass.red)]
    st = make_state(objs, [counting()])
    act(st, "delete", object_ids=["f"])
    c = compute_counts(st.analysis)["slide"]
    assert (c.red_only, c.green_only, c.yellow, c.uncertain) == (2, 1, 1, 1)
    assert (c.red_positive, c.green_positive, c.total_objects_exclusive) == (3, 2, 4)
    assert c.red_only + c.green_only + c.yellow == c.total_objects_exclusive  # uncertain not included
    st.analysis.reporting_mode = ReportingMode.channel_inclusive
    assert compute_counts(st.analysis)["slide"].reporting_mode == ReportingMode.channel_inclusive
    regs = compute_counts(st.analysis)["regions"]
    assert len(regs) == 1 and regs[0].scope == "c1" and regs[0].red_only == 2


def test_counts_exclude_objects_outside_counting_regions():
    st = make_state([obj("a", 10, 10), obj("b", 110, 20)], [counting(), exclusion()])
    assert st.analysis.objects[1].region_id is None  # inside exclusion
    assert compute_counts(st.analysis)["slide"].red_only == 1


def test_review_queue_priority_and_ties():
    R = ReviewReason
    objs = [
        obj("edge", 10, 10, review_reasons=[R.near_edge]),
        obj("dense", 10, 30, review_reasons=[R.dense_cluster]),
        obj("weak2", 50, 30, review_reasons=[R.weak_signal]),
        obj("weak1", 20, 30, review_reasons=[R.near_threshold]),
        obj("sat", 90, 90, review_reasons=[R.saturated, R.near_edge]),
        obj("amb", 100, 100, ObjectClass.uncertain, review_reasons=[R.ambiguous_match]),
        obj("shape", 30, 60, review_reasons=[R.suspicious_shape]),
        obj("fine", 30, 70),
        obj("done", 30, 80, review_reasons=[R.weak_signal], reviewed=True),
    ]
    st = make_state(objs, [counting()])
    q = review_queue(st.analysis)
    assert [i["object_id"] for i in q] == ["amb", "sat", "weak1", "weak2", "dense", "shape", "edge"]
    assert q[2]["priority"] == q[3]["priority"] == REASON_PRIORITY[R.weak_signal]


def test_accept_and_approve_gates():
    st = make_state([obj("a", 10, 10), obj("u", 30, 30, ObjectClass.uncertain)], [counting()])
    with pytest.raises(ReviewError) as e:
        act(st, "accept_region", region_id="c1")
    assert e.value.code == "region_not_inspected"
    with pytest.raises(ReviewError) as e:
        act(st, "approve_region", region_id="c1")
    assert e.value.code == "region_not_inspected"
    act(st, "region_inspected", region_id="c1", inspected=True)
    act(st, "accept_region", region_id="c1")
    assert st.analysis.objects[0].reviewed and not st.analysis.objects[1].reviewed  # uncertain not accepted
    with pytest.raises(ReviewError) as e:
        act(st, "approve_region", region_id="c1")
    assert e.value.code == "unresolved_uncertain"
    act(st, "reclassify", object_ids=["u"], cls="green")
    act(st, "approve_region", region_id="c1")
    r = st.analysis.regions[0]
    assert r.approved and r.approval_fingerprint and r.approved_at


@pytest.mark.parametrize("edit", [
    lambda st: act(st, "add", x=50, y=50, cls="red"),
    lambda st: act(st, "move", object_id="a", x=12, y=12),
    lambda st: act(st, "delete", object_ids=["a"]),
    lambda st: act(st, "reclassify", object_ids=["a"], cls="green"),
    lambda st: act(st, "merge", object_ids=["a", "b"]),
    lambda st: act(st, "split", object_id="a", positions=[[10, 10], [14, 10]], classes=["red", "red"]),
])
def test_any_object_edit_clears_that_regions_approval_only(edit):
    st = make_state([obj("a", 10, 10), obj("b", 30, 10), obj("z", 130, 100)],
                    [counting("c1", rect(5, 5, 60, 60)), counting("c2", rect(100, 70, 150, 110))])
    for rid in ("c1", "c2"):
        inspect_and_approve(st, rid)
    edit(st)
    r = {r.id: r for r in st.analysis.regions}
    assert not r["c1"].approved and r["c2"].approved
    assert r["c1"].approval_fingerprint is None


def test_confirm_does_not_clear_approval_and_undo_restores_it():
    st = make_state([obj("a", 10, 10)], [counting()])
    inspect_and_approve(st)
    act(st, "confirm", object_ids=["a"])
    assert st.analysis.regions[0].approved
    act(st, "move", object_id="a", x=11, y=11)
    assert not st.analysis.regions[0].approved
    engine.undo(st)  # exact state restored -> approval valid again
    assert st.analysis.regions[0].approved


def test_region_edit_invalidates_changed_and_overlapped():
    st = make_state([obj("a", 10, 10), obj("b", 130, 100)],
                    [counting("c1", rect(5, 5, 60, 60)), counting("c2", rect(100, 70, 150, 110))])
    inspect_and_approve(st, "c1"); inspect_and_approve(st, "c2")
    # unchanged geometry (rename only): approvals survive, detection not stale
    regs = [r.model_copy() for r in st.analysis.regions]
    regs[0].name = "renamed"
    engine.set_regions(st, regs)
    assert all(r.approved for r in st.analysis.regions) and not st.analysis.provenance.stale_detection
    # new exclusion overlapping c1 only
    regs = [r.model_copy() for r in st.analysis.regions] + [exclusion("x1", rect(40, 40, 70, 70))]
    engine.set_regions(st, regs)
    r = {r.id: r for r in st.analysis.regions}
    assert not r["c1"].approved and r["c2"].approved
    assert st.analysis.provenance.stale_detection
    # client-supplied approval flags are ignored
    regs = [x.model_copy(update={"approved": True}) for x in st.analysis.regions]
    engine.set_regions(st, regs)
    assert not {r.id: r for r in st.analysis.regions}["c1"].approved
    # object inside new exclusion loses its region
    assert st.analysis.objects[0].region_id == "c1"  # (10,10) is outside x1
    # removing the exclusion again also changes fingerprints of overlapped regions
    st.detection_sig = engine.input_signature(st.analysis)  # (as after a re-run)
    engine.recompute_stale(st)
    inspect_and_approve(st, "c1")
    engine.set_regions(st, [r for r in st.analysis.regions if r.id != "x1"])
    assert not {r.id: r for r in st.analysis.regions}["c1"].approved


def test_region_edit_undo():
    st = make_state([obj("a", 10, 10)], [counting()])
    inspect_and_approve(st)
    engine.set_regions(st, [counting("c1", rect(5, 5, 50, 50))])
    assert not st.analysis.regions[0].approved
    engine.undo(st)
    assert st.analysis.regions[0].polygon[2] == (155, 115) and st.analysis.regions[0].approved


def test_invalid_regions_rejected():
    st = make_state([], [])
    with pytest.raises(ReviewError):
        engine.set_regions(st, [counting("c1", [(0, 0), (1, 1)])])
    with pytest.raises(ReviewError):
        engine.set_regions(st, [counting("c1", rect(0, 0, 10, 10)), counting("c1", rect(20, 20, 30, 30))])
    with pytest.raises(ReviewError):
        engine.set_regions(st, [counting("c1", [(0, 0), (5, 5), (10, 10)])])  # zero area


def test_unapprove_and_uninspect():
    st = make_state([obj("a", 10, 10)], [counting()])
    inspect_and_approve(st)
    act(st, "unapprove_region", region_id="c1")
    assert not st.analysis.regions[0].approved
    inspect_and_approve(st)
    act(st, "region_inspected", region_id="c1", inspected=False)
    assert not st.analysis.regions[0].approved


def test_finalize_gates_freeze_and_reopen():
    st = make_state([obj("a", 10, 10)], [counting()])
    with pytest.raises(ReviewError) as e:
        engine.finalize(st)
    assert e.value.code == "regions_not_approved"
    inspect_and_approve(st)
    engine.finalize(st)
    a = st.analysis
    assert a.finalized_at and a.status.value == "finalized"
    for t, p in [("add", dict(x=1, y=1, cls="red")), ("delete", dict(object_ids=["a"])),
                 ("unapprove_region", dict(region_id="c1"))]:
        with pytest.raises(ReviewError) as e:
            act(st, t, **p)
        assert e.value.code == "analysis_finalized"
    with pytest.raises(ReviewError):
        engine.undo(st)
    with pytest.raises(ReviewError):
        engine.set_regions(st, [])
    assert compute_counts(a)["slide"].is_final
    engine.reopen(st)
    assert a.finalized_at is None and a.status.value == "in_review"
    assert a.regions[0].approved  # approvals kept ...
    act(st, "move", object_id="a", x=12, y=12)
    assert not a.regions[0].approved  # ... until an edit invalidates them
    with pytest.raises(ReviewError):
        engine.finalize(st)


def test_finalize_requires_mapping_nonstale_and_regions():
    st = make_state([obj("a", 10, 10)], [])
    with pytest.raises(ReviewError) as e:
        engine.finalize(st)
    assert e.value.code == "no_counting_region"
    st = make_state([obj("a", 10, 10)], [counting()], mapped=False)
    st.analysis.regions[0].approved = True
    codes = {c for c, _ in engine.finalize_problems(st.analysis)}
    assert "channel_mapping_unconfirmed" in codes
    st = make_state([obj("a", 10, 10)], [counting()])
    inspect_and_approve(st)
    st.analysis.provenance.stale_detection = True
    with pytest.raises(ReviewError) as e:
        engine.finalize(st)
    assert e.value.code == "stale_detection"


def test_mapping_change_after_detection_marks_stale_and_clears_approvals():
    st = make_state([obj("a", 10, 10)], [counting()])
    inspect_and_approve(st)
    engine.set_channel_mapping(st, [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}])
    assert st.analysis.regions[0].approved and not st.analysis.provenance.stale_detection  # unchanged
    engine.set_channel_mapping(st, [{"index": 0, "role": "green"}, {"index": 1, "role": "red"}])
    assert not st.analysis.regions[0].approved and st.analysis.provenance.stale_detection
    assert st.analysis.undo_depth == 0
    with pytest.raises(ReviewError):
        engine.set_channel_mapping(st, [{"index": 0, "role": "red"}, {"index": 1, "role": "red"}])
    with pytest.raises(ReviewError):
        engine.set_channel_mapping(st, [{"index": 0, "role": "red"}])


def test_rerun_preserves_manual_edited_reviewed_and_drops_nearby():
    old = [obj("auto_keep", 20, 20, ObjectClass.red),            # replaced
           obj("edited", 40, 40, ObjectClass.red),               # will be edited
           obj("rev", 60, 40, ObjectClass.green),                # will be confirmed
           obj("gone", 80, 40, ObjectClass.red)]                 # deleted by user (tombstone)
    st = make_state(old, [counting()])
    act(st, "move", object_id="edited", x=41, y=41)
    act(st, "confirm", object_ids=["rev"])
    act(st, "delete", object_ids=["gone"])
    act(st, "add", x=100, y=80, cls="yellow")
    manual = [o for o in st.analysis.objects if o.origin == Origin.manual][0]
    inspect_and_approve(st)
    new = [obj("n1", 20.5, 20.5), obj("n2", 42, 42),          # n2 within 3px of edited -> dropped
           obj("n3", 61, 41, ObjectClass.green),                 # near reviewed -> dropped
           obj("n4", 81, 40),                                    # near user-deleted -> suppressed
           obj("n5", 101, 81, ObjectClass.yellow),               # near manual -> dropped
           obj("n6", 120, 20), obj("out", 3, 3)]                 # out is outside the counting region
    from polony_app.schemas import DetectionRequest

    engine.apply_detection_result(st, new, DetectionRequest(), {"big-fish": "x"}, [])
    ids = {o.id for o in live(st)}
    assert {"edited", "rev", manual.id} <= ids
    assert "auto_keep" not in ids
    auto = [o for o in live(st) if o.origin == Origin.auto and o.id not in ("rev",)]
    assert sorted(o.id for o in auto) == ["n1", "n6"]
    act_ = st.log[-1]
    assert act_.payload["dropped_near_preserved"] == 3 and act_.payload["dropped_near_deleted"] == 1
    assert act_.payload["dropped_outside_valid_area"] == 1
    assert not st.analysis.regions[0].approved and not st.analysis.regions[0].inspected
    assert not st.analysis.provenance.stale_detection
    # undo restores prior objects exactly
    engine.undo(st)
    assert "auto_keep" in {o.id for o in live(st)} and "n1" not in {o.id for o in st.analysis.objects}


def test_rerun_identical_results_keeps_valid_approval_but_new_ids_do_not():
    st = make_state([obj("a", 20, 20)], [counting()])
    inspect_and_approve(st)
    from polony_app.schemas import DetectionRequest

    engine.apply_detection_result(st, [obj("a", 20, 20)], DetectionRequest(), {}, [])
    assert st.analysis.regions[0].approved  # nothing changed in the region
    engine.apply_detection_result(st, [obj("a", 20, 20), obj("b", 70, 70)], DetectionRequest(), {}, [])
    assert not st.analysis.regions[0].approved


def test_history_and_fingerprint_stable_under_reviewed_flag():
    st = make_state([obj("a", 10, 10)], [counting()])
    fp1 = engine.region_fingerprint(st.analysis, st.analysis.regions[0])
    st.analysis.objects[0].reviewed = True
    assert engine.region_fingerprint(st.analysis, st.analysis.regions[0]) == fp1
    st.analysis.objects[0].cls = ObjectClass.green
    assert engine.region_fingerprint(st.analysis, st.analysis.regions[0]) != fp1


def test_invalid_payloads_and_unknown_action():
    st = make_state([obj("a", 10, 10)], [counting()])
    for t, p, code in [("delete", {}, "invalid_payload"), ("delete", {"object_ids": ["nope"]}, "object_not_found"),
                       ("reclassify", {"object_ids": ["a"], "cls": "blue"}, "invalid_payload"),
                       ("add", {"x": "a", "y": 1, "cls": "red"}, "invalid_payload"),
                       ("approve_region", {"region_id": "zz"}, "region_not_found"),
                       ("bogus", {}, "invalid_action"), ("finalize", {}, "invalid_action")]:
        with pytest.raises(ReviewError) as e:
            act(st, t, **p)
        assert e.value.code == code, (t, e.value.code)
    act(st, "delete", object_ids=["a"])
    with pytest.raises(ReviewError) as e:
        act(st, "delete", object_ids=["a"])
    assert e.value.code == "object_deleted"
