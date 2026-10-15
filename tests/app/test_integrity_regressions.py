"""Regression tests for the adversarial-review integrity findings (1-9, 14, 16)."""
import io
import json
import threading
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from fakes import counting, detector, exclusion, make_state as _make_state, obj, rect, wait_job, write_two_channel_tiff  # noqa: F401
from polony_app.api.main import create_app
from polony_app.export import exporter
from polony_app.review import engine
from polony_app.review.state import DetectionCancelled, ReviewError
from polony_app.schemas import DetectionRequest, JobStatus, ObjectClass, Origin

pytestmark = pytest.mark.filterwarnings("ignore")

MAP = [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]
SWAP = [{"index": 0, "role": "green"}, {"index": 1, "role": "red"}]


def req(spot=None):
    d = {} if spot is None else {"red": {"spot_radius_px": spot}}
    return DetectionRequest.model_validate(d)


def new_state(objs=(), regions=None, detected=False):
    """Mapped state; detection (if any) is applied through the real merge path."""
    st = _make_state([], regions if regions is not None else [counting()], detected=False)
    if detected:
        engine.apply_detection_result(st, list(objs), st.analysis.detection_request, {"numpy": "x"}, [])
    return st


def act(st, t, **p):
    return engine.apply_action(st, t, p)


def inspect_and_approve(st, rid="c1"):
    act(st, "region_inspected", region_id=rid, inspected=True)
    act(st, "approve_region", region_id=rid)


def live(st):
    return [o for o in st.analysis.objects if not o.deleted]


def codes(st):
    return {c for c, _ in engine.finalize_problems(st)}


# ---------------------------------------------------------------- 1: mapping / alignment invalidate
def test_f1_mapping_swap_without_detection_clears_approval_inspection_undo():
    st = new_state()
    act(st, "add", x=30, y=30, cls="red")
    act(st, "add", x=80, y=60, cls="green")
    inspect_and_approve(st)
    assert st.analysis.regions[0].approved and st.undo
    engine.set_channel_mapping(st, SWAP)
    r = st.analysis.regions[0]
    assert not r.approved and not r.inspected and r.approval_fingerprint is None
    assert st.undo == [] and st.redo == []
    assert "regions_not_approved" in codes(st)
    with pytest.raises(ReviewError):
        engine.finalize(st)


def test_f1_alignment_change_without_detection_clears_approval():
    st = new_state()
    act(st, "add", x=30, y=30, cls="yellow")
    inspect_and_approve(st)
    engine.set_alignment(st, 3, 3)
    assert not st.analysis.regions[0].approved and not st.analysis.regions[0].inspected and st.undo == []
    with pytest.raises(ReviewError):
        engine.finalize(st)
    ev = [e for e in st.events if e["event"] == "alignment_changed"][-1]
    assert ev["approvals_cleared"] == ["c1"] and ev["dy"] == 3


def test_f1_fingerprint_itself_covers_roles_and_alignment():
    st = new_state()
    act(st, "add", x=30, y=30, cls="red")
    inspect_and_approve(st)
    st.analysis.image.channels[0].role, st.analysis.image.channels[1].role = (
        st.analysis.image.channels[1].role, st.analysis.image.channels[0].role)  # bypass set_channel_mapping
    assert engine.revalidate_approvals(st.analysis) == ["c1"]
    st = new_state()
    act(st, "add", x=30, y=30, cls="red")
    inspect_and_approve(st)
    st.analysis.alignment_dx = 1  # bypass set_alignment
    assert engine.revalidate_approvals(st.analysis) == ["c1"]


def test_f1_first_confirmation_with_nothing_dependent_is_a_plain_event():
    st = _make_state([], [counting()], mapped=False, detected=False)
    engine.set_channel_mapping(st, MAP)
    ev = st.events[-1]
    assert ev["event"] == "channel_mapping_confirmed" and ev["had_dependents"] is False


# ---------------------------------------------------------------- 2/3/16: derived staleness
def test_f2_stale_is_derived_and_clears_when_parameters_set_back():
    st = new_state([obj("a", 30, 30)], detected=True)
    assert not st.analysis.provenance.stale_detection
    engine.set_settings(st, detection_request=req(9.0))
    assert st.analysis.provenance.stale_detection
    assert "stale_detection" in codes(st)
    engine.set_settings(st, detection_request=req())  # back to the used parameters (16)
    assert not st.analysis.provenance.stale_detection and "stale_detection" not in codes(st)


def test_f2_settings_changed_during_job_make_the_result_stale():
    st = new_state()
    used = req()
    st.analysis.detection_request = req(9.0)  # user changed settings while the job ran
    engine.apply_detection_result(st, [obj("a", 30, 30)], used, {}, [])
    assert st.analysis.provenance.stale_detection
    assert st.analysis.provenance.detection_request == used
    assert "stale_detection" in codes(st)


def test_f2_finalize_problems_checks_request_equality_explicitly():
    st = new_state([obj("a", 30, 30)], detected=True)
    inspect_and_approve(st)
    assert engine.finalize_problems(st) == []
    # flag + signature say "fine" but the parameters differ from the recorded ones
    st.analysis.detection_request = req(9.0)
    st.analysis.provenance.stale_detection = False
    st.detection_sig = None
    assert "stale_detection" in {c for c, _ in engine.finalize_problems(st.analysis)}
    with pytest.raises(ReviewError):
        engine.finalize(st)


def test_f2_regions_mapping_alignment_staleness_and_revert():
    st = new_state([obj("a", 30, 30)], detected=True)
    engine.set_alignment(st, 1, 0)
    assert st.analysis.provenance.stale_detection
    engine.set_alignment(st, 0, 0)  # back to what the detection used
    assert not st.analysis.provenance.stale_detection
    engine.set_channel_mapping(st, SWAP)
    assert st.analysis.provenance.stale_detection


def test_f3_undo_of_region_edit_does_not_erase_settings_staleness():
    st = new_state([obj("a", 30, 30), obj("b", 80, 60)], detected=True)
    inspect_and_approve(st)
    engine.set_regions(st, [r.model_copy() for r in st.analysis.regions] + [exclusion("x", rect(0, 0, 3, 3))])
    engine.set_settings(st, detection_request=req(9.0))
    assert st.analysis.provenance.stale_detection
    engine.undo(st)  # undoes the region edit
    assert st.analysis.provenance.stale_detection
    engine.redo(st)
    assert st.analysis.provenance.stale_detection
    assert "stale_detection" in codes(st)


def test_f3_undo_of_rerun_restores_previous_signature_and_staleness():
    st = new_state([obj("a", 30, 30)], detected=True)
    sig1 = json.loads(json.dumps(st.detection_sig))
    engine.set_settings(st, detection_request=req(9.0))
    engine.apply_detection_result(st, [obj("a", 30, 30)], req(9.0), {}, [])
    assert not st.analysis.provenance.stale_detection and st.detection_sig != sig1
    engine.undo(st)
    assert st.detection_sig == sig1
    assert st.analysis.provenance.stale_detection  # settings (9.0) != the restored run (default)
    engine.redo(st)
    assert not st.analysis.provenance.stale_detection


def test_signature_and_bookkeeping_survive_persistence():
    from polony_app.review.state import AnalysisState

    st = new_state([obj("a", 30, 30)], detected=True)
    act(st, "delete", object_ids=["a"])
    st2 = AnalysisState.from_json(json.loads(json.dumps(st.to_json())), st.log)
    assert st2.detection_sig == st.detection_sig and st2.user_deleted == ["a"]


# ---------------------------------------------------------------- 4: tombstones of any origin
def test_f4_user_deleted_objects_of_any_origin_are_not_resurrected():
    st = new_state([obj("a", 30, 30), obj("b", 80, 60), obj("c", 100, 100)], detected=True)
    act(st, "move", object_id="a", x=31, y=31)            # origin -> edited
    act(st, "delete", object_ids=["a"])
    act(st, "confirm", object_ids=["b"])                  # reviewed auto
    act(st, "delete", object_ids=["b"])
    act(st, "add", x=50, y=50, cls="red")                 # manual
    mid = [o.id for o in st.analysis.objects if o.origin == Origin.manual][0]
    act(st, "delete", object_ids=[mid])
    engine.apply_detection_result(
        st, [obj("a", 31, 31), obj("b", 80, 60), obj("n", 50, 50), obj("c", 100, 100)], req(), {}, [])
    assert [o.id for o in live(st)] == ["c"] or {(round(o.x), round(o.y)) for o in live(st)} == {(100, 100)}


def test_f4_deleted_after_moving_still_suppresses_original_detection_spot():
    st = new_state([obj("a", 30, 30)], detected=True)
    act(st, "move", object_id="a", x=40, y=40)  # 14 px away from where it was detected
    act(st, "delete", object_ids=["a"])
    engine.apply_detection_result(st, [obj("a2", 30, 30)], req(), {}, [])
    assert live(st) == []


def test_f4_merge_and_split_deletions_are_not_tombstones_and_undo_clears_marker():
    st = new_state([obj("a", 30, 30), obj("b", 32, 32)], detected=True)
    act(st, "merge", object_ids=["a", "b"])
    assert st.user_deleted == []
    engine.apply_detection_result(st, [obj("n", 30, 30)], req(), {}, [])
    # the merged (edited) object is preserved; new object near it dropped; but a spot where only
    # merged-away originals were is not a tombstone by itself
    st = new_state([obj("a", 30, 30)], detected=True)
    act(st, "delete", object_ids=["a"])
    assert st.user_deleted == ["a"]
    engine.undo(st)
    assert st.user_deleted == []
    engine.redo(st)
    assert st.user_deleted == ["a"]


# ---------------------------------------------------------------- 5: red/green swap
def test_f5_swap_mirrors_classes_swaps_evidence_and_requires_rereview():
    objs = [obj("r", 30, 30, ObjectClass.red), obj("g", 60, 30, ObjectClass.green),
            obj("y", 90, 30, ObjectClass.yellow), obj("u", 30, 80, ObjectClass.uncertain,
                                                      red=obj("t", 30, 80, ObjectClass.red).red)]
    st = new_state(objs, detected=True)
    act(st, "confirm", object_ids=["r", "g", "y"])
    act(st, "add", x=100, y=100, cls="red")
    act(st, "region_inspected", region_id="c1", inspected=True)
    mid = [o.id for o in st.analysis.objects if o.origin == Origin.manual][0]
    act(st, "reclassify", object_ids=["u"], cls="uncertain")
    act(st, "delete", object_ids=["g"])
    by = {o.id: o for o in st.analysis.objects}
    red_ev_r, green_ev_y = by["r"].red.model_copy(), by["y"].green.model_copy()
    engine.set_channel_mapping(st, SWAP)
    by = {o.id: o for o in st.analysis.objects}
    assert by["r"].cls == ObjectClass.green and by["r"].green == red_ev_r and not by["r"].red.detected
    assert by["y"].cls == ObjectClass.yellow and by["y"].red == green_ev_y
    assert by["u"].cls == ObjectClass.uncertain and by["u"].green.detected and not by["u"].red.detected
    assert by[mid].cls == ObjectClass.green
    assert by["g"].cls == ObjectClass.green and by["g"].deleted  # deleted objects untouched
    assert all(not o.reviewed for o in live(st))
    assert all(not r.approved and not r.inspected for r in st.analysis.regions)
    assert st.analysis.provenance.stale_detection and st.undo == []
    ev = [e for e in st.events if e["event"] == "channel_mapping_changed"][-1]
    assert ev["red_green_swapped"] == 4
    # re-run after the swap is consistent: auto/unreviewed replaced, manual (mirrored) preserved
    engine.apply_detection_result(
        st, [obj("r", 30, 30, ObjectClass.green), obj("y", 90, 30, ObjectClass.yellow),
             obj("m2", 100, 100, ObjectClass.green)], req(), {}, [])
    classes = {(round(o.x), round(o.y)): o.cls for o in live(st)}
    assert classes[(30, 30)] == ObjectClass.green and classes[(100, 100)] == ObjectClass.green
    assert len([o for o in live(st) if (round(o.x), round(o.y)) == (100, 100)]) == 1
    assert not st.analysis.provenance.stale_detection


def test_f5_non_swap_mapping_change_keeps_classes_but_invalidates():
    from polony_app.schemas import ChannelInfo, ChannelRole

    st = new_state([obj("r", 30, 30)], detected=True)
    st.analysis.image.channels.append(ChannelInfo(index=2, source_file="a.tif", page=2, role=ChannelRole.ignore, role_confirmed=True))
    # red channel 0 -> ignore, ignore channel 2 -> red: not a swap
    engine.set_channel_mapping(st, [{"index": 0, "role": "ignore"}, {"index": 1, "role": "green"}, {"index": 2, "role": "red"}])
    assert st.analysis.objects[0].cls == ObjectClass.red
    assert st.analysis.provenance.stale_detection


# ---------------------------------------------------------------- 6: export completeness
def test_f6_bundle_has_full_action_log_history_info_and_versions(tmp_path):
    st = new_state([obj("a", 30, 30), obj("b", 80, 60)], detected=True)
    act(st, "move", object_id="a", x=31, y=31)
    act(st, "delete", object_ids=["b"])
    engine.undo(st)
    for i in range(60):  # more than the old last-50 truncation
        engine.set_alignment(st, i + 1, 0)
    st.analysis.image.tiff_info_text = "ImageJ=1.53\nunit=um"
    st.analysis.provenance.package_versions = {"bigfish": "0.6.2", "skimage": "0.25", "numpy": "2.0"}
    orig = tmp_path / "originals"
    orig.mkdir()
    (orig / "upload_manifest.json").write_text(json.dumps({"a.tif": "my scan (1).tif"}))
    z = zipfile.ZipFile(io.BytesIO(exporter.bundle_zip(st, orig)))
    names = set(z.namelist())
    assert {"actions.jsonl", "objects_history.csv", "provenance.json"} <= names
    lines = [json.loads(x) for x in z.read("actions.jsonl").decode().splitlines()]
    assert [(l["seq"], l["type"]) for l in lines] == [(a.seq, a.type.value) for a in st.log]
    assert all(l["ts"] and "payload" in l and l["inverse_omitted"] for l in lines)
    assert lines[-1]["payload"].get("object_ids") == ["b"] or lines[-2]["payload"]
    hist = z.read("objects_history.csv").decode().splitlines()
    assert hist[0].startswith("object_id,step,timestamp,event")
    assert any(l.startswith("a,") and "moved from" in l for l in hist)
    assert any(l.startswith("b,") and "created by detection" in l for l in hist)
    p = json.loads(z.read("provenance.json"))
    assert p["image"]["tiff_info_text"] == "ImageJ=1.53\nunit=um"
    assert p["image"]["original_upload_filenames"] == {"a.tif": "my scan (1).tif"}
    assert p["image"]["channel_mapping"][0]["original_upload_filename"] == "my scan (1).tif"
    assert len([e for e in p["events"] if e["event"] == "alignment_changed"]) == 60
    assert len(p["action_log_summary"]["events"]) == len(st.events) > 50
    assert any(e["event"] == "undo" for e in p["events"])
    assert "package_versions" not in p
    det, now_ = p["package_versions_at_detection"], p["package_versions_at_export"]
    assert det["big-fish"] == "0.6.2" and det["scikit-image"] == "0.25" and "bigfish" not in det
    assert "numpy" in now_ and "polony_app" in now_ and set(det) - {"polony_app"} <= set(now_) | {"big-fish", "scikit-image"}
    assert "actions.jsonl" in z.read("README.txt").decode()


def test_f6_missing_manifest_is_tolerated(tmp_path):
    st = new_state([obj("a", 30, 30)], detected=True)
    p = json.loads(exporter.provenance_json(st, tmp_path))
    assert p["image"]["original_upload_filenames"] is None
    p = json.loads(exporter.provenance_json(st))
    assert p["image"]["original_upload_filenames"] is None


# ---------------------------------------------------------------- 7: cancel race
def test_f7_cancelled_before_commit_commits_nothing():
    st = new_state([obj("a", 30, 30)], detected=True)
    before = st.analysis.model_dump(mode="json")
    sig, nlog = json.dumps(st.detection_sig), len(st.log)
    with pytest.raises(DetectionCancelled):
        engine.apply_detection_result(st, [obj("z", 60, 60)], req(), {}, [], cancelled=lambda: True)
    assert st.analysis.model_dump(mode="json") == before and json.dumps(st.detection_sig) == sig
    assert len(st.log) == nlog


def test_f7_runner_never_downgrades_a_returned_result_and_cancel_return_is_cancelled(tmp_path):
    from polony_app.store.db import Store
    from polony_app.store.jobs import CANCELLED_MESSAGE, JobRunner

    store = Store(tmp_path / "d")
    runner = JobRunner(store)
    aid = store.create_analysis("x")

    def committed_then_cancel(progress, cancelled):
        for ev in list(runner._cancel.values()):
            ev.set()  # cancel arrives after the work was committed
        return "1 objects detected"

    j = runner.submit(aid, committed_then_cancel)
    t0 = time.time()
    while runner.get(j.id).status not in (JobStatus.done, JobStatus.cancelled, JobStatus.failed) and time.time() - t0 < 5:
        time.sleep(0.01)
    assert runner.get(j.id).status == JobStatus.done
    t0 = time.time()
    while runner.active_job(aid) and time.time() - t0 < 5:
        time.sleep(0.01)
    j2 = runner.submit(aid, lambda p, c: CANCELLED_MESSAGE)
    t0 = time.time()
    while runner.get(j2.id).status not in (JobStatus.done, JobStatus.cancelled) and time.time() - t0 < 5:
        time.sleep(0.01)
    assert runner.get(j2.id).status == JobStatus.cancelled
    runner.shutdown()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as c:
        yield c


@pytest.fixture
def tiff(tmp_path):
    p = tmp_path / "scan.tif"
    write_two_channel_tiff(p)
    return p


def api_ready(client, tiff):
    with open(tiff, "rb") as f:
        r = client.post("/api/analyses", files=[("files", (tiff.name, f, "image/tiff"))], data={"name": "n"})
    aid = r.json()["id"]
    assert client.put(f"/api/analyses/{aid}/channels", json={"mapping": MAP}).status_code == 200
    r = client.put(f"/api/analyses/{aid}/regions",
                   json={"regions": [counting("c1", rect(5, 5, 155, 115)).model_dump(mode="json")]})
    assert r.status_code == 200
    return aid


def api_act(client, aid, t, **p):
    return client.post(f"/api/analyses/{aid}/actions", json={"type": t, "payload": p})


@pytest.mark.parametrize("when", ["after_commit", "before_commit"])
def test_f7_api_job_status_agrees_with_persisted_objects(client, tiff, detector, when):
    aid = api_ready(client, tiff)
    runner = client.app.state.runner
    orig = engine.apply_detection_result

    def hooked(st, *a, **k):
        if when == "before_commit":
            for ev in list(runner._cancel.values()):
                ev.set()
            return orig(st, *a, **k)
        r = orig(st, *a, **k)
        for ev in list(runner._cancel.values()):
            ev.set()
        return r

    engine.apply_detection_result = hooked
    try:
        detector.objects = [obj("a", 30, 30)]
        r = client.post(f"/api/analyses/{aid}/detect", json={})
        j = wait_job(client, r.json()["id"])
    finally:
        engine.apply_detection_result = orig
    a = client.get(f"/api/analyses/{aid}").json()
    n = len([o for o in a["objects"] if not o["deleted"]])
    assert not (j["status"] == "cancelled" and n > 0)
    assert not (j["status"] == "done" and n == 0)
    if when == "after_commit":
        assert j["status"] == "done" and n == 1


# ---------------------------------------------------------------- 8: status flips
def test_f8_edit_during_job_keeps_detecting_and_flag_persists():
    from polony_app.review.state import AnalysisState

    st = new_state([obj("a", 30, 30)], detected=True)
    engine.refresh_status(st, job_running=True)
    assert st.analysis.status.value == "detecting"
    act(st, "move", object_id="a", x=35, y=35)
    engine.undo(st)
    engine.set_settings(st, reporting_mode=st.analysis.reporting_mode)
    assert st.analysis.status.value == "detecting"
    st2 = AnalysisState.from_json(json.loads(json.dumps(st.to_json())), st.log)
    engine.refresh_status(st2)
    assert st2.analysis.status.value == "detecting"
    engine.refresh_status(st2, job_running=False)
    assert st2.analysis.status.value == "in_review"


def test_f8_api_edits_during_running_job_do_not_flip_status(client, tiff, detector):
    aid = api_ready(client, tiff)
    detector.objects = [obj("a", 30, 30)]
    r = client.post(f"/api/analyses/{aid}/detect", json={})
    jid = r.json()["id"]
    wait_job(client, jid)
    detector.slow = True
    r = client.post(f"/api/analyses/{aid}/detect", json={})
    jid = r.json()["id"]
    time.sleep(0.1)
    assert api_act(client, aid, "add", x=100, y=100, cls="green").status_code == 200
    assert client.get(f"/api/analyses/{aid}").json()["status"] == "detecting"
    wait_job(client, jid)
    detector.slow = False
    assert client.get(f"/api/analyses/{aid}").json()["status"] == "in_review"


def test_f8_orphan_recovery_clears_persisted_flag(tmp_path):
    from polony_app.store.db import Store

    store = Store(tmp_path / "d")
    aid = store.create_analysis("x")
    with store.edit(aid) as st:
        engine.refresh_status(st, job_running=True)
    store.close()
    store2 = Store(tmp_path / "d")
    with store2.view(aid) as st:
        assert not st.job_active
    store2.close()


# ---------------------------------------------------------------- 9: split evidence
def test_f9_split_children_keep_only_matching_evidence():
    st = new_state([obj("y", 30, 30, ObjectClass.yellow)], detected=True)
    act(st, "split", object_id="y", positions=[[29, 29], [33, 33]], classes=["red", "green"])
    kids = [o for o in st.analysis.objects if o.history and "split from" in o.history[0]]
    by = {k.cls: k for k in kids}
    assert by[ObjectClass.red].red.detected and not by[ObjectClass.red].green.detected
    assert by[ObjectClass.green].green.detected and not by[ObjectClass.green].red.detected
    # yellow child keeps both
    st = new_state([obj("y", 30, 30, ObjectClass.yellow)], detected=True)
    act(st, "split", object_id="y", positions=[[29, 29], [33, 33]], classes=["yellow", "red"])
    kids = {o.cls: o for o in st.analysis.objects if o.history and "split from" in o.history[0]}
    assert kids[ObjectClass.yellow].red.detected and kids[ObjectClass.yellow].green.detected


# ---------------------------------------------------------------- 14: unique ids
def test_f14_ids_are_unique_even_on_collision(monkeypatch):
    st = new_state([obj("a", 30, 30), obj("b", 32, 32)], detected=True)
    act(st, "add", x=50, y=50, cls="red")
    existing = {o.id for o in st.analysis.objects}
    colliding = sorted(existing)[0]
    seq = iter([colliding, colliding, "fresh1", "fresh2", "fresh3", "fresh4"])
    monkeypatch.setattr(engine, "_short", lambda prefix: next(seq))
    act(st, "add", x=60, y=60, cls="red")
    ids = [o.id for o in st.analysis.objects]
    assert len(ids) == len(set(ids)) and "fresh1" in ids
    act(st, "merge", object_ids=["a", "b"])
    ids = [o.id for o in st.analysis.objects]
    assert len(ids) == len(set(ids)) and "fresh2" in ids
    # detection ids colliding with preserved ones
    seq2 = iter(["fresh2", "x9"])
    monkeypatch.setattr(engine, "_short", lambda prefix: next(seq2))
    engine.apply_detection_result(st, [obj("fresh2", 120, 100)], req(), {}, [])
    ids = [o.id for o in st.analysis.objects]
    assert len(ids) == len(set(ids)) and "x9" in ids
