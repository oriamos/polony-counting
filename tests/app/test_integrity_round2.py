"""Regression tests for the second adversarial integrity review (legacy state, swap alignment,
detect race, merged tombstones, export naming, API strictness)."""
import io
import json
import sqlite3
import threading
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from fakes import counting, detector, make_state, obj, rect, wait_job, write_two_channel_tiff  # noqa: F401
from polony_app.api.main import create_app
from polony_app.export import exporter
from polony_app.review import engine
from polony_app.review.state import AnalysisState
from polony_app.schemas import DetectionRequest, ObjectClass

pytestmark = pytest.mark.filterwarnings("ignore")

MAP = [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]
SWAP = [{"index": 0, "role": "green"}, {"index": 1, "role": "red"}]


@pytest.fixture
def tiff(tmp_path):
    p = tmp_path / "scan.tif"
    write_two_channel_tiff(p)
    return p


def ready(c, tiff):
    with open(tiff, "rb") as f:
        aid = c.post("/api/analyses", files=[("files", (tiff.name, f, "image/tiff"))],
                     data={"name": "n"}).json()["id"]
    assert c.put(f"/api/analyses/{aid}/channels", json={"mapping": MAP}).status_code == 200
    r = c.put(f"/api/analyses/{aid}/regions",
              json={"regions": [counting("c1", rect(5, 5, 155, 115)).model_dump(mode="json")]})
    assert r.status_code == 200
    return aid


def act(c, aid, t, **p):
    return c.post(f"/api/analyses/{aid}/actions", json={"type": t, "payload": p})


def detect(c, det, aid, objs):
    det.objects = objs
    r = c.post(f"/api/analyses/{aid}/detect", json={})
    assert r.status_code == 200, r.text
    wait_job(c, r.json()["id"])
    return c.get(f"/api/analyses/{aid}").json()


def live(a):
    return sorted(o["id"] for o in a["objects"] if not o["deleted"])


# ------------------------------------------------------------ 1 legacy persisted format
def _legacy_rewrite(path, aid, stale_flag=False):
    db = sqlite3.connect(str(path / "app.sqlite"))
    row = json.loads(db.execute("SELECT state FROM analyses WHERE id=?", (aid,)).fetchone()[0])
    for k in ("detection_sig", "user_deleted", "job_active"):
        row.pop(k, None)
    row["analysis"]["provenance"]["stale_detection"] = stale_flag
    db.execute("UPDATE analyses SET state=? WHERE id=?", (json.dumps(row), aid))
    db.commit()
    db.close()


def test_legacy_state_without_signature_becomes_stale_on_region_and_alignment_change(tmp_path, tiff, detector):
    data = tmp_path / "data"
    with TestClient(create_app(data)) as c:
        aid = ready(c, tiff)
        detect(c, detector, aid, [obj("a", 30, 30), obj("b", 80, 60)])
        act(c, aid, "region_inspected", region_id="c1", inspected=True)
        assert act(c, aid, "approve_region", region_id="c1").status_code == 200
    _legacy_rewrite(data, aid)
    with TestClient(create_app(data)) as c:
        a = c.get(f"/api/analyses/{aid}").json()
        assert a["regions"][0]["approved"] and not a["provenance"]["stale_detection"]  # L0
        regs = a["regions"]
        regs[0]["polygon"][2] = [150, 110]
        r = c.put(f"/api/analyses/{aid}/regions", json={"regions": regs}).json()
        assert r["provenance"]["stale_detection"]  # L1
        act(c, aid, "region_inspected", region_id="c1", inspected=True)
        assert act(c, aid, "approve_region", region_id="c1").status_code == 409  # L2
        assert c.post(f"/api/analyses/{aid}/finalize").status_code == 409  # L3
        c.post(f"/api/analyses/{aid}/alignment", json={"dy": 1, "dx": 0})  # L4
        assert c.get(f"/api/analyses/{aid}").json()["provenance"]["stale_detection"]
        # re-running detection records a real signature and clears staleness
        a = detect(c, detector, aid, [obj("a", 30, 30)])
        assert not a["provenance"]["stale_detection"]


def test_legacy_state_with_stale_flag_stays_stale_until_rerun(tmp_path, tiff, detector):
    data = tmp_path / "data"
    with TestClient(create_app(data)) as c:
        aid = ready(c, tiff)
        detect(c, detector, aid, [obj("a", 30, 30)])
    _legacy_rewrite(data, aid, stale_flag=True)
    with TestClient(create_app(data)) as c:
        assert c.get(f"/api/analyses/{aid}").json()["provenance"]["stale_detection"]
        act(c, aid, "add", x=50, y=50, cls="red")
        assert c.get(f"/api/analyses/{aid}").json()["provenance"]["stale_detection"]  # sticky
        assert not detect(c, detector, aid, [obj("a", 30, 30)])["provenance"]["stale_detection"]


def test_migrate_legacy_state_unit():
    st = make_state([obj("a", 30, 30)], [counting()])
    assert st.detection_sig is None and st.analysis.provenance.detection_ran_at
    assert engine.migrate_legacy_state(st) and st.detection_sig is not None
    assert not engine.recompute_stale(st)
    engine.set_alignment(st, 2, 0)
    assert st.analysis.provenance.stale_detection
    # already has a signature / never ran: untouched
    assert not engine.migrate_legacy_state(st)
    st2 = make_state([], [counting()], detected=False)
    assert not engine.migrate_legacy_state(st2) and st2.detection_sig is None


# ------------------------------------------------------------ 2 swap negates alignment
def test_swap_negates_alignment_and_logs_it_and_consumers_see_it(tmp_path, tiff, detector):
    with TestClient(create_app(tmp_path / "d")) as c:
        aid = ready(c, tiff)
        c.post(f"/api/analyses/{aid}/alignment", json={"dy": 2, "dx": -1})
        detect(c, detector, aid, [obj("r", 30, 30)])
        assert detector.last_alignment == (2, -1)
        r = c.put(f"/api/analyses/{aid}/channels", json={"mapping": SWAP}).json()
        assert (r["alignment_dy"], r["alignment_dx"]) == (-2, 1)
        ev = [e for e in c.get(f"/api/analyses/{aid}/export", params={"kind": "provenance"}).json()["events"]
              if e["event"] == "channel_mapping_changed"][-1]
        assert ev["alignment_negated"] == {"previous": {"dy": 2, "dx": -1}, "now": {"dy": -2, "dx": 1}}
        detect(c, detector, aid, [obj("r", 30, 30)])
        assert detector.last_alignment == (-2, 1)           # detection receives the corrected shift
        st = c.app.state.store
        with st.view(aid) as s:
            assert c.app.state.images.raw_key(s, "green")[4] == (-2, 1)   # image service too
        # swapping back restores the original shift
        r = c.put(f"/api/analyses/{aid}/channels", json={"mapping": MAP}).json()
        assert (r["alignment_dy"], r["alignment_dx"]) == (2, -1)


# ------------------------------------------------------------ 3 concurrent detect
def test_concurrent_detect_second_refused_and_status_stays_detecting(tmp_path, tiff, detector):
    with TestClient(create_app(tmp_path / "d")) as c:
        aid = ready(c, tiff)
        runner = c.app.state.runner
        orig = runner.submit
        gate = threading.Event()

        def slow_submit(*a, **k):
            gate.wait(1.0)  # widens the window between a check and the submit
            return orig(*a, **k)

        runner.submit = slow_submit
        detector.slow = True
        detector.objects = [obj("a", 30, 30)]
        res = []
        ts = [threading.Thread(target=lambda: res.append(c.post(f"/api/analyses/{aid}/detect", json={})))
              for _ in range(2)]
        [t.start() for t in ts]
        time.sleep(0.3)
        gate.set()
        [t.join() for t in ts]
        runner.submit = orig
        assert sorted(r.status_code for r in res) == [200, 409]
        time.sleep(0.2)
        assert c.get(f"/api/analyses/{aid}").json()["status"] == "detecting"
        jid = [r.json()["id"] for r in res if r.status_code == 200][0]
        detector.slow = False
        wait_job(c, jid)
        assert c.get(f"/api/analyses/{aid}").json()["status"] == "in_review"


# ------------------------------------------------------------ 4 merged tombstones
def _state_with_merge_deleted():
    st = make_state([], [counting()], detected=False)
    engine.apply_detection_result(
        st, [obj("a", 30, 30), obj("b", 80, 60), obj("c", 100, 100)], st.analysis.detection_request, {}, [])
    engine.apply_action(st, "merge", {"object_ids": ["b", "c"]})
    mid = [o.id for o in st.analysis.objects if "merged from" in " ".join(o.history)][0]
    engine.apply_action(st, "delete", {"object_ids": [mid]})
    return st, mid


def _rerun(st):
    engine.apply_detection_result(
        st, [obj("b", 80, 60), obj("c", 100, 100), obj("m", 90, 80), obj("a", 30, 30)],
        st.analysis.detection_request, {}, [])
    return sorted(o.id for o in st.analysis.objects if not o.deleted)


def test_deleting_merged_object_suppresses_all_constituents_on_rerun():
    st, mid = _state_with_merge_deleted()
    assert {"b", "c", mid} <= set(st.user_deleted)
    assert [o.x for o in _rerun_objs(st)] == [30.0]  # only 'a' (never deleted) comes back


def _rerun_objs(st):
    _rerun(st)
    return [o for o in st.analysis.objects if not o.deleted]


def test_merged_tombstones_survive_undo_redo_and_persistence():
    st, mid = _state_with_merge_deleted()
    engine.undo(st)
    assert mid not in st.user_deleted and "b" not in st.user_deleted and "c" not in st.user_deleted
    engine.redo(st)
    st2 = AnalysisState.from_json(json.loads(json.dumps(st.to_json())), st.log)
    assert {"b", "c", mid} <= set(st2.user_deleted)
    assert [o.x for o in _rerun_objs(st2)] == [30.0]


def test_merge_alone_still_creates_no_tombstones():
    st = make_state([], [counting()], detected=False)
    engine.apply_detection_result(st, [obj("a", 30, 30), obj("b", 32, 32)], st.analysis.detection_request, {}, [])
    engine.apply_action(st, "merge", {"object_ids": ["a", "b"]})
    assert st.user_deleted == []


# ------------------------------------------------------------ 6 export naming
def test_bundle_members_and_download_names_share_the_status_rule(tmp_path, tiff, detector):
    with TestClient(create_app(tmp_path / "d")) as c:
        aid = ready(c, tiff)
        detect(c, detector, aid, [obj("a", 30, 30)])
        for kind, ext in (("objects", "csv"), ("history", "csv"), ("actions", "jsonl"), ("bundle", "zip")):
            r = c.get(f"/api/analyses/{aid}/export", params={"kind": kind})
            assert f"_{kind}_DRAFT.{ext}" in r.headers["content-disposition"]
        z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/analyses/{aid}/export?kind=bundle").content))
        assert {"objects_DRAFT.csv", "regions_DRAFT.csv", "slide_DRAFT.csv"} <= set(z.namelist())
        hist = z.read("objects_history.csv").decode().splitlines()
        assert hist[0].endswith(",status") and all(l.endswith(",DRAFT") for l in hist[1:])
        lines = [json.loads(l) for l in z.read("actions.jsonl").decode().splitlines()]
        assert lines and all(l["status"] == "DRAFT" for l in lines)
        readme = z.read("README.txt").decode()
        assert "<DRAFT|FINAL>" in readme and "actions.jsonl line" in readme


def test_export_filename_is_the_single_rule():
    st = make_state([obj("a", 30, 30)], [counting()])
    a = st.analysis
    for kind, ext in (("objects", "csv"), ("provenance", "json"), ("bundle", "zip")):
        assert exporter.export_filename(a, kind) == f"t_{kind}_DRAFT.{ext}"
    a.finalized_at = "2026-01-01T00:00:00Z"
    assert exporter.export_filename(a, "objects") == "t_objects_FINAL.csv"
    assert exporter.member_name(a, "slide") == "slide_FINAL.csv"


# ------------------------------------------------------------ 7 strict API input
def test_settings_rejects_preview_window_and_unknown_keys(tmp_path, tiff, detector):
    with TestClient(create_app(tmp_path / "d")) as c:
        aid = ready(c, tiff)
        r = c.put(f"/api/analyses/{aid}/settings", json={"detection_request": {"preview_window": [0, 0, 10, 10]}})
        assert r.status_code == 422
        r = c.put(f"/api/analyses/{aid}/settings", json={"detection_request": {"red": {"spot_radius": 4}}})
        assert r.status_code == 422 and r.json()["detail"]["fields"] == ["detection_request.red.spot_radius"]
        r = c.put(f"/api/analyses/{aid}/settings", json={"bogus": 1})
        assert r.status_code == 422 and "bogus" in r.json()["message"]
        r = c.post(f"/api/analyses/{aid}/detect", json={"green": {"background": {"sigma": 3}}, "x": 1})
        assert r.status_code == 422
        assert set(r.json()["detail"]["fields"]) == {"green.background.sigma", "x"}
        # valid payloads still work
        assert c.put(f"/api/analyses/{aid}/settings",
                     json={"detection_request": {"red": {"spot_radius_px": 4}}}).status_code == 200
        assert c.post(f"/api/analyses/{aid}/detect", json={"red": {"spot_radius_px": 4}}).status_code == 200
