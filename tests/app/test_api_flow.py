"""API round trips with TestClient (detection is faked until/unless noted)."""
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from fakes import H, W, counting, detector, exclusion, obj, rect, wait_job, write_two_channel_tiff  # noqa: F401
from polony_app.api.main import create_app
from polony_app.schemas import ObjectClass, ReviewReason


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def client(data_dir):
    with TestClient(create_app(data_dir)) as c:
        yield c


@pytest.fixture
def tiff(tmp_path):
    p = tmp_path / "scan one.tif"
    write_two_channel_tiff(p)
    return p


def upload(client, tiff, name="slide A"):
    with open(tiff, "rb") as f:
        r = client.post("/api/analyses", files=[("files", (tiff.name, f, "image/tiff"))], data={"name": name})
    assert r.status_code == 200, r.text
    return r.json()


def to_ready(client, tiff, regions=None):
    a = upload(client, tiff)
    aid = a["id"]
    r = client.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
    assert r.status_code == 200, r.text
    regs = regions or [counting("c1", rect(5, 5, 155, 115)).model_dump(mode="json")]
    r = client.put(f"/api/analyses/{aid}/regions", json={"regions": regs})
    assert r.status_code == 200, r.text
    return aid


def detect(client, aid, detector, objs):
    detector.objects = objs
    r = client.post(f"/api/analyses/{aid}/detect", json={})
    assert r.status_code == 200, r.text
    j = wait_job(client, r.json()["id"])
    assert j["status"] == "done", j
    return client.get(f"/api/analyses/{aid}").json()


def act(client, aid, type_, **payload):
    return client.post(f"/api/analyses/{aid}/actions", json={"type": type_, "payload": payload})


DETS = lambda: [obj("d1", 30, 30), obj("d2", 80, 60, ObjectClass.yellow), obj("d3", 120, 90, ObjectClass.green,
                review_reasons=[ReviewReason.weak_signal]), obj("d4", 40, 95, ObjectClass.uncertain,
                review_reasons=[ReviewReason.ambiguous_match])]


def test_health(client):
    j = client.get("/api/health").json()
    assert j["ok"] and j["version"]


def test_full_round_trip_with_restart_and_export(data_dir, tiff, detector):
    with TestClient(create_app(data_dir)) as client:
        a = upload(client, tiff)
        aid = a["id"]
        assert a["status"] == "needs_channel_mapping"
        assert all(c["role"] is None and not c["role_confirmed"] for c in a["image"]["channels"])
        # detection refused before mapping is confirmed
        r = client.post(f"/api/analyses/{aid}/detect", json={})
        assert r.status_code == 409 and r.json()["code"] == "channel_mapping_unconfirmed"
        assert client.get(f"/api/analyses/{aid}/image/composite").status_code == 409
        r = client.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]})
        assert r.json()["status"] == "ready"
        r = client.put(f"/api/analyses/{aid}/regions", json={"regions": [counting("c1").model_dump(mode="json")]})
        assert r.json()["slide"]["counting_area_px2"] > 10000
        det = detect(client, aid, detector, DETS())
        assert det["status"] == "in_review" and len(det["objects"]) == 4
        assert det["provenance"]["detection_ran_at"] and not det["provenance"]["stale_detection"]
        assert all(o["region_id"] == "c1" for o in det["objects"])

        q = client.get(f"/api/analyses/{aid}/review-queue").json()
        assert [i["object_id"] for i in q] == ["d4", "d3"]
        c = client.get(f"/api/analyses/{aid}/counts").json()
        assert c["slide"]["yellow"] == 1 and c["slide"]["uncertain"] == 1 and c["slide"]["total_objects_exclusive"] == 3
        assert c["regions"][0]["scope"] == "c1"

        # approval gated by inspection and uncertain objects
        assert act(client, aid, "approve_region", region_id="c1").json()["code"] == "region_not_inspected"
        assert act(client, aid, "region_inspected", region_id="c1", inspected=True).status_code == 200
        r = act(client, aid, "approve_region", region_id="c1")
        assert r.status_code == 409 and r.json()["code"] == "unresolved_uncertain"
        assert act(client, aid, "reclassify", object_ids=["d4"], cls="red").status_code == 200
        r = act(client, aid, "add", x=60, y=100, cls="green")
        assert r.status_code == 200 and r.headers["X-Created-Ids"]
        newid = r.headers["X-Created-Ids"]
        assert client.post(f"/api/analyses/{aid}/undo").json()["undo_depth"] == 4
        assert all(o["id"] != newid for o in client.get(f"/api/analyses/{aid}").json()["objects"])
        assert client.post(f"/api/analyses/{aid}/redo").json()["redo_depth"] == 0
        assert act(client, aid, "delete", object_ids=[newid]).status_code == 200
        assert act(client, aid, "accept_region", region_id="c1").status_code == 200
        r = act(client, aid, "approve_region", region_id="c1")
        assert r.status_code == 200 and r.json()["regions"][0]["approved"]

        # DRAFT export before finalize
        r = client.get(f"/api/analyses/{aid}/export", params={"kind": "objects"})
        assert "DRAFT" in r.headers["content-disposition"] and r.text.splitlines()[1].endswith(",DRAFT")
        fin = client.post(f"/api/analyses/{aid}/finalize")
        assert fin.status_code == 200 and fin.json()["status"] == "finalized" and fin.json()["finalized_at"]
        before = client.get(f"/api/analyses/{aid}").json()
        # frozen
        r = act(client, aid, "delete", object_ids=["d1"])
        assert r.status_code == 409 and r.json()["code"] == "analysis_finalized"
        assert client.post(f"/api/analyses/{aid}/undo").json()["code"] == "analysis_finalized"
        assert client.put(f"/api/analyses/{aid}/regions", json={"regions": []}).status_code == 409
        assert client.post(f"/api/analyses/{aid}/detect", json={}).status_code == 409

    # ---- restart: brand new app on the same data dir
    with TestClient(create_app(data_dir)) as client2:
        lst = client2.get("/api/analyses").json()
        assert len(lst) == 1 and lst[0]["status"] == "finalized" and lst[0]["counts"]["yellow"] == 1
        after = client2.get(f"/api/analyses/{aid}").json()
        assert after == before
        # image still renders from stored originals
        r = client2.get(f"/api/analyses/{aid}/image/composite", params={"max_dim": 64})
        assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
        # export FINAL
        r = client2.get(f"/api/analyses/{aid}/export", params={"kind": "slide"})
        assert "DRAFT" not in r.headers["content-disposition"] and r.text.splitlines()[1].endswith(",FINAL")
        z = zipfile.ZipFile(io.BytesIO(client2.get(f"/api/analyses/{aid}/export?kind=bundle").content))
        names = z.namelist()
        assert {"README.txt", "objects_FINAL.csv", "regions_FINAL.csv", "slide_FINAL.csv", "provenance.json",
                "actions.jsonl", "objects_history.csv"} == set(names)
        assert z.read("README.txt").decode().startswith("FINAL")
        prov = json.loads(z.read("provenance.json"))
        assert prov["status"] == "FINAL" and prov["image"]["sha256"] and prov["image"]["bit_depth"] == 16
        assert prov["action_log_summary"]["by_type"]["approve_region"] == 1
        # reopen: approvals kept, edit clears it
        r = client2.post(f"/api/analyses/{aid}/reopen")
        assert r.json()["status"] == "in_review" and r.json()["finalized_at"] is None and r.json()["regions"][0]["approved"]
        prov2 = client2.get(f"/api/analyses/{aid}/export", params={"kind": "provenance"}).json()
        assert prov2["action_log_summary"]["by_type"]["reopen"] == 1
        # the undo history was dropped on finalize
        assert client2.post(f"/api/analyses/{aid}/undo").json()["code"] == "nothing_to_undo"
        r = act(client2, aid, "move", object_id="d1", x=31, y=31)
        assert r.status_code == 200 and not r.json()["regions"][0]["approved"]
        assert client2.post(f"/api/analyses/{aid}/finalize").json()["code"] == "regions_not_approved"
        # undo history survives restart too
    with TestClient(create_app(data_dir)) as client3:
        a3 = client3.get(f"/api/analyses/{aid}").json()
        assert a3["undo_depth"] == 1
        assert client3.post(f"/api/analyses/{aid}/undo").json()["regions"][0]["approved"]


def test_approval_invalidation_via_api(client, tiff, detector):
    aid = to_ready(client, tiff, [counting("c1", rect(5, 5, 70, 115)).model_dump(mode="json"),
                                  counting("c2", rect(80, 5, 155, 115)).model_dump(mode="json")])
    detect(client, aid, detector, [obj("a", 30, 30), obj("b", 120, 30)])
    for rid in ("c1", "c2"):
        act(client, aid, "region_inspected", region_id=rid, inspected=True)
        assert act(client, aid, "approve_region", region_id=rid).status_code == 200
    r = act(client, aid, "move", object_id="a", x=31, y=31).json()
    assert [x["approved"] for x in r["regions"]] == [False, True]
    # exclusion overlapping c2 invalidates c2 and marks detection stale
    act(client, aid, "approve_region", region_id="c1")
    regs = client.get(f"/api/analyses/{aid}").json()["regions"] + [exclusion("x", rect(140, 100, 155, 115)).model_dump(mode="json")]
    r = client.put(f"/api/analyses/{aid}/regions", json={"regions": regs}).json()
    assert [x["approved"] for x in r["regions"] if x["kind"] == "counting"] == [True, False]
    assert r["provenance"]["stale_detection"] is True
    assert act(client, aid, "approve_region", region_id="c2").json()["code"] == "stale_detection"
    assert client.post(f"/api/analyses/{aid}/finalize").json()["code"] in ("stale_detection", "regions_not_approved")
    # re-detect clears staleness; mapping change makes it stale again and clears approvals
    detect(client, aid, detector, [obj("a", 30, 30), obj("b", 120, 30)])
    r = client.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "green"}, {"index": 1, "role": "red"}]}).json()
    assert r["provenance"]["stale_detection"] and not any(x["approved"] for x in r["regions"])


def test_rerun_keeps_manual_work_via_api(client, tiff, detector):
    aid = to_ready(client, tiff)
    detect(client, aid, detector, [obj("a", 30, 30), obj("b", 60, 60)])
    act(client, aid, "move", object_id="a", x=33, y=33)
    r = act(client, aid, "add", x=100, y=100, cls="red")
    detect(client, aid, detector, [obj("a", 33.5, 33.5), obj("n", 60, 61), obj("m", 100.5, 100)])
    objs = [o for o in client.get(f"/api/analyses/{aid}").json()["objects"] if not o["deleted"]]
    assert sorted((round(o["x"]), round(o["y"]), o["origin"]) for o in objs) == [
        (33, 33, "edited"), (60, 61, "auto"), (100, 100, "manual")]


def test_settings_slide_validation_and_stale(client, tiff, detector):
    aid = to_ready(client, tiff)
    r = client.put(f"/api/analyses/{aid}/settings", json={"slide": {"sample_volume_ul": -1}})
    assert r.status_code == 422 and r.json()["code"] == "invalid_slide"
    r = client.put(f"/api/analyses/{aid}/settings", json={"reporting_mode": "channel_inclusive", "slide": {"slide_id": "S9", "sample_volume_ul": 7.5}})
    assert r.json()["reporting_mode"] == "channel_inclusive" and r.json()["slide"]["sample_volume_ul"] == 7.5
    detect(client, aid, detector, [obj("a", 30, 30)])
    req = {"red": {"spot_radius_px": 7.5}}
    r = client.put(f"/api/analyses/{aid}/settings", json={"detection_request": req}).json()
    assert r["provenance"]["stale_detection"] is True
    # counts carry the selected mode
    assert client.get(f"/api/analyses/{aid}/counts").json()["slide"]["reporting_mode"] == "channel_inclusive"


def test_error_responses(client, tiff, tmp_path):
    r = client.get("/api/analyses/doesnotexist")
    assert r.status_code == 404 and r.json()["code"] == "analysis_not_found"
    assert client.get("/api/analyses/../etc/passwd").status_code == 404
    assert client.get("/api/analyses/%2e%2e%2f").status_code == 404
    assert client.get("/api/nope").json()["code"] == "not_found"
    assert client.get("/api/jobs/abc").json()["code"] == "job_not_found"
    assert client.get("/api/csv-imports/zzz").status_code == 404
    # bad file types / empty / corrupt
    r = client.post("/api/analyses", files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 415 and r.json()["code"] == "unsupported_file"
    r = client.post("/api/analyses", files=[("files", ("a.tif", b"", "image/tiff"))])
    assert r.status_code == 422 and r.json()["code"] == "empty_file"
    r = client.post("/api/analyses", files=[("files", ("a.tif", b"not a tiff at all", "image/tiff"))])
    assert r.status_code == 422 and r.json()["code"] in ("ingest_error", "unreadable_image")
    assert client.get("/api/analyses").json() == []  # failed uploads leave nothing behind
    r = client.post("/api/analyses")
    assert r.status_code == 422 and r.json()["code"] == "validation_error"
    aid = upload(client, tiff)["id"]
    assert client.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "red"}]}).json()["code"] == "invalid_mapping"
    assert client.put(f"/api/analyses/{aid}/channels", json={"mapping": [{"index": 0, "role": "purple"}]}).json()["code"] == "validation_error"
    assert act(client, aid, "bogus").json()["code"] == "invalid_action"
    assert client.get(f"/api/analyses/{aid}/export", params={"kind": "xml"}).status_code == 422
    assert client.put(f"/api/analyses/{aid}/regions", json={"regions": [{"id": "r", "kind": "counting", "name": "r", "polygon": [[0, 0], [1, 1]]}]}).json()["code"] == "invalid_regions"


def test_upload_size_limit_and_safe_names(data_dir, tiff, monkeypatch):
    monkeypatch.setenv("POLONY_MAX_UPLOAD_MB", "1")
    with TestClient(create_app(data_dir)) as c:
        big = b"II*\x00" + b"\0" * (2 * 1024 * 1024)
        r = c.post("/api/analyses", files=[("files", ("big.tif", big, "image/tiff"))])
        assert r.status_code == 413 and r.json()["code"] == "too_large"
        with open(tiff, "rb") as f:
            r = c.post("/api/analyses", files=[("files", ("../../etc/pass wd.tif", f, "image/tiff"))])
        assert r.status_code == 200
        a = r.json()
        stored = a["image"]["channels"][0]["source_file"]
        assert "/" not in stored and ".." not in stored
        assert list((data_dir / "originals" / a["id"]).glob("*.tif"))
        assert not any(p.name == "pass wd.tif" for p in data_dir.rglob("*"))


def test_image_endpoints(client, tiff, detector):
    aid = to_ready(client, tiff)
    for view in ("red", "green", "composite", "processed_red", "processed_green"):
        r = client.get(f"/api/analyses/{aid}/image/{view}", params={"lo": 100, "hi": 3000, "gamma": 1.2, "x0": 10, "y0": 20, "x1": 90, "y1": 70})
        assert r.status_code == 200, (view, r.text)
        assert r.headers["X-Image-Width"] == str(W) and r.headers["X-Image-Height"] == str(H)
        assert r.headers["X-Crop"] == "10,20,90,70"
        assert float(r.headers["X-Auto-Hi"]) > float(r.headers["X-Auto-Lo"])
        from PIL import Image

        assert Image.open(io.BytesIO(r.content)).size == (80, 50)
    assert client.get(f"/api/analyses/{aid}/image/bogus").status_code == 422
    assert client.get(f"/api/analyses/{aid}/image/red", params={"x0": 5}).json()["code"] == "invalid_crop"
    h = client.get(f"/api/analyses/{aid}/histogram/red").json()
    assert len(h["bins"]) == len(h["counts"]) == 256 and sum(h["counts"]) == H * W and h["hi"] > h["p995"] > h["lo"]
    detect(client, aid, detector, [obj("d1", 30, 30)])
    r = client.get(f"/api/analyses/{aid}/thumb/d1", params={"size": 32})
    assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).size == (32, 32)
    assert client.get(f"/api/analyses/{aid}/thumb/zz").status_code == 404
    # display contrast never changes stored data
    assert client.get(f"/api/analyses/{aid}/counts").json()["slide"]["red_only"] == 1


def test_alignment_endpoint(client, tiff, detector):
    aid = to_ready(client, tiff)
    r = client.post(f"/api/analyses/{aid}/alignment", json={"estimate": True}).json()
    assert r["applied"] is False and {"dy", "dx", "peak_corr"} <= set(r)
    detect(client, aid, detector, [obj("a", 30, 30)])
    r = client.post(f"/api/analyses/{aid}/alignment", json={"dy": 1, "dx": -2}).json()
    assert r["applied"] is True and (r["dy"], r["dx"]) == (1, -2)
    assert client.get(f"/api/analyses/{aid}").json()["provenance"]["stale_detection"] is True
    assert client.post(f"/api/analyses/{aid}/alignment", json={}).json()["code"] == "invalid_alignment"
    z = client.get(f"/api/analyses/{aid}/export", params={"kind": "provenance"}).json()
    assert z["alignment_applied_to_green_px"] == {"dy": 1, "dx": -2}
    an = client.get(f"/api/analyses/{aid}").json()
    assert (an["alignment_dy"], an["alignment_dx"]) == (1, -2)


def test_job_failure_cancellation_and_orphans(data_dir, tiff, detector):
    with TestClient(create_app(data_dir)) as client:
        aid = to_ready(client, tiff)
        detector.fail = True
        r = client.post(f"/api/analyses/{aid}/detect", json={}).json()
        j = wait_job(client, r["id"])
        assert j["status"] == "failed" and "boom" in j["error"]
        assert client.get(f"/api/analyses/{aid}").json()["status"] == "ready"
        detector.fail = False
        detector.slow = True
        detector.objects = [obj("a", 30, 30)]
        r = client.post(f"/api/analyses/{aid}/detect", json={}).json()
        assert client.get(f"/api/analyses/{aid}").json()["status"] == "detecting"
        assert client.post(f"/api/analyses/{aid}/detect", json={}).json()["code"] == "job_in_progress"
        assert client.delete(f"/api/analyses/{aid}").status_code == 409
        c = client.post(f"/api/jobs/{r['id']}/cancel").json()
        j = wait_job(client, r["id"])
        assert j["status"] == "cancelled"
        a = client.get(f"/api/analyses/{aid}").json()
        assert a["status"] == "ready" and a["objects"] == []
        # a job left 'running' by a dead process is failed on startup
        detector.slow = False
        jid = client.app.state.store.create_job(aid).id
        j = client.app.state.store.get_job(jid)
        from polony_app.schemas import JobStatus
        j.status = JobStatus.running
        client.app.state.store.save_job(j)
    with TestClient(create_app(data_dir)) as c2:
        j = c2.get(f"/api/jobs/{jid}").json()
        assert j["status"] == "failed" and "restarted" in j["error"]


def test_preview_not_persisted(client, tiff, detector):
    aid = to_ready(client, tiff)
    detector.objects = [obj("p1", 30, 30)]
    r = client.post(f"/api/analyses/{aid}/detect/preview", json={"preview_window": [0, 0, 80, 60]})
    assert r.status_code == 200 and len(r.json()["objects"]) == 1 and detector.last_window == (0, 0, 80, 60)
    assert client.get(f"/api/analyses/{aid}").json()["objects"] == []
    assert client.post(f"/api/analyses/{aid}/detect/preview", json={}).status_code == 422
    assert client.post(f"/api/analyses/{aid}/detect/preview", json={"preview_window": [0, 0, 999, 60]}).json()["code"] == "invalid_window"


def test_delete_analysis_removes_files(client, tiff, data_dir):
    aid = upload(client, tiff)["id"]
    assert (data_dir / "originals" / aid).exists()
    assert client.delete(f"/api/analyses/{aid}").json()["ok"]
    assert not (data_dir / "originals" / aid).exists()
    assert client.get(f"/api/analyses/{aid}").status_code == 404


def test_csv_import_api(client):
    csv_bytes = "﻿slide ID,SAMPLE VOLUM[ul],field 1,field 2\nA,2.5,0,\n".encode()
    r = client.post("/api/csv-imports", files={"file": ("T7A_x.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    c = r.json()
    assert c["color_mapping"] is None and c["rows"][0]["fields"] == [0, None]
    cid = c["id"]
    assert client.put(f"/api/csv-imports/{cid}", json={"color_mapping": "red"}).json()["color_mapping"] == "red"
    assert client.put(f"/api/csv-imports/{cid}", json={"color_mapping": "T7A"}).status_code == 422
    assert client.get(f"/api/csv-imports/{cid}").json()["color_mapping"] == "red"
    assert [x["id"] for x in client.get("/api/csv-imports").json()] == [cid]
    assert client.post("/api/csv-imports", files={"file": ("e.csv", b"  ", "text/csv")}).json()["code"] == "empty_file"


def test_cancel_arriving_just_before_commit_commits_nothing(client, tiff, detector, monkeypatch):
    from polony_app.review import engine

    aid = to_ready(client, tiff)
    detector.objects = [obj("a", 30, 30)]
    orig = engine.apply_detection_result

    def cancel_then_apply(*a, **k):
        for ev in list(client.app.state.runner._cancel.values()):
            ev.set()  # cancel lands after the pre-commit check, before the merge
        return orig(*a, **k)

    monkeypatch.setattr(engine, "apply_detection_result", cancel_then_apply)
    r = client.post(f"/api/analyses/{aid}/detect", json={}).json()
    j = wait_job(client, r["id"])
    assert j["status"] == "cancelled", j
    an = client.get(f"/api/analyses/{aid}").json()
    assert an["objects"] == [] and an["status"] == "ready" and an["provenance"]["detection_ran_at"] is None


def test_settings_changed_during_job_leaves_result_stale(client, tiff, detector, monkeypatch):
    aid = to_ready(client, tiff)
    detector.objects = [obj("a", 30, 30)]
    # change the stored request between submit and commit
    from polony_app.detection import pipeline
    real = pipeline.run_detection

    def det_and_change(*a, **k):
        res = real(*a, **k)
        r = client.put(f"/api/analyses/{aid}/settings", json={"detection_request": {"match_tolerance_px": 9.0}})
        assert r.status_code == 200, r.text
        return res

    monkeypatch.setattr(pipeline, "run_detection", det_and_change)
    r = client.post(f"/api/analyses/{aid}/detect", json={}).json()
    j = wait_job(client, r["id"])
    assert j["status"] == "done", j
    an = client.get(f"/api/analyses/{aid}").json()
    assert an["provenance"]["stale_detection"] is True


def test_provenance_carries_original_upload_filename(client, tmp_path):
    p = tmp_path / "My Slide 2026 (T7A).tif"
    write_two_channel_tiff(p)
    a = upload(client, p)
    prov = client.get(f"/api/analyses/{a['id']}/export", params={"kind": "provenance"}).json()
    names = prov["image"]["original_upload_filenames"]
    assert "My Slide 2026 (T7A).tif" in names.values()
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/analyses/{a['id']}/export?kind=bundle").content))
    assert "My Slide 2026 (T7A).tif" in json.loads(z.read("provenance.json"))["image"]["original_upload_filenames"].values()


def test_new_export_kinds_and_status_in_filename(client, tiff, detector):
    aid = to_ready(client, tiff)
    for kind, ext in (("actions", "jsonl"), ("history", "csv"), ("objects", "csv")):
        r = client.get(f"/api/analyses/{aid}/export", params={"kind": kind})
        assert r.status_code == 200
        assert f"_{kind}_DRAFT.{ext}" in r.headers["content-disposition"]


def test_manifest_written_for_two_file_and_composite_uploads(client, tmp_path):
    import numpy as np
    import tifffile

    r16 = (np.random.default_rng(1).random((40, 50)) * 4000).astype("uint16")
    pa, pb = tmp_path / "Red 635 (A).tif", tmp_path / "Green 532 (A).tif"
    tifffile.imwrite(pa, r16)
    tifffile.imwrite(pb, r16)
    pc = tmp_path / "Comp (RGB) 1.tif"
    tifffile.imwrite(pc, np.stack([r16 // 16, r16 // 16, r16 // 16], -1).astype("uint8"))
    for paths, expect in (([pa, pb], {"Red 635 (A).tif", "Green 532 (A).tif"}), ([pc], {"Comp (RGB) 1.tif"})):
        files = [("files", (p.name, open(p, "rb"), "image/tiff")) for p in paths]
        r = client.post("/api/analyses", files=files, data={"name": "m"})
        assert r.status_code == 200, r.text
        prov = client.get(f"/api/analyses/{r.json()['id']}/export", params={"kind": "provenance"}).json()
        assert set(prov["image"]["original_upload_filenames"].values()) == expect
