"""Experiments (stage 3 backend): CRUD, import with mapping copy, unattended batch run (skip + continue,
done_with_failures, cancel between slides, only_missing, restart), experiment CSV + provenance, lab export."""
import csv
import io
import json
import sqlite3
import sys
import time
import types

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from fakes import H, W, detector, obj, rect, write_two_channel_tiff  # noqa: F401
from polony_app.api.main import create_app
from polony_app.export import slide_save as ss
from polony_app.schemas import ObjectClass, Region, RegionKind

TERMINAL = ("done", "done_with_failures", "failed", "cancelled")


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def client(data_dir):
    with TestClient(create_app(data_dir)) as c:
        yield c


@pytest.fixture
def tiffs(tmp_path):
    out = []
    for i in range(3):
        p = tmp_path / f"slide_{i}.tif"
        write_two_channel_tiff(p, seed=i)
        out.append(p)
    return out


def new_exp(client, **kw):
    r = client.post("/api/experiments", json={"name": "Exp 1", **kw})
    assert r.status_code == 200, r.text
    return r.json()


def import_files(client, eid, paths):
    files = [("files", (p.name, open(p, "rb"), "image/tiff")) for p in paths]
    try:
        r = client.post(f"/api/experiments/{eid}/import", files=files)
    finally:
        for _, (_, fh, _) in files:
            fh.close()
    return r


MAPPING = {"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]}


def fields(n=2):
    """n numbered grid fields side by side."""
    w = 150 // n
    return [Region(id=f"f{i}", kind=RegionKind.counting, name=f"F{i}", field_index=i,
                   polygon=rect(5 + (i - 1) * w, 5, 5 + i * w, 115)).model_dump(mode="json") for i in range(1, n + 1)]


def ready_experiment(client, tiffs, n_fields=2):
    """Experiment with the 3 slides imported, mapped and with grid fields."""
    e = new_exp(client)
    eid = e["id"]
    created = import_files(client, eid, tiffs).json()["created"]
    ids = [c["id"] for c in created]
    for aid in ids:
        assert client.put(f"/api/analyses/{aid}/channels", json=MAPPING).status_code == 200
        assert client.put(f"/api/analyses/{aid}/regions", json={"regions": fields(n_fields)}).status_code == 200
    return eid, ids


def wait_batch(client, eid, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = client.get(f"/api/experiments/{eid}/run").json()
        if v["batch"] and v["batch"]["status"] in TERMINAL:
            return v
        time.sleep(0.02)
    raise AssertionError("batch did not finish")


class AidDetector:
    """Fake detector that fails / is slow for chosen analyses (the pipeline receives analysis_id)."""

    def __init__(self, base):
        self.base, self.fail, self.slow, self.seen = base, set(), 0.0, []

    def __call__(self, *a, **kw):
        aid = kw.get("analysis_id")
        self.seen.append(aid)
        if aid in self.fail:
            raise RuntimeError(f"engine exploded on {aid}")
        if self.slow:
            time.sleep(self.slow)
        return self.base(*a, **kw)


@pytest.fixture
def aid_detector(detector, monkeypatch):
    from polony_app.detection import pipeline

    d = AidDetector(detector)
    detector.objects = [obj("o1", 30, 30), obj("o2", 120, 90, ObjectClass.green), obj("o3", 80, 60, ObjectClass.yellow)]
    monkeypatch.setattr(pipeline, "run_detection", d)
    return d


# ------------------------------------------------------------------ CRUD
def test_crud_and_detach_on_delete(client, tiffs):
    assert client.post("/api/experiments", json={"name": "  "}).json()["code"] == "invalid_name"
    assert client.post("/api/experiments", json={"name": "x", "nope": 1}).json()["code"] == "unknown_fields"
    assert client.post("/api/experiments", json={"name": "x", "default_detection": {"engin": "x"}}).status_code == 422
    e = new_exp(client, notes="n1", default_grid={"rows": 4, "cols": 6})
    eid = e["id"]
    assert e["status"] == "open" and e["slides"] == [] and e["last_batch"] is None
    assert e["default_detection"]["engine"] == "bigfish" and e["default_grid"] == {"rows": 4, "cols": 6}
    lst = client.get("/api/experiments").json()
    assert [x["id"] for x in lst] == [eid] and lst[0]["n_slides"] == 0
    r = client.put(f"/api/experiments/{eid}", json={"name": "Renamed", "default_detection": {"engine": "hybrid"}})
    assert r.status_code == 200, r.text
    g = client.get(f"/api/experiments/{eid}").json()
    assert g["name"] == "Renamed" and g["notes"] == "n1" and g["default_detection"]["engine"] == "hybrid"
    assert g["default_grid"] == {"rows": 4, "cols": 6}          # untouched: key not in the PUT
    assert client.put(f"/api/experiments/{eid}", json={"default_grid": None}).json()["default_grid"] is None
    assert client.put(f"/api/experiments/{eid}", json={"default_detection": {"preview_window": [0, 0, 5, 5]}}
                      ).json()["code"] == "invalid_request"
    assert client.get("/api/experiments/ffffffffffff").status_code == 404
    # attach an existing analysis
    with open(tiffs[0], "rb") as f:
        aid = client.post("/api/analyses", files=[("files", (tiffs[0].name, f, "image/tiff"))]).json()["id"]
    v = client.post(f"/api/experiments/{eid}/slides", json={"analysis_ids": [aid]}).json()
    assert [s["id"] for s in v["slides"]] == [aid] and v["slides"][0]["order_in_experiment"] == 1
    row = next(x for x in client.get("/api/analyses").json() if x["id"] == aid)
    assert row["experiment_id"] == eid and row["order_in_experiment"] == 1
    assert client.get(f"/api/analyses/{aid}").json()["experiment_id"] == eid
    other = new_exp(client)["id"]
    assert client.post(f"/api/experiments/{other}/slides", json={"analysis_ids": [aid]}).json()["code"] == \
        "in_other_experiment"
    # delete: the slide is detached, not deleted
    d = client.delete(f"/api/experiments/{eid}").json()
    assert d["detached_slides"] == [aid] and d["deleted_slides"] == []
    assert client.get(f"/api/experiments/{eid}").status_code == 404
    a = client.get(f"/api/analyses/{aid}").json()
    assert a["experiment_id"] is None and a["order_in_experiment"] == 0


def test_delete_with_slides_and_detach_one(client, tiffs):
    eid = new_exp(client)["id"]
    ids = [c["id"] for c in import_files(client, eid, tiffs[:2]).json()["created"]]
    v = client.delete(f"/api/experiments/{eid}/slides/{ids[0]}").json()
    assert [s["id"] for s in v["slides"]] == [ids[1]]
    d = client.delete(f"/api/experiments/{eid}", params={"delete_slides": "true"}).json()
    assert d["deleted_slides"] == [ids[1]]
    assert client.get(f"/api/analyses/{ids[1]}").status_code == 404
    assert client.get(f"/api/analyses/{ids[0]}").status_code == 200


def test_existing_sqlite_db_is_migrated(tmp_path):
    """A data dir created before experiments existed: the columns are added, old rows read as NULL."""
    d = tmp_path / "old"
    d.mkdir()
    db = sqlite3.connect(str(d / "app.sqlite"))
    db.executescript("""
      CREATE TABLE analyses (id TEXT PRIMARY KEY, name TEXT, status TEXT, created_at TEXT, updated_at TEXT,
        finalized_at TEXT, summary TEXT, state TEXT NOT NULL);
      CREATE TABLE jobs (id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL, status TEXT, progress REAL,
        message TEXT, error TEXT, started_at TEXT, finished_at TEXT, created_at TEXT);
      INSERT INTO jobs VALUES ('aaaaaaaaaaaa','bbbbbbbbbbbb','done',1.0,'ok',NULL,NULL,NULL,'2026-01-01');""")
    db.close()
    from polony_app.store.db import Store

    s = Store(d)
    j = s.get_job("aaaaaaaaaaaa")
    assert j.kind == "detect" and j.parent_id is None and j.experiment_id is None
    cols = {r[1] for r in s._exec("PRAGMA table_info(analyses)").fetchall()}
    assert {"experiment_id", "order_in_experiment"} <= cols
    s.close()
    Store(d).close()   # idempotent


# ------------------------------------------------------------------ import
def test_import_three_files_and_mapping_copy(client, tiffs, tmp_path):
    eid = new_exp(client)["id"]
    first = import_files(client, eid, tiffs[:1]).json()["created"][0]
    assert first["mapping_copied_from"] is None and first["status"] == "needs_channel_mapping"
    swapped = {"mapping": [{"index": 0, "role": "green"}, {"index": 1, "role": "red"}]}
    assert client.put(f"/api/analyses/{first['id']}/channels", json=swapped).status_code == 200
    # different layout (8-bit): not copied; a non-TIFF: reported, the others still imported
    odd = tmp_path / "eight_bit.tif"
    tifffile.imwrite(str(odd), (np.random.default_rng(0).random((2, H, W)) * 200).astype(np.uint8))
    bad = tmp_path / "notes.txt"
    bad.write_text("x")
    r = import_files(client, eid, [tiffs[1], tiffs[2], odd, bad])
    assert r.status_code == 200, r.text
    out = r.json()
    assert [c["order_in_experiment"] for c in out["created"]] == [2, 3, 4]
    assert [c["mapping_copied_from"] for c in out["created"]] == [first["id"], first["id"], None]
    assert out["errors"] == [{"file": "notes.txt", "code": "unsupported_file",
                              "message": "'notes.txt': only .tif/.tiff files are accepted"}]
    for c in out["created"][:2]:
        a = client.get(f"/api/analyses/{c['id']}").json()
        assert a["status"] == "ready" and [ch["role"] for ch in a["image"]["channels"]] == ["green", "red"]
        assert all(ch["role_confirmed"] for ch in a["image"]["channels"])
    a = client.get(f"/api/analyses/{out['created'][2]['id']}").json()
    assert a["status"] == "needs_channel_mapping"
    v = client.get(f"/api/experiments/{eid}").json()
    assert [s["name"] for s in v["slides"]] == ["slide_0", "slide_1", "slide_2", "eight_bit"]
    assert [s["mapping_confirmed"] for s in v["slides"]] == [True, True, True, False]
    # only bad files: 422 and nothing created
    r = import_files(client, eid, [bad])
    assert r.status_code == 422 and r.json()["code"] == "import_failed"
    assert len(client.get(f"/api/experiments/{eid}").json()["slides"]) == 4
    # reorder
    ids = [s["id"] for s in v["slides"]]
    assert client.put(f"/api/experiments/{eid}/order", json={"analysis_ids": ids[:2]}).status_code == 422
    v = client.put(f"/api/experiments/{eid}/order", json={"analysis_ids": ids[::-1]}).json()
    assert [s["id"] for s in v["slides"]] == ids[::-1]


# ------------------------------------------------------------------ batch run
def test_run_skips_failed_slide_and_continues(client, tiffs, aid_detector):
    eid, ids = ready_experiment(client, tiffs)
    aid_detector.fail = {ids[1]}
    r = client.post(f"/api/experiments/{eid}/run", json={})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["kind"] == "batch" and b["experiment_id"] == eid and b["analysis_id"] == f"experiment:{eid}"
    v = wait_batch(client, eid)
    assert v["batch"]["status"] == "done_with_failures"
    assert v["batch"]["message"].startswith("2 / 3 done, 1 failed")
    assert [s["status"] for s in v["slides"]] == ["done", "failed", "done"]
    assert "engine exploded" in v["slides"][1]["error"]
    assert aid_detector.seen == ids            # sequential, in experiment order, the third ran after the failure
    assert v["summary"] == {"total": 3, "done": 2, "failed": 1, "cancelled": 0, "running": 0, "queued": 0,
                            "pending": 0, "not_run": 0, "skipped": 0}
    # child jobs are ordinary jobs with parent_id
    child = client.get(f"/api/jobs/{v['slides'][0]['job_id']}").json()
    assert child["parent_id"] == b["id"] and child["kind"] == "detect" and child["status"] == "done"
    e = client.get(f"/api/experiments/{eid}").json()
    assert e["status"] == "open" and e["last_batch"]["id"] == b["id"]
    assert [s["status"] for s in e["slides"]] == ["in_review", "ready", "in_review"]
    assert e["slides"][1]["last_job"]["status"] == "failed"
    assert e["slides"][0]["counts"]["red_only"] == 1 and e["slides"][0]["n_fields"] == 2
    # retry only the failed slide
    aid_detector.fail = set()
    r = client.post(f"/api/experiments/{eid}/run", json={"analysis_ids": [ids[1]]})
    assert r.status_code == 200, r.text
    v = wait_batch(client, eid)
    assert v["batch"]["status"] == "done" and [s["status"] for s in v["slides"]] == ["done"]


def test_run_refusals_are_failed_children(client, tiffs, aid_detector):
    """No confirmed mapping / no counting region: failed with the reason, the batch continues."""
    eid = new_exp(client)["id"]
    ids = [c["id"] for c in import_files(client, eid, tiffs).json()["created"]]
    client.put(f"/api/analyses/{ids[0]}/channels", json=MAPPING)       # mapped, no region
    for aid in ids[2:]:
        client.put(f"/api/analyses/{aid}/channels", json=MAPPING)
        client.put(f"/api/analyses/{aid}/regions", json={"regions": fields()})
    client.post(f"/api/experiments/{eid}/run", json={})
    v = wait_batch(client, eid)
    assert v["batch"]["status"] == "done_with_failures"
    assert [s["status"] for s in v["slides"]] == ["failed", "failed", "done"]
    assert v["slides"][0]["error"].startswith("no_counting_region:")
    assert v["slides"][1]["error"].startswith("channel_mapping_unconfirmed:")
    assert aid_detector.seen == [ids[2]]


def test_run_validation(client, tiffs, aid_detector):
    eid, ids = ready_experiment(client, tiffs)
    assert client.post(f"/api/experiments/{eid}/run", json={"bogus": 1}).json()["code"] == "unknown_fields"
    assert client.post(f"/api/experiments/{eid}/run", json={"analysis_ids": ["ffffffffffff"]}).json()["code"] == \
        "not_in_experiment"
    assert client.post(f"/api/experiments/{eid}/run",
                       json={"overrides": {ids[0]: {"preview_window": [0, 0, 4, 4]}}}).json()["code"] == "invalid_request"
    empty = new_exp(client)["id"]
    assert client.post(f"/api/experiments/{empty}/run", json={}).json()["code"] == "nothing_to_run"
    assert client.get(f"/api/experiments/{empty}/run").json() == {"batch": None, "slides": [], "summary": None}


def test_cancel_between_slides_and_delete_refused(client, tiffs, aid_detector):
    eid, ids = ready_experiment(client, tiffs)
    aid_detector.slow = 0.4
    b = client.post(f"/api/experiments/{eid}/run", json={}).json()
    t0 = time.time()
    while not aid_detector.seen and time.time() - t0 < 10:
        time.sleep(0.01)
    assert client.get(f"/api/experiments/{eid}").json()["status"] == "running"
    assert client.delete(f"/api/experiments/{eid}").json()["code"] == "batch_in_progress"
    assert client.post(f"/api/experiments/{eid}/run", json={}).json()["code"] == "batch_in_progress"
    client.post(f"/api/jobs/{b['id']}/cancel")
    v = wait_batch(client, eid)
    assert v["batch"]["status"] == "cancelled" and v["batch"]["message"].startswith("cancelled after 1 of 3")
    assert [s["status"] for s in v["slides"]] == ["done", "not_run", "not_run"]   # the current slide finished
    assert aid_detector.seen == [ids[0]]
    assert client.delete(f"/api/experiments/{eid}").status_code == 200


def test_only_missing_and_finalized_are_skipped(client, tiffs, aid_detector):
    eid, ids = ready_experiment(client, tiffs)
    client.post(f"/api/experiments/{eid}/run", json={"analysis_ids": [ids[0]]})
    assert wait_batch(client, eid)["batch"]["status"] == "done"
    aid_detector.seen.clear()
    client.post(f"/api/experiments/{eid}/run", json={"only_missing": True})
    v = wait_batch(client, eid)
    assert [(s["status"], s["skip_reason"]) for s in v["slides"]] == [("skipped", "already_detected"),
                                                                     ("done", None), ("done", None)]
    assert aid_detector.seen == ids[1:] and v["summary"]["total"] == 2 and v["summary"]["skipped"] == 1
    assert v["batch"]["message"] == "2 / 2 done, 0 failed, 1 skipped"
    assert client.post(f"/api/experiments/{eid}/run", json={"only_missing": True}).json()["code"] == "nothing_to_run"


def test_overrides_and_spotiflow_prepared_once(client, tiffs, aid_detector, monkeypatch):
    calls = []
    stub = types.ModuleType("polony_app.engines.spotiflow_adapter")
    stub.ensure_model = lambda name, cache, progress: calls.append(name) or types.SimpleNamespace(cached=True)
    stub.model_cache_dir = lambda: "/nonexistent"
    monkeypatch.setitem(sys.modules, "polony_app.engines.spotiflow_adapter", stub)
    eid, ids = ready_experiment(client, tiffs)
    client.put(f"/api/experiments/{eid}", json={"default_detection": {"engine": "hybrid"}})
    from polony_app.detection import pipeline

    seen = []
    base = pipeline.run_detection

    def spy(red, green, regions, request, *a, **kw):
        seen.append((kw.get("analysis_id"), request.engine.value, request.red.threshold))
        return base(red, green, regions, request, *a, **kw)

    monkeypatch.setattr(pipeline, "run_detection", spy)
    r = client.post(f"/api/experiments/{eid}/run",
                    json={"overrides": {ids[1]: {"engine": "bigfish", "red": {"threshold": 123.0}}}})
    assert r.status_code == 200, r.text
    v = wait_batch(client, eid)
    assert v["batch"]["status"] == "done", v
    assert calls == ["general"]                                  # once, before the first slide
    assert seen == [(ids[0], "hybrid", None), (ids[1], "bigfish", 123.0), (ids[2], "hybrid", None)]
    assert [s["override"] for s in v["slides"]] == [False, True, False]
    assert "model 'general' ready" in v["batch"]["message"]


def test_batch_left_running_fails_on_restart(data_dir, tiffs, aid_detector):
    with TestClient(create_app(data_dir)) as c:
        eid, ids = ready_experiment(c, tiffs)
        st = c.app.state.store
        j = st.create_job(f"experiment:{eid}", kind="batch", experiment_id=eid,
                          detail={"slides": [{"analysis_id": ids[0], "name": "s", "order": 1, "skip": None}]})
        from polony_app.schemas import JobStatus

        j.status = JobStatus.running
        st.save_job(j)
    with TestClient(create_app(data_dir)) as c2:
        v = c2.get(f"/api/experiments/{eid}/run").json()
        assert v["batch"]["status"] == "failed" and "server restarted" in v["batch"]["error"]
        assert [s["status"] for s in v["slides"]] == ["not_run"]
        assert c2.get(f"/api/experiments/{eid}").json()["status"] == "open"


# ------------------------------------------------------------------ exports
def test_export_csv_and_provenance(client, tiffs, aid_detector):
    eid, ids = ready_experiment(client, tiffs)
    aid_detector.fail = {ids[2]}           # third slide: no counts -> not in the CSV
    client.post(f"/api/experiments/{eid}/run", json={})
    wait_batch(client, eid)
    r = client.get(f"/api/experiments/{eid}/export.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == 'attachment; filename="Exp_1_experiment.csv"'
    assert r.headers["x-slides-included"] == "2" and r.headers["x-slides-skipped"] == "1"
    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert rows[0] == ["Slide name", "What counted", "NofPol", "NofGrids"] + [f"Field{i}" for i in range(1, 25)] + \
        ["Experiment", "Status", "Engine"]
    assert rows[0][:28] == ss.EXPERIMENT_HEADER
    blank = [""] * 22
    # objects: red at x=30 (field 1), green at x=120 (field 2), yellow at x=80 (field 2)
    assert rows[1:] == [
        ["slide_0", "red, not yellow", "1", "2", "1", "0"] + blank + ["Exp 1", "draft", "bigfish"],
        ["slide_0", "green, not yellow", "1", "2", "0", "1"] + blank + ["Exp 1", "draft", "bigfish"],
        ["slide_0", "yellow", "1", "2", "0", "1"] + blank + ["Exp 1", "draft", "bigfish"],
        ["slide_1", "red, not yellow", "1", "2", "1", "0"] + blank + ["Exp 1", "draft", "bigfish"],
        ["slide_1", "green, not yellow", "1", "2", "0", "1"] + blank + ["Exp 1", "draft", "bigfish"],
        ["slide_1", "yellow", "1", "2", "0", "1"] + blank + ["Exp 1", "draft", "bigfish"],
    ]
    r = client.get(f"/api/experiments/{eid}/export.csv", params=[("series", "red,yellow")])
    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert [x[:6] for x in rows[1:]] == [["slide_0", "red and yellow", "2", "2", "1", "1"],
                                         ["slide_1", "red and yellow", "2", "2", "1", "1"]]
    assert client.get(f"/api/experiments/{eid}/export.csv", params={"series": "blue"}).status_code == 422
    # provenance
    r = client.get(f"/api/experiments/{eid}/provenance.json")
    assert r.headers["content-disposition"] == 'attachment; filename="Exp_1_provenance.json"'
    p = r.json()
    batch = client.get(f"/api/experiments/{eid}/run").json()
    assert p["experiment"]["id"] == eid and p["experiment"]["default_detection"]["engine"] == "bigfish"
    assert p["last_batch"]["job_id"] == batch["batch"]["id"] and p["last_batch"]["status"] == "done_with_failures"
    assert [s["analysis_id"] for s in p["slides"]] == ids
    assert [s["batch"]["status"] for s in p["slides"]] == ["done", "done", "failed"]
    assert all(s["batch"]["batch_job_id"] == batch["batch"]["id"] for s in p["slides"])
    assert p["slides"][0]["counted_fields"] == [1, 2] and "grid" in p["slides"][0]
    assert p["slides"][0]["detection_request"]["engine"] == "bigfish"
    assert p["csv"]["skipped_slides"] == [{"analysis_id": ids[2], "name": "slide_2", "reason": "no_detection"}]
    assert [x["what_counted"] for x in p["csv"]["series"]] == ["red, not yellow", "green, not yellow", "yellow"]


def test_lab_export_by_experiment(client, tiffs, aid_detector):
    eid, ids = ready_experiment(client, tiffs)
    assert client.post("/api/lab-export", json={"experiment_id": eid}).json()["code"] == "no_finalized_slides"
    assert client.post("/api/lab-export", json={"experiment_id": eid, "analysis_ids": ids}).json()["code"] == \
        "invalid_request"
    assert client.post("/api/lab-export", json={}).json()["code"] == "invalid_request"
    assert client.post("/api/lab-export", json={"experiment_id": "ffffffffffff"}).status_code == 404
    client.post(f"/api/experiments/{eid}/run", json={"analysis_ids": [ids[0]]})
    wait_batch(client, eid)
    a = client.get(f"/api/analyses/{ids[0]}").json()
    for r in a["regions"]:
        assert client.post(f"/api/analyses/{ids[0]}/actions", json={
            "type": "region_inspected", "payload": {"region_id": r["id"], "inspected": True}}).status_code == 200
        assert client.post(f"/api/analyses/{ids[0]}/actions",
                           json={"type": "approve_region", "payload": {"region_id": r["id"]}}).status_code == 200
    assert client.post(f"/api/analyses/{ids[0]}/finalize").status_code == 200
    # the experiment selection reaches the lab exporter: same answer as asking for the one finalized slide
    by_exp = client.post("/api/lab-export", json={"experiment_id": eid, "format": "csv", "series": "T7A"})
    by_ids = client.post("/api/lab-export", json={"analysis_ids": [ids[0]], "format": "csv", "series": "T7A"})
    assert by_exp.status_code == by_ids.status_code
    assert by_exp.content == by_ids.content
    csv_rows = client.get(f"/api/experiments/{eid}/export.csv").content.decode("utf-8-sig").splitlines()
    assert csv_rows[1].endswith("Exp 1,finalized,bigfish")
    # a finalized slide is skipped by a run, and delete_slides refuses while it is finalized
    v = client.post(f"/api/experiments/{eid}/run", json={})
    assert v.status_code == 200
    v = wait_batch(client, eid)
    assert v["slides"][0]["skip_reason"] == "finalized"
    assert client.delete(f"/api/experiments/{eid}", params={"delete_slides": True}).json()["code"] == \
        "analysis_finalized"
    assert json.loads(client.get(f"/api/experiments/{eid}").content)["slides"][0]["export_status"] == "finalized"


# ------------------------------------------------------------------ Postgres
PG_MIGRATION_TABLES = {"experiments"}


def test_pg_migration_adds_experiment_columns(tmp_path, pg_schema):
    """Needs POLONY_TEST_DATABASE_URL (skipped otherwise, like test_store_pg.py): an existing v2 schema gets the
    experiments table and the new analyses/jobs columns; old rows stay valid."""
    import os
    import uuid

    import psycopg

    from polony_app.config import Settings
    from polony_app.store import pg as pgmod

    url = os.environ["POLONY_TEST_DATABASE_URL"]
    with psycopg.connect(url, autocommit=True) as c:
        c.execute(f'CREATE SCHEMA "{pg_schema}"')
        c.execute(f'CREATE TABLE "{pg_schema}".schema_version (version integer PRIMARY KEY, '
                  "applied_at timestamptz NOT NULL DEFAULT now())")
        for v in (1, 2):
            for stmt in pgmod.MIGRATIONS[v]:
                c.execute(stmt.format(s=f'"{pg_schema}"'))
            c.execute(f'INSERT INTO "{pg_schema}".schema_version (version) VALUES (%s)', (v,))
        c.execute(f'INSERT INTO "{pg_schema}".jobs (id,analysis_id,status,created_at) '
                  "VALUES ('aaaaaaaaaaaa','bbbbbbbbbbbb','done','2026-01-01')")
    s = pgmod.PgStore(tmp_path / "pg", Settings(database_url=url, database_schema=pg_schema,
                                                machine_id=uuid.uuid4().hex), recover=False)
    j = s.get_job("aaaaaaaaaaaa")
    assert j.kind == "detect" and j.parent_id is None and j.experiment_id is None
    from polony_app.schemas import Experiment

    e = Experiment(id="cccccccccccc", name="E", created_at="t", updated_at="t", default_grid={"rows": 2})
    s.create_experiment(e)
    assert s.get_experiment(e.id).default_grid == {"rows": 2}
    aid = s.create_analysis("x")
    with s.edit(aid) as st:
        st.analysis.experiment_id, st.analysis.order_in_experiment = e.id, 1
    assert s.experiment_slide_ids(e.id) == [aid]
    s.close()
