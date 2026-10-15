"""Two-engine API: hybrid detect wiring (fake detector), audit, staleness, endpoints, exports."""
import csv
import io
import json
import zipfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fakes import counting, detector, obj, rect, wait_job, write_two_channel_tiff  # noqa: F401
from polony_app.api.main import create_app
from polony_app.schemas import ChannelEvidence, ConsensusStatus as CS, EngineName, EngineRunInfo, ObjectClass, RawDetection, ReviewReason as R


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def client(data_dir):
    with TestClient(create_app(data_dir)) as c:
        yield c


@pytest.fixture
def aid(client, tmp_path):
    p = tmp_path / "s.tif"
    write_two_channel_tiff(p)
    with open(p, "rb") as f:
        a = client.post("/api/analyses", files=[("files", (p.name, f, "image/tiff"))], data={"name": "s"}).json()
    i = a["id"]
    assert client.put(f"/api/analyses/{i}/channels",
                      json={"mapping": [{"index": 0, "role": "red"}, {"index": 1, "role": "green"}]}).status_code == 200
    assert client.put(f"/api/analyses/{i}/regions",
                      json={"regions": [counting("c1", rect(5, 5, 155, 115)).model_dump(mode="json")]}).status_code == 200
    return i


def cobj(oid, x, y, status, cls=ObjectClass.red, **kw):
    o = obj(oid, x, y, cls, **kw)
    for ch in ("red", "green"):
        e = getattr(o, ch)
        if e.detected:
            setattr(o, ch, e.model_copy(update={"consensus_status": status, "bigfish_x": x, "bigfish_y": y,
                                                 "bigfish_score": 4.2, "spotiflow_x": x + .5, "spotiflow_y": y,
                                                 "spotiflow_score": 0.87, "match_distance_px": 0.5}))
    return o


class HybridFake:
    """Replaces pipeline.run_detection: returns objects + engine runs (+ a failed Spotiflow run)."""

    def __init__(self):
        self.objects, self.fail_spotiflow, self.calls, self.kwargs = [], False, 0, {}
        self.request = None

    def __call__(self, red, green, regions, request, progress, cancelled, window=None, alignment=(0, 0), **kw):
        self.calls += 1
        self.kwargs, self.request = kw, request
        aid = kw.get("analysis_id", "x" * 12)
        n = self.calls
        runs = [EngineRunInfo(run_id=f"bf{n}", analysis_id=aid, engine=EngineName.bigfish, page=0,
                              channel_role="red", engine_version="1", n_raw=2, n_valid=2, started_at="t",
                              params={"threshold_scale": 1.0}),
                EngineRunInfo(run_id=f"sp{n}", analysis_id=aid, engine=EngineName.spotiflow, page=0,
                              channel_role="red", model_name=request.spotiflow.model_name, model_version="9",
                              model_checksum="cd" * 32, device="cpu", started_at="t",
                              status="failed" if self.fail_spotiflow else "ok",
                              error="model could not be loaded" if self.fail_spotiflow else None,
                              preprocessing={"normalizer": "auto"})]
        raws = [RawDetection(id=f"rb{n}{i}", run_id=f"bf{n}", analysis_id=aid, engine=EngineName.bigfish, page=0,
                             channel_role="red", x=30.0 + i, y=30.0, score=3.1, score_kind="bigfish_log_response_noise_units")
                for i in range(2)]
        if not self.fail_spotiflow:
            raws += [RawDetection(id=f"rs{n}0", run_id=f"sp{n}", analysis_id=aid, engine=EngineName.spotiflow, page=0,
                                  channel_role="red", x=30.5, y=30.0, score=0.8, score_kind="spotiflow_prob_raw", valid=False,
                                  excluded_reason="outside")]
        return SimpleNamespace(objects=[o.model_copy(deep=True) for o in self.objects],
                               package_versions={"big-fish": "fake"}, notes=["fake hybrid"],
                               engine_runs=runs, raw_detections=raws)


@pytest.fixture
def hybrid(detector, monkeypatch):
    from polony_app.detection import pipeline

    h = HybridFake()
    monkeypatch.setattr(pipeline, "run_detection", h)
    return h


def run_detect(client, aid, **body):
    r = client.post(f"/api/analyses/{aid}/detect", json=body)
    assert r.status_code == 200, r.text
    return wait_job(client, r.json()["id"])


OBJS = lambda: [cobj("a1", 30, 30, CS.agreement), cobj("a2", 50, 30, CS.agreement),
                cobj("b1", 70, 30, CS.bigfish_only, review_reasons=[R.bigfish_only]),
                cobj("s1", 90, 30, CS.spotiflow_only, ObjectClass.green, review_reasons=[R.spotiflow_only]),
                obj("w", 110, 30, review_reasons=[R.weak_signal])]


def test_detect_accepts_engine_params_and_persists_results(client, aid, hybrid):
    hybrid.objects = OBJS()
    j = run_detect(client, aid, engine="hybrid", spotiflow={"model_name": "general", "prob_thresh": 0.4},
                   consensus_tolerance_px=3.0)
    assert j["status"] == "done" and "fake hybrid" not in j["message"]
    assert hybrid.request.engine.value == "hybrid" and hybrid.request.spotiflow.prob_thresh == 0.4
    assert hybrid.request.consensus_tolerance_px == 3.0
    assert hybrid.kwargs["analysis_id"] == aid and hybrid.kwargs["image_sha256"]
    a = client.get(f"/api/analyses/{aid}").json()
    assert a["provenance"]["engine"] == "hybrid" and "raw_detections" not in a
    assert len(json.dumps(a)) < 40_000
    runs = client.get(f"/api/analyses/{aid}/engine-runs").json()
    assert {r["engine"] for r in runs} == {"bigfish", "spotiflow"} and runs[1]["model_checksum"] == "cd" * 32
    r = client.get(f"/api/analyses/{aid}/raw-detections")
    assert len(r.json()) == 3 and r.headers["X-Total-Count"] == "3"
    assert len(client.get(f"/api/analyses/{aid}/raw-detections?engine=spotiflow").json()) == 1
    assert len(client.get(f"/api/analyses/{aid}/raw-detections?valid=false").json()) == 1
    assert len(client.get(f"/api/analyses/{aid}/raw-detections?run_id=bf1&limit=1").json()) == 1
    assert client.get(f"/api/analyses/{aid}/raw-detections?limit=0").status_code == 422
    assert client.get(f"/api/analyses/{aid}/raw-detections?engine=nope").status_code == 422
    assert client.get("/api/analyses/000000000000/engine-runs").status_code == 404


def test_unknown_engine_keys_still_refused(client, aid, hybrid):
    r = client.post(f"/api/analyses/{aid}/detect", json={"engine": "hybrid", "spotiflow": {"modell": "x"}})
    assert r.status_code == 422 and "spotiflow.modell" in r.json()["detail"]["fields"]
    assert client.post(f"/api/analyses/{aid}/detect", json={"engine": "magic"}).status_code == 422


def test_failed_spotiflow_in_hybrid_does_not_fail_job(client, aid, hybrid):
    hybrid.objects = OBJS()
    hybrid.fail_spotiflow = True
    j = run_detect(client, aid, engine="hybrid")
    assert j["status"] == "done" and j["error"] is None
    assert "WARNING" in j["message"] and "spotiflow" in j["message"] and "model could not be loaded" in j["message"]
    assert len(client.get(f"/api/analyses/{aid}").json()["objects"]) == 5
    failed = [r for r in client.get(f"/api/analyses/{aid}/engine-runs").json() if r["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["error"] == "model could not be loaded"


def test_audit_survives_reruns_and_actions_log_untouched(client, aid, hybrid):
    hybrid.objects = OBJS()
    run_detect(client, aid, engine="hybrid")
    client.post(f"/api/analyses/{aid}/actions", json={"type": "delete", "payload": {"object_ids": ["a1"]}})
    client.post(f"/api/analyses/{aid}/actions", json={"type": "confirm", "payload": {"object_ids": ["a2"]}})
    before = client.get(f"/api/analyses/{aid}/export?kind=actions").text
    run_detect(client, aid, engine="hybrid")
    runs = client.get(f"/api/analyses/{aid}/engine-runs").json()
    assert [r["run_id"] for r in runs] == ["bf1", "sp1", "bf2", "sp2"]      # older runs never deleted
    assert len(client.get(f"/api/analyses/{aid}/raw-detections", params={"all_runs": "true"}).json()) == 6  # audit: all kept
    assert len(client.get(f"/api/analyses/{aid}/raw-detections").json()) == 3                               # default: latest run only
    after = client.get(f"/api/analyses/{aid}/export?kind=actions").text
    assert after.startswith(before) and after.count("\n") == before.count("\n") + 1   # + rerun_detection only
    a = client.get(f"/api/analyses/{aid}").json()
    objs = {o["id"]: o for o in a["objects"]}
    assert objs["a1"]["deleted"] and objs["a2"]["reviewed"]                 # tombstone + reviewed preserved


def test_engine_params_change_marks_detection_stale(client, aid, hybrid):
    hybrid.objects = OBJS()
    run_detect(client, aid, engine="hybrid", spotiflow={"model_name": "general"})
    assert client.get(f"/api/analyses/{aid}").json()["provenance"]["stale_detection"] is False

    def settings(**req):
        r = client.put(f"/api/analyses/{aid}/settings", json={"detection_request": req})
        assert r.status_code == 200, r.text
        return r.json()["provenance"]["stale_detection"]

    base = {"engine": "hybrid", "spotiflow": {"model_name": "general"}}
    assert settings(**base) is False
    assert settings(engine="bigfish", spotiflow={"model_name": "general"}) is True
    assert settings(engine="hybrid", spotiflow={"model_name": "other_model"}) is True
    assert settings(engine="hybrid", spotiflow={"model_name": "general", "prob_thresh": 0.3}) is True
    assert settings(**base, consensus_tolerance_px=9.0) is True
    assert settings(**base) is False                                       # set back => not stale


def test_consensus_summary_and_qc_sample_endpoints(client, aid, hybrid):
    hybrid.objects = OBJS() + [cobj(f"q{i}", 10 + 2 * i, 60, CS.agreement) for i in range(40)]
    run_detect(client, aid, engine="hybrid")
    client.post(f"/api/analyses/{aid}/actions", json={"type": "delete", "payload": {"object_ids": ["b1"]}})
    client.post(f"/api/analyses/{aid}/actions", json={"type": "add", "payload": {"x": 100, "y": 100, "cls": "red"}})
    s = client.get(f"/api/analyses/{aid}/consensus/summary").json()
    red = next(p for p in s["pages"] if p["channel_role"] == "red")
    assert red["page"] == 0 and red["human_added"] == 1 and red["human_rejected"] == 1
    assert s["total"]["spotiflow_only"] == 1 and s["total"]["agreement"] == 42
    # unflagged agreements are not queued; disagreements are
    ids = [q["object_id"] for q in client.get(f"/api/analyses/{aid}/review-queue").json()]
    assert "s1" in ids and "a1" not in ids and "b1" not in ids
    r1 = client.post(f"/api/analyses/{aid}/consensus/qc-sample", json={"n": 10, "seed": 3}).json()
    r2 = client.post(f"/api/analyses/{aid}/consensus/qc-sample", json={"n": 10, "seed": 3}).json()
    assert r1["marked"] == r2["marked"] == r1["n"] == 10 and r2["newly_marked"] == 0
    assert r1["qc"]["sample_size"] == 10 and r1["qc"]["ci95"] is None and "not ground truth" in r1["note"]
    assert s["qc"]["sample_size"] == 0 and s["qc"]["n_for_margin_0_05"] > 0
    # legacy body (fraction only) still works and maps to max(40, fraction * N) clamped to the eligible agreements
    rf = client.post(f"/api/analyses/{aid}/consensus/qc-sample", json={"fraction": 0.05, "seed": 3}).json()
    assert rf["n"] == min(40, rf["eligible"]) and rf["fraction"] == 0.05
    r1 = rf
    q = client.get(f"/api/analyses/{aid}/review-queue").json()
    qc = [i for i in q if i["reasons"] == ["qc_sample"]]
    assert len(qc) == r1["marked"] and q[-1]["reasons"] == ["qc_sample"]      # after every disagreement
    assert client.post(f"/api/analyses/{aid}/consensus/qc-sample", json={"fraction": 2}).status_code == 422
    assert client.post(f"/api/analyses/{aid}/consensus/qc-sample", json={"n": -1}).status_code == 422
    assert client.get(f"/api/analyses/{aid}/consensus/summary").json()["qc"]["sample_size"] == r1["marked"]
    assert client.post(f"/api/analyses/{aid}/undo").status_code == 200


def test_engines_listing_and_prepare(client, monkeypatch):
    e = client.get("/api/engines").json()
    bf, sp = e
    assert bf["name"] == "bigfish" and bf["available"] is True
    assert sp["name"] == "spotiflow" and isinstance(sp["available"], bool)
    assert sp["available"] or sp["reason"]
    import sys
    import types

    ad = types.ModuleType("polony_app.engines.spotiflow_adapter")
    calls = []
    ad.available = lambda: (True, "")
    ad.version = lambda: "9.9"
    ad.list_models = lambda: [{"name": "general", "cached": True, "version": "1", "checksum": "ab"}]

    def ensure_model(name, cache_dir, progress):
        calls.append((name, str(cache_dir)))
        progress(0.5, "downloading")
        return SimpleNamespace(name=name, cached=False, checksum_sha256="f" * 64)

    ad.ensure_model = ensure_model
    import polony_app.engines as pkg

    monkeypatch.setitem(sys.modules, "polony_app.engines.spotiflow_adapter", ad)
    monkeypatch.setattr(pkg, "spotiflow_adapter", ad, raising=False)
    sp = client.get("/api/engines").json()[1]
    assert sp["available"] is True and sp["version"] == "9.9" and sp["models"][0]["name"] == "general"
    r = client.post("/api/engines/spotiflow/prepare", json={"model_name": "general"})
    assert r.status_code == 200
    j = wait_job(client, r.json()["id"])
    assert j["status"] == "done" and "general" in j["message"] and calls and calls[0][0] == "general"


def test_prepare_without_adapter_is_409(client, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "polony_app.engines.spotiflow_adapter", None)   # import raises
    r = client.post("/api/engines/spotiflow/prepare", json={"model_name": "general"})
    assert r.status_code == 409 and r.json()["code"] == "engine_unavailable"
    sp = client.get("/api/engines").json()[1]
    assert sp["available"] is False and sp["reason"]


def test_exports_columns_bundle_and_provenance(client, aid, hybrid):
    hybrid.objects = OBJS()
    run_detect(client, aid, engine="hybrid")
    csv_text = client.get(f"/api/analyses/{aid}/export?kind=objects").text
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    for ch in ("red", "green"):
        for c in ("consensus_status", "bigfish_x", "bigfish_y", "bigfish_loG_noise_units", "spotiflow_x",
                  "spotiflow_y", "spotiflow_model_score", "engine_match_distance_px"):
            assert f"{ch}_{c}" in rows[0]
    by = {r["id"]: r for r in rows}
    assert by["a1"]["red_consensus_status"] == "agreement" and by["a1"]["red_spotiflow_model_score"] == "0.87"
    assert by["a1"]["red_bigfish_loG_noise_units"] == "4.2" and by["a1"]["red_engine_match_distance_px"] == "0.5"
    assert by["s1"]["green_consensus_status"] == "spotiflow_only" and by["s1"]["red_consensus_status"] == ""
    assert by["w"]["red_consensus_status"] == "" and by["w"]["red_spotiflow_x"] == ""
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/analyses/{aid}/export?kind=bundle").content))
    names = z.namelist()
    assert "raw_detections.csv" in names and "engine_runs.json" in names
    raw = list(csv.DictReader(io.StringIO(z.read("raw_detections.csv").decode())))
    assert len(raw) == 3 and {"model_score", "loG_noise_units", "score_kind", "valid", "excluded_reason"} <= set(raw[0])
    sp = next(r for r in raw if r["engine"] == "spotiflow")
    assert sp["model_score"] == "0.8" and sp["loG_noise_units"] == "" and sp["valid"] == "false" and sp["excluded_reason"] == "outside"
    bf = next(r for r in raw if r["engine"] == "bigfish")
    assert bf["loG_noise_units"] == "3.1" and bf["model_score"] == ""
    assert len(json.loads(z.read("engine_runs.json"))) == 2
    readme = z.read("README.txt").decode()
    assert "AGREEMENT IS NOT TRUTH" in readme and "MODEL SCORE IS NOT A PROBABILITY" in readme
    blob = " ".join(z.read(n).decode().lower() for n in names if n.endswith(".csv"))
    assert "confidence" not in blob and "ci_" not in blob
    prov = json.loads(z.read("provenance.json"))
    sp_run = next(r for r in prov["engine_runs"] if r["engine"] == "spotiflow")
    assert sp_run["model_checksum"] == "cd" * 32 and sp_run["model_version"] == "9" and sp_run["device"] == "cpu"
    assert sp_run["preprocessing"] == {"normalizer": "auto"}
    assert json.loads(client.get(f"/api/analyses/{aid}/export?kind=provenance").text)["engine_runs"]


def test_bigfish_only_flow_without_engine_data_unchanged(client, aid, detector):
    detector.objects = [obj("d1", 30, 30)]
    j = run_detect(client, aid)
    assert j["status"] == "done" and j["message"] == "1 objects detected" or "WARNING" not in j["message"]
    assert client.get(f"/api/analyses/{aid}/engine-runs").json() == []
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/analyses/{aid}/export?kind=bundle").content))
    assert "raw_detections.csv" not in z.namelist()
    assert json.loads(z.read("provenance.json"))["engine_runs"] == []


def test_raw_detections_default_to_latest_run_per_engine_page(client, aid, hybrid):
    hybrid.objects = OBJS()
    run_detect(client, aid, engine="hybrid")
    run_detect(client, aid, engine="hybrid")           # a rerun: older runs stay stored for audit
    base = f"/api/analyses/{aid}/raw-detections"
    latest = client.get(base, params={"engine": "bigfish", "page": 0})
    assert [d["run_id"] for d in latest.json()] == ["bf2", "bf2"] and latest.headers["X-Total-Count"] == "2"
    every = client.get(base, params={"engine": "bigfish", "page": 0, "all_runs": "true"})
    assert sorted({d["run_id"] for d in every.json()}) == ["bf1", "bf2"] and every.headers["X-Total-Count"] == "4"
    old = client.get(base, params={"run_id": "bf1"})
    assert {d["run_id"] for d in old.json()} == {"bf1"}
    both = client.get(base).json()                       # no filters: latest run of every (engine, page)
    assert {d["run_id"] for d in both} <= {"bf2", "sp2"}
