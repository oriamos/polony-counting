"""Engine runs + raw detections persistence (SQLite and, when configured, PostgreSQL)."""
import pytest

from polony_app.config import load_settings
from polony_app.schemas import EngineName, EngineRunInfo, RawDetection
from polony_app.store import NotFound, open_store


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def store(data_dir):
    s = open_store(data_dir, load_settings(persist_machine_id=False))
    yield s
    s.close()


def run(rid, engine=EngineName.bigfish, page=0, status="ok", aid="x", **kw):
    return EngineRunInfo(run_id=rid, analysis_id=aid, engine=engine, page=page, status=status,
                         started_at="2026-01-01T00:00:00Z", **kw)


def det(i, rid, engine=EngineName.bigfish, page=0, valid=True, aid="x", x=1.0, y=2.0, score=3.5):
    return RawDetection(id=i, run_id=rid, analysis_id=aid, engine=engine, page=page, x=x, y=y, score=score,
                        score_kind="s", valid=valid, channel_role="red" if page == 0 else "green")


def test_round_trip_filters_and_paging(store):
    aid = store.create_analysis("a")
    runs = [run("r1", model_name=None, params={"k": 1}, preprocessing={"n": "none"}, device="cpu"),
            run("r2", EngineName.spotiflow, 0, model_name="general", model_version="1", model_checksum="ab" * 32),
            run("r3", EngineName.bigfish, 1)]
    dets = [det(f"b{i}", "r1", valid=i % 2 == 0, x=float(i)) for i in range(6)]
    dets += [det(f"s{i}", "r2", EngineName.spotiflow, score=0.9) for i in range(3)]
    dets += [det("g0", "r3", page=1)]
    store.save_engine_results(aid, runs, dets)

    got = store.get_engine_runs(aid)
    assert [r.run_id for r in got] == ["r1", "r2", "r3"]
    assert got[0].params == {"k": 1} and got[0].device == "cpu"
    assert got[1].model_checksum == "ab" * 32 and got[1].model_name == "general"

    assert len(store.get_raw_detections(aid)) == 10
    assert len(store.get_raw_detections(aid, engine="spotiflow")) == 3
    assert len(store.get_raw_detections(aid, engine=EngineName.bigfish, page=1)) == 1
    assert len(store.get_raw_detections(aid, run_id="r1")) == 6
    assert len(store.get_raw_detections(aid, run_id="r1", valid=True)) == 3
    assert len(store.get_raw_detections(aid, run_id="r1", valid=False)) == 3
    assert store.count_raw_detections(aid, run_id="r1", valid=True) == 3
    page1 = store.get_raw_detections(aid, run_id="r1", limit=4)
    page2 = store.get_raw_detections(aid, run_id="r1", limit=4, offset=4)
    assert [d.id for d in page1 + page2] == [f"b{i}" for i in range(6)]
    d = store.get_raw_detections(aid, engine="spotiflow", limit=1)[0]
    assert d.score == 0.9 and d.score_kind == "s" and d.engine == EngineName.spotiflow


def test_append_only_across_reruns_and_idempotent_run_id(store):
    aid = store.create_analysis("a")
    store.save_engine_results(aid, [run("r1")], [det("a", "r1"), det("b", "r1")])
    store.save_engine_results(aid, [run("r2")], [det("a", "r2")])      # same detection id, other run: kept apart
    assert [r.run_id for r in store.get_engine_runs(aid)] == ["r1", "r2"]
    assert len(store.get_raw_detections(aid, run_id="r1")) == 2 and len(store.get_raw_detections(aid, run_id="r2")) == 1
    store.save_engine_results(aid, [run("r2", n_raw=7)], [det("z", "r2")])   # retry of r2 replaces ONLY r2
    assert [d.id for d in store.get_raw_detections(aid, run_id="r2")] == ["z"]
    assert len(store.get_raw_detections(aid, run_id="r1")) == 2
    assert [r.n_raw for r in store.get_engine_runs(aid)] == [0, 7]
    store.save_engine_results(aid, [run("r4", status="failed", error="boom", engine=EngineName.spotiflow)], [])
    failed = [r for r in store.get_engine_runs(aid) if r.status == "failed"]
    assert failed and failed[0].error == "boom"


def test_analysis_state_does_not_grow_with_raw_detections(store):
    aid = store.create_analysis("a")
    before = len(store.view(aid).__enter__().analysis.model_dump_json())
    store.save_engine_results(aid, [run("r1")], [det(f"d{i}", "r1") for i in range(500)])
    with store.view(aid) as st:
        assert len(st.analysis.model_dump_json()) == before
        assert "raw_detections" not in st.analysis.model_dump()


def test_delete_analysis_deletes_engine_data_and_isolation(store):
    a, b = store.create_analysis("a"), store.create_analysis("b")
    store.save_engine_results(a, [run("r1", aid=a)], [det("d", "r1", aid=a)])
    store.save_engine_results(b, [run("r1", aid=b)], [det("d", "r1", aid=b)])
    store.delete_analysis(a)
    assert store.get_engine_runs(a) == [] and store.get_raw_detections(a) == []
    assert len(store.get_engine_runs(b)) == 1 and len(store.get_raw_detections(b)) == 1


def test_unknown_analysis_refused(store):
    with pytest.raises(NotFound):
        store.save_engine_results("0" * 12, [run("r1")], [])
