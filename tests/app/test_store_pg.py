"""Shared PostgreSQL backend: several computers (= several PgStore instances, different data dirs)
working on the SAME schema.  Needs POLONY_TEST_DATABASE_URL; uses a throw-away ``test_<uuid>`` schema."""
import os
import threading
import time
import uuid

import psycopg
import pytest

from fakes import counting, make_state
from polony_app.config import Settings
from polony_app.review import engine
from polony_app.schemas import AnalysisStatus, JobStatus
from polony_app.store.jobs import JobRunner
from polony_app.store.pg import DatabaseUnreachable, PgStore

URL = os.environ.get("POLONY_TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not URL, reason="set POLONY_TEST_DATABASE_URL to run the Postgres tests")


def mk(tmp_path, schema, name=None, **kw):
    kw.setdefault("view_ttl", 0.0)
    kw.setdefault("recover", False)
    s = Settings(database_url=URL, database_schema=schema, machine_id=uuid.uuid4().hex)
    return PgStore(tmp_path / (name or uuid.uuid4().hex[:6]), s, **kw)


def prepare(store) -> str:
    aid = store.create_analysis("shared")
    base = make_state()
    with store.edit(aid) as st:
        st.analysis.image = base.analysis.image
        st.analysis.regions = [counting()]
        engine.assign_region_ids(st.analysis)
        st.analysis.status = AnalysisStatus.in_review
    return aid


def add(store, aid, x, y):
    with store.edit(aid) as st:
        engine.apply_action(st, "add", {"x": x, "y": y, "cls": "red"})


def sql(schema, q, args=()):
    with psycopg.connect(URL, autocommit=True) as c:
        c.execute(f'SET search_path TO "{schema}"')
        cur = c.execute(q, args)
        return cur.fetchall() if cur.description else None


def test_migrations_are_idempotent_and_versioned(tmp_path, pg_schema):
    a, b = mk(tmp_path, pg_schema), mk(tmp_path, pg_schema)
    rows = sql(pg_schema, "SELECT version FROM schema_version")
    assert rows == [(1,), (2,), (3,)]
    assert {r[0] for r in sql(pg_schema, "SELECT table_name FROM information_schema.tables WHERE table_schema=%s",
                              (pg_schema,))} >= {"analyses", "actions", "jobs", "csv_imports", "machines", "schema_version"}
    assert len(sql(pg_schema, "SELECT 1 FROM machines")) == 2
    a.close(); b.close()


def test_other_computer_sees_data_and_info(tmp_path, pg_schema):
    a, b = mk(tmp_path, pg_schema), mk(tmp_path, pg_schema)
    aid = prepare(a)
    add(a, aid, 20, 20)
    assert [x["id"] for x in b.list_analyses()] == [aid]
    with b.view(aid) as st:
        assert len(st.analysis.objects) == 1 and st.analysis.undo_depth == 1
    add(b, aid, 30, 30)          # B edits what A created
    with a.view(aid) as st:      # A sees B's edit (version check -> reload)
        assert len(st.analysis.objects) == 2 and len(st.log) == 2
    info = a.info()
    assert info["backend"] == "postgres" and info["connected"] and info["schema"] == pg_schema
    a.close(); b.close()


def test_concurrent_edits_from_two_computers_lose_nothing(tmp_path, pg_schema):
    a, b = mk(tmp_path, pg_schema), mk(tmp_path, pg_schema)
    aid = prepare(a)
    errs: list = []

    def worker(store, k):
        try:
            for i in range(6):
                add(store, aid, 10 + k * 10 + i, 20 + k)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=worker, args=(s, k)) for k, s in enumerate([a, a, b, b])]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs, errs
    with a.view(aid) as st:
        ids = [o.id for o in st.analysis.objects]
        assert len(ids) == 24 and len(set(ids)) == 24           # no lost update, ids unique
        assert st.analysis.undo_depth == 24 and len(st.undo) == 24
    seqs = [r[0] for r in sql(pg_schema, "SELECT seq FROM actions WHERE analysis_id=%s ORDER BY seq", (aid,))]
    assert seqs == list(range(1, 25))
    assert sql(pg_schema, "SELECT version FROM analyses WHERE id=%s", (aid,))[0][0] == 26   # create + prepare + 24
    # undo on B is seen consistently by A
    for _ in range(5):
        with b.edit(aid) as st:
            engine.undo(st)
    with a.view(aid) as st:
        assert st.analysis.undo_depth == 19 and st.analysis.redo_depth == 5
        assert len([o for o in st.analysis.objects if not o.deleted]) == 19
    a.close(); b.close()


def test_failed_edit_rolls_back_everything(tmp_path, pg_schema):
    a = mk(tmp_path, pg_schema)
    aid = prepare(a)
    add(a, aid, 20, 20)
    with pytest.raises(Exception):
        with a.edit(aid) as st:
            engine.apply_action(st, "add", {"x": 40, "y": 40, "cls": "red"})
            raise RuntimeError("boom")
    with a.view(aid) as st:
        assert len(st.analysis.objects) == 1 and len(st.log) == 1
    assert sql(pg_schema, "SELECT count(*) FROM actions")[0][0] == 1
    a.close()


def test_edit_with_stale_cache_refreshes_under_lock(tmp_path, pg_schema):
    a, b = mk(tmp_path, pg_schema, view_ttl=60.0), mk(tmp_path, pg_schema, view_ttl=60.0)
    aid = prepare(a)
    with a.view(aid): pass          # A caches
    add(b, aid, 50, 50)             # B changes it
    add(a, aid, 60, 60)             # A's edit must start from B's version (cache valid by TTL, but lock re-checks)
    with b.view(aid) as st:
        pass
    b2 = mk(tmp_path, pg_schema)
    with b2.view(aid) as st:
        assert len(st.analysis.objects) == 2 and [x.seq for x in st.log] == [1, 2]
    a.close(); b.close(); b2.close()


def test_delete_cascades(tmp_path, pg_schema):
    a = mk(tmp_path, pg_schema)
    aid = prepare(a)
    add(a, aid, 20, 20)
    a.delete_analysis(aid)
    assert sql(pg_schema, "SELECT count(*) FROM actions")[0][0] == 0 and not a.exists(aid)
    a.close()


# ------------------------------------------------------------------ jobs
def test_orphan_recovery_only_touches_own_or_stale_jobs(tmp_path, pg_schema):
    a = mk(tmp_path, pg_schema, stale_seconds=60)
    b = mk(tmp_path, pg_schema, stale_seconds=60)
    aid1, aid2 = prepare(a), prepare(a)
    j_a = a.create_job(aid1)         # owned by A, fresh heartbeat
    j_b = b.create_job(aid2)         # owned by B
    for j in (j_a, j_b):
        j.status = JobStatus.running
    a.save_job(j_a); b.save_job(j_b)
    with a.edit(aid1) as st:
        engine.refresh_status(st, job_running=True)
    # B restarts: only B's job may be failed; A's fresh job on another machine must survive
    b2 = PgStore(tmp_path / "b2", Settings(database_url=URL, database_schema=pg_schema,
                                           machine_id=b.machine_id), stale_seconds=60)
    assert a.get_job(j_a.id).status == JobStatus.running
    assert a.get_job(j_b.id).status == JobStatus.failed
    with a.view(aid1) as st:
        assert st.job_active and st.analysis.status == AnalysisStatus.detecting     # untouched
    # A's heartbeat becomes old (machine died) -> any machine may recover it
    sql(pg_schema, "UPDATE jobs SET heartbeat_at = now() - interval '5 minutes' WHERE id=%s", (j_a.id,))
    assert a.get_job(j_a.id).status == JobStatus.failed       # reported as dead immediately
    ids = b2.recover_orphans()
    assert j_a.id in ids
    with b2.view(aid1) as st:
        assert not st.job_active and st.analysis.status != AnalysisStatus.detecting
    a.close(); b.close(); b2.close()


def test_one_active_job_per_analysis_across_machines_and_stale_takeover(tmp_path, pg_schema):
    a, b = mk(tmp_path, pg_schema, stale_seconds=60), mk(tmp_path, pg_schema, stale_seconds=60)
    aid = prepare(a)
    j = a.create_job(aid)
    assert b.find_active_job(aid) == j.id
    with pytest.raises(RuntimeError, match="job_in_progress"):
        b.create_job(aid)
    sql(pg_schema, "UPDATE jobs SET heartbeat_at = now() - interval '5 minutes'")
    assert b.find_active_job(aid) is None
    j2 = b.create_job(aid)           # stale job is failed, a new one may start
    assert a.get_job(j.id).status == JobStatus.failed and j2.id != j.id
    a.close(); b.close()


def test_cancel_flag_crosses_machines_via_heartbeat(tmp_path, pg_schema):
    a, b = mk(tmp_path, pg_schema), mk(tmp_path, pg_schema)
    aid = prepare(a)
    runner = JobRunner(a)
    runner.heartbeat_interval = 0.1
    started = threading.Event()

    def work(progress, cancelled):
        started.set()
        t0 = time.time()
        while not cancelled() and time.time() - t0 < 10:
            time.sleep(0.02)
        return "cancelled" if cancelled() else "finished"

    job = runner.submit(aid, work)
    assert started.wait(5)
    assert JobRunner(b).active_job(aid) == job.id                   # visible from the other computer
    with pytest.raises(RuntimeError):
        runner.submit(aid, work)
    JobRunner(b).cancel(job.id)                                     # cancel pressed on the other computer
    t0 = time.time()
    while runner.get(job.id).status == JobStatus.running and time.time() - t0 < 5:
        time.sleep(0.05)
    assert runner.get(job.id).status == JobStatus.cancelled
    hb_before = sql(pg_schema, "SELECT heartbeat_at FROM jobs WHERE id=%s", (job.id,))[0][0]
    assert hb_before is not None
    runner.shutdown(); a.close(); b.close()


def test_heartbeat_keeps_long_job_alive(tmp_path, pg_schema):
    a = mk(tmp_path, pg_schema, stale_seconds=1.5)
    aid = prepare(a)
    runner = JobRunner(a)
    runner.heartbeat_interval = 0.3
    release = threading.Event()
    job = runner.submit(aid, lambda p, c: (release.wait(5), "ok")[1])
    time.sleep(2.5)                  # longer than stale_seconds without any progress call
    assert a.find_active_job(aid) == job.id and a.get_job(job.id).status == JobStatus.running
    release.set()
    t0 = time.time()
    while runner.get(job.id).status == JobStatus.running and time.time() - t0 < 5:
        time.sleep(0.05)
    assert runner.get(job.id).status == JobStatus.done
    runner.shutdown(); a.close()


def test_csv_import_roundtrip_and_blank_stays_none(tmp_path, pg_schema):
    from polony_app.store.csv_import import parse_csv

    a, b = mk(tmp_path, pg_schema), mk(tmp_path, pg_schema)
    imp = parse_csv(b"slide ID,field 1,field 2\nT7A,3,\n", "c.csv", "abc123abc123")
    a.save_csv_import(imp)
    got = b.get_csv_import("abc123abc123")
    assert got.rows[0].fields == [3, None] and got.rows[0].raw_fields == ["3", ""]
    assert len(b.list_csv_imports()) == 1
    a.close(); b.close()


def test_drop_schema_refuses_real_schema(tmp_path):
    s = Settings(database_url=URL, database_schema="polony", machine_id="c" * 32)
    st = PgStore.__new__(PgStore)
    st.schema = "polony"
    with pytest.raises(RuntimeError):
        st.drop_schema()
