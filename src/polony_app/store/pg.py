"""Shared PostgreSQL (Neon) persistence backend: same interface as ``db.Store``.

* All tables live in one dedicated schema (default ``polony``) -> a restricted role needs
  rights on that schema only.  Migrations are idempotent and versioned (``schema_version``).
* Several computers can open the same analysis.  Every mutation is ONE transaction that takes
  ``SELECT ... FOR UPDATE`` on the analysis row, so two people editing the same analysis are
  serialised and no update is lost; undo/redo stacks live in the persisted state, so they stay
  consistent.  ``version`` increases on every committed edit.
* Round trips per request (pipelined, so each pipeline is one network round trip):
    view  (cache fresh, < VIEW_TTL s)        0
    view  (cache older)                      1   (version check; full reload only if changed)
    edit                                     2   (BEGIN+lock+read | UPDATE+INSERT actions+COMMIT)
* Original TIFFs are NOT in the database: they stay in ``data_dir/originals/<id>/`` on each
  computer; the database holds metadata + sha256 (inside the analysis state).
* The connection string is a secret: it is never logged, error messages are redacted.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlsplit

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from polony_app.config import DEFAULT_SCHEMA, Settings, _valid_schema, redact
from polony_app.review.counts import compute_counts
from polony_app.review.engine import migrate_legacy_state, now, refresh_status
from polony_app.review.state import AnalysisState, ReviewError
from polony_app.schemas import (
    Analysis,
    AnalysisStatus,
    CsvImport,
    EngineRunInfo,
    Experiment,
    Job,
    JobStatus,
    RawDetection,
    ReviewAction,
)

from .db import BATCH_RESTART_ERROR, NotFound, experiment_args, experiment_from_row, new_id, new_job

JOB_STALE_SECONDS = 120.0       # a running job whose heartbeat is older than this is orphaned
VIEW_TTL = 1.5                  # seconds a cached analysis may be served without a version check
SCHEMA_VERSION = 3

MIGRATIONS: dict[int, list[str]] = {
    1: [
        """CREATE TABLE IF NOT EXISTS {s}.analyses (
             id text PRIMARY KEY, name text, status text,
             created_at text, updated_at text, finalized_at text,
             summary jsonb, state jsonb NOT NULL,
             version integer NOT NULL DEFAULT 1,
             updated_by text, db_updated_at timestamptz NOT NULL DEFAULT now())""",
        """CREATE TABLE IF NOT EXISTS {s}.actions (
             analysis_id text NOT NULL REFERENCES {s}.analyses(id) ON DELETE CASCADE,
             seq integer NOT NULL, ts text, type text, payload jsonb,
             machine_id text, data jsonb NOT NULL,
             PRIMARY KEY (analysis_id, seq))""",
        """CREATE TABLE IF NOT EXISTS {s}.jobs (
             id text PRIMARY KEY, analysis_id text NOT NULL, status text, progress double precision,
             message text, error text, started_at text, finished_at text, created_at text,
             owner_machine_id text, heartbeat_at timestamptz NOT NULL DEFAULT now(),
             cancel_requested boolean NOT NULL DEFAULT false)""",
        "CREATE INDEX IF NOT EXISTS jobs_analysis_idx ON {s}.jobs (analysis_id)",
        # at most ONE active job per analysis, across all machines
        """CREATE UNIQUE INDEX IF NOT EXISTS jobs_one_active_idx ON {s}.jobs (analysis_id)
             WHERE status IN ('queued','running')""",
        """CREATE TABLE IF NOT EXISTS {s}.csv_imports (
             id text PRIMARY KEY, created_at text, data jsonb NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS {s}.machines (
             machine_id text PRIMARY KEY, first_seen timestamptz NOT NULL DEFAULT now(),
             last_seen timestamptz NOT NULL DEFAULT now(), app_version text)""",
    ],
    2: [   # two-engine architecture: append-only audit of engine runs + raw detections
        """CREATE TABLE IF NOT EXISTS {s}.engine_runs (
             analysis_id text NOT NULL REFERENCES {s}.analyses(id) ON DELETE CASCADE,
             run_id text NOT NULL, engine text, page integer, seq bigserial,
             data jsonb NOT NULL, PRIMARY KEY (analysis_id, run_id))""",
        "CREATE INDEX IF NOT EXISTS engine_runs_aid_idx ON {s}.engine_runs (analysis_id, engine, page)",
        """CREATE TABLE IF NOT EXISTS {s}.raw_detections (
             analysis_id text NOT NULL REFERENCES {s}.analyses(id) ON DELETE CASCADE,
             run_id text NOT NULL, id text NOT NULL, engine text, page integer, valid boolean,
             seq bigserial, data jsonb NOT NULL, PRIMARY KEY (analysis_id, run_id, id))""",
        "CREATE INDEX IF NOT EXISTS raw_det_aid_idx ON {s}.raw_detections (analysis_id, engine, page, valid)",
        "CREATE INDEX IF NOT EXISTS raw_det_run_idx ON {s}.raw_detections (analysis_id, run_id)",
    ],
    3: [   # experiments + batch runs (existing analyses / jobs stay valid: the new columns are NULL)
        """CREATE TABLE IF NOT EXISTS {s}.experiments (
             id text PRIMARY KEY, name text NOT NULL, created_at text, updated_at text, notes text,
             default_grid jsonb, default_detection jsonb, status text)""",
        "ALTER TABLE {s}.analyses ADD COLUMN IF NOT EXISTS experiment_id text",
        "ALTER TABLE {s}.analyses ADD COLUMN IF NOT EXISTS order_in_experiment integer",
        "CREATE INDEX IF NOT EXISTS analyses_experiment_idx ON {s}.analyses (experiment_id)",
        "ALTER TABLE {s}.jobs ADD COLUMN IF NOT EXISTS kind text DEFAULT 'detect'",
        "ALTER TABLE {s}.jobs ADD COLUMN IF NOT EXISTS parent_id text",
        "ALTER TABLE {s}.jobs ADD COLUMN IF NOT EXISTS experiment_id text",
        "ALTER TABLE {s}.jobs ADD COLUMN IF NOT EXISTS detail jsonb",
        "CREATE INDEX IF NOT EXISTS jobs_parent_idx ON {s}.jobs (parent_id)",
        "CREATE INDEX IF NOT EXISTS jobs_experiment_idx ON {s}.jobs (experiment_id, kind)",
    ],
}


class DatabaseUnreachable(ReviewError):
    def __init__(self, message: str = "the shared database is not reachable"):
        super().__init__("database_unreachable", message, 503)


def _summary(st: AnalysisState) -> str:
    try:
        return compute_counts(st.analysis)["slide"].model_dump_json()
    except Exception:  # pragma: no cover
        return "null"


class _Entry:
    __slots__ = ("st", "version", "checked", "saved_len")

    def __init__(self, st: AnalysisState, version: int):
        self.st, self.version, self.checked, self.saved_len = st, version, time.monotonic(), len(st.log)


class PgStore:
    backend = "postgres"
    supports_heartbeat = True

    def __init__(self, data_dir: str | Path, settings: Settings, *, stale_seconds: float = JOB_STALE_SECONDS,
                 pool_size: int = 10, recover: bool = True, view_ttl: Optional[float] = None):
        if not settings.database_url:
            raise ValueError("PgStore needs settings.database_url")
        self.data_dir = Path(data_dir).resolve()
        for sub in ("originals", "csv_imports", "tmp"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)
        self._url = settings.database_url
        self.schema = _valid_schema(settings.database_schema or DEFAULT_SCHEMA)
        self.machine_id = settings.machine_id
        self.stale_seconds = float(stale_seconds)
        self.view_ttl = float(os.environ.get("POLONY_PG_VIEW_TTL", VIEW_TTL) if view_ttl is None else view_ttl)
        self._pipeline = os.environ.get("POLONY_PG_PIPELINE", "1") != "0"
        s = self.schema
        self._A, self._X, self._J, self._C, self._M, self._V = (
            f'"{s}".analyses', f'"{s}".actions', f'"{s}".jobs', f'"{s}".csv_imports',
            f'"{s}".machines', f'"{s}".schema_version',
        )
        self._E = f'"{s}".experiments'
        self._cache: dict[str, _Entry] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()
        self._tls = threading.local()
        self._ready_ok = False
        self._last_attempt = 0.0
        self.last_error: Optional[str] = None
        self.connected = False
        self.stats = {"round_trips_hint": 0}
        u = urlsplit(self._url)
        self.host, self.database = u.hostname, (u.path or "/").lstrip("/")
        self._pool = ConnectionPool(
            self._url,
            kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row,
                    "connect_timeout": 8, "application_name": "polony-review"},
            min_size=1, max_size=int(pool_size), timeout=12, max_idle=90, max_lifetime=900,
            open=False, name="polony-pg",
        )
        self._pool.open(wait=False)
        try:
            self._ready()
            if recover:
                self.recover_orphans()
        except DatabaseUnreachable:
            pass  # the app still starts; every operation reports 503 until the database is back

    # ----------------------------------------------------------- connection handling
    def _fail(self, e: BaseException) -> DatabaseUnreachable:
        self.connected = False
        self.last_error = redact(f"{type(e).__name__}: {e}", self._url)[:300]
        self._ready_ok = False
        return DatabaseUnreachable(f"the shared database is not reachable ({self.last_error})")

    def _ready(self) -> None:
        if self._ready_ok:
            return
        with self._guard:
            if self._ready_ok:
                return
            if time.monotonic() - self._last_attempt < 2.0 and self.last_error:
                raise DatabaseUnreachable(f"the shared database is not reachable ({self.last_error})")
            self._last_attempt = time.monotonic()
            try:
                self._migrate()
                self.connected, self.last_error, self._ready_ok = True, None, True
            except (psycopg.Error, PoolTimeout) as e:
                raise self._fail(e) from None

    @contextmanager
    def _conn(self) -> Iterator[psycopg.Connection]:
        self._ready()
        try:
            with self._pool.connection() as conn:
                yield conn
            self.connected = True
        except (psycopg.OperationalError, psycopg.InterfaceError, PoolTimeout) as e:
            raise self._fail(e) from None

    def _pipe(self, conn):
        return conn.pipeline() if self._pipeline else nullcontext()

    def _q(self, sql: str, args: tuple = (), fetch: str = "none"):
        """One autocommit statement, one retry on a dead pooled connection."""
        for attempt in (0, 1):
            was_ready = self._ready_ok
            try:
                with self._conn() as conn:
                    cur = conn.execute(sql, args)
                    if fetch == "one":
                        return cur.fetchone()
                    if fetch == "all":
                        return cur.fetchall()
                    return cur.rowcount
            except DatabaseUnreachable:
                if attempt or not was_ready:
                    raise
                self._ready_ok = True  # retry once with a fresh connection

    def _migrate(self) -> None:
        s = self.schema
        with psycopg.connect(self._url, autocommit=True, prepare_threshold=None, row_factory=dict_row,
                             connect_timeout=8, application_name="polony-review") as conn:
            exists = conn.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name=%s", (s,)).fetchone()
            if not exists:
                conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{s}"')
            conn.execute(f'CREATE TABLE IF NOT EXISTS {self._V} (version integer PRIMARY KEY, '
                         "applied_at timestamptz NOT NULL DEFAULT now())")
            cur = conn.execute(f"SELECT coalesce(max(version),0) AS v FROM {self._V}").fetchone()["v"]
            if cur < SCHEMA_VERSION:
                with conn.transaction():
                    conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"polony:{s}",))
                    cur = conn.execute(f"SELECT coalesce(max(version),0) AS v FROM {self._V}").fetchone()["v"]
                    for v in sorted(MIGRATIONS):
                        if v > cur:
                            for stmt in MIGRATIONS[v]:
                                conn.execute(stmt.format(s=f'"{s}"'))
                            conn.execute(f"INSERT INTO {self._V} (version) VALUES (%s)", (v,))
            conn.execute(
                f"INSERT INTO {self._M} (machine_id, app_version) VALUES (%s, %s) "
                "ON CONFLICT (machine_id) DO UPDATE SET last_seen=now()",
                (self.machine_id, _app_version()))

    def close(self) -> None:
        try:
            self._pool.close(timeout=3)
        except Exception:  # noqa: BLE001
            pass

    def info(self) -> dict:
        latency = None
        try:
            t = time.perf_counter()
            self._q("SELECT 1", fetch="one")
            latency = round((time.perf_counter() - t) * 1000, 1)
        except DatabaseUnreachable:
            pass
        return {"backend": "postgres", "host": self.host, "database": self.database, "schema": self.schema,
                "machine_id": self.machine_id, "connected": self.connected and latency is not None,
                "latency_ms": latency, "last_error": self.last_error}

    def drop_schema(self) -> None:
        """Test helper: only for throw-away ``test_*`` schemas."""
        if not self.schema.startswith("test_"):
            raise RuntimeError("refusing to drop a non-test schema")
        with psycopg.connect(self._url, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')

    # ----------------------------------------------------------- locks
    def _lock_for(self, aid: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(aid, threading.RLock())

    # ----------------------------------------------------------- recovery
    def recover_orphans(self) -> list[str]:
        """Fail only jobs that are provably dead: owned by THIS machine (we just started, so
        they are gone) or whose heartbeat is older than ``stale_seconds`` (their machine
        crashed / lost network).  Jobs of other machines with a fresh heartbeat are untouched."""
        msg = "interrupted: the application was restarted while this job was running"
        rows = self._q(
            f"UPDATE {self._J} SET status='failed', error=CASE WHEN kind='batch' THEN %s ELSE %s END, finished_at=%s "
            "WHERE status IN ('queued','running') AND (owner_machine_id=%s OR "
            "heartbeat_at < now() - make_interval(secs => %s)) RETURNING id",
            (BATCH_RESTART_ERROR, msg, now(), self.machine_id, self.stale_seconds), fetch="all")
        ids = [r["id"] for r in rows]
        # an experiment marked running whose batch is no longer alive (on any machine) is open again
        self._q(f"UPDATE {self._E} e SET status='open' WHERE status='running' AND NOT EXISTS ("
                f"SELECT 1 FROM {self._J} j WHERE j.experiment_id=e.id AND j.kind='batch' "
                "AND j.status IN ('queued','running'))")
        stuck = self._q(
            f"SELECT a.id FROM {self._A} a WHERE (a.status='detecting' OR a.state->>'job_active'='true') "
            f"AND NOT EXISTS (SELECT 1 FROM {self._J} j WHERE j.analysis_id=a.id "
            "AND j.status IN ('queued','running'))", fetch="all")
        for r in stuck:
            with self.edit(r["id"]) as st:
                refresh_status(st, job_running=False)
        return ids

    # ----------------------------------------------------------- analyses
    def originals_dir(self, aid: str) -> Path:
        return self.data_dir / "originals" / aid

    def create_analysis(self, name: str) -> str:
        aid = new_id()
        ts = now()
        a = Analysis(id=aid, name=name, created_at=ts, updated_at=ts, status=AnalysisStatus.created)
        st = AnalysisState(analysis=a)
        st.sync_depths()
        self.originals_dir(aid).mkdir(parents=True, exist_ok=True)
        with self._lock_for(aid):
            self._q(
                f"INSERT INTO {self._A} (id,name,status,created_at,updated_at,finalized_at,summary,state,"
                "version,updated_by,experiment_id,order_in_experiment) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,1,%s,%s,%s)",
                (aid, a.name, a.status.value, a.created_at, a.updated_at, a.finalized_at,
                 _summary(st), json.dumps(st.to_json()), self.machine_id, a.experiment_id, a.order_in_experiment))
            self._cache[aid] = _Entry(st, 1)
        return aid

    def _build(self, row: dict, log: list[ReviewAction]) -> AnalysisState:
        st = AnalysisState.from_json(row["state"], log)
        migrate_legacy_state(st)
        return st

    def _fresh(self, aid: str) -> AnalysisState:
        """Cached state if the DB version still matches, else (re)loaded. Caller holds the lock."""
        ent = self._cache.get(aid)
        if ent is not None and time.monotonic() - ent.checked < self.view_ttl:
            return ent.st
        if ent is not None:
            r = self._q(f"SELECT version FROM {self._A} WHERE id=%s", (aid,), fetch="one")
            if r is None:
                self._cache.pop(aid, None)
                raise NotFound("analysis", aid)
            if r["version"] == ent.version:
                ent.checked = time.monotonic()
                return ent.st
        return self._reload(aid, ent)

    def _reload(self, aid: str, ent: Optional[_Entry]) -> AnalysisState:
        last = ent.saved_len and ent.st.log[ent.saved_len - 1].seq if ent and ent.saved_len else 0
        for attempt in (0, 1):
            was_ready = self._ready_ok
            try:
                with self._conn() as conn, self._pipe(conn):
                    c1 = conn.execute(f"SELECT version, state FROM {self._A} WHERE id=%s", (aid,))
                    c2 = conn.execute(f"SELECT data FROM {self._X} WHERE analysis_id=%s AND seq>%s ORDER BY seq",
                                      (aid, last))
                row, acts = c1.fetchone(), c2.fetchall()
                break
            except DatabaseUnreachable:
                if attempt or not was_ready:
                    raise
                self._ready_ok = True
        if row is None:
            self._cache.pop(aid, None)
            raise NotFound("analysis", aid)
        new = [ReviewAction.model_validate(a["data"]) for a in acts]
        log = (list(ent.st.log[: ent.saved_len]) if ent else []) + new
        st = self._build(row, log)
        self._cache[aid] = _Entry(st, row["version"])
        return st

    @contextmanager
    def view(self, aid: str) -> Iterator[AnalysisState]:
        """Locked read access. Do not mutate."""
        with self._lock_for(aid):
            yield self._fresh(aid)

    @contextmanager
    def edit(self, aid: str) -> Iterator[AnalysisState]:
        """One transaction: lock the analysis row, read, mutate (caller), write, commit."""
        with self._lock_for(aid):
            held = getattr(self._tls, "held", None)
            if held is None:
                held = self._tls.held = set()
            if aid in held:
                raise RuntimeError("nested store.edit() on the same analysis")
            ent = self._cache.get(aid)
            last = ent.st.log[ent.saved_len - 1].seq if ent and ent.saved_len else 0
            held.add(aid)
            try:
                with self._conn() as conn:
                    try:
                        st = self._locked_read(conn, aid, ent, last)
                        ent = self._cache[aid]
                        try:
                            yield st
                        except BaseException:
                            self._cache.pop(aid, None)  # discard possibly half-mutated state
                            raise
                        self._commit(conn, aid, st, ent)
                    except BaseException:
                        self._cache.pop(aid, None)
                        self._rollback(conn)
                        raise
            finally:
                held.discard(aid)

    def _rollback(self, conn) -> None:
        try:
            if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
                conn.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            pass

    def _locked_read(self, conn, aid: str, ent: Optional[_Entry], last: int) -> AnalysisState:
        with self._pipe(conn):
            conn.execute("BEGIN")
            conn.execute("SET LOCAL lock_timeout='30s'")
            conn.execute("SET LOCAL idle_in_transaction_session_timeout='180s'")
            c1 = conn.execute(f"SELECT version, state FROM {self._A} WHERE id=%s FOR UPDATE", (aid,))
            c2 = conn.execute(f"SELECT data FROM {self._X} WHERE analysis_id=%s AND seq>%s ORDER BY seq",
                              (aid, last))
        row = c1.fetchone()
        acts = c2.fetchall()
        if row is None:
            self._rollback(conn)
            raise NotFound("analysis", aid)
        if ent is not None and ent.version == row["version"]:
            ent.checked = time.monotonic()
            return ent.st
        new = [ReviewAction.model_validate(a["data"]) for a in acts]
        log = (list(ent.st.log[: ent.saved_len]) if ent else []) + new
        st = self._build(row, log)
        self._cache[aid] = _Entry(st, row["version"])
        return st

    def _commit(self, conn, aid: str, st: AnalysisState, ent: _Entry) -> None:
        a = st.analysis
        rows = [
            (aid, act.seq, act.ts, act.type.value if hasattr(act.type, "value") else str(act.type),
             json.dumps(act.payload), self.machine_id, act.model_dump_json())
            for act in st.log[ent.saved_len:]
        ]
        with self._pipe(conn):
            cu = conn.execute(
                f"UPDATE {self._A} SET name=%s,status=%s,updated_at=%s,finalized_at=%s,summary=%s::jsonb,"
                "state=%s::jsonb,version=version+1,updated_by=%s,db_updated_at=now(),experiment_id=%s,"
                "order_in_experiment=%s WHERE id=%s AND version=%s",
                (a.name, a.status.value, a.updated_at, a.finalized_at, _summary(st),
                 json.dumps(st.to_json()), self.machine_id, a.experiment_id, a.order_in_experiment, aid, ent.version))
            if rows:
                conn.cursor().executemany(
                    f"INSERT INTO {self._X} (analysis_id,seq,ts,type,payload,machine_id,data) "
                    "VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb) "
                    "ON CONFLICT (analysis_id,seq) DO UPDATE SET data=EXCLUDED.data", rows)
            conn.execute("COMMIT")
        if cu.rowcount != 1:  # pragma: no cover - the row lock makes this impossible
            self._cache.pop(aid, None)
            raise ReviewError("conflict", "the analysis was changed by someone else; reload and retry", 409)
        ent.version += 1
        ent.saved_len = len(st.log)
        ent.checked = time.monotonic()

    def exists(self, aid: str) -> bool:
        return self._q(f"SELECT 1 AS x FROM {self._A} WHERE id=%s", (aid,), fetch="one") is not None

    def list_analyses(self) -> list[dict]:
        rows = self._q(f"SELECT id,name,status,updated_at,finalized_at,summary,experiment_id,order_in_experiment "
                       f"FROM {self._A} ORDER BY updated_at DESC", fetch="all")
        return [{"id": r["id"], "name": r["name"], "status": r["status"], "updated_at": r["updated_at"],
                 "finalized_at": r["finalized_at"], "experiment_id": r["experiment_id"],
                 "order_in_experiment": r["order_in_experiment"] or 0, "counts": r["summary"]} for r in rows]

    def delete_analysis(self, aid: str) -> None:
        with self._lock_for(aid):
            n = self._q(f"DELETE FROM {self._A} WHERE id=%s", (aid,))
            if not n:
                raise NotFound("analysis", aid)
            self._q(f"DELETE FROM {self._J} WHERE analysis_id=%s", (aid,))
            self._cache.pop(aid, None)
            shutil.rmtree(self.originals_dir(aid), ignore_errors=True)

    # ----------------------------------------------------------- engine results (audit, append-only)
    def save_engine_results(self, aid: str, runs: list[EngineRunInfo],
                            detections: list[RawDetection]) -> None:
        """Append engine runs + raw detections of one detect job (one transaction). Earlier runs are
        never touched; only an identical run_id is replaced. Not part of the Analysis JSON."""
        if not self.exists(aid):
            raise NotFound("analysis", aid)
        E, D = f'"{self.schema}".engine_runs', f'"{self.schema}".raw_detections'
        try:
            with self._conn() as conn:
                with conn.transaction():
                    for r in runs:
                        conn.execute(f"DELETE FROM {D} WHERE analysis_id=%s AND run_id=%s", (aid, r.run_id))
                        conn.execute(f"DELETE FROM {E} WHERE analysis_id=%s AND run_id=%s", (aid, r.run_id))
                        conn.execute(
                            f"INSERT INTO {E} (analysis_id,run_id,engine,page,data) VALUES (%s,%s,%s,%s,%s::jsonb)",
                            (aid, r.run_id, r.engine.value, r.page, r.model_dump_json()))
                    if detections:
                        conn.cursor().executemany(
                            f"INSERT INTO {D} (analysis_id,run_id,id,engine,page,valid,data) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT (analysis_id,run_id,id) "
                            "DO UPDATE SET data=EXCLUDED.data, valid=EXCLUDED.valid",
                            [(aid, d.run_id, d.id, d.engine.value, d.page, bool(d.valid), d.model_dump_json())
                             for d in detections])
        except psycopg.errors.ForeignKeyViolation:
            raise NotFound("analysis", aid) from None

    def get_engine_runs(self, aid: str) -> list[EngineRunInfo]:
        rows = self._q(f'SELECT data FROM "{self.schema}".engine_runs WHERE analysis_id=%s ORDER BY seq',
                       (aid,), fetch="all")
        return [EngineRunInfo.model_validate(r["data"]) for r in rows]

    @staticmethod
    def _det_where(aid, engine, page, run_id, valid) -> tuple[str, list]:
        where, args = ["analysis_id=%s"], [aid]
        for col, v in (("engine", engine), ("page", page), ("run_id", run_id), ("valid", valid)):
            if v is not None:
                where.append(f"{col}=%s")
                args.append(getattr(v, "value", v))
        return " AND ".join(where), args

    def get_raw_detections(self, aid: str, engine=None, page: int | None = None, run_id: str | None = None,
                           valid: bool | None = None, limit: int | None = None, offset: int = 0) -> list[RawDetection]:
        w, args = self._det_where(aid, engine, page, run_id, valid)
        sql = f'SELECT data FROM "{self.schema}".raw_detections WHERE {w} ORDER BY seq'
        if limit is not None:
            sql += " LIMIT %s"
            args.append(int(limit))
        if offset:
            sql += " OFFSET %s"
            args.append(int(offset))
        rows = self._q(sql, tuple(args), fetch="all")
        return [RawDetection.model_validate(r["data"]) for r in rows]

    def count_raw_detections(self, aid: str, engine=None, page: int | None = None, run_id: str | None = None,
                             valid: bool | None = None) -> int:
        w, args = self._det_where(aid, engine, page, run_id, valid)
        r = self._q(f'SELECT count(*) AS n FROM "{self.schema}".raw_detections WHERE {w}', tuple(args), fetch="one")
        return int(r["n"])

    # ----------------------------------------------------------- jobs
    def create_job(self, aid: str, kind: str = "detect", parent_id: Optional[str] = None,
                   experiment_id: Optional[str] = None, detail: Optional[dict] = None,
                   status: JobStatus = JobStatus.queued, error: Optional[str] = None) -> Job:
        j = new_job(aid, kind, parent_id, experiment_id, detail, status, error)
        try:
            with self._conn() as conn:
                with conn.transaction():
                    conn.execute(
                        f"UPDATE {self._J} SET status='failed', error='stale: owner stopped sending heartbeats', "
                        "finished_at=%s WHERE analysis_id=%s AND status IN ('queued','running') "
                        "AND heartbeat_at < now() - make_interval(secs => %s)",
                        (now(), aid, self.stale_seconds))
                    conn.execute(
                        f"INSERT INTO {self._J} (id,analysis_id,status,progress,message,error,started_at,finished_at,"
                        "created_at,owner_machine_id,kind,parent_id,experiment_id,detail)"
                        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)",
                        (j.id, aid, j.status.value, j.progress, j.message, error, j.started_at, j.finished_at,
                         now(), self.machine_id, kind, parent_id, experiment_id,
                         json.dumps(detail) if detail is not None else None))
        except psycopg.errors.UniqueViolation:
            raise RuntimeError("job_in_progress") from None
        return j

    def save_job(self, j: Job) -> bool:
        """Persist progress + heartbeat. Returns True when a cancel was requested (by any machine)."""
        r = self._q(
            f"UPDATE {self._J} SET status=%s,progress=%s,message=%s,error=%s,started_at=%s,finished_at=%s,"
            "detail=%s::jsonb,heartbeat_at=now() WHERE id=%s RETURNING cancel_requested",
            (j.status.value, j.progress, j.message, j.error, j.started_at, j.finished_at,
             json.dumps(j.detail) if j.detail is not None else None, j.id), fetch="all")
        return bool(r and r[0]["cancel_requested"])

    def touch_job(self, jid: str) -> bool:
        r = self._q(f"UPDATE {self._J} SET heartbeat_at=now() WHERE id=%s AND status IN ('queued','running') "
                    "RETURNING cancel_requested", (jid,), fetch="all")
        return bool(r and r[0]["cancel_requested"])

    def request_cancel(self, jid: str) -> None:
        self._q(f"UPDATE {self._J} SET cancel_requested=true WHERE id=%s AND status IN ('queued','running')",
                (jid,))

    def find_active_job(self, aid: str) -> Optional[str]:
        r = self._q(f"SELECT id FROM {self._J} WHERE analysis_id=%s AND status IN ('queued','running') "
                    "AND heartbeat_at >= now() - make_interval(secs => %s) LIMIT 1",
                    (aid, self.stale_seconds), fetch="one")
        return r["id"] if r else None

    def get_job(self, jid: str) -> Job:
        r = self._q(f"SELECT *, extract(epoch from now() - heartbeat_at) AS age FROM {self._J} WHERE id=%s",
                    (jid,), fetch="one")
        if r is None:
            raise NotFound("job", jid)
        return self._job_row(r)

    def _job_row(self, r: dict) -> Job:
        status = JobStatus(r["status"])
        error = r["error"]
        if status in (JobStatus.queued, JobStatus.running):
            # an orphan whose owner vanished must not look alive forever
            if r["age"] > self.stale_seconds:
                status, error = JobStatus.failed, "stale: owner stopped sending heartbeats"
        return Job(id=r["id"], analysis_id=r["analysis_id"], status=status, progress=r["progress"] or 0.0,
                   message=r["message"] or "", error=error, started_at=r["started_at"],
                   finished_at=r["finished_at"], kind=r.get("kind") or "detect", parent_id=r.get("parent_id"),
                   experiment_id=r.get("experiment_id"), detail=r.get("detail"))

    def list_jobs(self, *, analysis_id: Optional[str] = None, parent_id: Optional[str] = None,
                  experiment_id: Optional[str] = None, kind: Optional[str] = None, limit: int = 500) -> list[Job]:
        """Matching jobs, newest first."""
        where, args = [], []
        for col, v in (("analysis_id", analysis_id), ("parent_id", parent_id), ("experiment_id", experiment_id)):
            if v is not None:
                where.append(f"{col}=%s")
                args.append(v)
        if kind is not None:
            where.append("coalesce(kind,'detect')=%s")
            args.append(kind)
        sql = f"SELECT *, extract(epoch from now() - heartbeat_at) AS age FROM {self._J}"
        sql += (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at DESC, id DESC LIMIT %s"
        return [self._job_row(r) for r in self._q(sql, (*args, int(limit)), fetch="all")]

    def delete_jobs(self, *, experiment_id: str, kind: str) -> None:
        self._q(f"DELETE FROM {self._J} WHERE experiment_id=%s AND kind=%s", (experiment_id, kind))

    # ----------------------------------------------------------- experiments
    def create_experiment(self, e: Experiment) -> Experiment:
        self._q(f"INSERT INTO {self._E} (id,name,created_at,updated_at,notes,default_grid,default_detection,status)"
                " VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)", experiment_args(e))
        return e

    def save_experiment(self, e: Experiment) -> None:
        n = self._q(f"UPDATE {self._E} SET name=%s,created_at=%s,updated_at=%s,notes=%s,default_grid=%s::jsonb,"
                    "default_detection=%s::jsonb,status=%s WHERE id=%s", experiment_args(e)[1:] + (e.id,))
        if not n:
            raise NotFound("experiment", e.id)

    def get_experiment(self, eid: str) -> Experiment:
        r = self._q(f"SELECT * FROM {self._E} WHERE id=%s", (eid,), fetch="one")
        if r is None:
            raise NotFound("experiment", eid)
        return experiment_from_row(r)

    def list_experiments(self) -> list[Experiment]:
        return [experiment_from_row(r) for r in self._q(f"SELECT * FROM {self._E} ORDER BY updated_at DESC",
                                                        fetch="all")]

    def delete_experiment(self, eid: str) -> None:
        if not self._q(f"DELETE FROM {self._E} WHERE id=%s", (eid,)):
            raise NotFound("experiment", eid)

    def experiment_slide_ids(self, eid: str) -> list[str]:
        """Analyses of an experiment, in experiment order."""
        rows = self._q(f"SELECT id FROM {self._A} WHERE experiment_id=%s "
                       "ORDER BY coalesce(order_in_experiment,0), created_at, id", (eid,), fetch="all")
        return [r["id"] for r in rows]

    # ----------------------------------------------------------- csv imports
    def csv_dir(self, cid: str) -> Path:
        return self.data_dir / "csv_imports" / cid

    def save_csv_import(self, c: CsvImport, created: bool = False) -> None:
        self._q(f"INSERT INTO {self._C} (id,created_at,data) VALUES (%s,%s,%s::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET data=EXCLUDED.data", (c.id, now(), c.model_dump_json()))

    def get_csv_import(self, cid: str) -> CsvImport:
        r = self._q(f"SELECT data FROM {self._C} WHERE id=%s", (cid,), fetch="one")
        if r is None:
            raise NotFound("csv_import", cid)
        return CsvImport.model_validate(r["data"])

    def list_csv_imports(self) -> list[CsvImport]:
        rows = self._q(f"SELECT data FROM {self._C} ORDER BY created_at DESC", fetch="all")
        return [CsvImport.model_validate(r["data"]) for r in rows]


def _app_version() -> str:
    try:
        import polony_app

        return str(polony_app.__version__)
    except Exception:  # noqa: BLE001
        return ""
