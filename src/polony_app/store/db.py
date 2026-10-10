"""SQLite + local-file persistence. Thread safe.

Layout under data_dir:
  app.sqlite                      analyses / actions / jobs / csv_imports
  originals/<analysis_id>/        original image files (never modified)
  csv_imports/<id>/               verbatim uploaded CSV
  tmp/                            upload staging
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from polony_app.review.counts import compute_counts
from polony_app.review.engine import now, refresh_status
from polony_app.review.state import AnalysisState, ReviewError
from polony_app.schemas import (
    Analysis,
    AnalysisStatus,
    CsvImport,
    EngineRunInfo,
    Job,
    JobStatus,
    RawDetection,
    ReviewAction,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
  id TEXT PRIMARY KEY, name TEXT, status TEXT, created_at TEXT, updated_at TEXT,
  finalized_at TEXT, summary TEXT, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS actions (
  analysis_id TEXT NOT NULL, seq INTEGER NOT NULL, data TEXT NOT NULL,
  PRIMARY KEY (analysis_id, seq));
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL, status TEXT, progress REAL,
  message TEXT, error TEXT, started_at TEXT, finished_at TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS csv_imports (
  id TEXT PRIMARY KEY, created_at TEXT, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS engine_runs (
  analysis_id TEXT NOT NULL, run_id TEXT NOT NULL, engine TEXT, page INTEGER, data TEXT NOT NULL,
  PRIMARY KEY (analysis_id, run_id));
CREATE INDEX IF NOT EXISTS engine_runs_aid_idx ON engine_runs (analysis_id, engine, page);
CREATE TABLE IF NOT EXISTS raw_detections (
  analysis_id TEXT NOT NULL, run_id TEXT NOT NULL, id TEXT NOT NULL, engine TEXT, page INTEGER,
  valid INTEGER, data TEXT NOT NULL,
  PRIMARY KEY (analysis_id, run_id, id));
CREATE INDEX IF NOT EXISTS raw_det_aid_idx ON raw_detections (analysis_id, engine, page, valid);
CREATE INDEX IF NOT EXISTS raw_det_run_idx ON raw_detections (analysis_id, run_id);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class NotFound(ReviewError):
    def __init__(self, what: str, ident: str):
        super().__init__(f"{what}_not_found", f"{what} '{ident}' not found", 404)


class Store:
    backend = "sqlite"
    supports_heartbeat = False   # jobs of other machines do not exist locally
    machine_id = ""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir).resolve()
        for sub in ("originals", "csv_imports", "tmp"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            str(self.data_dir / "app.sqlite"), check_same_thread=False, isolation_level=None
        )
        self._db.row_factory = sqlite3.Row
        self._dblock = threading.RLock()
        with self._dblock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(SCHEMA)
        self._cache: dict[str, AnalysisState] = {}
        self._saved_log_len: dict[str, int] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()
        self.recover_orphans()

    # ----------------------------------------------------------- low level
    def _exec(self, sql: str, args: tuple = ()):
        with self._dblock:
            return self._db.execute(sql, args)

    def _lock_for(self, aid: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(aid, threading.RLock())

    def close(self) -> None:
        with self._dblock:
            self._db.close()

    def info(self) -> dict:
        return {"backend": "sqlite", "host": None, "database": "app.sqlite", "schema": None,
                "machine_id": None, "connected": True, "latency_ms": 0.0, "last_error": None}

    # cross-machine job hooks (no-ops for the single-process SQLite backend)
    def touch_job(self, jid: str) -> bool:
        return False

    def request_cancel(self, jid: str) -> None:
        return None

    def find_active_job(self, aid: str) -> str | None:
        return None

    # ----------------------------------------------------------- startup recovery
    def recover_orphans(self) -> list[str]:
        """Jobs left queued/running by a crashed/stopped process become failed;
        analyses stuck in 'detecting' return to review/ready."""
        with self._dblock:
            rows = self._exec(
                "SELECT id, analysis_id FROM jobs WHERE status IN ('queued','running')"
            ).fetchall()
            ids = [r["id"] for r in rows]
            for r in rows:
                self._exec(
                    "UPDATE jobs SET status='failed', error=?, finished_at=? WHERE id=?",
                    ("interrupted: the application was restarted while this job was running", now(), r["id"]),
                )
            stuck = self._exec(
                "SELECT id FROM analyses WHERE status='detecting' OR state LIKE '%\"job_active\": true%'"
            ).fetchall()
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
            self._insert(st)
            self._cache[aid] = st
            self._saved_log_len[aid] = 0
        return aid

    def _summary(self, st: AnalysisState) -> str:
        try:
            return compute_counts(st.analysis)["slide"].model_dump_json()
        except Exception:  # pragma: no cover
            return "null"

    def _insert(self, st: AnalysisState) -> None:
        a = st.analysis
        self._exec(
            "INSERT INTO analyses (id,name,status,created_at,updated_at,finalized_at,summary,state)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (a.id, a.name, a.status.value, a.created_at, a.updated_at, a.finalized_at,
             self._summary(st), json.dumps(st.to_json())),
        )

    def _save(self, st: AnalysisState) -> None:
        a = st.analysis
        aid = a.id
        with self._dblock:
            self._exec("BEGIN")
            try:
                self._exec(
                    "UPDATE analyses SET name=?,status=?,updated_at=?,finalized_at=?,summary=?,state=? WHERE id=?",
                    (a.name, a.status.value, a.updated_at, a.finalized_at, self._summary(st),
                     json.dumps(st.to_json()), aid),
                )
                start = self._saved_log_len.get(aid, 0)
                for act in st.log[start:]:
                    self._exec(
                        "INSERT OR REPLACE INTO actions (analysis_id,seq,data) VALUES (?,?,?)",
                        (aid, act.seq, act.model_dump_json()),
                    )
                self._exec("COMMIT")
            except Exception:
                self._exec("ROLLBACK")
                raise
        self._saved_log_len[aid] = len(st.log)

    def _load(self, aid: str) -> AnalysisState:
        st = self._cache.get(aid)
        if st is not None:
            return st
        with self._dblock:
            row = self._exec("SELECT state FROM analyses WHERE id=?", (aid,)).fetchone()
            if row is None:
                raise NotFound("analysis", aid)
            acts = self._exec(
                "SELECT data FROM actions WHERE analysis_id=? ORDER BY seq", (aid,)
            ).fetchall()
        log = [ReviewAction.model_validate_json(r["data"]) for r in acts]
        st = AnalysisState.from_json(json.loads(row["state"]), log)
        from polony_app.review.engine import migrate_legacy_state

        migrate_legacy_state(st)  # old persisted format without detection_sig
        self._cache[aid] = st
        self._saved_log_len[aid] = len(log)
        return st

    @contextmanager
    def view(self, aid: str) -> Iterator[AnalysisState]:
        """Locked read access. Do not mutate."""
        with self._lock_for(aid):
            yield self._load(aid)

    @contextmanager
    def edit(self, aid: str) -> Iterator[AnalysisState]:
        """Locked read-modify-write; saved on success, cache evicted on error."""
        with self._lock_for(aid):
            st = self._load(aid)
            try:
                yield st
                self._save(st)
            except BaseException:
                self._cache.pop(aid, None)  # discard possibly half-mutated state
                raise

    def exists(self, aid: str) -> bool:
        with self._dblock:
            return self._exec("SELECT 1 FROM analyses WHERE id=?", (aid,)).fetchone() is not None

    def list_analyses(self) -> list[dict]:
        with self._dblock:
            rows = self._exec(
                "SELECT id,name,status,updated_at,finalized_at,summary FROM analyses ORDER BY updated_at DESC"
            ).fetchall()
        out = []
        for r in rows:
            d = {k: r[k] for k in ("id", "name", "status", "updated_at", "finalized_at")}
            s = r["summary"]
            d["counts"] = json.loads(s) if s and s != "null" else None
            out.append(d)
        return out

    def delete_analysis(self, aid: str) -> None:
        with self._lock_for(aid):
            if not self.exists(aid):
                raise NotFound("analysis", aid)
            with self._dblock:
                self._exec("DELETE FROM analyses WHERE id=?", (aid,))
                self._exec("DELETE FROM actions WHERE analysis_id=?", (aid,))
                self._exec("DELETE FROM jobs WHERE analysis_id=?", (aid,))
                self._exec("DELETE FROM engine_runs WHERE analysis_id=?", (aid,))
                self._exec("DELETE FROM raw_detections WHERE analysis_id=?", (aid,))
            self._cache.pop(aid, None)
            self._saved_log_len.pop(aid, None)
            shutil.rmtree(self.originals_dir(aid), ignore_errors=True)

    # ----------------------------------------------------------- engine results (audit, append-only)
    def save_engine_results(self, aid: str, runs: list[EngineRunInfo],
                            detections: list[RawDetection]) -> None:
        """Append the engine runs + raw detections of one detect job. Runs of earlier jobs are never
        touched (audit); only an identical run_id is replaced (idempotent retry). Kept outside the
        Analysis JSON on purpose (it can hold 100k+ points)."""
        if not self.exists(aid):
            raise NotFound("analysis", aid)
        run_ids = [r.run_id for r in runs]
        with self._dblock:
            self._exec("BEGIN")
            try:
                for rid in run_ids:
                    self._exec("DELETE FROM raw_detections WHERE analysis_id=? AND run_id=?", (aid, rid))
                    self._exec("DELETE FROM engine_runs WHERE analysis_id=? AND run_id=?", (aid, rid))
                for r in runs:
                    self._exec(
                        "INSERT INTO engine_runs (analysis_id,run_id,engine,page,data) VALUES (?,?,?,?,?)",
                        (aid, r.run_id, r.engine.value, r.page, r.model_dump_json()))
                self._db.executemany(
                    "INSERT OR REPLACE INTO raw_detections (analysis_id,run_id,id,engine,page,valid,data)"
                    " VALUES (?,?,?,?,?,?,?)",
                    [(aid, d.run_id, d.id, d.engine.value, d.page, 1 if d.valid else 0, d.model_dump_json())
                     for d in detections])
                self._exec("COMMIT")
            except Exception:
                self._exec("ROLLBACK")
                raise

    def get_engine_runs(self, aid: str) -> list[EngineRunInfo]:
        with self._dblock:
            rows = self._exec("SELECT data FROM engine_runs WHERE analysis_id=? ORDER BY rowid", (aid,)).fetchall()
        return [EngineRunInfo.model_validate_json(r["data"]) for r in rows]

    @staticmethod
    def _det_where(aid, engine, page, run_id, valid) -> tuple[str, list]:
        where, args = ["analysis_id=?"], [aid]
        for col, v in (("engine", engine), ("page", page), ("run_id", run_id)):
            if v is not None:
                where.append(f"{col}=?")
                args.append(getattr(v, "value", v))
        if valid is not None:
            where.append("valid=?")
            args.append(1 if valid else 0)
        return " AND ".join(where), args

    def get_raw_detections(self, aid: str, engine=None, page: int | None = None, run_id: str | None = None,
                           valid: bool | None = None, limit: int | None = None, offset: int = 0) -> list[RawDetection]:
        w, args = self._det_where(aid, engine, page, run_id, valid)
        sql = f"SELECT data FROM raw_detections WHERE {w} ORDER BY rowid"
        if limit is not None or offset:
            sql += " LIMIT ? OFFSET ?"
            args += [-1 if limit is None else int(limit), int(offset)]
        with self._dblock:
            rows = self._exec(sql, tuple(args)).fetchall()
        return [RawDetection.model_validate_json(r["data"]) for r in rows]

    def count_raw_detections(self, aid: str, engine=None, page: int | None = None, run_id: str | None = None,
                             valid: bool | None = None) -> int:
        w, args = self._det_where(aid, engine, page, run_id, valid)
        with self._dblock:
            return int(self._exec(f"SELECT count(*) AS n FROM raw_detections WHERE {w}", tuple(args)).fetchone()["n"])

    # ----------------------------------------------------------- jobs
    def create_job(self, aid: str) -> Job:
        j = Job(id=new_id(), analysis_id=aid, status=JobStatus.queued)
        self._exec(
            "INSERT INTO jobs (id,analysis_id,status,progress,message,created_at) VALUES (?,?,?,?,?,?)",
            (j.id, aid, j.status.value, 0.0, "", now()),
        )
        return j

    def save_job(self, j: Job) -> None:
        self._exec(
            "UPDATE jobs SET status=?,progress=?,message=?,error=?,started_at=?,finished_at=? WHERE id=?",
            (j.status.value, j.progress, j.message, j.error, j.started_at, j.finished_at, j.id),
        )

    def get_job(self, jid: str) -> Job:
        with self._dblock:
            r = self._exec("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if r is None:
            raise NotFound("job", jid)
        return Job(
            id=r["id"], analysis_id=r["analysis_id"], status=JobStatus(r["status"]),
            progress=r["progress"] or 0.0, message=r["message"] or "", error=r["error"],
            started_at=r["started_at"], finished_at=r["finished_at"],
        )

    # ----------------------------------------------------------- csv imports
    def csv_dir(self, cid: str) -> Path:
        return self.data_dir / "csv_imports" / cid

    def save_csv_import(self, c: CsvImport, created: bool = False) -> None:
        self._exec(
            "INSERT OR REPLACE INTO csv_imports (id,created_at,data) VALUES (?,?,?)",
            (c.id, now(), c.model_dump_json()),
        )

    def get_csv_import(self, cid: str) -> CsvImport:
        with self._dblock:
            r = self._exec("SELECT data FROM csv_imports WHERE id=?", (cid,)).fetchone()
        if r is None:
            raise NotFound("csv_import", cid)
        return CsvImport.model_validate_json(r["data"])

    def list_csv_imports(self) -> list[CsvImport]:
        with self._dblock:
            rows = self._exec("SELECT data FROM csv_imports ORDER BY created_at DESC").fetchall()
        return [CsvImport.model_validate_json(r["data"]) for r in rows]
