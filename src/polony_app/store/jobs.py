"""Threaded job runner with progress + cooperative cancellation."""
from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from polony_app.review.engine import now
from polony_app.schemas import Job, JobStatus

from .db import Store

# fn(progress, cancelled) -> None | message | (final JobStatus, message)
JobFn = Callable[[Callable[[float, str], None], Callable[[], bool]], "str | tuple[JobStatus, str] | None"]


class JobCancelled(Exception):
    pass


# A job function reports "I stopped because of a cancel request, nothing was committed"
# by returning this message (or by raising JobCancelled / DetectionCancelled).
CANCELLED_MESSAGE = "cancelled"


class JobRunner:
    def __init__(self, store: Store, max_workers: int = 2):
        self.store = store
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="polony-job")
        self._cancel: dict[str, threading.Event] = {}
        self._jobs: dict[str, Job] = {}
        self._active_by_analysis: dict[str, str] = {}
        self._lock = threading.RLock()
        self._flush_every = 1.0 if store.supports_heartbeat else 0.2   # fewer round trips to a remote DB
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []   # dedicated (batch) jobs, outside the worker pool
        self.heartbeat_interval = 5.0

    def active_job(self, analysis_id: str) -> str | None:
        with self._lock:
            local = self._active_by_analysis.get(analysis_id)
        if local:
            return local
        return self.store.find_active_job(analysis_id)   # a job running on another computer

    def _heartbeat_loop(self) -> None:
        """Shared DB only: keep the heartbeat of our running jobs fresh (so other computers do
        not mistake them for orphans) and pick up cancel requests made from another computer."""
        while not self._hb_stop.wait(self.heartbeat_interval):
            with self._lock:
                ids = list(self._active_by_analysis.values())
            for jid in ids:
                try:
                    if self.store.touch_job(jid):
                        ev = self._cancel.get(jid)
                        if ev is not None:
                            ev.set()
                except Exception:  # noqa: BLE001  (database briefly unreachable: keep trying)
                    pass

    def submit(self, analysis_id: str, fn: JobFn, on_finish: Callable[[Job], None] | None = None, *,
               kind: str = "detect", parent_id: str | None = None, experiment_id: str | None = None,
               detail: dict | None = None, dedicated: bool = False) -> Job:
        """fn(progress, cancelled) -> optional final message, or (JobStatus, message) to choose the final
        status (a batch reports done_with_failures / cancelled that way). Raise JobCancelled
        (or anything with name DetectionCancelled) to end as cancelled.
        ``dedicated=True`` runs fn on its own thread instead of the worker pool: a batch job only
        sequences other jobs and waits for them, so it must never hold a pool worker they need."""
        with self._lock:
            if analysis_id in self._active_by_analysis:
                raise RuntimeError("job_in_progress")
            # RuntimeError('job_in_progress') if another machine runs one
            job = self.store.create_job(analysis_id, kind=kind, parent_id=parent_id, experiment_id=experiment_id,
                                        detail=detail)
            ev = threading.Event()
            self._cancel[job.id] = ev
            self._jobs[job.id] = job
            self._active_by_analysis[analysis_id] = job.id
        if self.store.supports_heartbeat and self._hb_thread is None:
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="polony-heartbeat", daemon=True)
            self._hb_thread.start()
        if dedicated:
            t = threading.Thread(target=self._run, args=(job.id, fn, ev, on_finish), name=f"polony-{kind}-{job.id}",
                                 daemon=True)
            with self._lock:
                self._threads = [x for x in self._threads if x.is_alive()] + [t]
            t.start()
        else:
            self._pool.submit(self._run, job.id, fn, ev, on_finish)
        return job.model_copy()

    def _run(self, jid: str, fn: JobFn, ev: threading.Event, on_finish) -> None:
        job = self._jobs[jid]
        last_flush = [0.0]

        def progress(frac: float, msg: str = "") -> None:
            with self._lock:
                job.progress = max(0.0, min(1.0, float(frac)))
                job.message = msg or job.message
                t = time.monotonic()
                if t - last_flush[0] > self._flush_every:
                    last_flush[0] = t
                    if self.store.save_job(job):
                        ev.set()   # cancel requested from another computer

        with self._lock:
            if ev.is_set():
                job.status = JobStatus.cancelled
                job.finished_at = now()
            else:
                job.status = JobStatus.running
                job.started_at = now()
            self.store.save_job(job)
        try:
            if job.status == JobStatus.running:
                msg = fn(progress, ev.is_set)
                with self._lock:
                    # The job's outcome is decided by what fn did, not by a cancel flag that
                    # may have been set after fn already committed: a result that was committed
                    # is never reported as cancelled, a cancel-return commits nothing.
                    final = None
                    if isinstance(msg, tuple):
                        final, msg = msg
                    if final is None:
                        final = JobStatus.cancelled if msg == CANCELLED_MESSAGE else JobStatus.done
                    job.status = final
                    if final in (JobStatus.done, JobStatus.done_with_failures):
                        job.progress = 1.0
                    if msg:
                        job.message = msg
        except Exception as e:  # noqa: BLE001
            with self._lock:
                if isinstance(e, JobCancelled) or type(e).__name__ == "DetectionCancelled":
                    job.status = JobStatus.cancelled
                    job.message = "cancelled"
                else:
                    job.status = JobStatus.failed
                    job.error = f"{type(e).__name__}: {e}"
                    job.message = "failed"
                    self.last_trace = traceback.format_exc()
        finally:
            with self._lock:
                job.finished_at = now()
                try:
                    self.store.save_job(job)
                except Exception:  # noqa: BLE001  (shared DB unreachable: the stale-heartbeat rule cleans up)
                    pass
                self._active_by_analysis.pop(job.analysis_id, None)
            if on_finish:
                try:
                    on_finish(job)
                except Exception:  # noqa: BLE001
                    pass

    def get(self, jid: str) -> Job:
        with self._lock:
            j = self._jobs.get(jid)
            if j is not None:
                return j.model_copy()
        return self.store.get_job(jid)

    def active_jobs(self) -> list[Job]:
        """Queued/running jobs of this process (what the desktop window asks before it closes)."""
        with self._lock:
            return [self._jobs[j].model_copy() for j in self._active_by_analysis.values() if j in self._jobs]

    def cancel(self, jid: str) -> Job:
        with self._lock:
            ev = self._cancel.get(jid)
            if ev is not None:
                ev.set()
        try:
            self.store.request_cancel(jid)   # job owned by another computer: its heartbeat sees the flag
        except Exception:  # noqa: BLE001
            if ev is None:
                raise
        return self.get(jid)

    def shutdown(self) -> None:
        self._hb_stop.set()
        with self._lock:
            for ev in self._cancel.values():
                ev.set()
            threads = list(self._threads)
        for t in threads:   # a cancelled batch stops after its current slide, whose job was cancelled too
            t.join(timeout=30)
        self._pool.shutdown(wait=True, cancel_futures=True)
