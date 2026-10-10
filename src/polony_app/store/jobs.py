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

JobFn = Callable[[Callable[[float, str], None], Callable[[], bool]], str | None]


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

    def active_job(self, analysis_id: str) -> str | None:
        with self._lock:
            return self._active_by_analysis.get(analysis_id)

    def submit(self, analysis_id: str, fn: JobFn, on_finish: Callable[[Job], None] | None = None) -> Job:
        """fn(progress, cancelled) -> optional final message. Raise JobCancelled
        (or anything with name DetectionCancelled) to end as cancelled."""
        with self._lock:
            if analysis_id in self._active_by_analysis:
                raise RuntimeError("job_in_progress")
            job = self.store.create_job(analysis_id)
            ev = threading.Event()
            self._cancel[job.id] = ev
            self._jobs[job.id] = job
            self._active_by_analysis[analysis_id] = job.id
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
                if t - last_flush[0] > 0.2:
                    last_flush[0] = t
                    self.store.save_job(job)

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
                    if msg == CANCELLED_MESSAGE:
                        job.status = JobStatus.cancelled
                    else:
                        job.status = JobStatus.done
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
                self.store.save_job(job)
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

    def cancel(self, jid: str) -> Job:
        with self._lock:
            ev = self._cancel.get(jid)
            if ev is not None:
                ev.set()
        return self.get(jid)

    def shutdown(self) -> None:
        with self._lock:
            for ev in self._cancel.values():
                ev.set()
        self._pool.shutdown(wait=True, cancel_futures=True)
