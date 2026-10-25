from __future__ import annotations

import logging
import shutil
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import UUID


logger = logging.getLogger(__name__)


class QueueFullError(RuntimeError):
    pass


class JobNotFoundError(LookupError):
    pass


class JobConflictError(RuntimeError):
    pass


class JobCancelledError(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    work_dir: Path
    input_sha256: str
    status: str = "queued"
    progress: int = 0
    stage: str = "queued"
    error: str | None = None
    result_path: Path | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future[None] | None = field(default=None, repr=False)

    def public(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "input_sha256": self.input_sha256,
            "error": self.error,
            "created_at": self.created_at,
        }


class JobContext:
    def __init__(self, job: Job, manager: JobManager):
        self.job = job
        self._manager = manager

    @property
    def work_dir(self) -> Path:
        return self.job.work_dir

    @property
    def cancel_event(self) -> threading.Event:
        return self.job.cancel_event

    def update(self, progress: int, stage: str) -> None:
        self._manager.update(self.job.id, progress, stage)

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelledError("Job was cancelled")


class JobManager:
    def __init__(self, max_workers: int = 1, max_active_jobs: int = 8):
        if max_workers < 1 or max_active_jobs < max_workers:
            raise ValueError("max_active_jobs must be >= max_workers >= 1")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="voice-job"
        )
        self._max_active_jobs = max_active_jobs
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    def submit(
        self,
        job_id: str,
        work_dir: Path,
        input_sha256: str,
        task: Callable[[JobContext], Path],
    ) -> Job:
        UUID(job_id)
        with self._lock:
            active = sum(
                job.status in {"queued", "running"} for job in self._jobs.values()
            )
            if active >= self._max_active_jobs:
                raise QueueFullError("The voice-conversion queue is full")
            if job_id in self._jobs:
                raise JobConflictError("Job id already exists")
            job = Job(job_id, work_dir, input_sha256)
            self._jobs[job_id] = job
            job.future = self._executor.submit(self._run, job, task)
            return job

    def _run(self, job: Job, task: Callable[[JobContext], Path]) -> None:
        cancelled_before_start = False
        with self._lock:
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.stage = "cancelled"
                cancelled_before_start = True
            else:
                job.status = "running"
                job.stage = "starting"
        if cancelled_before_start:
            self._cleanup_work_dir(job.work_dir)
            return

        try:
            result_path = task(JobContext(job, self))
            if not result_path.is_file():
                raise RuntimeError("Conversion finished without creating an output file")
        except JobCancelledError:
            with self._lock:
                job.status = "cancelled"
                job.stage = "cancelled"
        except Exception as error:
            with self._lock:
                if job.cancel_event.is_set():
                    job.status = "cancelled"
                    job.stage = "cancelled"
                else:
                    logger.exception("Voice-conversion job %s failed", job.id)
                    job.status = "failed"
                    job.stage = "failed"
                    job.error = f"{type(error).__name__}: {error}"
        else:
            with self._lock:
                if job.cancel_event.is_set():
                    job.status = "cancelled"
                    job.stage = "cancelled"
                else:
                    job.status = "succeeded"
                    job.stage = "complete"
                    job.progress = 100
                    job.result_path = result_path.resolve()
        finally:
            with self._lock:
                succeeded = job.status == "succeeded"
                result_path = job.result_path if succeeded else None
            self._cleanup_work_dir(job.work_dir, result_path)

    def update(self, job_id: str, progress: int, stage: str) -> None:
        if not 0 <= progress <= 100 or not stage:
            raise ValueError("progress must be 0..100 and stage must not be empty")
        with self._lock:
            job = self._get_unlocked(job_id)
            if job.status == "running":
                job.progress = progress
                job.stage = stage

    def get(self, job_id: str) -> Job:
        with self._lock:
            return self._get_unlocked(job_id)

    def cancel(self, job_id: str) -> Job:
        cleanup = False
        with self._lock:
            job = self._get_unlocked(job_id)
            if job.status not in {"queued", "running"}:
                raise JobConflictError(f"Cannot cancel a {job.status} job")
            job.cancel_event.set()
            if job.status == "queued":
                if job.future is not None:
                    job.future.cancel()
                job.status = "cancelled"
                job.stage = "cancelled"
                cleanup = True
            else:
                job.stage = "cancelling"
        if cleanup:
            self._cleanup_work_dir(job.work_dir)
        return job

    def _get_unlocked(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError as error:
            raise JobNotFoundError(job_id) from error

    @staticmethod
    def _cleanup_work_dir(work_dir: Path, result_path: Path | None = None) -> None:
        try:
            if result_path is None or not work_dir.is_dir():
                shutil.rmtree(work_dir, ignore_errors=True)
                return

            resolved_work_dir = work_dir.resolve()
            resolved_result = result_path.resolve()
            if resolved_result.parent != resolved_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
                return

            for child in work_dir.iterdir():
                if child.resolve() == resolved_result:
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not clean voice-conversion job directory %s", work_dir)


def cleanup_stale_job_directories(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    for child in work_dir.iterdir():
        if child.is_symlink() or not child.is_dir():
            continue
        try:
            if str(UUID(child.name)) != child.name:
                continue
        except ValueError:
            continue
        try:
            shutil.rmtree(child)
        except OSError:
            logger.exception("Could not remove stale job directory %s", child)
