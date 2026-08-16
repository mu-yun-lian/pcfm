"""后台任务执行器：长任务异步化 + 进度 + 取消。

P1 用 JSON 文件持久化任务状态；P2 迁 SQLite。
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class Job:
    job_id: str
    type: str
    person_id: str | None
    status: str
    progress: float
    stage: str
    message: str
    error_code: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    result: dict = field(default_factory=dict)


class JobCancelled(Exception):
    """任务被取消。"""


class JobStore:
    """SQLite 持久化的任务存储（job 表由 db.Database 建好）。"""

    def __init__(self, db) -> None:
        self._db = db
        self._lock = threading.Lock()

    def create(self, type_: str, person_id: str | None) -> Job:
        job = Job(
            job_id=f"job-{uuid.uuid4().hex[:12]}",
            type=type_,
            person_id=person_id,
            status=JobStatus.queued.value,
            progress=0.0,
            stage="queued",
            message="",
            error_code=None,
            error_message=None,
            created_at=_now(),
            updated_at=_now(),
        )
        with self._lock:
            self._db.conn.execute(
                "INSERT INTO job (job_id, type, person_id, status, progress, stage, message, error_code, error_message, result, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (job.job_id, job.type, job.person_id, job.status, job.progress, job.stage, job.message, job.error_code, job.error_message, json.dumps(job.result), job.created_at, job.updated_at),
            )
            self._db.conn.commit()
        return job

    def get(self, job_id: str) -> Job | None:
        row = self._db.conn.execute("SELECT * FROM job WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return Job(
            job_id=row["job_id"], type=row["type"], person_id=row["person_id"],
            status=row["status"], progress=row["progress"], stage=row["stage"],
            message=row["message"], error_code=row["error_code"], error_message=row["error_message"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            result=json.loads(row["result"]) if row["result"] else {},
        )

    def update(self, job_id: str, **changes: Any) -> Job | None:
        with self._lock:
            job = self.get(job_id)
            if job is None:
                return None
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()
            self._db.conn.execute(
                "UPDATE job SET status=?, progress=?, stage=?, message=?, error_code=?, error_message=?, result=?, updated_at=? WHERE job_id=?",
                (job.status, job.progress, job.stage, job.message, job.error_code, job.error_message, json.dumps(job.result), job.updated_at, job.job_id),
            )
            self._db.conn.commit()
            return job


class JobRunner:
    """线程池执行器，并发上限 max_workers，支持进度与取消。"""

    def __init__(self, store: JobStore, max_workers: int = 2) -> None:
        self._store = store
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cancel_events: dict[str, threading.Event] = {}

    def submit(self, type_: str, person_id: str | None, fn: Callable, *args: Any, **kwargs: Any) -> Job:
        job = self._store.create(type_, person_id)
        cancel_event = threading.Event()
        self._cancel_events[job.job_id] = cancel_event
        self._executor.submit(self._run, job.job_id, cancel_event, fn, *args, **kwargs)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._store.get(job_id)

    def cancel(self, job_id: str) -> bool:
        event = self._cancel_events.get(job_id)
        if event is None:
            return False
        event.set()
        return True

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait)

    def _run(self, job_id: str, cancel_event: threading.Event, fn: Callable, *args: Any, **kwargs: Any) -> None:
        if cancel_event.is_set():
            self._store.update(job_id, status=JobStatus.cancelled.value, stage="cancelled")
            return
        self._store.update(job_id, status=JobStatus.running.value, stage="running")
        progress = self._make_progress(job_id, cancel_event)
        try:
            result = fn(progress=progress, cancel=cancel_event, *args, **kwargs)
            if isinstance(result, dict):
                self._store.update(job_id, status=JobStatus.succeeded.value, progress=1.0, stage="done", result=result)
            else:
                self._store.update(job_id, status=JobStatus.succeeded.value, progress=1.0, stage="done")
        except JobCancelled:
            self._store.update(job_id, status=JobStatus.cancelled.value, stage="cancelled")
        except Exception as error:
            self._store.update(
                job_id,
                status=JobStatus.failed.value,
                error_code=type(error).__name__,
                error_message=str(error),
            )

    def _make_progress(self, job_id: str, cancel_event: threading.Event) -> Callable:
        def progress(value: float, stage: str = "", message: str = "") -> None:
            if cancel_event.is_set():
                raise JobCancelled()
            self._store.update(job_id, progress=value, stage=stage, message=message)
        return progress
