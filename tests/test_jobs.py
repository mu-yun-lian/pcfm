# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from pcfm.persistence.db import Database
from pcfm.jobs import JobRunner, JobStatus, JobStore


class JobRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "pcfm.db")
        self.store = JobStore(self.db)
        self.runner = JobRunner(self.store, max_workers=2)

    def tearDown(self) -> None:
        self.runner.shutdown(wait=True)
        self.db.close()
        self.tmp.cleanup()

    def _wait_status(self, job_id: str, status: str, timeout: float = 15.0):
        """轮询直到任务达到指定状态或超时(高负载下线程池调度可能延迟)。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.store.get(job_id)
            if state is not None and state.status == status:
                return state
            time.sleep(0.02)
        return self.store.get(job_id)

    def test_success(self) -> None:
        job = self.runner.submit("test", None, lambda progress=None, cancel=None: "ok")
        state = self._wait_status(job.job_id, JobStatus.succeeded.value)
        self.assertEqual(state.status, JobStatus.succeeded.value)
        self.assertEqual(state.progress, 1.0)

    def test_failure(self) -> None:
        def boom(progress=None, cancel=None):
            raise RuntimeError("坏了")
        job = self.runner.submit("test", None, boom)
        state = self._wait_status(job.job_id, JobStatus.failed.value)
        self.assertEqual(state.status, JobStatus.failed.value)
        self.assertEqual(state.error_message, "坏了")

    def test_progress_callback(self) -> None:
        def work(progress=None, cancel=None):
            progress(0.5, "half", "进行中")
            return {"done": True}
        job = self.runner.submit("test", None, work)
        state = self._wait_status(job.job_id, JobStatus.succeeded.value)
        self.assertEqual(state.result, {"done": True})


if __name__ == "__main__":
    unittest.main()
