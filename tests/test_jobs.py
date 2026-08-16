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

    def test_success(self) -> None:
        job = self.runner.submit("test", None, lambda progress=None, cancel=None: "ok")
        for _ in range(50):
            state = self.store.get(job.job_id)
            if state.status == JobStatus.succeeded.value:
                break
            time.sleep(0.05)
        self.assertEqual(self.store.get(job.job_id).status, JobStatus.succeeded.value)
        self.assertEqual(self.store.get(job.job_id).progress, 1.0)

    def test_failure(self) -> None:
        def boom(progress=None, cancel=None):
            raise RuntimeError("坏了")
        job = self.runner.submit("test", None, boom)
        for _ in range(50):
            if self.store.get(job.job_id).status == JobStatus.failed.value:
                break
            time.sleep(0.05)
        state = self.store.get(job.job_id)
        self.assertEqual(state.status, JobStatus.failed.value)
        self.assertEqual(state.error_message, "坏了")

    def test_progress_callback(self) -> None:
        def work(progress=None, cancel=None):
            progress(0.5, "half", "进行中")
            return {"done": True}
        job = self.runner.submit("test", None, work)
        for _ in range(50):
            if self.store.get(job.job_id).status == JobStatus.succeeded.value:
                break
            time.sleep(0.05)
        self.assertEqual(self.store.get(job.job_id).result, {"done": True})


if __name__ == "__main__":
    unittest.main()
