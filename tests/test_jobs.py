"""STEP-06 任务管理测试。"""

from __future__ import annotations

import threading
import time

from app.jobs import JobCancelled, JobManager


def _wait(jm: JobManager, job_id: str, timeout: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jm.get(job_id)
        if job is not None and job.status in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job timeout")


def test_job_result_and_active() -> None:
    jm = JobManager()
    started = threading.Event()
    release = threading.Event()

    def run(progress, cancel):
        started.set()
        release.wait(1.0)
        return {"ok": True}

    try:
        job_id = jm.submit("sync", run)
        assert started.wait(1.0)
        assert jm.is_active("sync") is True
        release.set()
        job = _wait(jm, job_id)
        assert job.status == "done"
        assert job.result == {"ok": True}
        assert jm.is_active("sync") is False
    finally:
        jm.shutdown()


def test_job_cancel() -> None:
    jm = JobManager()
    try:
        def run(progress, cancel):
            while not cancel.is_set():
                time.sleep(0.01)
            raise JobCancelled("cancelled")

        job_id = jm.submit("sync", run)
        time.sleep(0.05)
        assert jm.cancel(job_id) is True
        job = _wait(jm, job_id)
        assert job.status == "cancelled"
    finally:
        jm.shutdown()


def test_submit_if_idle_rejects_duplicate() -> None:
    import pytest

    from app.jobs import BusyError

    jm = JobManager()
    started = threading.Event()
    release = threading.Event()

    def run(progress, cancel):
        started.set()
        release.wait(1.0)
        return {"ok": True}

    try:
        jm.submit("sync", run)
        assert started.wait(1.0)
        with pytest.raises(BusyError):
            jm.submit_if_idle("sync", lambda progress, cancel: {})
        release.set()
    finally:
        release.set()
        jm.shutdown()
