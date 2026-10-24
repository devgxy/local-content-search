"""单 worker 后台任务管理。

scan/sync/reindex/delete 都提交到这里串行执行；每个任务有进度、结果、
错误和取消事件。单 worker 是刻意的：本地 LanceDB/SQLite 写入不并发，
避免同一 doc 的 run_id 交叉。

Step: STEP-06
Doc: docs/steps/STEP-06-入库流水线.md §5
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class JobCancelled(RuntimeError):
    """任务被用户/退出流程取消。"""


class BusyError(RuntimeError):
    """已有同 kind 任务在排队/运行，拒绝重复提交。"""


@dataclass
class Job:
    """一个后台任务的完整状态。

    Attributes:
        id: UUID。
        kind: scan/sync/reindex/delete。
        status: queued/running/done/failed/cancelled。
        progress: 0..1。
        message: 当前步骤描述。
        processed: 已处理数。
        total: 总数。
        result: 成功后的结构化结果。
        error: 失败/取消原因。
        created_at: 创建时间戳。
        finished_at: 终止时间戳。
        cancel_event: 协作式取消事件；任务循环需主动检查。
    """

    id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    processed: int = 0
    total: int = 0
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict:
        """转换为 API JobDTO 可序列化字典。

        Returns:
            包含 id/kind/status/progress/result/error/时间戳的 dict。
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "processed": self.processed,
            "total": self.total,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    """单 worker 任务队列。

    Attributes:
        max_jobs: 内存里最多保留的任务数，超出后裁剪已结束任务。
        max_workers: 固定为 1，保证写任务串行。
    """

    def __init__(self, max_jobs: int = 50, max_workers: int = 1):
        """初始化任务管理器。

        Args:
            max_jobs: 任务历史上限。
            max_workers: 工作线程数；默认 1。
        """
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.RLock()
        self._max_jobs = max_jobs
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running_kinds: set[str] = set()

    def _submit_locked(self, kind: str, fn: Callable) -> str:
        """调用方已持有 _lock 时创建 Job 并提交线程池。

        Args:
            kind: 任务类型。
            fn: ``fn(progress, cancel_event) -> dict``。

        Returns:
            job_id。
        """
        job = Job(id=uuid.uuid4().hex, kind=kind)
        self._jobs[job.id] = job
        self._trim()
        self._executor.submit(self._run, job, fn)
        logger.info("任务入队 job_id=%s kind=%s", job.id, kind)
        return job.id

    def submit(self, kind: str, fn: Callable) -> str:
        """无条件提交任务。

        Args:
            kind: 任务类型。
            fn: 任务函数。

        Returns:
            job_id。
        """
        with self._lock:
            return self._submit_locked(kind, fn)

    def submit_if_idle(
        self,
        kind: str,
        fn: Callable,
        *,
        blocked_kinds: set[str] | None = None,
    ) -> str:
        """原子地“没有同类任务才提交”。

        Args:
            kind: 新任务类型。
            fn: 任务函数。
            blocked_kinds: 被视为冲突的任务类型；默认只有 kind。

        Returns:
            job_id。

        Raises:
            BusyError: 已有冲突任务在 queued/running。
        """
        blocked = blocked_kinds or {kind}
        with self._lock:
            if any(
                job.kind in blocked and job.status in {"queued", "running"}
                for job in self._jobs.values()
            ):
                raise BusyError("已有同类任务在运行")
            return self._submit_locked(kind, fn)

    def _run(self, job: Job, fn: Callable) -> None:
        """线程池入口：执行任务并维护状态机。

        Args:
            job: 已入队的 Job。
            fn: 任务函数。
        """
        with self._lock:
            job.status = "running"
            self._running_kinds.add(job.kind)
        logger.info("任务开始 job_id=%s kind=%s", job.id, job.kind)

        def progress(processed: int, total: int, message: str = "") -> None:
            """线程安全地更新进度。

            Args:
                processed: 已处理数。
                total: 总数。
                message: 当前步骤描述。
            """
            with self._lock:
                job.processed = processed
                job.total = total
                job.progress = (processed / total) if total else 0.0
                if message:
                    job.message = message

        try:
            if job.cancel_event.is_set():
                raise JobCancelled("任务已取消")
            result = fn(progress, job.cancel_event)
            with self._lock:
                job.status = "done"
                job.result = result if isinstance(result, dict) else {"value": result}
                job.progress = 1.0 if job.status == "done" else job.progress
        except JobCancelled:
            logger.info("任务取消 job_id=%s kind=%s", job.id, job.kind)
            with self._lock:
                job.status = "cancelled"
                job.error = "已取消"
        except Exception as exc:  # noqa: BLE001
            logger.exception("任务失败 job_id=%s kind=%s", job.id, job.kind)
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
        finally:
            with self._lock:
                job.finished_at = time.time()
                self._running_kinds.discard(job.kind)
                logger.info("任务结束 job_id=%s kind=%s status=%s", job.id, job.kind, job.status)
                self._trim()

    def _trim(self) -> None:
        """裁剪最老的已结束任务，保持内存上限。"""
        if len(self._jobs) <= self._max_jobs:
            return
        for job_id, job in list(self._jobs.items()):
            if len(self._jobs) <= self._max_jobs:
                break
            if job.status in {"done", "failed", "cancelled"}:
                del self._jobs[job_id]

    def get(self, job_id: str) -> Job | None:
        """按 ID 查询任务；不存在返回 None。"""
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """返回全部任务快照（按入队顺序）。"""
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> bool:
        """请求取消排队/运行中的任务。

        Args:
            job_id: 任务 ID。

        Returns:
            True 表示已设置取消事件；任务不存在或已结束返回 False。
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in {"queued", "running"}:
                return False
            job.cancel_event.set()
            return True

    def is_running(self, kind: str) -> bool:
        """是否已有 running 的同类任务。"""
        with self._lock:
            return kind in self._running_kinds

    def is_active(self, kind: str) -> bool:
        """包含排队中的任务，用于阻止重复提交。"""
        with self._lock:
            return any(
                job.kind == kind and job.status in {"queued", "running"}
                for job in self._jobs.values()
            )

    def shutdown(self, wait: bool = True) -> None:
        """取消所有活动任务并关闭线程池。

        Args:
            wait: True 时等待运行中的任务退出；关闭流程使用 True。
        """
        with self._lock:
            for job in self._jobs.values():
                if job.status in {"queued", "running"}:
                    job.cancel_event.set()
        self._executor.shutdown(wait=wait, cancel_futures=True)
