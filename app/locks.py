"""跨进程写锁。

Step: STEP-06
Doc: docs/08-代码评审与修复记录.md §3
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def write_lock(lock_path: Path):
    """同一 data 目录下的 CLI/Web 写入串行化。

    Args:
        lock_path: 锁文件路径，通常为 data/.write.lock。

    Yields:
        无；进入 with 时持有排他 flock，退出/异常时释放。
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        # flock 是进程级建议锁；跨进程同时 ingest/delete 时在这里排队。
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
