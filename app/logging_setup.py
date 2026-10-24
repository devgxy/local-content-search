"""日志初始化：控制台按 LOG_LEVEL，文件固定 DEBUG + 轮转。

文件日志用于排障，保留 7 个 10MB 分片；控制台默认 INFO，避免刷屏。

Step: STEP-00
Doc: docs/steps/STEP-00-项目骨架与环境校验.md §4.3
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import Settings

_CONFIGURED = False


def redact_path(path: Path | str, max_len: int = 80) -> str:
    """日志中使用的路径脱敏。

    Args:
        path: 原始路径。
        max_len: 最大显示长度。

    Returns:
        过长时保留头尾的字符串。
    """
    text = str(path)
    if len(text) <= max_len:
        return text
    head = max_len // 2 - 2
    tail = max_len - head - 3
    return f"{text[:head]}...{text[-tail:]}"


def setup_logging(settings: Settings, force: bool = False) -> None:
    """初始化根日志。

    Args:
        settings: 配置（log_level/log_dir）。
        force: True 时移除已有 handler 重新配置；测试/重启使用。
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        settings.log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    _CONFIGURED = True
