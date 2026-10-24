"""PDF 解析子进程隔离：坏 PDF 的 segfault 不杀死主进程。"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import tempfile
from pathlib import Path

from app.parsing.base import ParseError

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def _worker(path: str, settings, result_path: str) -> None:  # noqa: ANN001
    try:
        from app.parsing.registry import get_parser

        doc = get_parser(path).parse(Path(path), settings)
        payload = ("ok", doc)
    except ParseError as exc:
        payload = ("parse_error", exc.code, exc.message, exc.hint)
    except BaseException as exc:  # noqa: BLE001
        payload = ("error", type(exc).__name__, str(exc))
    with open(result_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _stop(process) -> bool:  # noqa: ANN001
    """有界终止子进程；返回是否确认退出。"""
    if process.is_alive():
        process.terminate()
        process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)
    return not process.is_alive()


def parse_pdf_isolated(path: Path, settings, *, timeout: float = 180.0):
    """在 spawn 子进程中解析 PDF，结果经临时文件回传。"""
    handle = tempfile.NamedTemporaryFile(
        prefix="local-search-parse-", suffix=".pickle", delete=False
    )
    result_path = handle.name
    handle.close()
    context = mp.get_context("spawn")
    process = context.Process(target=_worker, args=(str(path), settings, result_path))
    try:
        process.start()
        process.join(timeout)
        if process.is_alive():
            raise ParseError("PARSER_TIMEOUT", "PDF 解析超时，已终止子进程")
        if process.exitcode != 0:
            raise ParseError(
                "PARSER_CRASH",
                f"PDF 解析进程异常退出：{process.exitcode}",
                "文件可能损坏；原始文件未修改",
            )
        try:
            with open(result_path, "rb") as result_file:
                message = pickle.load(result_file)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            raise ParseError("PARSER_CRASH", f"PDF 解析结果不完整：{exc}") from exc
        if message[0] == "ok":
            return message[1]
        if message[0] == "parse_error":
            _, code, text, hint = message
            raise ParseError(code, text, hint)
        raise ParseError("PARSE_FAILED", f"{message[1]}: {message[2]}")
    finally:
        import sys

        cleanup_error = ""
        try:
            cleanup_failed = not _stop(process)
        except Exception as exc:  # noqa: BLE001
            cleanup_failed = True
            cleanup_error = str(exc)
        try:
            os.unlink(result_path)
        except OSError as exc:
            cleanup_error = f"{cleanup_error}; {exc}".strip("; ")
            logger.warning("清理 PDF 解析临时文件失败：%s", exc)
        if cleanup_failed:
            logger.error("PDF 解析子进程回收失败：%s", cleanup_error or "unknown")
            if sys.exc_info()[0] is None:
                raise ParseError(
                    "PARSER_CRASH",
                    f"PDF 解析子进程无法回收：{cleanup_error or 'unknown'}",
                )
