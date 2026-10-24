"""稳定 ID 与哈希工具。

ID 只依赖路径/内容，不依赖时间，因此重建后仍能稳定关联清单与向量行。

Step: STEP-00
Doc: docs/02-代码结构与追溯矩阵.md §2.2
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """计算字节 SHA-256。

    Args:
        data: 原始字节。

    Returns:
        64 位十六进制字符串。
    """
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """计算 UTF-8 文本 SHA-256。"""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path | str, chunk_size: int = _CHUNK) -> str:
    """流式计算文件哈希。

    Args:
        path: 文件路径。
        chunk_size: 每次读取字节数，默认 1MB。

    Returns:
        SHA-256 十六进制字符串。

    Raises:
        OSError: 文件不可读。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def normalize_path(path: Path | str) -> Path:
    """展开 ~ 并解析为绝对路径，作为文件身份。"""
    return Path(path).expanduser().resolve()


def make_doc_id(path: Path | str) -> str:
    """文档 ID：规范化绝对路径的 SHA-256 前 20 位。"""
    return sha256_text(str(normalize_path(path)))[:20]


def make_content_hash(text: str) -> str:
    """生成 chunk 内容哈希（前 16 位），用于去重/稳定 chunk ID。"""
    return sha256_text(text)[:16]


def make_chunk_id(doc_id: str, chunk_index: int, content_hash: str) -> str:
    """生成稳定 chunk ID。

    Args:
        doc_id: 文档 ID。
        chunk_index: 块序号；图片 asset 可用 -1。
        content_hash: 内容哈希。

    Returns:
        32 位十六进制 ID。
    """
    return sha256_text(f"{doc_id}:{chunk_index}:{content_hash}")[:32]


def escape_sql_literal(value: str) -> str:
    """把字符串安全地放进 LanceDB SQL 过滤条件。

    Args:
        value: 原始字符串。

    Returns:
        单引号包裹并转义后的字面量。
    """
    return "'" + value.replace("'", "''") + "'"
