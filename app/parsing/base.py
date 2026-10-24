"""解析层统一数据结构。

Step: STEP-03
Doc: docs/steps/STEP-03-解析层.md §4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class ParseError(Exception):
    """解析/OCR 失败，带错误码。"""
    def __init__(self, code: str, message: str, hint: str = "") -> None:
        """初始化解析异常。

        Args:
            code: 错误码。
            message: 中文说明。
            hint: 可选修复建议。
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


@dataclass
class ParsedBlock:
    """解析后的结构块。"""
    text: str
    kind: str = "paragraph"  # heading/paragraph/table/code/ocr/note
    page: int = 0
    slide: int = 0
    heading_path: tuple[str, ...] = ()
    char_start: int = 0
    char_end: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """解析后的文档：块、全文、标题、警告。"""
    path: Path
    file_type: str
    title: str
    blocks: list[ParsedBlock]
    text: str = ""
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class Parser(Protocol):
    """解析器协议。"""
    extensions: tuple[str, ...]

    def parse(self, path: Path, settings) -> ParsedDocument:
        """解析单个文件并返回 ParsedDocument。

        Args:
            path: 文件绝对路径。
            settings: 解析/OCR 配置。

        Returns:
            ParsedDocument。

        Raises:
            ParseError: 不支持/损坏/加密/空文档。
        """
        ...  # noqa: ANN001


def normalize_text(text: str) -> str:
    """统一换行/去空/去 NUL。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def finalize_document(doc: ParsedDocument) -> ParsedDocument:
    """清理空块、计算字符偏移、生成全文。"""
    blocks: list[ParsedBlock] = []
    cursor = 0
    for block in doc.blocks:
        text = normalize_text(block.text)
        if not text:
            continue
        block.text = text
        block.char_start = cursor
        block.char_end = cursor + len(text)
        cursor = block.char_end + 2
        blocks.append(block)

    doc.blocks = blocks
    doc.text = "\n\n".join(block.text for block in blocks)
    if not doc.title:
        doc.title = doc.path.stem
    return doc
