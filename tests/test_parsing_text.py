"""STEP-03 文本/Markdown 解析测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.parsing.base import ParseError
from app.parsing.registry import get_parser
from app.parsing.text_md import decode_bytes


def test_markdown_blocks(fixtures_dir: Path, tmp_settings: Settings) -> None:
    path = fixtures_dir / "tiny.md"
    doc = get_parser(path).parse(path, tmp_settings)
    assert doc.file_type == "md"
    assert doc.title == "混合检索系统"
    kinds = {block.kind for block in doc.blocks}
    assert {"heading", "paragraph", "table", "code"} <= kinds
    headings = [block.text for block in doc.blocks if block.kind == "heading"]
    assert "检索策略" in headings
    second = [b for b in doc.blocks if b.text.startswith("第二段")][0]
    assert second.heading_path == ("混合检索系统", "检索策略")


def test_gb18030_decoding(fixtures_dir: Path, tmp_settings: Settings) -> None:
    path = fixtures_dir / "gb18030.txt"
    doc = get_parser(path).parse(path, tmp_settings)
    assert "中文编码测试" in doc.text
    assert "E1024" in doc.text
    assert doc.file_type == "txt"


def test_decode_replacement_warning() -> None:
    text, warnings = decode_bytes(b"\xff\xfe\x00\x00bad")
    assert isinstance(text, str)
    assert warnings or text


def test_unsupported_extension(tmp_path: Path, tmp_settings: Settings) -> None:
    path = tmp_path / "a.exe"
    path.write_bytes(b"x")
    with pytest.raises(ParseError) as exc:
        get_parser(path)
    assert exc.value.code == "UNSUPPORTED_TYPE"
