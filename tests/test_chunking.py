"""STEP-04 切片测试。"""

from __future__ import annotations

from pathlib import Path

from app.chunking import chunk_document
from app.config import Settings
from app.parsing.base import ParsedBlock, ParsedDocument, finalize_document


def _doc(tmp_path: Path, blocks: list[ParsedBlock], title: str = "测试文档") -> ParsedDocument:
    doc = ParsedDocument(
        path=tmp_path / "a.md",
        file_type="md",
        title=title,
        blocks=blocks,
    )
    return finalize_document(doc)


def _small_settings() -> Settings:
    return Settings(chunk_size=100, chunk_overlap=20, chunk_min=20, chunk_max=150)


def test_long_document(tmp_path: Path) -> None:
    body = "".join(f"这是第{i}句，讲混合检索。" for i in range(80))
    doc = _doc(
        tmp_path,
        [
            ParsedBlock(text="标题", kind="heading", heading_path=("标题",)),
            ParsedBlock(text=body, kind="paragraph", heading_path=("标题",)),
        ],
    )
    chunks = chunk_document(doc, _small_settings())
    assert len(chunks) > 1
    assert all(len(chunk.display_text) <= 160 for chunk in chunks)
    assert chunks[0].text.startswith("标题\n")
    assert chunks[0].chunk_index == 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_overlap_exists(tmp_path: Path) -> None:
    body = "".join(f"第{i}句内容比较长一些。" for i in range(40))
    doc = _doc(tmp_path, [ParsedBlock(text=body, kind="paragraph", heading_path=("H",))])
    chunks = chunk_document(doc, _small_settings())
    assert len(chunks) > 2
    overlapped = any(
        chunks[i].display_text[-10:] in chunks[i + 1].display_text
        for i in range(len(chunks) - 1)
    )
    assert overlapped


def test_page_boundary(tmp_path: Path) -> None:
    doc = _doc(
        tmp_path,
        [
            ParsedBlock(text="第一页内容。" * 10, kind="paragraph", page=1),
            ParsedBlock(text="第二页内容。" * 10, kind="paragraph", page=2),
        ],
    )
    chunks = chunk_document(doc, _small_settings())
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert all("第一页" not in c.display_text for c in chunks if c.page == 2)


def test_table_header_repeat(tmp_path: Path) -> None:
    lines = ["列1 | 列2"] + [f"值{i} | 说明{i}" for i in range(100)]
    doc = _doc(tmp_path, [ParsedBlock(text="\n".join(lines), kind="table", heading_path=("表",))])
    chunks = chunk_document(doc, _small_settings())
    assert len(chunks) > 1
    assert all(chunk.display_text.startswith("列1 | 列2") for chunk in chunks)
    assert all(len(chunk.display_text) <= 150 for chunk in chunks)


def test_short_document(tmp_path: Path) -> None:
    doc = _doc(tmp_path, [ParsedBlock(text="很短的一句话。", kind="paragraph")])
    chunks = chunk_document(doc, _small_settings())
    assert len(chunks) == 1
    assert chunks[0].display_text == "很短的一句话。"


def test_empty_document(tmp_path: Path) -> None:
    doc = _doc(tmp_path, [])
    assert chunk_document(doc, _small_settings()) == []


def test_deterministic(tmp_path: Path) -> None:
    body = "".join(f"第{i}句。" for i in range(100))
    doc = _doc(tmp_path, [ParsedBlock(text=body, kind="paragraph", heading_path=("H",))])
    first = chunk_document(doc, _small_settings())
    second = chunk_document(doc, _small_settings())
    assert [(c.display_text, c.chunk_index) for c in first] == [
        (c.display_text, c.chunk_index) for c in second
    ]


def test_offsets_point_into_document(tmp_path: Path) -> None:
    body = "".join(f"第{i}句内容。" for i in range(30))
    doc = _doc(tmp_path, [ParsedBlock(text=body, kind="paragraph", heading_path=("H",))])
    chunks = chunk_document(doc, _small_settings())
    for chunk in chunks:
        assert 0 <= chunk.char_start < chunk.char_end <= len(doc.text)
        assert chunk.display_text[:10] in doc.text[chunk.char_start : chunk.char_end]


def test_short_pages_not_merged(tmp_path: Path) -> None:
    doc = _doc(
        tmp_path,
        [
            ParsedBlock(text="第一页短句。", kind="paragraph", page=1),
            ParsedBlock(text="第二页短句。", kind="paragraph", page=2),
        ],
    )
    chunks = chunk_document(doc, _small_settings())
    assert len(chunks) == 2
    assert {chunk.page for chunk in chunks} == {1, 2}


def test_non_contiguous_sections_keep_order(tmp_path: Path) -> None:
    doc = _doc(
        tmp_path,
        [
            ParsedBlock(text="A1 内容。", kind="paragraph", heading_path=("A",)),
            ParsedBlock(text="B1 内容。", kind="paragraph", heading_path=("B",)),
            ParsedBlock(text="A2 内容。", kind="paragraph", heading_path=("A",)),
        ],
    )
    chunks = chunk_document(doc, _small_settings())
    assert [chunk.display_text for chunk in chunks] == ["A1 内容。", "B1 内容。", "A2 内容。"]


def test_overlap_respects_max_length(tmp_path: Path) -> None:
    body = "".join(f"第{i}段" + "长" * 300 + "。" for i in range(5))
    doc = _doc(tmp_path, [ParsedBlock(text=body, kind="paragraph", heading_path=("H",))])
    settings = Settings(chunk_size=320, chunk_overlap=60, chunk_min=20, chunk_max=360)
    chunks = chunk_document(doc, settings)
    assert len(chunks) > 1
    assert all(len(chunk.display_text) <= 360 for chunk in chunks)


def test_duplicate_body_in_different_sections_kept(tmp_path: Path) -> None:
    doc = _doc(
        tmp_path,
        [
            ParsedBlock(text="相同正文。", kind="paragraph", heading_path=("A",)),
            ParsedBlock(text="相同正文。", kind="paragraph", heading_path=("B",)),
        ],
    )
    chunks = chunk_document(doc, _small_settings())
    assert len(chunks) == 2
    assert {chunk.heading for chunk in chunks} == {"A", "B"}


def test_newline_preserved(tmp_path: Path) -> None:
    doc = _doc(tmp_path, [ParsedBlock(text="hello\nworld", kind="paragraph")])
    chunks = chunk_document(doc, _small_settings())
    assert chunks[0].display_text == "hello\nworld"


def test_adaptive_keeps_paragraph_under_max(tmp_path: Path) -> None:
    text = "甲" * 220 + "。" + "乙" * 220 + "。"
    doc = _doc(tmp_path, [ParsedBlock(text=text, kind="paragraph", heading_path=("H",))])
    adaptive = Settings(
        chunk_size=400,
        chunk_overlap=50,
        chunk_min=80,
        chunk_max=600,
        chunk_adaptive=True,
    )
    fixed = Settings(
        chunk_size=400,
        chunk_overlap=50,
        chunk_min=80,
        chunk_max=600,
        chunk_adaptive=False,
    )
    assert len(chunk_document(doc, adaptive)) == 1
    assert len(chunk_document(doc, fixed)) > 1


def test_overlap_is_capped_after_long_paragraph(tmp_path: Path) -> None:
    first = "甲" * 300 + "。"
    second = "乙" * 300 + "。"
    doc = _doc(
        tmp_path,
        [
            ParsedBlock(text=first, kind="paragraph", heading_path=("H",)),
            ParsedBlock(text=second, kind="paragraph", heading_path=("H",)),
        ],
    )
    settings = Settings(chunk_size=400, chunk_overlap=50, chunk_min=80, chunk_max=600)
    chunks = chunk_document(doc, settings)
    assert len(chunks) == 2
    assert len(chunks[1].display_text) <= settings.chunk_max
    assert chunks[1].display_text.startswith(chunks[0].display_text[-50:])


def test_long_table_row_repeats_header_and_row_key(tmp_path: Path) -> None:
    cell = "甲" * 300
    table = f"| 模块 | 说明 |\n|---|---|\n| A | {cell} |\n| B | 短 |"
    doc = _doc(tmp_path, [ParsedBlock(text=table, kind="table", heading_path=("T",))])
    settings = Settings(
        chunk_size=100,
        chunk_overlap=30,
        chunk_min=40,
        chunk_max=120,
    )
    chunks = chunk_document(doc, settings)

    assert len(chunks) >= 3
    assert all(chunk.display_text.startswith("| 模块 | 说明 |") for chunk in chunks)
    assert not any(
        chunk.display_text.strip() == "| 模块 | 说明 |" for chunk in chunks
    )
    assert all(len(chunk.display_text) <= settings.chunk_max for chunk in chunks)
    long_chunks = [chunk for chunk in chunks if "A" in chunk.display_text]
    assert len(long_chunks) >= 2
    assert all("| A" in chunk.display_text for chunk in long_chunks)
    assert any("（续）" in chunk.display_text for chunk in long_chunks)
    assert sum(chunk.display_text.count("甲") for chunk in chunks) == 300


def test_table_gets_short_preceding_context(tmp_path: Path) -> None:
    doc = _doc(
        tmp_path,
        [
            ParsedBlock(text="下表给出错误码说明。", kind="paragraph"),
            ParsedBlock(text="代码 | 含义\nE1024 | 超时", kind="table"),
        ],
    )
    chunks = chunk_document(doc, Settings(special_context=True))
    table = next(chunk for chunk in chunks if chunk.source_kind == "table")
    assert "下表给出错误码说明。" in table.text
    assert "下表给出错误码说明。" not in table.display_text


def test_special_context_can_be_disabled(tmp_path: Path) -> None:
    doc = _doc(
        tmp_path,
        [
            ParsedBlock(text="前一句说明。", kind="paragraph"),
            ParsedBlock(text="A | B\n1 | 2", kind="table"),
        ],
    )
    chunks = chunk_document(doc, Settings(special_context=False))
    table = next(chunk for chunk in chunks if chunk.source_kind == "table")
    assert "前一句说明。" not in table.text
