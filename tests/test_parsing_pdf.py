"""STEP-03 PDF 解析测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app.config import Settings
from app.parsing.base import ParseError
from app.parsing.registry import get_parser


def test_text_pdf(fixtures_dir: Path, tmp_settings: Settings) -> None:
    path = fixtures_dir / "sample.pdf"
    doc = get_parser(path).parse(path, tmp_settings)
    assert doc.file_type == "pdf"
    assert len(doc.blocks) >= 1
    assert doc.blocks[0].page == 1
    assert "混合检索系统设计文档" in doc.text
    assert "E1024" in doc.text


def test_scanned_pdf_ocr(fixtures_dir: Path, tmp_settings: Settings) -> None:
    path = fixtures_dir / "scanned.pdf"
    doc = get_parser(path).parse(path, tmp_settings)
    assert doc.blocks
    assert doc.blocks[0].kind == "ocr"
    assert "E1024" in doc.text


def test_broken_pdf_does_not_crash(fixtures_dir: Path, tmp_settings: Settings) -> None:
    path = fixtures_dir / "broken.pdf"
    doc = get_parser(path).parse(path, tmp_settings)
    assert doc.file_type == "pdf"
    assert doc.blocks == []
    assert doc.warnings


def test_encrypted_pdf(tmp_path: Path, tmp_settings: Settings) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "secret")
    path = tmp_path / "encrypted.pdf"
    doc.save(
        str(path),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    doc.close()

    with pytest.raises(ParseError) as exc:
        get_parser(path).parse(path, tmp_settings)
    assert exc.value.code == "ENCRYPTED_PDF"


def test_short_text_pdf_keeps_text_on_ocr_failure(
    tmp_path: Path, tmp_settings: Settings, monkeypatch
) -> None:
    import pymupdf

    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if not font.exists():
        pytest.skip("测试字体不存在")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_font(fontname="cjk", fontfile=str(font))
    page.insert_text((72, 100), "短文本", fontname="cjk", fontsize=20)
    path = tmp_path / "short.pdf"
    doc.save(str(path))
    doc.close()

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise ParseError("OCR_UNAVAILABLE", "boom")

    monkeypatch.setattr("app.parsing.pdf.ocr_image_bytes_lines", boom)
    parsed = get_parser(path).parse(path, tmp_settings)
    assert "短文本" in parsed.text
    assert any("OCR 失败" in warning for warning in parsed.warnings)


def test_repeated_header_footer_removed(tmp_path: Path, tmp_settings: Settings) -> None:
    import pymupdf

    doc = pymupdf.open()
    for index in range(4):
        page = doc.new_page()
        page.insert_text((72, 30), "CONFIDENTIAL HEADER")
        page.insert_text((72, 100), f"Unique body page {index + 1}")
        page.insert_text((72, 805), f"Page {index + 1}")
    path = tmp_path / "header-footer.pdf"
    doc.save(str(path))
    doc.close()

    parsed = get_parser(path).parse(path, tmp_settings)
    assert "CONFIDENTIAL HEADER" not in parsed.text
    assert "Page 1" not in parsed.text
    assert "Unique body page 1" in parsed.text
    assert parsed.meta["header_footer_removed"] >= 8
    assert any("页眉/页脚" in warning for warning in parsed.warnings)


def test_single_page_header_is_preserved(tmp_path: Path, tmp_settings: Settings) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 30), "SINGLE PAGE HEADER")
    page.insert_text((72, 100), "Body")
    path = tmp_path / "single-header.pdf"
    doc.save(str(path))
    doc.close()

    parsed = get_parser(path).parse(path, tmp_settings)
    assert "SINGLE PAGE HEADER" in parsed.text
    assert parsed.meta["header_footer_removed"] == 0


def test_pdf_table_structure(tmp_path: Path, tmp_settings: Settings) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), "Module")
    page.insert_text((160, 90), "Desc")
    page.insert_text((72, 120), "Search")
    page.insert_text((160, 120), "Hybrid")
    # 画 2x2 表格线，PyMuPDF find_tables 依赖可见表格线。
    for y in (80, 105, 130):
        page.draw_line((60, y), (300, y))
    for x in (60, 150, 300):
        page.draw_line((x, 80), (x, 130))
    path = tmp_path / "table.pdf"
    doc.save(str(path))
    doc.close()

    no_ocr = replace(tmp_settings, ocr_enabled=False)
    parsed = get_parser(path).parse(path, no_ocr)
    tables = [block for block in parsed.blocks if block.kind == "table"]
    assert tables
    assert "Module" in tables[0].text
    assert "Hybrid" in tables[0].text

    # 关闭页眉页脚不应让表格提取失效。
    no_strip = replace(no_ocr, pdf_strip_header_footer=False)
    parsed_no_strip = get_parser(path).parse(path, no_strip)
    assert any(block.kind == "table" for block in parsed_no_strip.blocks)
