"""STEP-03 OCR 测试。"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.parsing.image_ocr import ImageParser, ocr_blob
from app.parsing.ocr import ocr_available, ocr_image

pytestmark = pytest.mark.skipif(not ocr_available(), reason="macOS Vision OCR 不可用")


def test_chinese_image(fixtures_dir: Path) -> None:
    text, warnings = ocr_image(fixtures_dir / "sample.png")
    assert "混合检索" in text
    assert "E1024" in text


def test_image_parser(fixtures_dir: Path, tmp_settings: Settings) -> None:
    path = fixtures_dir / "sample.png"
    doc = ImageParser().parse(path, tmp_settings)
    assert doc.file_type == "image"
    assert doc.blocks
    assert "混合检索" in doc.text


def test_tiny_blob_skipped(tmp_settings: Settings) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="PNG")
    text, warnings = ocr_blob(buffer.getvalue(), min_px=200)
    assert text == ""
    assert warnings


def test_ocr_lines_have_bbox(fixtures_dir: Path) -> None:
    from app.parsing.ocr import ocr_image_lines

    lines, warnings = ocr_image_lines(fixtures_dir / "sample.png")
    assert lines
    assert all(len(line.bbox) == 4 for line in lines)
    assert "E1024" in "\n".join(line.text for line in lines)


def test_scanned_layout_header_footer_table(tmp_path: Path, tmp_settings: Settings) -> None:
    import pymupdf

    from app.parsing.pdf import PdfParser

    source = pymupdf.open()
    for index in range(3):
        page = source.new_page(width=420, height=560)
        page.insert_text((40, 35), "CONFIDENTIAL HEADER", fontsize=12)
        page.insert_text((40, 75), f"Body page {index + 1} unique text", fontsize=14)
        page.insert_text((40, 525), f"Page {index + 1}", fontsize=10)
        if index == 0:
            for y in (150, 190, 230):
                page.draw_line((40, y), (380, y))
            for x in (40, 190, 380):
                page.draw_line((x, 150), (x, 230))
            page.insert_text((55, 175), "Module", fontsize=12)
            page.insert_text((205, 175), "Desc", fontsize=12)
            page.insert_text((55, 215), "Search", fontsize=12)
            page.insert_text((205, 215), "Hybrid", fontsize=12)
    scanned = pymupdf.open()
    for index in range(3):
        pix = source[index].get_pixmap(dpi=180)
        page = scanned.new_page(width=420, height=560)
        page.insert_image(page.rect, stream=pix.tobytes("png"))
    path = tmp_path / "layout.pdf"
    scanned.save(path)
    scanned.close()
    source.close()

    parsed = PdfParser().parse(path, tmp_settings)
    assert "Body page 1 unique text" in parsed.text
    assert "CONFIDENTIAL HEADER" not in parsed.text
    assert "Page 1" not in parsed.text
    tables = [block for block in parsed.blocks if block.kind == "table"]
    assert tables
    assert any("Module" in block.text and "Hybrid" in block.text for block in tables)
