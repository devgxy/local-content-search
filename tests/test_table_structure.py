"""离线段落：Table Transformer 检测/结构/单元格 OCR。"""

from __future__ import annotations

import io

import pymupdf
import pytest
from PIL import Image

from app.config import Settings
from app.parsing.ocr import ocr_available
from app.table_structure import TableStructureService, models_ready

pytestmark = [
    pytest.mark.skipif(not models_ready(Settings()), reason="表格模型未下载"),
    pytest.mark.skipif(not ocr_available(), reason="macOS Vision OCR 不可用"),
]


def _table_image() -> Image.Image:
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=560)
    for y in (120, 170, 220):
        page.draw_line((40, y), (380, y))
    for x in (40, 200, 380):
        page.draw_line((x, 120), (x, 220))
    page.insert_text((55, 150), "Module", fontsize=12)
    page.insert_text((215, 150), "Desc", fontsize=12)
    page.insert_text((55, 200), "Search", fontsize=12)
    page.insert_text((215, 200), "Hybrid", fontsize=12)
    pix = page.get_pixmap(dpi=180)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return image


def test_table_detection_and_structure() -> None:
    service = TableStructureService(Settings())
    service.ensure_ready()
    image = _table_image()
    tables = service.detect(image)
    assert tables
    result = service.recognize(image, tables[0][0])
    assert result.cells
    assert result.html.startswith("<table>")
    text = "\n".join(cell.text for cell in result.cells)
    assert "Module" in text or "Search" in text
    assert result.confidence > 0.5
    assert result.warnings == []


def test_merged_cell_html() -> None:
    from app.table_structure import TableCell, _to_html

    cells = [
        TableCell(0, 0, "合并表头", (0, 0, 10, 10), colspan=2, confidence=0.9),
        TableCell(1, 0, "A", (0, 10, 5, 20)),
        TableCell(1, 1, "B", (5, 10, 10, 20)),
    ]
    html = _to_html(cells, 2, 2)
    assert 'colspan="2"' in html
    assert "合并表头" in html
