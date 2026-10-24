"""OCR 版面排序测试（不依赖 OCR 服务）。"""

from __future__ import annotations

from app.parsing.ocr import OcrLine
from app.parsing.ocr_layout import order_lines_columns, strip_repeated_margins


def _line(text: str, x: float, y: float) -> OcrLine:
    return OcrLine(text=text, confidence=1.0, bbox=(x, y, 0.2, 0.03))


def test_column_reading_order() -> None:
    lines = [
        _line("right-1", 0.60, 0.80),
        _line("left-1", 0.05, 0.80),
        _line("right-2", 0.60, 0.70),
        _line("left-2", 0.05, 0.70),
    ]
    ordered = order_lines_columns(lines)
    assert [line.text for line in ordered] == ["left-1", "left-2", "right-1", "right-2"]


def test_cross_page_margin_removal() -> None:
    pages = {
        0: [_line("CONFIDENTIAL", 0.1, 0.94), _line("body-1", 0.1, 0.5)],
        1: [_line("CONFIDENTIAL", 0.1, 0.94), _line("body-2", 0.1, 0.5)],
    }
    cleaned, removed = strip_repeated_margins(pages)
    assert removed == 2
    assert [line.text for line in cleaned[0]] == ["body-1"]
    assert [line.text for line in cleaned[1]] == ["body-2"]
