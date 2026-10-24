"""生成可控的表格/版面验证 fixture。

输出 data/eval/layout/：
- merged.pdf / merged.png：合并单元格带线表格
- wireless.pdf / wireless.png：无表格线表格
- two_column.pdf / two_column.png：双栏正文
- merged_rot8.pdf：8° 旋转扫描页
- ground_truth.json：可控文本/结构
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "eval" / "layout"


def _table_page(*, lines: bool, merged: bool) -> pymupdf.Page:
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=700)
    page.insert_text((40, 50), "Table validation", fontsize=16)
    x0, y0, cell_w, cell_h = 50, 100, 150, 70
    rows, cols = 3, 3
    # merged header: row 0, col 0 spans 2 columns
    if lines:
        for r in range(rows + 1):
            page.draw_line((x0, y0 + r * cell_h), (x0 + cols * cell_w, y0 + r * cell_h))
        for c in range(cols + 1):
            page.draw_line((x0 + c * cell_w, y0), (x0 + c * cell_w, y0 + rows * cell_h))
        if merged:
            page.draw_line((x0 + cell_w, y0), (x0 + cell_w, y0 + cell_h), color=(1, 1, 1))
    cells = []
    texts = [
        ["Merged Header", "", "Notes"],
        ["A", "B", "C"],
        ["D", "E", "F"],
    ]
    for r, row in enumerate(texts):
        for c, text in enumerate(row):
            if not text:
                continue
            page.insert_text(
                (x0 + c * cell_w + 10, y0 + r * cell_h + 42),
                text,
                fontsize=14,
            )
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "text": text,
                    "rowspan": 1,
                    "colspan": 2 if merged and r == 0 and c == 0 else 1,
                }
            )
    return page, {"rows": rows, "cols": cols, "cells": cells}


def _two_column_page() -> pymupdf.Page:
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((40, 50), "Two column validation", fontsize=16)
    left = ["Left column one", "Left column two", "Left column three"]
    right = ["Right column one", "Right column two", "Right column three"]
    for i, text in enumerate(left):
        page.insert_text((40, 120 + i * 40), text, fontsize=12)
    for i, text in enumerate(right):
        page.insert_text((320, 120 + i * 40), text, fontsize=12)
    return page, {"columns": [left, right]}


def _render(page: pymupdf.Page, path_pdf: Path, path_png: Path) -> None:
    doc = page.parent
    doc.save(path_pdf)
    pix = page.get_pixmap(dpi=180)
    Image.open(io.BytesIO(pix.tobytes("png"))).save(path_png)
    doc.close()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    truth = {}
    page, truth["merged"] = _table_page(lines=True, merged=True)
    _render(page, OUT / "merged.pdf", OUT / "merged.png")
    page, truth["wireless"] = _table_page(lines=False, merged=False)
    _render(page, OUT / "wireless.pdf", OUT / "wireless.png")
    page, truth["two_column"] = _two_column_page()
    _render(page, OUT / "two_column.pdf", OUT / "two_column.png")
    # 8° 倾斜图片 PDF
    image = Image.open(OUT / "merged.png").convert("RGB").rotate(8, expand=False, fillcolor="white")
    rotated = pymupdf.open()
    rpage = rotated.new_page(width=image.width, height=image.height)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    rpage.insert_image(rpage.rect, stream=buf.getvalue())
    rotated.save(OUT / "merged_rot8.pdf")
    rotated.close()
    truth["merged_rot8"] = {"angle": 8}
    (OUT / "ground_truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "fixtures": list(truth)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
