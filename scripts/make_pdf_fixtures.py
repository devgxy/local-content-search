"""把指定 Markdown 文档转成合成 PDF 验证集。

用途：
- clean：纯文本层 PDF；
- header：重复页眉 + 页码；
- scanned：前两页栅格化为图片 PDF，触发整页 OCR。

输出目录默认在 data/ 下，不提交二进制文件。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pymupdf
from markdown_it import MarkdownIt
from playwright.sync_api import sync_playwright

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DEFAULT_DOCS = (
    "docs/16-数据清洗分块与向量化.md",
    "docs/03-评测方案.md",
    "docs/14-数据清洗与自适应切片.md",
    "docs/steps/STEP-03-解析层.md",
    "docs/steps/STEP-10-评测与基准.md",
)

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm; }
body { font-family: -apple-system, "PingFang SC", "Hiragino Sans GB", sans-serif;
       font-size: 10.5pt; line-height: 1.6; color: #111; }
h1, h2, h3 { page-break-after: avoid; }
h1 { font-size: 22pt; margin: 0 0 14pt; }
h2 { font-size: 16pt; margin-top: 20pt; border-bottom: 1px solid #ddd; }
h3 { font-size: 12.5pt; margin-top: 14pt; }
pre { background: #f6f8fa; padding: 8px; white-space: pre-wrap; word-break: break-word;
      font-size: 8.5pt; line-height: 1.45; }
code { font-family: "SFMono-Regular", Menlo, monospace; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; page-break-inside: auto; }
th, td { border: 1px solid #bbb; padding: 3px 5px; vertical-align: top; }
tr { page-break-inside: avoid; }
img { max-width: 100%; }
"""


def _html(path: Path) -> str:
    md = MarkdownIt("commonmark")
    md.enable("table")
    md.enable("strikethrough")
    body = md.render(path.read_text(encoding="utf-8"))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<style>{CSS}</style></head><body>{body}</body></html>"""


def _print_pdf(page, html: str, out: Path, *, header: bool) -> None:
    page.set_content(html, wait_until="load")
    page.emulate_media(media="print")
    kwargs = {
        "path": str(out),
        "format": "A4",
        "print_background": True,
        "margin": {"top": "22mm", "bottom": "20mm", "left": "16mm", "right": "16mm"},
    }
    if header:
        kwargs.update(
            display_header_footer=True,
            header_template="<div style='font-size:8px;width:100%;text-align:center;color:#666;'>本地检索系统 · 合成验证 PDF</div>",
            footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#666;'>第 <span class='pageNumber'></span> 页 / <span class='totalPages'></span> 页</div>",
        )
    page.pdf(**kwargs)


def _scanned(src: Path, out: Path, pages: int = 2) -> None:
    source = pymupdf.open(src)
    target = pymupdf.open()
    try:
        for page_no in range(min(pages, source.page_count)):
            page = source[page_no]
            pix = page.get_pixmap(dpi=170)
            new_page = target.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=pix.tobytes("png"))
        target.save(out)
    finally:
        source.close()
        target.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/eval/pdf-synth/root")
    parser.add_argument("--docs", nargs="*", default=list(DEFAULT_DOCS))
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        if not args.reset:
            raise SystemExit(f"输出目录已存在：{out}；用 --reset")
        shutil.rmtree(out)
    for name in ("clean", "header", "scanned"):
        (out / name).mkdir(parents=True)

    docs = [Path(item) for item in args.docs]
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True)
        page = browser.new_page()
        try:
            for doc in docs:
                html = _html(doc)
                clean = out / "clean" / f"{doc.stem}.pdf"
                header = out / "header" / f"{doc.stem}.pdf"
                _print_pdf(page, html, clean, header=False)
                _print_pdf(page, html, header, header=True)
                print(f"generated {clean} {header}")
            browser.close()
        finally:
            try:
                browser.close()
            except Exception:
                pass

    # 只对第一份长文档做 2 页图片 PDF，控制 OCR 成本。
    first_clean = out / "clean" / f"{docs[0].stem}.pdf"
    if first_clean.exists():
        _scanned(first_clean, out / "scanned" / f"{docs[0].stem}.pdf")
        print("generated scanned", out / "scanned" / f"{docs[0].stem}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
