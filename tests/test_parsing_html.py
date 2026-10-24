"""STEP-03 HTML 解析测试。"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.parsing.registry import get_parser


def test_html_structure(tmp_path: Path, tmp_settings: Settings) -> None:
    path = tmp_path / "page.html"
    path.write_text(
        """<!doctype html><html><head><title>检索设计</title>
        <script>ignore()</script></head><body>
        <h1>架构</h1><h2>检索层</h2>
        <p>使用混合检索。</p>
        <table><tr><th>模块</th><th>说明</th></tr>
        <tr><td>FTS</td><td>关键词</td></tr></table>
        <pre class="language-python">print(1)</pre>
        <img alt="架构图" src="x.png">
        <ul><li>第一项</li><li>第二项</li></ul>
        </body></html>""",
        encoding="utf-8",
    )
    doc = get_parser(path).parse(path, tmp_settings)
    assert doc.file_type == "html"
    assert doc.title == "检索设计"
    kinds = [block.kind for block in doc.blocks]
    assert kinds[:2] == ["heading", "heading"]
    assert any(block.kind == "table" and "模块 | 说明" in block.text for block in doc.blocks)
    assert any(block.kind == "code" and "print(1)" in block.text for block in doc.blocks)
    assert any("架构图" in block.text for block in doc.blocks)
    assert "ignore()" not in doc.text
    paragraph = next(block for block in doc.blocks if block.text == "使用混合检索。")
    assert paragraph.heading_path == ("架构", "检索层")


def test_html_only_heading(tmp_path: Path, tmp_settings: Settings) -> None:
    path = tmp_path / "empty.html"
    path.write_text("<html><body><h1>只有标题</h1></body></html>", encoding="utf-8")
    doc = get_parser(path).parse(path, tmp_settings)
    # parser 返回标题 block；ingest 后续会判定没有可用正文
    assert doc.title == "只有标题"
    assert doc.blocks


def test_unsupported_html_extension(tmp_path: Path, tmp_settings: Settings) -> None:
    path = tmp_path / "page.htm"
    path.write_text("<p>htm 支持</p>", encoding="utf-8")
    doc = get_parser(path).parse(path, tmp_settings)
    assert doc.file_type == "html"


def test_html_prefers_main_content(tmp_path: Path, tmp_settings: Settings) -> None:
    path = tmp_path / "main.html"
    path.write_text(
        """<html><head><title>正文页</title></head><body>
        <nav>导航噪声</nav><main><h1>正文标题</h1><p>正文内容。</p></main>
        <footer>页脚噪声</footer></body></html>""",
        encoding="utf-8",
    )
    doc = get_parser(path).parse(path, tmp_settings)
    assert "正文内容。" in doc.text
    assert "导航噪声" not in doc.text
    assert "页脚噪声" not in doc.text
    assert doc.meta["content_selector"] == "//main"
