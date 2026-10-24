"""HTML 解析：标题/段落/列表/表格/代码/图片 alt。

Step: STEP-03
Doc: docs/steps/STEP-03-解析层.md §5.5
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree, html

from app.parsing.base import (
    ParsedBlock,
    ParsedDocument,
    ParseError,
    finalize_document,
    normalize_text,
)
from app.parsing.text_md import decode_bytes

_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCK_TAGS = {
    "p",
    "blockquote",
    "li",
    "dt",
    "dd",
    "figcaption",
    "caption",
}
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "nav"}


class HtmlParser:
    """HTML 解析器：保留标题路径、表格文本、代码块和 alt text。"""

    extensions = (".html", ".htm")

    def parse(self, path: Path, settings) -> ParsedDocument:  # noqa: ANN001
        """解析 HTML 文件。

        Args:
            path: HTML 绝对路径。
            settings: 解析配置（当前不使用 OCR）。

        Returns:
            ParsedDocument。

        Raises:
            ParseError: 文件不可读或 HTML 无法解析。
        """
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ParseError("BROKEN_FILE", f"读取失败：{exc}") from exc
        text, warnings = decode_bytes(raw)
        try:
            root = html.fromstring(text)
        except (etree.ParserError, ValueError) as exc:
            raise ParseError("BROKEN_FILE", f"HTML 解析失败：{exc}") from exc

        title = ""
        content_root = root
        content_selector = "body"
        # 优先正文容器；只有内容足够长才采用，避免误选空 main。
        for selector in (
            "//main",
            "//article",
            "//*[@role='main']",
            "//*[@id='content']",
            "//*[contains(concat(' ', normalize-space(@class), ' '), ' content ')]",
        ):
            nodes = root.xpath(selector)
            if not nodes:
                continue
            candidate = max(nodes, key=lambda node: len(normalize_text(node.text_content())))
            if len(normalize_text(candidate.text_content())) >= 4:
                content_root = candidate
                content_selector = selector
                break
        title_nodes = root.xpath("//title")
        if title_nodes:
            title = normalize_text(title_nodes[0].text_content())
        blocks: list[ParsedBlock] = []
        heading_stack: list[str] = []
        first_h1 = ""

        def add(kind: str, value: str, **kwargs) -> None:  # noqa: ANN003
            body = normalize_text(value)
            if body:
                blocks.append(
                    ParsedBlock(
                        text=body,
                        kind=kind,
                        heading_path=tuple(heading_stack),
                        **kwargs,
                    )
                )

        def walk(node) -> None:  # noqa: ANN001
            """按文档顺序遍历 DOM。"""
            nonlocal first_h1
            tag = str(node.tag).lower() if isinstance(node.tag, str) else ""
            if tag in _SKIP_TAGS or tag in {"head", "title"}:
                return
            if tag in _HEADINGS:
                level = int(tag[1])
                value = normalize_text(node.text_content())
                if value:
                    heading_stack[:] = heading_stack[: level - 1] + [value]
                    if tag == "h1" and not first_h1:
                        first_h1 = value
                    add("heading", value)
                return
            if tag == "table":
                rows: list[str] = []
                for row in node.xpath(".//tr"):
                    cells = [
                        normalize_text(cell.text_content())
                        for cell in row.xpath("./th|./td")
                    ]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    add("table", "\n".join(rows))
                return
            if tag == "pre":
                lang = ""
                classes = str(node.get("class") or "").split()
                for item in classes:
                    if item.startswith("language-"):
                        lang = item.removeprefix("language-")
                        break
                add("code", node.text_content(), meta={"language": lang})
                return
            if tag == "img":
                alt = normalize_text(node.get("alt") or "")
                if alt:
                    add("paragraph", alt, meta={"source": "html_alt"})
                return
            if tag in _BLOCK_TAGS:
                add("paragraph", node.text_content())
                return
            if node.text and normalize_text(node.text):
                add("paragraph", node.text)
            for child in node:
                walk(child)

        walk(content_root)
        doc = ParsedDocument(
            path=path,
            file_type="html",
            title=title or first_h1 or path.stem,
            blocks=blocks,
            warnings=warnings,
            meta={"content_selector": content_selector},
        )
        return finalize_document(doc)
