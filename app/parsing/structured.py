"""JSON / CSV / LOG 轻量解析。

动态目录里常见的结构化文本先转成“可检索文本”：
- JSON 递归展开 key/value；
- CSV 按行转 `a | b`，首行作为表头；
- LOG 保留行结构。

Step: STEP-03
Doc: docs/steps/STEP-03-解析层.md §5.7
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from app.parsing.base import ParsedBlock, ParsedDocument, ParseError, finalize_document
from app.parsing.text_md import decode_bytes


def _flatten_json(value, prefix: str = "") -> list[str]:
    """把 JSON 递归展开成可检索的 key: value 行。"""
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, (dict, list)):
                lines.extend(_flatten_json(item, next_prefix))
            elif item is not None:
                lines.append(f"{next_prefix}: {item}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            if isinstance(item, (dict, list)):
                lines.extend(_flatten_json(item, next_prefix))
            elif item is not None:
                lines.append(f"{next_prefix}: {item}")
    elif value is not None:
        lines.append(f"{prefix}: {value}")
    return lines


class StructuredTextParser:
    """JSON/CSV/LOG 解析器。"""

    extensions = (".json", ".csv", ".log")

    def parse(self, path: Path, settings) -> ParsedDocument:  # noqa: ANN001
        """解析结构化文本；失败时抛 ParseError。"""
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ParseError("BROKEN_FILE", f"读取失败：{exc}") from exc
        text, warnings = decode_bytes(raw)
        suffix = path.suffix.lower()
        if suffix == ".json":
            file_type = "json"
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ParseError("BROKEN_FILE", f"JSON 解析失败：{exc}") from exc
            lines = _flatten_json(value)
            blocks = [ParsedBlock(text="\n".join(lines), kind="paragraph")] if lines else []
        elif suffix == ".csv":
            file_type = "csv"
            rows: list[str] = []
            try:
                for row in csv.reader(io.StringIO(text)):
                    cells = [cell.strip() for cell in row]
                    if any(cells):
                        rows.append(" | ".join(cells))
            except csv.Error as exc:
                raise ParseError("BROKEN_FILE", f"CSV 解析失败：{exc}") from exc
            blocks = [ParsedBlock(text="\n".join(rows), kind="table")] if rows else []
        else:
            file_type = "txt"
            blocks = [ParsedBlock(text=text, kind="paragraph")] if text.strip() else []
        return finalize_document(
            ParsedDocument(
                path=path,
                file_type=file_type,
                title=path.stem,
                blocks=blocks,
                warnings=warnings,
            )
        )
