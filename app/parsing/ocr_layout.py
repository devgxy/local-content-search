"""OCR 版面：跨页页边统计、行列分组、表格候选。

输入均为 ``OcrLine``；坐标是 Vision 归一化 bbox，原点在左下。
"""

from __future__ import annotations

import re
from collections import Counter

from app.parsing.ocr import OcrLine

_PAGE_NUMBER_RE = re.compile(
    r"^(?:第\s*)?\d{1,4}(?:\s*页)?(?:\s*/\s*\d{1,4})?$|"
    r"^\d{1,4}\s*/\s*\d{1,4}$|"
    r"^page\s*\d{1,4}(?:\s*of\s*\d{1,4})?$",
    re.IGNORECASE,
)


def _key(text: str) -> str:
    return re.sub(r"\d+", "#", text.casefold()).strip()


def _zone(line: OcrLine, ratio: float) -> str | None:
    _, y, _, height = line.bbox
    top = y + height
    if top >= 1.0 - ratio:
        return "top"
    if y <= ratio:
        return "bottom"
    return None


def strip_repeated_margins(
    pages: dict[int, list[OcrLine]],
    *,
    ratio: float = 0.08,
) -> tuple[dict[int, list[OcrLine]], int]:
    """跨页删除重复页眉/页脚/页码；只处理页面边缘区域。"""
    if len(pages) < 2:
        return pages, 0
    counts: Counter[tuple[str, str]] = Counter()
    for lines in pages.values():
        for line in lines:
            zone = _zone(line, ratio)
            if zone:
                counts[(zone, _key(line.text))] += 1
    threshold = max(2, int(len(pages) * 0.3 + 0.999))
    repeated = {key for key, count in counts.items() if count >= threshold}
    cleaned: dict[int, list[OcrLine]] = {}
    removed = 0
    for page_no, lines in pages.items():
        kept: list[OcrLine] = []
        for line in lines:
            zone = _zone(line, ratio)
            drop = bool(
                zone
                and ((zone, _key(line.text)) in repeated or _PAGE_NUMBER_RE.fullmatch(line.text.strip()))
            )
            if drop:
                removed += 1
            else:
                kept.append(line)
        cleaned[page_no] = kept
    return cleaned, removed


def group_rows(lines: list[OcrLine], *, tolerance: float = 0.018) -> list[list[OcrLine]]:
    """按 bbox 中心 y 分行；行内按 x 排序。"""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda line: -(line.bbox[1] + line.bbox[3] / 2))
    rows: list[list[OcrLine]] = []
    centers: list[float] = []
    for line in ordered:
        center = line.bbox[1] + line.bbox[3] / 2
        target = None
        for index, row_center in enumerate(centers):
            if abs(center - row_center) <= tolerance:
                target = index
                break
        if target is None:
            centers.append(center)
            rows.append([line])
        else:
            rows[target].append(line)
    for row in rows:
        row.sort(key=lambda line: line.bbox[0])
    return rows


def table_candidates(
    rows: list[list[OcrLine]], *, min_cells: int = 2
) -> list[list[list[OcrLine]]]:
    """连续多行、每行至少 min_cells 个 x 分离文本，视为表格候选。

    这是保守启发式；不是完整表格结构识别。
    """
    groups: list[list[list[OcrLine]]] = []
    current: list[list[OcrLine]] = []
    for row in rows:
        if len(row) >= min_cells:
            current.append(row)
        else:
            if len(current) >= 2:
                groups.append(current)
            current = []
    if len(current) >= 2:
        groups.append(current)
    return groups


def table_text(table: list[list[OcrLine]]) -> str:
    return "\n".join(" | ".join(line.text for line in row) for row in table)


def order_lines_columns(lines: list[OcrLine], *, gap: float = 0.12) -> list[OcrLine]:
    """按 x 方向大间距分栏，再按栏从左到右、栏内从上到下排序。

    这是版面启发式；只有至少两栏、每栏至少两行时才改变顺序。
    """
    if len(lines) < 4:
        return sorted(lines, key=lambda line: (-(line.bbox[1] + line.bbox[3]), line.bbox[0]))
    ordered = sorted(lines, key=lambda line: line.bbox[0])
    groups: list[list[OcrLine]] = []
    for line in ordered:
        center = line.bbox[0] + line.bbox[2] / 2
        if not groups:
            groups.append([line])
            continue
        previous = groups[-1][-1]
        prev_center = previous.bbox[0] + previous.bbox[2] / 2
        if center - prev_center > gap:
            groups.append([line])
        else:
            groups[-1].append(line)
    usable = [group for group in groups if len(group) >= 2]
    if len(usable) < 2:
        return sorted(lines, key=lambda line: (-(line.bbox[1] + line.bbox[3]), line.bbox[0]))
    result: list[OcrLine] = []
    for group in usable:
        result.extend(sorted(group, key=lambda line: (-(line.bbox[1] + line.bbox[3]), line.bbox[0])))
    return result
