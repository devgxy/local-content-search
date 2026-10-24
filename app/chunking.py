"""结构感知切片。

流程：解析块按 section 分组 → 句子/行 span → 按 400 字目标打包 →
短块合并 → 表格/代码按 chunk_max 硬切 → 去重/截断 → ChunkDraft。
标题会拼进 text 供 FTS/向量使用，display_text 保留纯正文。

Step: STEP-04
Doc: docs/steps/STEP-04-切片层.md §5
"""

# =============================================================================
# 代码导读：结构感知切片
# =============================================================================
#
# 职责：把 ParsedDocument 的 block 转成 ChunkDraft；不读文件、不编码、不写库。
#
# 输入：ParsedDocument（heading/paragraph/table/code/ocr/note）
# 输出：list[ChunkDraft]（text/display_text/heading/page/slide/offset/hash）
#
# 总体流程：
#
#   chunk_document(doc, settings)
#     -> 按 (heading_path, page, slide) 分 section（只合并连续相同 key）
#     -> section 内：
#          heading       -> 跳过（只影响 heading_path）
#          text 类       -> 文本缓冲，句子/整段 span，_pack_spans 打包
#          table/code    -> 先 flush 文本，再 _split_by_lines 按行切
#     -> _merge_short：同 section 短块合并
#     -> 去重：(content_hash, heading, page, slide)
#     -> 重排 chunk_index
#     -> 超过 max_chunks_per_doc 截断
#
# 示例：
#
#   输入 blocks：
#     heading "错误码"
#     paragraph "第一段。"
#     paragraph "第二段。"
#     table "表头：code | meaning\n行 1：E1024 | 连接超时"
#     code "print(1)"
#
#   处理：
#     1. heading 跳过；
#     2. 两个 paragraph 进入同一个 section 文本缓冲；
#        chunk_size=400，两个短段通常打包成一个 text chunk；
#     3. table flush 文本后，按 chunk_max 按行切；
#     4. code 再按行切；
#     5. table/code 不会和 paragraph 混拼。
#
#   输出示意：
#     chunk 0: text  "错误码\n第一段。第二段。"
#     chunk 1: table "错误码\n表头：..."
#     chunk 2: code  "错误码\nprint(1)"
#
# 关键规则：
#   - section key 含 heading_path/page/slide，PDF 每页、PPTX 每页独立；
#   - chunk_adaptive=True：段落 <= chunk_max 时整段作为 span，不按句拆；
#   - 段落 > chunk_max 才 _sentence_spans，单句再超长才 _hard_split；
#   - _pack_spans 目标 chunk_size，硬上限 chunk_max；
#   - overlap 只取上一块末尾最多 chunk_overlap 字；会挤爆 max 时放弃 overlap；
#   - table/code 按行切，table 重复首行表头；table/code 不共享 overlap；
#   - display_text 不带标题；text 前面拼 heading（最多 120 字）；
#   - 短块合并要求 page/slide/heading/source_kind 相同；
#   - 文本去重 key 含 heading/page/slide；table/code 不去重，避免长行续片丢数据。
#
# 已知限制：
#   - _pack_spans 直接用 "" 拼接 span，跨 paragraph 的空行不会保留；
#   - 表格重复表头时 char_start 是近似值；
#   - 表格/代码与前后说明段不共享 chunk（仅同 section 的检索上下文）；
#   - 无 heading 的 PDF 只能靠 page 分 section；
#   - 去重只针对同 section 的完全同正文；
#   - 截断发生在去重之后，截断后 chunk_index 重排。
# =============================================================================

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.config import Settings
from app.ids import make_content_hash
from app.parsing.base import ParsedBlock, ParsedDocument

logger = logging.getLogger(__name__)

_TEXT_KINDS = {"paragraph", "ocr", "note"}
_SPECIAL_KINDS = {"table", "code"}


@dataclass
class ChunkDraft:
    """切片中间结构，尚未写入 LanceDB。

    Attributes:
        chunk_index: 文档内顺序，由 chunk_document 最终赋值。
        text: 标题 + 正文，用于 FTS/向量。
        display_text: 展示正文，不含标题前缀。
        heading: 标题路径（最多 120 字）。
        page: PDF 页码。
        slide: PPTX 页码。
        char_start/char_end: 在解析全文中的字符偏移。
        source_kind: text/ocr/table/code 等。
        content_hash: 正文内容哈希，用于去重。
    """

    chunk_index: int
    text: str
    display_text: str
    heading: str
    page: int
    slide: int
    char_start: int
    char_end: int
    source_kind: str
    content_hash: str
    meta: dict = field(default_factory=dict)


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """按中英文句末标点/换行切句。

    只在段落超过 chunk_max（或关闭 adaptive）时使用；句末标点保留在
    前一句中，偏移相对当前 block 的 text，由调用方加上 block.char_start。

    Args:
        text: 块正文。

    Returns:
        ``[(句子, local_start, local_end)]``，偏移相对块首。
    """
    spans: list[tuple[str, int, int]] = []
    start = 0
    for index, char in enumerate(text):
        boundary = char in "。！？!?；;\n"
        if char == "." and (index + 1 == len(text) or text[index + 1].isspace()):
            boundary = True
        if not boundary:
            continue
        end = index + 1
        raw = text[start:end]
        if raw.strip():
            spans.append((raw, start, end))
        start = end
    if start < len(text):
        raw = text[start:]
        if raw.strip():
            spans.append((raw, start, len(text)))
    return spans


def _hard_split(text: str, start: int, max_len: int, overlap: int) -> list[tuple[str, int, int]]:
    """超长文本按固定长度硬切。

    Args:
        text: 待切文本。
        start: 在全文中的绝对起点。
        max_len: 单块最大长度。
        overlap: 相邻块重叠字符数。

    Returns:
        ``[(piece, abs_start, abs_end)]``。
    """
    pieces = []
    cursor = 0
    while cursor < len(text):
        end = min(cursor + max_len, len(text))
        pieces.append((text[cursor:end], start + cursor, start + end))
        if end == len(text):
            break
        # 下一块从 end-overlap 开始；cursor+1 防止 overlap>=max_len 时死循环。
        cursor = max(end - overlap, cursor + 1)
    return pieces


def _pack_spans(
    spans: list[tuple[str, int, int]],
    *,
    target: int,
    overlap: int,
    max_len: int,
) -> list[tuple[str, int, int]]:
    """按 target 打包句子 span，并保留 overlap。

    Args:
        spans: 句子/行 span。
        target: 目标块长。
        overlap: 相邻块重叠字符数。
        max_len: 硬上限，overlap 不得挤爆。

    Returns:
        ``[(body, abs_start, abs_end)]``。
    """
    chunks: list[tuple[str, int, int]] = []
    buffer: list[tuple[str, int, int]] = []
    buffer_len = 0

    def emit() -> None:
        """把当前 span 追加为候选块。"""
        nonlocal buffer, buffer_len
        if not buffer:
            return
        # 注意：这里直接拼接，不补换行/空行；跨 paragraph 的边界不会保留。
        body = "".join(item[0] for item in buffer)
        chunks.append((body, buffer[0][1], buffer[-1][2]))
        buffer = []
        buffer_len = 0

    for span in spans:
        text, start, end = span
        if len(text) > max_len:
            # 单句/单行已超过硬上限：先结束当前块，再按固定长度硬切。
            emit()
            for piece in _hard_split(text, start, max_len, overlap):
                chunks.append(piece)
            continue
        if buffer and buffer_len + len(text) > target:
            # 到达软目标：先 emit，再从上一块末尾取 overlap 作为下一块开头。
            previous = buffer
            emit()
            # overlap 只取上一块末尾最多 overlap 个字符；不能把整段长句
            # 全量带入下一块，否则会重复大量正文。
            tail: list[tuple[str, int, int]] = []
            remaining = overlap
            for item_text, item_start, item_end in reversed(previous):
                if remaining <= 0:
                    break
                if len(item_text) <= remaining:
                    tail.insert(0, (item_text, item_start, item_end))
                    remaining -= len(item_text)
                else:
                    tail.insert(
                        0,
                        (item_text[-remaining:], item_end - remaining, item_end),
                    )
                    remaining = 0
            buffer = tail
            buffer_len = sum(len(item[0]) for item in tail)
        if buffer and buffer_len + len(text) > max_len:
            # overlap 会挤爆上限时放弃 overlap，优先保证单块长度。
            buffer = []
            buffer_len = 0
        buffer.append(span)
        buffer_len += len(text)
    emit()
    return chunks


_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _is_table_separator(line: str) -> bool:
    """判断 Markdown 表格分隔行（|---|---|）。"""
    return "-" in line and bool(_TABLE_SEPARATOR_RE.match(line))


def _row_parts(row: str) -> tuple[str, str]:
    """拆出行键前缀和内容。

    Markdown:
        "| A | long cell |" -> ("| A |", "long cell |")
    普通/PPTX:
        "A | long cell"    -> ("A | ", "long cell")
        "行 1：A | long"   -> ("行 1：A | ", "long")
    """
    if row.startswith("|"):
        end = row.find("|", 1)
        if end > 0:
            # 保留第一个单元格后的空格，输出更接近原始表格。
            return row[: end + 1], row[end + 1 :]
    if " | " in row:
        prefix, content = row.split(" | ", 1)
        return prefix + " | ", content
    return "", row


def _continuation_prefix(prefix: str) -> str:
    """给行键加“续”标记，保留 Markdown/PPTX 的列结构。"""
    if not prefix:
        return "（续） "
    stripped = prefix.rstrip()
    if stripped.endswith("|"):
        return stripped[:-1].rstrip() + "（续） | "
    return prefix + "（续） "


def _split_table(block: ParsedBlock, *, limit: int) -> list[tuple[str, int, int]]:
    """表格行打包：不产生表头单独 chunk；超长行重复表头/行键。

    普通行按行打包；某行超过剩余空间时，把该行内容切成多片，每片都带
    表头和行键，续片行键加“（续）”。
    """
    lines = block.text.split("\n")
    if not lines:
        return []
    limit = max(limit, 1)
    header = lines[0]
    rows = [
        line
        for line in lines[1:]
        if line.strip() and not _is_table_separator(line)
    ]
    if not rows:
        # 只有表头没有数据行：不生成“表头单独成块”。
        return []

    chunks: list[tuple[str, int, int]] = []
    current: list[str] = [header]
    has_data = False
    current_start = block.char_start
    cursor = block.char_start + len(header) + 1

    def emit() -> None:
        """输出当前表格块；只有表头时不输出。"""
        nonlocal current, has_data, current_start
        if has_data:
            chunks.append(("\n".join(current), current_start, min(cursor, block.char_end)))
        current = [header]
        has_data = False
        current_start = cursor

    for row in rows:
        row_start = cursor
        row_len = len(row)
        # 普通行：表头 + 行能放下。
        if len(header) + 1 + row_len <= limit:
            if has_data and sum(len(item) + 1 for item in current) + row_len > limit:
                emit()
            current.append(row)
            has_data = True
            cursor = row_start + row_len + 1
            continue

        # 长行：先结束已有数据块；当前只剩表头，不单独输出。
        if has_data:
            emit()

        prefix, content = _row_parts(row)
        cont_prefix = _continuation_prefix(prefix)
        # 用两个前缀中较长者计算预算，保证续片也不超限。
        prefix_len = max(len(prefix), len(cont_prefix))
        budget = limit - len(header) - 1 - prefix_len
        if budget < 1:
            # 表头/行键本身超限：退化为整行硬切，避免死循环。
            pieces = _hard_split(row, row_start, limit, 0)
            for piece, start, end in pieces:
                chunks.append((piece, start, end))
        elif content:
            pieces = _hard_split(content, row_start + len(prefix), budget, 0)
            for index, (piece, _start, _end) in enumerate(pieces):
                line_prefix = prefix if index == 0 else cont_prefix
                body = f"{header}\n{line_prefix}{piece}"
                chunks.append((body, row_start, min(row_start + row_len, block.char_end)))
        else:
            # 行键本身超长且没有内容：整行硬切。
            pieces = _hard_split(row, row_start, limit, 0)
            for piece, start, end in pieces:
                chunks.append((piece, start, end))

        cursor = row_start + row_len + 1
        current = [header]
        has_data = False
        current_start = cursor

    emit()
    return chunks


def _split_by_lines(
    block: ParsedBlock,
    *,
    limit: int,
    repeat_header: bool,
) -> list[tuple[str, int, int]]:
    """表格/代码按行打包，超长行硬切。

    Args:
        block: 表格/代码块。
        limit: 单块上限（通常 chunk_max）。
        repeat_header: 是否重复首行表头。

    Returns:
        ``[(body, abs_start, abs_end)]``。
    """
    lines = block.text.split("\n")
    if not lines:
        return []
    limit = max(limit, 1)
    # 表格把首行当表头；后续每块重复它。代码块不重复。
    header = lines[0] if repeat_header and lines else ""
    chunks: list[tuple[str, int, int]] = []
    buffer: list[str] = []
    buffer_start = block.char_start
    cursor = block.char_start
    for line in lines:
        # 单行超过 limit 时硬切；这里不需要 overlap，避免表格行错位。
        pieces = [line] if len(line) <= limit else [
            piece[0] for piece in _hard_split(line, cursor, limit, 0)
        ]
        for piece in pieces:
            if buffer and sum(len(item) + 1 for item in buffer) + len(piece) > limit:
                body = "\n".join(buffer)
                chunks.append((body, buffer_start, cursor))
                # 新块以表头开头；如果表头+当前行本身超限，则放弃表头。
                buffer = [header] if header else []
                if buffer and sum(len(item) + 1 for item in buffer) + len(piece) > limit:
                    buffer = []
                buffer_start = cursor
            buffer.append(piece)
            cursor += len(piece) + 1
    if buffer:
        chunks.append(("\n".join(buffer), buffer_start, min(cursor, block.char_end)))
    return chunks


def _make_draft(
    body: str,
    start: int,
    end: int,
    block: ParsedBlock,
    *,
    page: int,
    slide: int,
    source_kind: str,
    context: str = "",
) -> ChunkDraft:
    """把正文片段包装成 ChunkDraft。

    Args:
        body: 纯正文。
        start/end: 字符偏移。
        block: 来源块。
        page/slide: 页码。
        source_kind: 来源类型。

    Returns:
        ChunkDraft；chunk_index 暂为 -1。
    """
    # heading 是检索上下文，最长 120 字；display_text 只保留正文。
    heading = " / ".join(block.heading_path)[:120]
    parts = [part for part in (heading, context.strip(), body) if part]
    text = "\n".join(parts)
    return ChunkDraft(
        chunk_index=-1,
        text=text,
        display_text=body,
        heading=heading,
        page=page,
        slide=slide,
        char_start=start,
        char_end=end,
        source_kind=source_kind,
        content_hash=make_content_hash(body),
        meta=dict(block.meta),
    )


def _merge_short(chunks: list[ChunkDraft], *, min_len: int, max_len: int) -> list[ChunkDraft]:
    """合并同 section 的过短块。

    Args:
        chunks: 已切片列表。
        min_len: 低于该长度尝试合并。
        max_len: 合并后不得超过。

    Returns:
        新列表；不修改入参。
    """
    merged: list[ChunkDraft] = []
    for chunk in chunks:
        # 只和紧邻的上一块合并；page/slide/heading/source_kind 必须一致。
        if (
            merged
            and len(merged[-1].display_text) < min_len
            and len(merged[-1].display_text) + len(chunk.display_text) + 1 <= max_len
            and merged[-1].page == chunk.page
            and merged[-1].slide == chunk.slide
            and merged[-1].heading == chunk.heading
            and merged[-1].source_kind == chunk.source_kind
        ):
            previous = merged[-1]
            previous.display_text = f"{previous.display_text}\n{chunk.display_text}"
            previous.text = (
                f"{previous.heading}\n{previous.display_text}"
                if previous.heading
                else previous.display_text
            )
            previous.char_end = chunk.char_end
            previous.content_hash = make_content_hash(previous.display_text)
            continue
        merged.append(chunk)
    return merged


def chunk_document(doc: ParsedDocument, settings: Settings) -> list[ChunkDraft]:
    """把 ParsedDocument 切成 ChunkDraft 列表。

    Args:
        doc: 解析后的文档。
        settings: chunk_size/overlap/min/max/adaptive 等。

    Returns:
        去重、编号、截断后的 chunk 列表；空文档返回 []。
    """
    logger.info(
        "开始切片: path=%s blocks=%d adaptive=%s target=%d overlap=%d max=%d",
        doc.path,
        len(doc.blocks),
        settings.chunk_adaptive,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.chunk_max,
    )
    sections: list[tuple[tuple, list[ParsedBlock]]] = []
    for block in doc.blocks:
        key = (block.heading_path, block.page, block.slide)
        if not sections or sections[-1][0] != key:
            sections.append((key, []))
        sections[-1][1].append(block)
    logger.debug("切片分组: path=%s sections=%d", doc.path, len(sections))

    drafts: list[ChunkDraft] = []
    for (heading_path, page, slide), blocks in sections:
        logger.debug(
            "切片 section: path=%s page=%s slide=%s heading=%s blocks=%d",
            doc.path,
            page,
            slide,
            " / ".join(heading_path),
            len(blocks),
        )
        text_blocks: list[ParsedBlock] = []

        def flush_text(page_no: int = page, slide_no: int = slide) -> None:
            """把缓冲段刷成 text block。"""
            nonlocal text_blocks
            if not text_blocks:
                return
            spans: list[tuple[str, int, int]] = []
            for block in text_blocks:
                # 自适应：段落不超过硬上限时整段作为一个 span，避免把
                # 完整段落从中间拆开；超长段落才按句/硬切。
                if settings.chunk_adaptive and len(block.text) <= settings.chunk_max:
                    spans.append((block.text, block.char_start, block.char_end))
                    continue
                for text, start, end in _sentence_spans(block.text):
                    spans.append((text, block.char_start + start, block.char_start + end))
            packed = _pack_spans(
                spans,
                target=settings.chunk_size,
                overlap=settings.chunk_overlap,
                max_len=settings.chunk_max,
            )
            logger.debug(
                "切片文本: path=%s page=%s slide=%s blocks=%d spans=%d packed=%d",
                doc.path,
                page_no,
                slide_no,
                len(text_blocks),
                len(spans),
                len(packed),
            )
            kind = "ocr" if all(block.kind == "ocr" for block in text_blocks) else "text"
            if any(block.kind == "note" for block in text_blocks) and kind == "text":
                kind = "note" if all(block.kind == "note" for block in text_blocks) else "text"
            for body, start, end in packed:
                if body.strip():
                    drafts.append(
                        _make_draft(
                            body,
                            start,
                            end,
                            text_blocks[0],
                            page=page_no,
                            slide=slide_no,
                            source_kind=kind,
                        )
                    )
            text_blocks = []

        for block in blocks:
            if block.kind == "heading":
                continue
            if block.kind in _TEXT_KINDS:
                text_blocks.append(block)
                continue
            if block.kind in _SPECIAL_KINDS:
                # 表格/代码前的最后一段说明，作为短上下文拼进检索 text；
                # display_text 仍只展示表格/代码本身。
                context = (
                    text_blocks[-1].text[-160:]
                    if settings.special_context and text_blocks
                    else ""
                )
                flush_text()
                if block.kind == "table":
                    pieces = _split_table(block, limit=settings.chunk_max)
                else:
                    pieces = _split_by_lines(
                        block,
                        limit=settings.chunk_max,
                        repeat_header=False,
                    )
                logger.debug(
                    "切片特殊块: path=%s kind=%s page=%s slide=%s chars=%d pieces=%d",
                    doc.path,
                    block.kind,
                    page,
                    slide,
                    len(block.text),
                    len(pieces),
                )
                for body, start, end in pieces:
                    if body.strip():
                        drafts.append(
                            _make_draft(
                                body,
                                start,
                                end,
                                block,
                                page=page,
                                slide=slide,
                                source_kind=block.kind,
                                context=context,
                            )
                        )
        flush_text()

    before_merge = len(drafts)
    drafts = _merge_short(drafts, min_len=settings.chunk_min, max_len=settings.chunk_max)
    logger.debug(
        "切片短块合并: path=%s before=%d after=%d",
        doc.path,
        before_merge,
        len(drafts),
    )

    deduped: list[ChunkDraft] = []
    seen: set[tuple] = set()
    for draft in drafts:
        # table/code 不做正文去重：长表格行的续片可能正文相同，但它们是
        # 同一行的不同内容片段，去重会直接丢数据。
        if draft.source_kind in {"table", "code"}:
            draft.chunk_index = len(deduped)
            deduped.append(draft)
            continue
        key = (draft.content_hash, draft.heading, draft.page, draft.slide)
        if key in seen:
            continue
        seen.add(key)
        draft.chunk_index = len(deduped)
        deduped.append(draft)
    logger.debug(
        "切片去重: path=%s before=%d after=%d",
        doc.path,
        len(drafts),
        len(deduped),
    )

    if len(deduped) > settings.max_chunks_per_doc:
        logger.warning(
            "文档切片数 %d 超过上限 %d，已截断：%s",
            len(deduped),
            settings.max_chunks_per_doc,
            doc.path,
        )
        deduped = deduped[: settings.max_chunks_per_doc]
        for index, draft in enumerate(deduped):
            draft.chunk_index = index

    for draft in deduped:
        logger.debug(
            "切片 chunk: path=%s index=%d kind=%s chars=%d page=%s slide=%s offsets=%d-%d",
            doc.path,
            draft.chunk_index,
            draft.source_kind,
            len(draft.display_text),
            draft.page,
            draft.slide,
            draft.char_start,
            draft.char_end,
        )
    if deduped:
        avg_chars = sum(len(draft.display_text) for draft in deduped) / len(deduped)
        max_chars = max(len(draft.display_text) for draft in deduped)
    else:
        avg_chars = 0.0
        max_chars = 0
    logger.info(
        "切片完成: path=%s chunks=%d avg_chars=%.1f max_chars=%d",
        doc.path,
        len(deduped),
        avg_chars,
        max_chars,
    )
    return deduped
