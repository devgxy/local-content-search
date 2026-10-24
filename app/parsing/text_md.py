"""TXT / Markdown 解析。

Step: STEP-03
Doc: docs/steps/STEP-03-解析层.md §5.1
"""

# =============================================================================
# 代码导读：TextMarkdownParser
# =============================================================================
#
# 这个文件只负责把 .txt/.md/.markdown 解析成 ParsedDocument，
# 不做切片、不做向量化、不写 LanceDB。
#
# 完整流程：
#
#   parse(path, settings)
#     -> path.read_bytes()
#     -> decode_bytes()        编码识别/回退
#     -> normalize_text()      换行/NUL/行尾空白
#     -> 按扩展名分流
#          .md/.markdown -> _parse_markdown()
#          其它          -> _parse_text()
#     -> finalize_document()   丢空 block/算 char 偏移/生成全文
#     -> ParsedDocument
#
# TXT 示例：
#
#   输入：
#     第一段第一行
#     第一段第二行
#
#     第二段
#
#   输出 blocks：
#     paragraph: "第一段第一行\n第一段第二行"
#     paragraph: "第二段"
#
#   TXT 只用空行分段，不识别 Markdown 结构。
#
# Markdown 示例：
#
#   输入：
#     # 错误码
#
#     下表列出错误码：
#
#     | code | meaning |
#     |---|---|
#     | E1024 | 连接超时 |
#
#     ```python
#     print("E1024")
#     ```
#
#     ## 子标题
#
#     第二段。
#
#   输出 blocks（按顺序）：
#     heading    "错误码"              heading_path=["错误码"]
#     paragraph  "下表列出错误码："      heading_path=["错误码"]
#     table      "| code | meaning |\n|---|---|\n| E1024 | 连接超时 |"
#                                     heading_path=["错误码"]
#     code       "print(\"E1024\")"    heading_path=["错误码"]
#                                     meta={"language": "python"}
#     heading    "子标题"              heading_path=["错误码", "子标题"]
#     paragraph  "第二段。"            heading_path=["错误码", "子标题"]
#
#   注意：heading block 不进入正文 chunk；后续 chunker 会跳过它，
#   但会把 heading_path 作为后续 paragraph/table/code 的标题前缀。
#
# Markdown 是逐行状态机，单行优先级：
#
#   1. ``` / ~~~ 代码围栏：进出 code，并记录语言
#   2. 代码块内部：原样加入 code，不解析标题/表格
#   3. # 标题：flush 段落/表格，更新 heading_stack，产出 heading block
#   4. |...| 表格行：flush 段落，加入 table 缓冲
#   5. 空行：flush 段落/表格
#   6. 普通行：flush 表格，加入 para 缓冲
#
# 文件结束时依次 flush_code/flush_para/flush_table。
#
# 已知边界：
#   - Setext 标题（标题下一行 ====）不识别；
#   - 列表/引用/HTML/图片不做结构解析，仍是 paragraph 文本；
#   - 表格只按“首尾都是 |”判断，不校验分隔行/单元格；
#   - 代码围栏未闭合时，文件剩余内容都算 code，且没有 warning；
#   - 只有标题没有正文时，后续 chunker 可能产出空列表，
#     ingest 会报 EMPTY_DOCUMENT。
#
# 下游关系：
#   - paragraph/ocr/note -> 文本缓冲 -> 400/600 切片；
#   - table/code        -> 单独按行切，table 重复表头；
#   - heading           -> 跳过，不占 chunk；
#   - heading_path      -> 进入 ChunkDraft.text 前缀；
#   - 表格/代码前的说明句目前不会自动进入 table/code chunk，
#     这是“表格/代码语义上下文偏弱”的主要原因。
# =============================================================================

from __future__ import annotations

import re
from pathlib import Path

from charset_normalizer import from_bytes

from app.parsing.base import (
    ParsedBlock,
    ParsedDocument,
    ParseError,
    finalize_document,
    normalize_text,
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^(```|~~~)(.*)$")

_FALLBACK_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "big5")


def decode_bytes(raw: bytes) -> tuple[str, list[str]]:
    """按编码回退解码文本。

    策略：
    1. charset_normalizer 自动识别；
    2. 失败后按 utf-8/utf-8-sig/gb18030/big5 依次尝试；
    3. 全部失败才使用 UTF-8 替换模式并记录 warning。
    """
    warnings: list[str] = []
    # best() 对短文本可能误判；只有它完全失败才走固定编码回退。
    best = from_bytes(raw).best()
    if best is not None and best.encoding:
        try:
            return str(best), warnings
        except Exception:  # noqa: BLE001
            pass
    for encoding in _FALLBACK_ENCODINGS:
        try:
            return raw.decode(encoding), warnings
        except UnicodeDecodeError:
            continue
    warnings.append("编码识别失败，使用 UTF-8 替换模式，可能有字符丢失")
    return raw.decode("utf-8", errors="replace"), warnings


class TextMarkdownParser:
    """TXT/Markdown 解析器。"""
    extensions = (".txt", ".md", ".markdown")

    def parse(self, path: Path, settings) -> ParsedDocument:  # noqa: ANN001
        """解析 TXT/Markdown。"""
        # 1) 读取原始字节；编码识别放在 decode_bytes，避免业务逻辑混入。
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ParseError("BROKEN_FILE", f"读取失败：{exc}") from exc

        # 2) 先解码，再统一换行/NUL/行尾空白；后续所有解析都基于规范化文本。
        text, warnings = decode_bytes(raw)
        text = normalize_text(text)
        # 3) 只有 .md/.markdown 走 Markdown 结构解析；.txt 一律按空行分段。
        if path.suffix.lower() in {".md", ".markdown"}:
            doc = self._parse_markdown(path, text)
        else:
            doc = self._parse_text(path, text)
        # 4) finalize 负责丢空 block、计算 char_start/end、生成 doc.text。
        doc.warnings.extend(warnings)
        return finalize_document(doc)

    def _parse_text(self, path: Path, text: str) -> ParsedDocument:
        """把纯文本按行/空行解析为段落块。"""
        # 只用空行分段；段内单个换行保留在 block.text 中。
        blocks = [
            ParsedBlock(text=normalize_text(chunk), kind="paragraph")
            for chunk in re.split(r"\n\s*\n", text)
            if chunk.strip()
        ]
        return ParsedDocument(path=path, file_type="txt", title=path.stem, blocks=blocks)

    def _parse_markdown(self, path: Path, text: str) -> ParsedDocument:
        """解析 Markdown 标题/段落/表格/代码块。"""
        # 状态机：同一时刻只可能处于 para/table/code 三种缓冲之一。
        # heading_stack 保存当前标题路径，flush 时会复制到每个 block。
        blocks: list[ParsedBlock] = []
        heading_stack: list[str] = []
        title = ""
        para: list[str] = []
        table: list[str] = []
        code: list[str] = []
        code_lang = ""
        in_code = False

        def flush_para() -> None:
            """刷出段落块；空段落不产生 block。"""
            nonlocal para
            if para:
                body = normalize_text("\n".join(para))
                if body:
                    blocks.append(
                        ParsedBlock(text=body, kind="paragraph", heading_path=tuple(heading_stack))
                    )
                para = []

        def flush_table() -> None:
            """刷出表格块；表格行原样保留，交给切片层按行处理。"""
            nonlocal table
            if table:
                body = normalize_text("\n".join(table))
                if body:
                    blocks.append(
                        ParsedBlock(text=body, kind="table", heading_path=tuple(heading_stack))
                    )
                table = []

        def flush_code() -> None:
            """刷出代码块；语言记录在 meta 中，正文不重复语言标记。"""
            nonlocal code
            if code:
                body = normalize_text("\n".join(code))
                if body:
                    blocks.append(
                        ParsedBlock(
                            text=body,
                            kind="code",
                            heading_path=tuple(heading_stack),
                            meta={"language": code_lang},
                        )
                    )
                code = []

        # 逐行扫描。所有 flush 都保持 block 的原始顺序。
        for line in text.split("\n"):
            # 优先级 1：代码围栏。进入/退出 code 时都要先结束其它缓冲。
            fence = FENCE_RE.match(line)
            if fence:
                if in_code:
                    flush_code()
                    in_code = False
                else:
                    flush_para()
                    flush_table()
                    in_code = True
                    code_lang = fence.group(2).strip()
                continue

            if in_code:
                # 优先级 2：围栏内原样保留，不解析标题/表格。
                code.append(line)
                continue

            # 优先级 3：ATX 标题（# / ## / ...），更新标题栈。
            heading = HEADING_RE.match(line)
            if heading:
                flush_para()
                flush_table()
                level = len(heading.group(1))
                title_text = heading.group(2).strip()
                # 用层级截断旧栈：H1 -> H2 -> H3 得到 [H1,H2,H3]；
                # 从 H1 跳到 H3 时保留 [H1,H3]，不伪造 H2。
                heading_stack = heading_stack[: level - 1] + [title_text]
                # 只把第一个 H1 作为文档标题；后续 H1 仍进入 heading_path。
                if level == 1 and not title:
                    title = title_text
                blocks.append(
                    ParsedBlock(
                        text=title_text,
                        kind="heading",
                        heading_path=tuple(heading_stack),
                    )
                )
                continue

            # 优先级 4：管道表格行。首尾都是 | 才识别为表格；
            # 不校验分隔行，也不解析单元格。
            if line.strip().startswith("|") and line.strip().endswith("|"):
                flush_para()
                table.append(line)
                continue

            # 优先级 5：空行结束段落/表格；代码块不受空行影响。
            if not line.strip():
                flush_para()
                flush_table()
                continue

            # 优先级 6：普通文本行；先结束表格，再进入段落缓冲。
            flush_table()
            para.append(line)

        # 文件结束：即使代码围栏未闭合，也把剩余 code 输出。
        flush_code()
        flush_para()
        flush_table()

        return ParsedDocument(
            path=path,
            file_type="md",
            title=title or path.stem,
            blocks=blocks,
        )
