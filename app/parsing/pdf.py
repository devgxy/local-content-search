"""PDF 解析：文本层优先，扫描页走 OCR，并清理重复页眉/页脚。

主流程：
1. 先收集每页文本行及 bbox；
2. 统计顶部/底部 8% 区域中跨页重复的行；
3. 删除重复行和页码，再重建页面文本；
4. 短文本页走 OCR；OCR 失败保留文本层。

Step: STEP-03
Doc: docs/steps/STEP-03-解析层.md §5.2
"""

# =============================================================================
# 代码导读：PdfParser
# =============================================================================
#
# 职责：把 PDF 转成 ParsedDocument；不做切片/向量化/写库。
#
# 输入：PDF 路径 + Settings
# 输出：
#   ParsedDocument(
#       file_type="pdf",
#       title=metadata.title 或文件名,
#       blocks=[每页一个 ParsedBlock],
#       warnings=[...],
#       meta={"pages": N, "header_footer_removed": M},
#   )
#
# 主流程：
#
#   parse()
#     -> pymupdf.open
#     -> 检查加密（只试空密码）
#     -> 页数上限 MAX_PDF_PAGES
#     -> 可选：逐页收集文本行 + bbox
#          -> 统计顶部/底部 8% 的重复页眉/页脚
#          -> 删除重复行和纯页码
#     -> 逐页：
#          raw_text = page.get_text("text")
#          text = 去页眉页脚后的文本
#          raw_text 太短且 OCR 开启 -> 200dpi OCR
#          OCR 成功 -> kind="ocr"；失败 -> 保留文本层
#          text 为空 -> warning，跳过该页
#          生成 ParsedBlock(page=n)
#     -> finalize_document：丢空 block/算偏移/生成 doc.text
#
# 示例 PDF（3 页）：
#
#   每页顶部：公司内部资料
#   每页正文：
#     第一页：系统支持 PDF 文本层解析，错误码 E1024 表示连接超时。
#     第二页：扫描 PDF 会走 OCR，表格示例：模块 | 说明；解析 | PDF。
#     第三页：重复页眉页脚清理后，再进行切片和向量化。
#   每页底部：第 1 页 / 第 2 页 / 第 3 页
#
#   解析结果：
#     meta = {"pages": 3, "header_footer_removed": 6}
#     warnings = ["已移除重复页眉/页脚/页码 6 行"]
#     blocks = 3 个 paragraph block，page=1/2/3
#     block 1.text = "第一页：系统支持 PDF 文本层解析，错误码 E1024 表示连接超时。"
#     block 2.text = "第二页：扫描 PDF 会走 OCR，表格示例：模块 | 说明；解析 | PDF。"
#     block 3.text = "第三页：重复页眉页脚清理后，再进行切片和向量化。"
#
#   注意：PDF 的 block 粒度是“页”，不是“段落”。
#   kind="paragraph" 只表示来自文本层；kind="ocr" 表示来自 OCR。
#
# 页眉页脚算法：
#   1. get_text("dict") 拿每行 spans + bbox；
#   2. 顶部/底部各占页高 8%；
#   3. 行文本做数字归一化（Page 1/Page 2 -> page #）；
#   4. 同一 (zone,key) 出现在 >=30% 页面且至少 2 页 -> 删除；
#   5. 页边纯页码（12/第3页/Page 3 of 10）直接删除；
#   6. 只清理文本层；扫描页 OCR 文本不清理。
#
# OCR 判定：
#   - 用 raw_text（清理前）的长度判断，避免“页眉 + 短正文”被 OCR
#     重新把页眉带回来；
#   - OCR 成功会替换整页文本；
#   - OCR 失败保留文本层，不丢已有内容。
#
# 下游切片：
#   - PDF 没有 heading_path，所以 chunk 没有标题前缀；
#   - section key = (heading_path, page, slide)，PDF 每页独立 section；
#   - 一页可能被切成多个 chunk；
#   - 不会跨页合并，也不会跨页扩展上下文。
#
# 已知边界：
#   - 多栏/旋转页面的行顺序可能不等于阅读顺序；
#   - 每页重复的表头可能被误判为页眉；
#   - 扫描页的页眉页脚由 OCR 产生，本轮不清理；
#   - 表格不结构化，PDF 文本页里的内嵌图片不逐张 OCR；
#   - 加密 PDF 只尝试空密码；
#   - 超过 MAX_PDF_PAGES 只解析前 N 页。
# =============================================================================

from __future__ import annotations

import io
import logging
import math
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from app.parsing.base import ParsedBlock, ParsedDocument, ParseError, finalize_document
from app.parsing.deskew import deskew_image
from app.parsing.ocr import ocr_image_bytes_lines
from app.parsing.ocr_layout import (
    group_rows,
    order_lines_columns,
    strip_repeated_margins,
    table_candidates,
    table_text,
)
from app.table_structure import TableStructureService, TableStructureUnavailable

logger = logging.getLogger(__name__)

PDF_MARGIN_RATIO = 0.08
_PAGE_NUMBER_RE = re.compile(
    r"^(?:第\s*)?\d{1,4}(?:\s*页)?(?:\s*/\s*\d{1,4})?$|"
    r"^\d{1,4}\s*/\s*\d{1,4}$|"
    r"^page\s*\d{1,4}(?:\s*of\s*\d{1,4})?$",
    re.IGNORECASE,
)


def _line_text(line: dict) -> str:
    """把 PyMuPDF line 的 spans 拼成一行文本。"""
    # 一行可能由多个 span 组成（不同字体/字号）；先拼接，再压缩空白。
    return " ".join("".join(span.get("text", "") for span in line.get("spans", [])).split())


def _line_key(text: str) -> str:
    """页眉页脚匹配键：数字归一化，避免“第 1 页/第 2 页”被当成不同行。"""
    # 例：Page 1 / Page 2 -> page #；第 1 页 / 第 2 页 -> 第 # 页。
    return re.sub(r"\d+", "#", text.casefold()).strip()


def _is_page_number(text: str) -> bool:
    """判断是否是纯页码/Page N。"""
    # 只匹配整行；正文里的数字不会被误删。
    return bool(_PAGE_NUMBER_RE.fullmatch(text.strip()))


def _iter_page_lines(page, ratio: float = PDF_MARGIN_RATIO):
    """遍历页面文本行，标记顶部/底部页边区域。

    Args:
        page: PyMuPDF page。
        ratio: 顶部/底部区域占页面高度比例。

    Yields:
        ``{text, key, zone, y0, y1}``；zone 为 top/bottom/None。
    """
    # dict 模式才有 bbox；text 模式只有纯文本，无法判断页边。
    data = page.get_text("dict")
    height = float(page.rect.height) or 1.0
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            text = _line_text(line)
            if not text:
                continue
            bbox = line.get("bbox") or (0, 0, 0, 0)
            y0, y1 = float(bbox[1]), float(bbox[3])
            # y 轴向下：y1 很小是顶部，y0 很大是底部。
            if y1 <= height * ratio:
                zone = "top"
            elif y0 >= height * (1 - ratio):
                zone = "bottom"
            else:
                zone = None
            yield {
                "text": text,
                "key": _line_key(text),
                "zone": zone,
                "bbox": tuple(float(v) for v in bbox),
            }


def _repeated_margin_keys(lines_by_page: dict[int, list[dict]], page_count: int) -> set[tuple[str, str]]:
    """统计跨页重复的页眉/页脚键。

    Args:
        lines_by_page: 页码 -> 行列表。
        page_count: 参与统计的页数。

    Returns:
        ``{(zone, key)}``；同一行至少出现在 30% 页面（至少 2 页）。
    """
    if page_count < 2:
        return set()
    # 按 (zone, key) 统计；同一页重复两次也会计 2 次，阈值是保守启发式。
    counts: Counter[tuple[str, str]] = Counter()
    for lines in lines_by_page.values():
        for line in lines:
            if line["zone"]:
                counts[(line["zone"], line["key"])] += 1
    threshold = max(2, math.ceil(page_count * 0.3))
    return {key for key, count in counts.items() if count >= threshold}


def _strip_margin_lines(
    lines: list[dict],
    repeated: set[tuple[str, str]],
) -> tuple[str, int]:
    """删除重复页眉/页脚/页码并重建页面文本。

    Args:
        lines: ``_iter_page_lines`` 结果。
        repeated: ``_repeated_margin_keys`` 结果。

    Returns:
        ``(清理后的文本, 删除行数)``。
    """
    kept: list[str] = []
    removed = 0
    for line in lines:
        zone = line["zone"]
        # 双条件：页边区域 + (跨页重复 或 纯页码)。
        should_remove = bool(
            zone
            and (
                (zone, line["key"]) in repeated
                or _is_page_number(line["text"])
            )
        )
        if should_remove:
            removed += 1
            continue
        kept.append(line["text"])
    return "\n".join(kept).strip(), removed


def _inside_table(bbox: tuple[float, float, float, float], boxes: list[tuple]) -> bool:
    """判断文本行中心点是否落在表格 bbox 内。"""
    if not boxes:
        return False
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    return any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for box in boxes)


def _page_tables(page, warnings: list[str]) -> tuple[list[ParsedBlock], list[tuple]]:
    """提取带线表格；返回 table blocks 和 bbox 列表。"""
    blocks: list[ParsedBlock] = []
    boxes: list[tuple] = []
    try:
        found = page.find_tables()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"表格检测失败：{exc}")
        return blocks, boxes
    for table in getattr(found, "tables", []) or []:
        try:
            rows = table.extract()
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"表格提取失败：{exc}")
            continue
        lines: list[str] = []
        for row in rows or []:
            cells = [str(cell or "").replace("\n", " ").strip() for cell in row]
            if any(cells):
                lines.append(" | ".join(cells))
        if len(lines) >= 2 and any("|" in line for line in lines):
            blocks.append(ParsedBlock(text="\n".join(lines), kind="table", page=0))
            boxes.append(tuple(table.bbox))
    return blocks, boxes


def _collect_ocr_pages(doc, total_pages: int, settings, warnings: list[str]) -> dict:
    """两阶段 OCR：先收集所有短文本页的行，再跨页清理页边。"""
    pages: dict[int, list] = {}
    for page_no in range(total_pages):
        page = doc[page_no]
        raw = page.get_text("text").strip()
        if len(raw) >= settings.pdf_ocr_text_threshold or not settings.ocr_enabled:
            continue
        try:
            pix = page.get_pixmap(dpi=200)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            if settings.ocr_deskew:
                image, skew = deskew_image(image, max_angle=settings.ocr_max_skew)
                if abs(skew) >= 0.5:
                    warnings.append(f"第 {page_no + 1} 页 deskew {skew:+.1f}°")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            lines, ocr_warnings = ocr_image_bytes_lines(
                buffer.getvalue(), min_confidence=settings.ocr_min_confidence
            )
            warnings.extend(f"第 {page_no + 1} 页：{w}" for w in ocr_warnings)
            if lines:
                pages[page_no] = lines
        except ParseError as exc:
            warnings.append(f"第 {page_no + 1} 页 OCR 失败：{exc.message}")
    cleaned, removed = strip_repeated_margins(pages)
    if removed:
        warnings.append(f"已移除 OCR 页眉/页脚/页码 {removed} 行")
    return cleaned


class PdfParser:
    """PDF 解析器：文本层 + 扫描页 OCR + 带线表格。"""
    extensions = (".pdf",)

    def parse(self, path: Path, settings) -> ParsedDocument:  # noqa: ANN001
        """解析 PDF 文件。"""
        # 延迟导入：只解析 PDF 时才加载 PyMuPDF。
        try:
            import pymupdf
        except ImportError as exc:
            raise ParseError("BROKEN_FILE", "未安装 pymupdf", "pip install pymupdf") from exc

        try:
            doc = pymupdf.open(str(path))
        except Exception as exc:  # noqa: BLE001
            raise ParseError("BROKEN_FILE", f"无法打开 PDF：{exc}") from exc

        warnings: list[str] = []
        blocks: list[ParsedBlock] = []
        metadata: dict = {}
        try:
            metadata = doc.metadata or {}
            # 只尝试空密码；有密码的 PDF 交给用户先解密。
            if doc.is_encrypted and not doc.authenticate(""):
                raise ParseError(
                    "ENCRYPTED_PDF",
                    "PDF 已加密且空密码无法打开",
                    "请先解除密码保护",
                )

            # 页数上限保护：大 PDF 可能 OCR 很久，先截断并明确 warning。
            total_pages = doc.page_count
            if total_pages > settings.max_pdf_pages:
                warnings.append(
                    f"PDF 共 {total_pages} 页，超过上限 {settings.max_pdf_pages}，仅解析前 "
                    f"{settings.max_pdf_pages} 页"
                )
                total_pages = settings.max_pdf_pages

            # 先收集所有页的行再统计重复；不能边删边统计，否则样本不完整。
            lines_by_page: dict[int, list[dict]] = {}
            repeated: set[tuple[str, str]] = set()
            if settings.pdf_strip_header_footer or settings.pdf_table_extract:
                try:
                    for page_no in range(total_pages):
                        lines_by_page[page_no] = list(_iter_page_lines(doc[page_no]))
                    if settings.pdf_strip_header_footer:
                        repeated = _repeated_margin_keys(lines_by_page, total_pages)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"行/页眉页脚检测失败，保留原文本：{exc}")
                    lines_by_page = {}
                    repeated = set()
            header_footer_removed = 0
            ocr_pages = _collect_ocr_pages(doc, total_pages, settings, warnings)
            table_service = None
            if settings.table_structure_enabled:
                try:
                    table_service = TableStructureService(settings)
                    table_service.ensure_ready()
                except TableStructureUnavailable as exc:
                    raise ParseError("TABLE_MODEL_UNAVAILABLE", str(exc)) from exc

            for page_no in range(total_pages):
                page = doc[page_no]
                # raw_text 是 OCR 判定依据；text 是清理后的正文。
                raw_text = page.get_text("text").strip()
                table_blocks: list[ParsedBlock] = []
                table_boxes: list[tuple] = []
                ocr_lines = ocr_pages.get(page_no, [])
                ocr_meta: list[dict] = []
                if ocr_lines:
                    ocr_meta = [
                        {"text": line.text, "bbox": list(line.bbox)}
                        for line in ocr_lines
                    ]
                    body_lines = list(ocr_lines)
                    model_tables: list[tuple[tuple, float]] = []
                    if table_service is not None:
                        try:
                            pix = page.get_pixmap(dpi=200)
                            page_image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                            if settings.ocr_deskew:
                                page_image, _ = deskew_image(page_image, max_angle=settings.ocr_max_skew)
                            model_tables = table_service.detect(page_image)
                            for table_bbox, det_score in model_tables:
                                try:
                                    result = table_service.recognize(page_image, table_bbox)
                                except TableStructureUnavailable as exc:
                                    warnings.append(f"第 {page_no + 1} 页表格结构失败：{exc}")
                                    continue
                                warnings.extend(
                                    f"第 {page_no + 1} 页表格：{w}" for w in result.warnings
                                )
                                if result.confidence < settings.table_confidence:
                                    warnings.append(
                                        f"第 {page_no + 1} 页表格置信度低：{result.confidence:.2f}"
                                    )
                                cells_by_row: dict[int, list] = {}
                                for cell in result.cells:
                                    cells_by_row.setdefault(cell.row, []).append(cell)
                                table_text_value = "\n".join(
                                    " | ".join(cell.text for cell in sorted(cells_by_row[row], key=lambda c: c.col))
                                    for row in sorted(cells_by_row)
                                )
                                has_cell_text = any(cell.text.strip() for cell in result.cells)
                                if not has_cell_text:
                                    warnings.append(
                                        f"第 {page_no + 1} 页表格 OCR 为空，保留原始 OCR 行"
                                    )
                                    continue
                                if result.warnings:
                                    warnings.append(
                                        f"第 {page_no + 1} 页表格部分 OCR 失败，只保留未覆盖的原始行"
                                    )
                                table_blocks.append(
                                    ParsedBlock(
                                        text=table_text_value,
                                        kind="table",
                                        page=page_no + 1,
                                        meta={
                                            "page": page_no + 1,
                                            "ocr_table": True,
                                            "html": result.html,
                                            "confidence": result.confidence,
                                            "detection_score": det_score,
                                        },
                                    )
                                )
                                # 只删除成功识别 cell 覆盖的 OCR 行，保留未覆盖原文。
                                width, height = page_image.size
                                x0 = int(round(table_bbox[0]))
                                y0 = int(round(table_bbox[1]))
                                covered = [
                                    (
                                        x0 + cell.bbox[0], y0 + cell.bbox[1],
                                        x0 + cell.bbox[2], y0 + cell.bbox[3],
                                    )
                                    for cell in result.cells
                                    if cell.text.strip() and cell.complete
                                ]
                                kept = []
                                for line in body_lines:
                                    lx, ly, lw, lh = line.bbox
                                    x_start, x_end = lx, lx + lw
                                    y_start, y_end = 1 - (ly + lh), 1 - ly
                                    replaced = any(
                                        bx0 / width <= x_start and x_end <= bx1 / width
                                        and by0 / height <= y_start and y_end <= by1 / height
                                        for bx0, by0, bx1, by1 in covered
                                    )
                                    if not replaced:
                                        kept.append(line)
                                body_lines = kept
                            if model_tables:
                                rows = group_rows(body_lines)
                                tables = []
                            else:
                                rows = group_rows(ocr_lines)
                                tables = table_candidates(rows)
                        except Exception as exc:  # noqa: BLE001
                            warnings.append(f"第 {page_no + 1} 页表格检测失败：{exc}")
                            rows = group_rows(ocr_lines)
                            tables = table_candidates(rows)
                    else:
                        rows = group_rows(ocr_lines)
                        tables = table_candidates(rows)
                    table_ids = {id(line) for table in tables for row in table for line in row}
                    body_lines = [line for line in body_lines if id(line) not in table_ids]
                    body_lines = order_lines_columns(body_lines)
                    text = "\n".join(line.text for line in body_lines).strip()
                    kind = "ocr"
                    for table in tables:
                        table_blocks.append(
                            ParsedBlock(
                                text=table_text(table),
                                kind="table",
                                page=page_no + 1,
                                meta={"page": page_no + 1, "ocr_table": True},
                            )
                        )
                else:
                    kind = "paragraph"
                    if settings.pdf_table_extract and lines_by_page:
                        table_blocks, table_boxes = _page_tables(page, warnings)

                if ocr_lines:
                    # OCR 已经完成阅读顺序；不再走文本层清洗。
                    pass
                elif lines_by_page and (settings.pdf_strip_header_footer or table_boxes):
                    page_lines = lines_by_page.get(page_no, [])
                    # 文本层表格不做整框删除：保留原文，避免单元格提取为空时丢字；
                    # table block 与原文的重复由下游控制。
                    if settings.pdf_strip_header_footer:
                        text, removed = _strip_margin_lines(page_lines, repeated)
                        header_footer_removed += removed
                    else:
                        text = "\n".join(line["text"] for line in page_lines).strip()
                else:
                    text = raw_text
                if not text and not table_blocks:
                    warnings.append(f"第 {page_no + 1} 页没有可用文本")
                    continue
                # 一页一个正文 block；表格作为独立 block 追加，并设置 page。
                if text:
                    block_meta = {"page": page_no + 1}
                    if kind == "ocr":
                        block_meta["ocr_lines"] = ocr_meta[:200]
                    blocks.append(
                        ParsedBlock(
                            text=text,
                            kind=kind,
                            page=page_no + 1,
                            meta=block_meta,
                        )
                    )
                for table in table_blocks:
                    table.page = page_no + 1
                    table.meta["page"] = page_no + 1
                    blocks.append(table)
        finally:
            doc.close()

        # PDF 元数据标题优先；为空用文件名。
        title = str(metadata.get("title") or "").strip()
        if header_footer_removed:
            warnings.append(f"已移除重复页眉/页脚/页码 {header_footer_removed} 行")

        return finalize_document(
            ParsedDocument(
                path=path,
                file_type="pdf",
                title=title or path.stem,
                blocks=blocks,
                warnings=warnings,
                meta={
                    "pages": total_pages,
                    "header_footer_removed": header_footer_removed,
                },
            )
        )
