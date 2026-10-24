"""图片 OCR 解析与内嵌图片 OCR 辅助。

Step: STEP-03
Doc: docs/steps/STEP-03-解析层.md §5.5
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

from app.parsing.base import ParsedBlock, ParsedDocument, ParseError, finalize_document
from app.parsing.deskew import deskew_image
from app.parsing.ocr import ocr_image_bytes, ocr_image_bytes_lines

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")


def image_dimensions(data: bytes) -> tuple[int, int]:
    """读取图片像素尺寸。"""
    with Image.open(io.BytesIO(data)) as image:
        return image.size


def image_to_png_bytes(data: bytes) -> bytes:
    """转 PNG 字节供 OCR。"""
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def ocr_blob(
    data: bytes,
    *,
    min_confidence: float = 0.25,
    min_px: int = 200,
) -> tuple[str, list[str]]:
    """对内嵌图片做 OCR；过小或失败时返回空串 + warning，不抛异常。"""
    warnings: list[str] = []
    try:
        width, height = image_dimensions(data)
    except Exception as exc:  # noqa: BLE001
        return "", [f"图片无法读取：{exc}"]
    if min(width, height) < min_px:
        return "", [f"图片过小（{width}x{height}），跳过 OCR"]
    try:
        png = image_to_png_bytes(data)
        text, ocr_warnings = ocr_image_bytes(png, min_confidence=min_confidence)
        warnings.extend(ocr_warnings)
        return text, warnings
    except Exception as exc:  # noqa: BLE001
        return "", [f"图片 OCR 失败：{exc}"]


class ImageParser:
    """图片 OCR 解析器。"""
    extensions = SUPPORTED_IMAGE_EXTENSIONS

    def parse(self, path: Path, settings) -> ParsedDocument:  # noqa: ANN001
        """OCR 图片并返回文本块。"""
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ParseError("BROKEN_FILE", f"读取图片失败：{exc}") from exc

        warnings: list[str] = []
        try:
            width, height = image_dimensions(data)
        except Exception as exc:  # noqa: BLE001
            raise ParseError("BROKEN_FILE", f"图片无法解码：{exc}") from exc

        text = ""
        ocr_meta: list[dict] = []
        lines = []
        image = None
        if settings.ocr_enabled:
            if min(width, height) < settings.ocr_min_image_px:
                warnings.append(f"图片较小（{width}x{height}），OCR 可能不准")
            try:
                image = Image.open(io.BytesIO(data)).convert("RGB")
                if settings.ocr_deskew:
                    image, _ = deskew_image(image, max_angle=settings.ocr_max_skew)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                lines, ocr_warnings = ocr_image_bytes_lines(
                    buffer.getvalue(),
                    min_confidence=settings.ocr_min_confidence,
                )
                text = "\n".join(line.text for line in lines)
                ocr_meta = [
                    {"text": line.text, "bbox": list(line.bbox)}
                    for line in lines
                ]
                warnings.extend(ocr_warnings)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"图片 OCR 失败：{exc}")
                text = ""
        else:
            warnings.append("OCR 已关闭")

        blocks = []
        table_blocks = []
        if settings.table_structure_enabled and image is not None:
            from app.table_structure import TableStructureService, TableStructureUnavailable

            try:
                table_service = TableStructureService(settings)
                table_service.ensure_ready()
                try:
                    table_boxes = table_service.detect(image)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"图片表格检测失败：{exc}")
                    table_boxes = []
                kept_lines = list(lines)
                for table_bbox, _score in table_boxes:
                    try:
                        result = table_service.recognize(image, table_bbox)
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"图片表格结构失败：{exc}")
                        continue
                    warnings.extend(f"图片表格：{w}" for w in result.warnings)
                    if result.confidence < settings.table_confidence:
                        warnings.append(f"图片表格置信度低：{result.confidence:.2f}")
                    rows_text: dict[int, list] = {}
                    for cell in result.cells:
                        rows_text.setdefault(cell.row, []).append(cell)
                    table_text = "\n".join(
                        " | ".join(cell.text for cell in sorted(rows_text[row], key=lambda c: c.col))
                        for row in sorted(rows_text)
                    )
                    has_cell_text = any(cell.text.strip() for cell in result.cells)
                    if not has_cell_text:
                        warnings.append("图片表格 OCR 为空，保留原始 OCR 行")
                        continue
                    table_blocks.append(
                        ParsedBlock(
                            text=table_text,
                            kind="table",
                            meta={
                                "html": result.html,
                                "confidence": result.confidence,
                                "ocr_table": True,
                            },
                        )
                    )
                    x0 = int(round(table_bbox[0]))
                    y0 = int(round(table_bbox[1]))
                    # 只删除“已经被完整识别单元格覆盖”的 OCR 行；
                    # 空单元格/未识别单元格保留原始文本，避免部分丢内容。
                    covered = [
                        (
                            x0 + cell.bbox[0], y0 + cell.bbox[1],
                            x0 + cell.bbox[2], y0 + cell.bbox[3],
                        )
                        for cell in result.cells
                        if cell.text.strip() and cell.complete
                    ]
                    filtered = []
                    for line in kept_lines:
                        lx, ly, lw, lh = line.bbox
                        # 行必须完整落在一个成功 cell 内，才允许删除；
                        # 跨 cell 边界或部分覆盖时保留原文。
                        x_start, x_end = lx, lx + lw
                        y_start, y_end = 1 - (ly + lh), 1 - ly
                        replaced = any(
                            bx0 / width <= x_start and x_end <= bx1 / width
                            and by0 / height <= y_start and y_end <= by1 / height
                            for bx0, by0, bx1, by1 in covered
                        )
                        if not replaced:
                            filtered.append(line)
                    kept_lines = filtered
                lines = kept_lines
                text = "\n".join(line.text for line in lines)
            except TableStructureUnavailable as exc:
                raise ParseError("TABLE_MODEL_UNAVAILABLE", str(exc)) from exc

        if text.strip():
            blocks.append(ParsedBlock(
                text=text,
                kind="ocr",
                meta={"width": width, "height": height, "ocr_lines": ocr_meta[:200]},
            ))
        blocks.extend(table_blocks)
        if not text.strip() and not table_blocks:
            warnings.append("图片没有识别出文本")

        return finalize_document(
            ParsedDocument(
                path=path,
                file_type="image",
                title=path.stem,
                blocks=blocks,
                warnings=warnings,
                meta={"width": width, "height": height},
            )
        )
