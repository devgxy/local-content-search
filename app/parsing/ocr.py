"""macOS Vision OCR 封装。

Step: STEP-03
Doc: docs/steps/STEP-03-解析层.md §5.6
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.parsing.base import ParseError

DEFAULT_LANGUAGES = ("zh-Hans", "en-US")


@dataclass(frozen=True)
class OcrLine:
    """OCR 单行：文本、置信度和归一化 bbox。"""

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


def ocr_available() -> bool:
    """检查 macOS Vision OCR 是否可用。"""
    return sys.platform == "darwin" and importlib.util.find_spec("ocrmac") is not None


def _write_temp(data: bytes) -> str:
    """把图片字节写入临时文件，供 macOS Vision 读取。"""
    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        handle.write(data)
        handle.flush()
        return handle.name
    finally:
        handle.close()


def ocr_image_bytes(
    data: bytes,
    *,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
    min_confidence: float = 0.25,
) -> tuple[str, list[str]]:
    """对图片字节做 OCR。"""
    if not ocr_available():
        raise ParseError(
            "OCR_UNAVAILABLE",
            "macOS Vision OCR 不可用",
            "确认在 macOS 上运行，并安装 ocrmac",
        )
    temp_path = _write_temp(data)
    try:
        return ocr_image(temp_path, languages=languages, min_confidence=min_confidence)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def sort_ocr_lines(lines: list[OcrLine]) -> list[OcrLine]:
    """按 Vision bbox 做从上到下、从左到右排序；bbox 非法时保持原序。"""
    def key(line: OcrLine) -> tuple:
        x, y, width, height = line.bbox
        return (-(y + height), x, line.text)

    try:
        return sorted(lines, key=key)
    except (TypeError, ValueError):
        return list(lines)


def ocr_image_lines(
    path: Path | str,
    *,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
    min_confidence: float = 0.25,
) -> tuple[list[OcrLine], list[str]]:
    """返回 OCR 行及归一化 bbox。"""
    if not ocr_available():
        raise ParseError(
            "OCR_UNAVAILABLE",
            "macOS Vision OCR 不可用",
            "确认在 macOS 上运行，并安装 ocrmac",
        )
    warnings: list[str] = []
    try:
        from ocrmac import ocrmac

        results = ocrmac.OCR(
            str(path),
            language_preference=list(languages),
        ).recognize()
    except Exception as exc:  # noqa: BLE001
        raise ParseError("OCR_UNAVAILABLE", f"OCR 失败：{exc}") from exc

    lines: list[OcrLine] = []
    dropped = 0
    for item in results:
        text = str(item[0]).strip()
        confidence = float(item[1])
        if not text:
            continue
        if confidence < min_confidence:
            dropped += 1
            continue
        raw_bbox = item[2] if len(item) > 2 else (0.0, 0.0, 0.0, 0.0)
        bbox = tuple(float(value) for value in raw_bbox[:4])
        lines.append(OcrLine(text=text, confidence=confidence, bbox=bbox))
    if dropped:
        warnings.append(f"OCR 丢弃 {dropped} 行低置信度文本")
    return lines, warnings


def ocr_image_bytes_lines(
    data: bytes,
    *,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
    min_confidence: float = 0.25,
) -> tuple[list[OcrLine], list[str]]:
    """图片字节 OCR，返回带 bbox 的行。"""
    temp_path = _write_temp(data)
    try:
        return ocr_image_lines(temp_path, languages=languages, min_confidence=min_confidence)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def ocr_image(
    path: Path | str,
    *,
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
    min_confidence: float = 0.25,
) -> tuple[str, list[str]]:
    """返回 (文本, warnings)。"""
    lines, warnings = ocr_image_lines(
        path, languages=languages, min_confidence=min_confidence
    )
    return "\n".join(line.text for line in lines).strip(), warnings
