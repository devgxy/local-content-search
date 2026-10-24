"""离线 Table Transformer：表格检测、结构识别、单元格 OCR。

模型权重放在 .models/table-transformer-{detection,structure}；
运行时只读本地目录，不访问网络。
"""

from __future__ import annotations

import html
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class TableStructureUnavailable(RuntimeError):
    """表格模型/权重/依赖不可用。"""


@dataclass
class TableCell:
    row: int
    col: int
    text: str
    bbox: tuple[float, float, float, float]
    rowspan: int = 1
    colspan: int = 1
    confidence: float = 0.0
    complete: bool = True


@dataclass
class TableResult:
    bbox: tuple[float, float, float, float]
    cells: list[TableCell] = field(default_factory=list)
    html: str = ""
    confidence: float = 0.0
    rows: list[tuple[tuple[float, float, float, float], float]] = field(default_factory=list)
    columns: list[tuple[tuple[float, float, float, float], float]] = field(default_factory=list)
    spans: list[tuple[tuple[float, float, float, float], float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def models_ready(settings) -> bool:  # noqa: ANN001
    return settings.table_detection_model_path.is_dir() and settings.table_structure_model_path.is_dir()


def _iou(a: tuple, b: tuple) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _load_processor(path: Path):  # noqa: ANN001
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(str(path), local_files_only=True)


def _load_structure_config(path: Path):  # noqa: ANN001
    from transformers import TableTransformerConfig

    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    # 旧权重 config 的 dilation=None 在当前 transformers 校验中非法。
    raw["dilation"] = False
    return TableTransformerConfig.from_dict(raw)


class TableStructureService:
    """本地表格结构服务；模型懒加载。"""

    def __init__(self, settings) -> None:  # noqa: ANN001
        self.settings = settings
        self._det_processor = None
        self._det_model = None
        self._str_processor = None
        self._str_model = None
        self._device = "cpu"

    def ensure_ready(self) -> None:
        """显式自检；模型缺失/加载失败抛 TableStructureUnavailable。"""
        if not models_ready(self.settings):
            raise TableStructureUnavailable(
                "表格模型目录不完整；需要 table-transformer-detection 和 table-transformer-structure"
            )
        self._load()

    def _load(self) -> None:
        if self._det_model is not None and self._str_model is not None:
            return
        try:
            import torch
            from transformers import TableTransformerForObjectDetection
        except Exception as exc:  # noqa: BLE001
            raise TableStructureUnavailable(f"加载 transformers/torch 失败：{exc}") from exc
        device = self.settings.table_structure_device
        if device == "mps" and not torch.backends.mps.is_available():
            raise TableStructureUnavailable("TABLE_STRUCTURE_DEVICE=mps 但 MPS 不可用")
        self._device = device
        try:
            det_path = self.settings.table_detection_model_path
            str_path = self.settings.table_structure_model_path
            self._det_processor = _load_processor(det_path)
            self._det_model = TableTransformerForObjectDetection.from_pretrained(
                str(det_path), local_files_only=True
            ).to(device).eval()
            # Table Transformer 结构权重使用与检测相同的 DETR 预处理；
            # 旧 structure processor 的 size 只有 longest_edge，会产生错误网格。
            self._str_processor = _load_processor(det_path)
            self._str_model = TableTransformerForObjectDetection.from_pretrained(
                str_path, config=_load_structure_config(str_path), local_files_only=True
            ).to(device).eval()
        except Exception as exc:  # noqa: BLE001
            raise TableStructureUnavailable(f"加载表格模型失败：{exc}") from exc

    @staticmethod
    def _tensor_to_boxes(processor, outputs, image: Image.Image, threshold: float):  # noqa: ANN001
        result = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[image.size[::-1]]
        )[0]
        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        labels = result["labels"].detach().cpu().numpy()
        return boxes, scores, labels

    def _detect_one(self, image: Image.Image, threshold: float) -> list[tuple[tuple, float]]:
        import torch

        inputs = self._det_processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._det_model(**inputs)
        boxes, scores, labels = self._tensor_to_boxes(
            self._det_processor, outputs, image, threshold
        )
        found = []
        for box, score, label in zip(boxes, scores, labels, strict=True):
            name = self._det_model.config.id2label.get(int(label), "")
            if name == "table":
                found.append((tuple(float(v) for v in box), float(score)))
        return found

    @staticmethod
    def _nms(items: list[tuple[tuple, float]], threshold: float = 0.5) -> list[tuple[tuple, float]]:
        kept: list[tuple[tuple, float]] = []
        for box, score in sorted(items, key=lambda item: -item[1]):
            if all(_iou(box, other) < threshold for other, _ in kept):
                kept.append((box, score))
        return kept

    def detect(self, image: Image.Image, *, threshold: float = 0.5) -> list[tuple[tuple, float]]:
        """检测页面中的表格 bbox；大图使用分块检测 + NMS。"""
        self._load()
        found = self._detect_one(image, threshold)
        if self.settings.table_tile_detection and min(image.width, image.height) >= 1200:
            step_x = image.width // 2
            step_y = image.height // 2
            pad_x = image.width // 10
            pad_y = image.height // 10
            for row in range(2):
                for col in range(2):
                    x0 = max(0, col * step_x - pad_x)
                    y0 = max(0, row * step_y - pad_y)
                    x1 = min(image.width, x0 + step_x + 2 * pad_x)
                    y1 = min(image.height, y0 + step_y + 2 * pad_y)
                    tile = image.crop((x0, y0, x1, y1))
                    for box, score in self._detect_one(tile, threshold):
                        mapped = (box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0)
                        # 与全图框重叠的分块结果通常是同一表格内部片段；
                        # 只补充全图未覆盖的区域。
                        if any(_iou(mapped, base) > 0.1 for base, _ in found):
                            continue
                        found.append((mapped, score))
        return self._nms(found)

    def recognize(
        self,
        image: Image.Image,
        table_bbox: tuple,
        *,
        threshold: float = 0.5,
        cell_ocr: bool | None = None,
    ) -> TableResult:
        """识别单个表格的行/列/单元格，并逐格 OCR。"""
        self._load()
        import torch

        left, top, right, bottom = [int(round(v)) for v in table_bbox]
        if right <= left or bottom <= top:
            raise TableStructureUnavailable("表格 bbox 为空")
        crop = image.crop((left, top, right, bottom)).convert("RGB")
        inputs = self._str_processor(images=crop, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._str_model(**inputs)
        boxes, scores, labels = self._tensor_to_boxes(
            self._str_processor, outputs, crop, threshold
        )
        rows: list[tuple[tuple, float]] = []
        cols: list[tuple[tuple, float]] = []
        header_rows: list[tuple[tuple, float]] = []
        header_cols: list[tuple[tuple, float]] = []
        spans: list[tuple[tuple, float]] = []
        for box, score, label in zip(boxes, scores, labels, strict=True):
            name = self._str_model.config.id2label.get(int(label), "")
            item = (tuple(float(v) for v in box), float(score))
            if name == "table row":
                rows.append(item)
            elif name == "table column":
                cols.append(item)
            elif name == "table projected row header":
                header_rows.append(item)
            elif name == "table column header":
                header_cols.append(item)
            elif name == "table spanning cell":
                spans.append(item)
        # 表头也是真实网格；合并后用 NMS 去掉同位置重复框。
        rows.extend(header_rows)
        cols.extend(header_cols)
        rows = self._nms(rows, 0.99)
        cols = self._nms(cols, 0.99)
        if not rows or not cols:
            raise TableStructureUnavailable("表格结构模型没有识别出行/列")
        rows.sort(key=lambda item: item[0][1])
        cols.sort(key=lambda item: item[0][0])
        warnings: list[str] = []
        cells: list[TableCell] = []
        do_cell_ocr = self.settings.table_cell_ocr if cell_ocr is None else cell_ocr
        for r_index, (r_box, _r_score) in enumerate(rows):
            for c_index, (c_box, _c_score) in enumerate(cols):
                x0 = max(r_box[0], c_box[0])
                y0 = max(r_box[1], c_box[1])
                x1 = min(r_box[2], c_box[2])
                y1 = min(r_box[3], c_box[3])
                if x1 <= x0 or y1 <= y0:
                    continue
                cell_img = crop.crop((int(x0), int(y0), int(x1), int(y1)))
                if cell_img.width < 2 or cell_img.height < 2:
                    warnings.append("单元格 bbox 过小，跳过 OCR")
                    continue
                text = ""
                cell_warnings: list[str] = []
                if do_cell_ocr:
                    buffer = io.BytesIO()
                    cell_img.save(buffer, format="PNG")
                    try:
                        from app.parsing.base import ParseError
                        from app.parsing.ocr import ocr_image_bytes_lines

                        lines, cell_warnings = ocr_image_bytes_lines(
                            buffer.getvalue(),
                            min_confidence=self.settings.ocr_min_confidence,
                        )
                        cell_warnings = list(cell_warnings)
                        warnings.extend(cell_warnings)
                        text = "\n".join(line.text for line in lines).strip()
                    except ParseError as exc:
                        warnings.append(f"单元格 OCR 失败：{exc.message}")
                cells.append(
                    TableCell(
                        row=r_index,
                        col=c_index,
                        text=text,
                        bbox=(x0, y0, x1, y1),
                        confidence=min(float(scores.max()), 1.0),
                        complete=not cell_warnings,
                    )
                )

        # spanning cell：按与行/列的交集覆盖对应单元格；保守处理，不强行恢复未知区域。
        for s_box, s_score in spans:
            hit_rows = [
                i for i, (r_box, _) in enumerate(rows)
                if min(r_box[2], s_box[2]) > max(r_box[0], s_box[0])
                and min(r_box[3], s_box[3]) > max(r_box[1], s_box[1])
            ]
            hit_cols = [
                i for i, (c_box, _) in enumerate(cols)
                if min(c_box[2], s_box[2]) > max(c_box[0], s_box[0])
                and min(c_box[3], s_box[3]) > max(c_box[1], s_box[1])
            ]
            for cell in cells:
                if cell.row == (hit_rows[0] if hit_rows else -1) and cell.col == (hit_cols[0] if hit_cols else -1):
                    cell.rowspan = max(1, len(hit_rows))
                    cell.colspan = max(1, len(hit_cols))
                    cell.confidence = max(cell.confidence, float(s_score))

        table_html = _to_html(cells, len(rows), len(cols))
        confidence = float(np.mean([score for _, score in rows + cols])) if rows or cols else 0.0
        return TableResult(
            bbox=tuple(float(v) for v in table_bbox),
            cells=cells,
            html=table_html,
            confidence=confidence,
            rows=rows,
            columns=cols,
            spans=spans,
            warnings=warnings,
        )


def _to_html(cells: list[TableCell], rows: int, cols: int) -> str:
    """把单元格转成带 rowspan/colspan 的 HTML；覆盖格跳过。"""
    by_pos = {(cell.row, cell.col): cell for cell in cells}
    covered: set[tuple[int, int]] = set()
    parts = ["<table>"]
    for r in range(rows):
        parts.append("<tr>")
        for c in range(cols):
            if (r, c) in covered:
                continue
            cell = by_pos.get((r, c))
            if cell is None:
                parts.append("<td></td>")
                continue
            span = f' rowspan="{cell.rowspan}"' if cell.rowspan > 1 else ""
            span += f' colspan="{cell.colspan}"' if cell.colspan > 1 else ""
            extra_text = []
            for rr in range(r, min(rows, r + cell.rowspan)):
                for cc in range(c, min(cols, c + cell.colspan)):
                    if (rr, cc) == (r, c):
                        continue
                    covered_cell = by_pos.get((rr, cc))
                    if covered_cell and covered_cell.text.strip():
                        # 按位置汇总，保留不同覆盖格的相同文本。
                        extra_text.append(covered_cell.text.strip())
                    covered.add((rr, cc))
            cell_text = "\n".join([cell.text.strip(), *extra_text]).strip()
            parts.append(f"<td{span}>{html.escape(cell_text)}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)
