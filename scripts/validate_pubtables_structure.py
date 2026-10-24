"""在 PubTables-1M Structure Test 上评估行/列/合并单元格检测。"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.table_structure import TableStructureService

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "eval" / "pubtables"
LABELS = {
    "table row": "rows",
    "table column": "columns",
    "table column header": "columns",
    "table projected row header": "rows",
    "table spanning cell": "spans",
}


def _iou(a, b) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def _grid_cells(rows: list[tuple], columns: list[tuple]) -> list[tuple]:
    """用 row/column 交点构造 cell 代理框（不含 span 合并）。"""
    cells = []
    for r_box, _ in rows:
        for c_box, _ in columns:
            x0 = max(r_box[0], c_box[0])
            y0 = max(r_box[1], c_box[1])
            x1 = min(r_box[2], c_box[2])
            y1 = min(r_box[3], c_box[3])
            if x1 > x0 and y1 > y0:
                cells.append((x0, y0, x1, y1))
    return cells


def _match(pred: list[tuple], gt: list[tuple], threshold: float = 0.5) -> tuple[int, int, int]:
    used: set[int] = set()
    tp = 0
    for box, _score in pred:
        best_i = -1
        best = 0.0
        for index, target in enumerate(gt):
            if index in used:
                continue
            value = _iou(box, target)
            if value > best:
                best = value
                best_i = index
        if best >= threshold and best_i >= 0:
            used.add(best_i)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def _read_xml(path: Path) -> tuple[tuple, dict[str, list[tuple]]]:
    root = ET.parse(path).getroot()
    table = None
    groups: dict[str, list[tuple]] = defaultdict(list)
    for obj in root.findall(".//object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        box = tuple(float(bnd.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
        if name == "table":
            table = box
        elif name in LABELS:
            groups[LABELS[name]].append(box)
    return table or (0, 0, 0, 0), groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default="reports/pubtables-structure.md")
    args = parser.parse_args()
    xmls = sorted((DATA / "ann").glob("*.xml"))[: args.limit]
    service = TableStructureService(Settings(table_structure_device="cpu"))
    service.ensure_ready()
    stats = {name: {"tp": 0, "fp": 0, "fn": 0, "images": 0} for name in ("rows", "columns", "spans", "cells")}
    details = []
    for xml in xmls:
        image_path = DATA / "images" / (xml.stem + ".jpg")
        if not image_path.exists():
            continue
        table, gt = _read_xml(xml)
        x0, y0, x1, y1 = [int(round(v)) for v in table]
        image = Image.open(image_path).convert("RGB")
        crop = image.crop((x0, y0, x1, y1))
        try:
            result = service.recognize(
                crop, (0, 0, crop.width, crop.height), cell_ocr=False
            )
        except Exception as exc:  # noqa: BLE001
            details.append({"image": image_path.name, "error": str(exc)})
            continue
        pred = {"rows": result.rows, "columns": result.columns, "spans": result.spans}
        pred["cells"] = [(box, 1.0) for box in _grid_cells(result.rows, result.columns)]
        # 结构模型输入是 table crop；GT 坐标也要平移到 crop 坐标系。
        gt = {
            name: [(b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0) for b in boxes]
            for name, boxes in gt.items()
        }
        gt["cells"] = _grid_cells(
            [(box, 1.0) for box in gt.get("rows", [])],
            [(box, 1.0) for box in gt.get("columns", [])],
        )
        item = {"image": image_path.name, "gt": {}, "pred": {}, "f1": {}}
        for name in stats:
            gboxes = gt.get(name, [])
            pboxes = pred[name]
            tp, fp, fn = _match(pboxes, gboxes)
            stats[name]["tp"] += tp
            stats[name]["fp"] += fp
            stats[name]["fn"] += fn
            stats[name]["images"] += 1
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            item["gt"][name] = len(gboxes)
            item["pred"][name] = len(pboxes)
            item["f1"][name] = round(f1, 3)
        details.append(item)
    summary = {}
    for name, values in stats.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary[name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    output = {"images": len(details), "summary": summary, "details": details}
    path = Path(args.out)
    lines = [
        "# PubTables-1M Structure 验证",
        "",
        f"- 图像：{len(details)}",
        "- 模型：Table Transformer structure",
        "",
        "| 类别 | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary.items():
        lines.append(
            f"| {name} | {value['tp']} | {value['fp']} | {value['fn']} | "
            f"{value['precision']} | {value['recall']} | {value['f1']} |"
        )
    lines += [
        "",
        "> PubTables 不提供逐 cell 标签；cell 指标由 row/column 交点构造代理框，不含合并跨越。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
