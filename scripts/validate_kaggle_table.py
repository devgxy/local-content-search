"""在 Kaggle table-detection val 集上验证 Table Transformer 检测。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.table_structure import TableStructureService

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "eval" / "kaggle" / "table-detection"


def iou(a, b) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="reports/kaggle-table-detection.md")
    args = parser.parse_args()

    truth: dict[str, list[tuple]] = defaultdict(list)
    with (DATA / "val.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 5:
                continue
            truth[row[0]].append(tuple(float(v) for v in row[1:5]))
    names = sorted(truth)[: args.limit]
    service = TableStructureService(Settings(table_structure_device="cpu"))
    service.ensure_ready()
    matched = 0
    total = 0
    extra = 0
    details = []
    for name in names:
        image = Image.open(DATA / "images" / name).convert("RGB")
        detections = service.detect(image, threshold=args.threshold)
        boxes = [item[0] for item in detections]
        gt = truth[name]
        total += len(gt)
        best_matches = []
        for target in gt:
            scores = [iou(target, box) for box in boxes]
            best = max(scores) if scores else 0.0
            best_matches.append(round(best, 3))
            if best >= 0.5:
                matched += 1
        extra += max(0, len(boxes) - len(gt))
        details.append({"image": name, "gt": len(gt), "det": len(boxes), "best_iou": best_matches})
    report = {
        "images": len(names),
        "gt_tables": total,
        "matched_iou50": matched,
        "recall_iou50": round(matched / total, 4) if total else 0.0,
        "extra_detections": extra,
        "details": details,
    }
    out = Path(args.out)
    lines = [
        "# Kaggle table-detection 验证",
        "",
        f"- 图像：{len(names)}",
        f"- GT 表格：{total}",
        f"- IoU≥0.5 命中：{matched}",
        f"- Recall：{report['recall_iou50']}",
        f"- 多余检测：{extra}",
        "",
        "| 图像 | GT | 检测 | 每个 GT 的最佳 IoU |",
        "|---|---:|---:|---|",
    ]
    for item in details:
        lines.append(f"| {item['image']} | {item['gt']} | {item['det']} | {item['best_iou']} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
