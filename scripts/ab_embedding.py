"""P0 A/B：BGE-large-zh vs EmbeddingGemma 2 文本塔。

Step: P0
Doc: docs/09-嵌入模型P0实施记录.md
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
CORPUS_ROOT = "docs"
DATASET = "examples/eval-p0.jsonl"

CONFIGS = {
    "bge": {
        "EMBED_PROVIDER": "bge",
        "EMBED_MODEL": "bge-large-zh-v1.5",
        "EMBED_DIM": "1024",
        "EMBED_DTYPE": "float32",
        "DATA_DIR": "/tmp/ab_bge",
        "OUT": "reports/ab-bge.md",
    },
    "gemma": {
        "EMBED_PROVIDER": "gemma",
        "EMBED_MODEL": "embeddinggemma-2",
        "EMBED_DIM": "768",
        "EMBED_DTYPE": "bfloat16",
        "DATA_DIR": "/tmp/ab_gemma",
        "OUT": "reports/ab-gemma.md",
    },
}


def run(cmd: list[str], env: dict[str, str]) -> str:
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(result.stdout[-4000:], flush=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {cmd[0]} ({result.returncode})")
    return result.stdout


def parse_report(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| hybrid"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 7:
            rows[cells[0]] = {
                "recall5": cells[1],
                "recall10": cells[2],
                "mrr": cells[3],
                "ndcg": cells[4],
                "p50": cells[5],
                "p95": cells[6],
            }
    return rows


def main() -> int:
    base_env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    results = {}
    for name, config in CONFIGS.items():
        data_dir = Path(config["DATA_DIR"])
        shutil.rmtree(data_dir, ignore_errors=True)
        env = {**base_env, **{k: v for k, v in config.items() if k != "OUT"}}
        run([PYTHON, "-m", "app.cli", "index", "--root", CORPUS_ROOT], env)
        run(
            [
                PYTHON,
                "scripts/benchmark.py",
                "--dataset",
                DATASET,
                "--configs",
                "hybrid,hybrid_rerank",
                "--top-k",
                "10",
                "--final-k",
                "10",
                "--out",
                config["OUT"],
            ],
            env,
        )
        results[name] = parse_report(ROOT / config["OUT"])

    lines = [
        "# P0 嵌入模型 A/B",
        "",
        "同一份 docs 语料 + 同一 7 条 query；仅替换 embedding 模型。",
        "",
        "| 模型 | 配置 | Recall@10 | MRR@10 | nDCG@10 | p50 | p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, rows in results.items():
        for config_name, row in rows.items():
            lines.append(
                f"| {name} | {config_name} | {row['recall10']} | {row['mrr']} | "
                f"{row['ndcg']} | {row['p50']} | {row['p95']} |"
            )
    out = ROOT / "reports/ab-embedding.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nA/B summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
