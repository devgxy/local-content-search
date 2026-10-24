"""评测 CLI。

Step: STEP-10
Doc: docs/steps/STEP-10-评测与基准.md §7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.db import connect_db, open_table
from app.embeddings import EmbeddingService
from app.evaluation import CONFIGS, evaluate, load_queries, write_report
from app.manifest import Manifest
from app.reranker import RerankerService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--root-path", default=None, help="只评测该索引根")
    parser.add_argument("--out", default="reports/eval.md")
    args = parser.parse_args()

    settings = load_settings()
    settings.ensure_dirs()
    queries = load_queries(Path(args.dataset))
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("数据集为空")
        return 1

    manifest = Manifest(settings.manifest_path)
    try:
        embedder = EmbeddingService(settings)
        table = open_table(connect_db(settings), settings, embedder.func)
        reranker = RerankerService(settings)
        configs = tuple(item.strip() for item in args.configs.split(",") if item.strip())
        payload = evaluate(
            settings,
            table,
            embedder,
            reranker,
            queries,
            configs=configs,
            top_k=args.top_k,
            final_k=args.final_k,
            manifest=manifest,
            root_path=args.root_path,
        )
        write_report(payload, settings, Path(args.out))
        print(json.dumps(
            {
                key: {
                    "recall@5": round(value.recall5, 4),
                    "recall@10": round(value.recall10, 4),
                    "mrr@10": round(value.mrr10, 4),
                    "ndcg@10": round(value.ndcg10, 4),
                    "chunk_recall@5": round(value.chunk_recall5, 4),
                    "chunk_recall@10": round(value.chunk_recall10, 4),
                    "chunk_mrr@10": round(value.chunk_mrr10, 4),
                    "chunk_ndcg@10": round(value.chunk_ndcg10, 4),
                    "chunk_queries": value.chunk_queries,
                    "p50_ms": round(value.p50_ms, 1),
                    "p95_ms": round(value.p95_ms, 1),
                }
                for key, value in payload["results"].items()
            },
            ensure_ascii=False,
            indent=2,
        ))
        print(f"报告：{Path(args.out).resolve()}")
        return 0
    finally:
        manifest.close()


if __name__ == "__main__":
    raise SystemExit(main())
