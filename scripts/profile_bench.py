"""P1 FAST / ACCURATE 档位基准。

Step: P1
Doc: docs/10-P1实施记录.md
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

from app.config import load_settings
from app.db import connect_db, open_table
from app.embeddings import EmbeddingService
from app.evaluation import _doc_ids, load_queries, mrr_at_k, ndcg_at_k, recall_at_k
from app.manifest import Manifest
from app.reranker import RerankerService
from app.search import SearchService


def main() -> int:
    settings = load_settings()
    manifest = Manifest(settings.manifest_path)
    embedder = EmbeddingService(settings)
    table = open_table(connect_db(settings), settings, embedder.func)
    service = SearchService(settings, table, embedder, RerankerService(settings), manifest)
    queries = load_queries(Path("examples/eval-p0.jsonl"))
    payload = {}
    for profile in ("fast", "accurate"):
        latencies = []
        recalls = []
        mrrs = []
        ndcgs = []
        for item in queries:
            start = time.perf_counter()
            result = service.search(
                item.query,
                top_k=20,
                final_k=10,
                profile=profile,
            )
            latencies.append((time.perf_counter() - start) * 1000)
            relevant = set(_doc_ids(item.relevant))
            retrieved = list(dict.fromkeys(hit.doc_id for hit in result.results))
            recalls.append(recall_at_k(relevant, retrieved, 10))
            mrrs.append(mrr_at_k(relevant, retrieved, 10))
            ndcgs.append(ndcg_at_k({doc_id: 1 for doc_id in relevant}, retrieved, 10))
        payload[profile] = {
            "recall@10": round(float(np.mean(recalls)), 4),
            "mrr@10": round(float(np.mean(mrrs)), 4),
            "ndcg@10": round(float(np.mean(ndcgs)), 4),
            "p50_ms": round(statistics.median(latencies), 1),
            "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        }
    manifest.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
