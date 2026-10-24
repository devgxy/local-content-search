"""Embedding 模型编码基准（P0）。

Step: P0
Doc: docs/09-嵌入模型P0实施记录.md
"""

from __future__ import annotations

import json
import statistics
import time

import numpy as np
import torch

from app.config import load_settings
from app.embeddings import EmbeddingService


def main() -> int:
    settings = load_settings()
    started = time.perf_counter()
    service = EmbeddingService(settings)
    _ = service.encode_query("预热查询")
    load_s = time.perf_counter() - started

    query_times = []
    for _ in range(10):
        start = time.perf_counter()
        service.encode_query("混合检索的延迟怎么优化？")
        query_times.append((time.perf_counter() - start) * 1000)

    base = (
        "混合检索系统需要同时利用向量语义召回和关键词精确匹配，"
        "LanceDB 在本地完成双路召回与 RRF 融合，重排模型再做精排。"
    )
    docs = [f"第{i}段。" + base for i in range(64)]
    start = time.perf_counter()
    vectors = service.encode_documents(docs, titles=[f"文档{i}" for i in range(64)])
    encode_s = time.perf_counter() - start

    payload = {
        "provider": settings.embed_provider,
        "model": settings.embed_model,
        "dtype": settings.embed_dtype,
        "dim": len(vectors[0]),
        "load_s": round(load_s, 2),
        "query_p50_ms": round(statistics.median(query_times), 1),
        "query_p95_ms": round(float(np.percentile(query_times, 95)), 1),
        "docs": len(docs),
        "encode_s": round(encode_s, 2),
        "chunk_per_s": round(len(docs) / encode_s, 1),
        "mps_allocated_gb": round(torch.mps.current_allocated_memory() / 1e9, 2),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
