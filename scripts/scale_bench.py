"""P2 规模化验证：256/768 维 10 万级向量检索（不加载模型）。

Step: P2
Doc: docs/11-P2实施记录.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from app.config import Settings
from app.db import connect_db


def run_dim(data_dir: Path, dim: int, rows: int, queries: int) -> dict:
    shutil.rmtree(data_dir, ignore_errors=True)
    settings = Settings(data_dir=data_dir)
    settings.ensure_dirs()
    db = connect_db(settings)
    rng = np.random.default_rng(42)
    started = time.perf_counter()
    vectors = rng.standard_normal((rows, dim), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vector_array = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()), dim
    )
    table = db.create_table(
        f"bench_{dim}",
        data=pa.table(
            {
                "id": pa.array([str(i) for i in range(rows)]),
                "vector": vector_array,
            }
        ),
    )
    build_s = time.perf_counter() - started

    qvec = rng.standard_normal(dim, dtype=np.float32)
    qvec /= np.linalg.norm(qvec)
    table.search(qvec).limit(20).to_list()
    latencies = []
    for _ in range(queries):
        start = time.perf_counter()
        table.search(qvec).limit(20).to_list()
        latencies.append((time.perf_counter() - start) * 1000)

    size_mb = sum(
        path.stat().st_size for path in data_dir.rglob("*") if path.is_file()
    ) / (1024 * 1024)
    return {
        "dim": dim,
        "rows": rows,
        "build_s": round(build_s, 2),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "size_mb": round(size_mb, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=20)
    args = parser.parse_args()
    results = []
    for dim in (256, 768):
        results.append(run_dim(Path(f"/tmp/scale_{dim}"), dim, args.rows, args.queries))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
