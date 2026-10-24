"""把 T2Retrieval（C-MTEB）parquet 转成可检索的本地子集。

输入：
  corpus.parquet   id/text
  queries.parquet  id/text
  qrels.parquet    qid/pid/score

输出：
  data/eval/public/t2retrieval/index/    LanceDB（passage-level）
  data/eval/public/t2retrieval/queries.jsonl

注意：子集不是官方全量成绩；只用于流程和相对比较。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path

import pyarrow.parquet as pq

from app.config import load_settings
from app.db import connect_db, open_table
from app.embeddings import EmbeddingService
from app.ids import make_chunk_id, make_content_hash


def _clean(text: str) -> str:
    """去掉 HTML 标签并压缩空白。"""
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/eval/public/t2retrieval")
    parser.add_argument("--data-dir", default="data/eval/public/t2retrieval/index")
    parser.add_argument("--queries-out", default="data/eval/public/t2retrieval/queries.jsonl")
    parser.add_argument("--query-limit", type=int, default=50)
    parser.add_argument("--negative-count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    source = Path(args.input_dir)
    corpus = pq.read_table(source / "corpus.parquet").to_pylist()
    queries = pq.read_table(source / "queries.parquet").to_pylist()
    qrels = pq.read_table(source / "qrels.parquet").to_pylist()

    corpus_text = {str(row["id"]): _clean(row["text"]) for row in corpus}
    query_text = {str(row["id"]): str(row["text"]) for row in queries}
    relevant_by_qid: dict[str, list[str]] = {}
    for row in qrels:
        relevant_by_qid.setdefault(str(row["qid"]), []).append(str(row["pid"]))

    selected_qids = [qid for qid in query_text if qid in relevant_by_qid][: args.query_limit]
    if not selected_qids:
        raise SystemExit("没有可用 qrels")

    relevant: set[str] = set()
    eval_rows = []
    for qid in selected_qids:
        pids = [pid for pid in relevant_by_qid[qid] if pid in corpus_text]
        if not pids:
            continue
        relevant.update(pids)
        eval_rows.append(
            {
                "qid": qid,
                "query": query_text[qid],
                "relevant_ids": pids,
                "type": "public",
            }
        )

    rng = random.Random(args.seed)
    negatives = [pid for pid in corpus_text if pid not in relevant]
    if len(negatives) > args.negative_count:
        negatives = rng.sample(negatives, args.negative_count)
    selected = list(dict.fromkeys(list(relevant) + negatives))
    print(f"queries={len(eval_rows)} relevant={len(relevant)} passages={len(selected)}")

    data_dir = Path(args.data_dir)
    if data_dir.exists():
        if not args.reset:
            raise SystemExit(f"数据目录已存在：{data_dir}；用 --reset 或换目录")
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    settings = load_settings({"DATA_DIR": str(data_dir)})
    settings.ensure_dirs()
    embedder = EmbeddingService(settings)
    table = open_table(connect_db(settings), settings, embedder.func)

    texts = [corpus_text[pid] for pid in selected]
    vectors = []
    batch = settings.embed_batch_size
    for start in range(0, len(texts), batch):
        vectors.extend(embedder.encode_documents(texts[start : start + batch]))
    rows = []
    for pid, text, vector in zip(selected, texts, vectors, strict=True):
        content_hash = make_content_hash(text)
        rows.append(
            {
                "id": make_chunk_id(pid, 0, content_hash),
                "doc_id": pid,
                "root_id": 0,
                "chunk_index": 0,
                "text": text,
                "vector": vector,
                "display_text": text,
                "file_name": pid,
                "file_path": f"public://t2retrieval/{pid}",
                "file_type": "md",
                "title": "T2Retrieval",
                "heading": "",
                "page": 0,
                "slide": 0,
                "char_start": 0,
                "char_end": len(text),
                "content_hash": content_hash,
                "doc_hash": "public",
                "run_id": "t2retrieval",
                "indexed_at": time.time(),
                "source_kind": "text",
                "meta_json": json.dumps({"source": "T2Retrieval"}, ensure_ascii=False),
            }
        )
    table.add(rows)
    queries_out = Path(args.queries_out)
    queries_out.parent.mkdir(parents=True, exist_ok=True)
    with queries_out.open("w", encoding="utf-8") as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {queries_out} rows={len(eval_rows)}")
    print(f"index {data_dir} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
