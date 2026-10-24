"""STEP-10 指标测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation import load_queries, mrr_at_k, ndcg_at_k, recall_at_k


def test_recall() -> None:
    assert recall_at_k({"a", "b", "c"}, ["x", "a", "y", "b"], 3) == pytest.approx(1 / 3)
    assert recall_at_k({"a"}, ["a"], 5) == 1.0
    assert recall_at_k(set(), ["a"], 5) == 0.0


def test_mrr() -> None:
    assert mrr_at_k({"a", "b"}, ["x", "a", "y", "b"], 3) == pytest.approx(1 / 2)
    assert mrr_at_k({"z"}, ["a", "b"], 10) == 0.0


def test_ndcg() -> None:
    assert ndcg_at_k({"a": 1, "b": 1}, ["a", "b"], 10) == pytest.approx(1.0)
    assert ndcg_at_k({"a": 1, "b": 1}, ["b", "a"], 10) == pytest.approx(1.0)
    assert ndcg_at_k({"a": 2}, ["x", "a"], 10) < 1.0
    assert ndcg_at_k({}, ["a"], 10) == 0.0


def test_load_queries(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps({"qid": "1", "query": "E1024", "relevant": ["/a.pdf"]}) + "\n",
        encoding="utf-8",
    )
    queries = load_queries(path)
    assert queries[0].qid == "1"
    assert queries[0].relevant == ["/a.pdf"]


def test_load_queries_with_chunk_labels(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps(
            {
                "qid": "c1",
                "query": "E1024",
                "relevant": ["/a.pdf"],
                "relevant_ids": ["doc1"],
                "relevant_chunks": ["/a.pdf#2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    queries = load_queries(path)
    assert queries[0].relevant_ids == ["doc1"]
    assert queries[0].relevant_chunks == ["/a.pdf#2"]


def test_load_queries_with_anchors(tmp_path: Path) -> None:
    path = tmp_path / "anchors.jsonl"
    path.write_text(
        json.dumps(
            {
                "qid": "a1",
                "query": "E1024",
                "relevant": [],
                "relevant_anchors": [
                    {
                        "path": "/a.pdf",
                        "quote": "连接超时",
                        "heading": "故障",
                        "doc_hash": "abc",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    queries = load_queries(path)
    assert queries[0].anchors[0].quote == "连接超时"
    assert queries[0].anchors[0].heading == "故障"
    assert queries[0].anchors[0].doc_hash == "abc"
