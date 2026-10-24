"""STEP-07 检索与重排测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.config import Settings
from app.db import connect_db, open_table
from app.reranker import RerankerUnavailable
from app.search import SearchError, SearchService


class FakeEmbedder:
    def __init__(self, func):
        self.func = func

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(self.func.generate_embeddings([query])[0], dtype=np.float32)


class FakeReranker:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if self.fail:
            raise RerankerUnavailable("boom")
        return np.asarray(
            [0.95 if "E1024" in passage else 0.1 for passage in passages],
            dtype=np.float32,
        )


def _row(func, *, id_, doc_id, idx, text, file_name, file_type, path, heading="") -> dict:
    return {
        "id": id_,
        "doc_id": doc_id,
        "root_id": 1,
        "chunk_index": idx,
        "text": f"{heading}\n{text}" if heading else text,
        "vector": func.generate_embeddings([text])[0],
        "display_text": text,
        "file_name": file_name,
        "file_path": path,
        "file_type": file_type,
        "title": file_name,
        "heading": heading,
        "page": 0,
        "slide": 0,
        "char_start": 0,
        "char_end": len(text),
        "content_hash": f"h{id_}",
        "doc_hash": "doc",
        "run_id": "run",
        "indexed_at": 0.0,
        "source_kind": "text",
        "meta_json": "{}",
    }


@pytest.fixture()
def search_stack(tmp_settings: Settings, fake_func):
    table = open_table(connect_db(tmp_settings), tmp_settings, fake_func)
    table.add(
        [
            _row(
                fake_func,
                id_="d1c0",
                doc_id="d1",
                idx=0,
                text="错误码 E1024：连接超时，请检查代理配置。",
                file_name="manual.pdf",
                file_type="pdf",
                path="/docs/manual.pdf",
                heading="故障排查",
            ),
            _row(
                fake_func,
                id_="d1c1",
                doc_id="d1",
                idx=1,
                text="补充说明：代理配置需要与端口一致。",
                file_name="manual.pdf",
                file_type="pdf",
                path="/docs/manual.pdf",
                heading="故障排查",
            ),
            _row(
                fake_func,
                id_="d2c0",
                doc_id="d2",
                idx=0,
                text="混合检索系统结合向量召回与关键词匹配。",
                file_name="guide.md",
                file_type="md",
                path="/docs/guide.md",
                heading="架构",
            ),
            _row(
                fake_func,
                id_="d3c0",
                doc_id="d3",
                idx=0,
                text="今天天气很好，适合散步。",
                file_name="life.txt",
                file_type="txt",
                path="/docs/life.txt",
            ),
        ]
    )
    settings = replace(tmp_settings, top_k=10, final_k=3, per_doc_limit=2)
    return SearchService(settings, table, FakeEmbedder(fake_func), FakeReranker()), settings


def test_keyword_hit_and_rerank(search_stack) -> None:
    service, _ = search_stack
    result = service.search("E1024")
    assert result.results
    assert result.results[0].file_name == "manual.pdf"
    assert result.results[0].rerank_score == pytest.approx(0.95)
    assert result.results[0].summary
    assert "**" not in result.results[0].summary
    assert result.results[0].rank_before >= 1


def test_per_doc_limit(search_stack) -> None:
    service, settings = search_stack
    settings = replace(settings, per_doc_limit=1, final_k=3)
    service.settings = settings
    result = service.search("E1024")
    doc_ids = [hit.doc_id for hit in result.results]
    assert doc_ids.count("d1") <= 1


def test_context_expansion(search_stack) -> None:
    service, _ = search_stack
    result = service.search("E1024")
    assert "代理配置需要与端口一致" in result.results[0].snippet


def test_filter_file_type(search_stack) -> None:
    service, _ = search_stack
    result = service.search("混合检索", file_types=["pdf"])
    assert result.results
    assert all(hit.file_type == "pdf" for hit in result.results)


def test_rerank_failure_is_explicit(search_stack) -> None:
    service, settings = search_stack
    service.reranker = FakeReranker(fail=True)
    with pytest.raises(SearchError) as exc:
        service.search("E1024")
    assert exc.value.code == "RERANK_FAILED"


def test_empty_query(search_stack) -> None:
    service, _ = search_stack
    with pytest.raises(SearchError):
        service.search("   ")


def test_no_candidates(search_stack) -> None:
    service, _ = search_stack
    result = service.search("E1024", root_id=999)
    assert result.results == []


def test_highlight_terms() -> None:
    terms = SearchService.highlight_terms("E1024 连接超时")
    assert "e1024" in terms
    assert any("连接" in term for term in terms)


def test_score_kind_by_mode(search_stack) -> None:
    service, _ = search_stack
    for mode, kind in (("vector", "distance"), ("fts", "bm25")):
        result = service.search("E1024", mode=mode)
        assert result.results
        assert result.results[0].score_kind == kind
        assert result.results[0].score > 0
    full = service.search("E1024", mode="full")
    assert full.results[0].score_kind == "rerank"


def test_overflow_refill_is_bounded(search_stack) -> None:
    service, settings = search_stack
    service.table.add(
        [
            _row(
                service.embedder.func,
                id_=f"d1c{i}",
                doc_id="d1",
                idx=i,
                text=f"错误码 E1024 的补充说明 {i}。",
                file_name="manual.pdf",
                file_type="pdf",
                path="/docs/manual.pdf",
            )
            for i in range(2, 5)
        ]
    )
    service.settings = replace(settings, per_doc_limit=1, final_k=5)
    result = service.search("E1024")
    assert [hit.doc_id for hit in result.results].count("d1") <= 3


def test_fast_profile_skips_rerank(search_stack) -> None:
    service, settings = search_stack
    service.settings = replace(settings, search_profile="fast")
    result = service.search("E1024", final_k=5)
    assert result.candidates <= 10
    assert result.results
    assert all(hit.rerank_score is None for hit in result.results)
    assert all(hit.score_kind == "rrf" for hit in result.results)


def test_parent_section_context(search_stack) -> None:
    service, _ = search_stack
    service.table.add(
        [
            _row(
                service.embedder.func,
                id_="d1c2",
                doc_id="d1",
                idx=2,
                text="第三段：同属故障排查章节的上下文。",
                file_name="manual.pdf",
                file_type="pdf",
                path="/docs/manual.pdf",
                heading="故障排查",
            )
        ]
    )
    result = service.search("E1024")
    assert "第三段" in result.results[0].snippet


def test_scope_doc_and_path_filters(search_stack) -> None:
    service, _ = search_stack
    by_doc = service.search("E1024", doc_ids=["d1"], mode="hybrid")
    assert by_doc.results
    assert all(hit.doc_id == "d1" for hit in by_doc.results)
    by_path = service.search("混合检索", file_paths=["/docs/guide.md"], mode="hybrid")
    assert by_path.results
    assert all(hit.file_path == "/docs/guide.md" for hit in by_path.results)


def test_rerank_uses_parent_section(search_stack) -> None:
    class RecordingReranker(FakeReranker):
        def __init__(self) -> None:
            super().__init__()
            self.passages: list[str] = []

        def score(self, query: str, passages: list[str]) -> np.ndarray:
            self.passages = passages
            return super().score(query, passages)

    service, settings = search_stack
    service.settings = replace(settings, rerank_parent_context=True)
    recorder = RecordingReranker()
    service.reranker = recorder
    result = service.search("E1024", mode="hybrid_rerank", top_k=10, final_k=3)
    assert result.results
    assert any("补充说明" in passage for passage in recorder.passages)


def test_hybrid_keeps_chunks_without_doc_level(search_stack) -> None:
    service, settings = search_stack
    service.settings = replace(settings, per_doc_limit=1, final_k=3)
    result = service.search("E1024", mode="hybrid", top_k=10, final_k=3)
    assert [hit.doc_id for hit in result.results].count("d1") >= 1


def test_metadata_boost_only_for_explicit_filename(search_stack) -> None:
    service, _ = search_stack
    candidate = {
        "file_name": "张三-个人薪酬说明表.docx",
        "file_path": "/资料/薪酬/张三-个人薪酬说明表.docx",
    }
    assert service._metadata_boost("张三-个人薪酬说明表", candidate) == 0.05
    assert service._metadata_boost("个人薪酬说明表", candidate) == 0.0


def test_doc_level_returns_one_chunk_per_doc(search_stack) -> None:
    service, settings = search_stack
    service.settings = replace(settings, per_doc_limit=2, final_k=5)
    result = service.search("E1024", mode="hybrid", top_k=10, final_k=5, doc_level=True)
    ids = [hit.doc_id for hit in result.results]
    assert len(ids) == len(set(ids))
