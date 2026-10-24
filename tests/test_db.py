"""STEP-01 数据层测试。"""

from __future__ import annotations

import json

import pytest
from lancedb.index import FTS
from lancedb.pydantic import LanceModel, Vector

from app.config import Settings
from app.db import (
    DBSchemaMismatch,
    connect_db,
    db_stats,
    ensure_fts,
    health_check,
    open_table,
    optimize,
)
from app.schema import EMBED_DIM, build_chunk_model


def _row(func, text: str, *, id_: str = "c1", doc_id: str = "d1", idx: int = 0) -> dict:
    vector = func.generate_embeddings([text])[0]
    return {
        "id": id_,
        "doc_id": doc_id,
        "root_id": 1,
        "chunk_index": idx,
        "text": text,
        "vector": vector,
        "display_text": text,
        "file_name": "a.md",
        "file_path": "/tmp/a.md",
        "file_type": "md",
        "title": "测试",
        "heading": "",
        "page": 0,
        "slide": 0,
        "char_start": 0,
        "char_end": len(text),
        "content_hash": f"h{idx}",
        "doc_hash": "doc",
        "run_id": "run",
        "indexed_at": 0.0,
        "source_kind": "text",
        "meta_json": "{}",
    }


def test_schema_fields(fake_func) -> None:
    model = build_chunk_model(fake_func)
    fields = model.model_fields
    for name in ("id", "text", "vector", "file_name", "file_path", "display_text"):
        assert name in fields
    assert EMBED_DIM == 1024


def test_open_table_idempotent(tmp_settings: Settings, fake_func) -> None:
    db = connect_db(tmp_settings)
    table1 = open_table(db, tmp_settings, fake_func)
    table2 = open_table(db, tmp_settings, fake_func)
    assert table1.name == table2.name == tmp_settings.table_name
    assert any(i.index_type == "FTS" for i in table2.list_indices())
    meta = json.loads(tmp_settings.schema_meta_path.read_text(encoding="utf-8"))
    assert meta["embed_dim"] == 1024


def test_chinese_fts_icu(tmp_settings: Settings, fake_func) -> None:
    db = connect_db(tmp_settings)
    table = open_table(db, tmp_settings, fake_func)
    table.add(
        [
            _row(fake_func, "混合检索系统结合向量召回与关键词匹配", id_="c1"),
            _row(fake_func, "LanceDB 是本地嵌入式向量数据库", id_="c2", idx=1),
            _row(fake_func, "今天天气很好适合散步", id_="c3", idx=2),
        ]
    )
    assert table.count_rows() == 3

    hits = table.search("混合检索", query_type="fts").limit(3).to_list()
    assert hits, "ICU 分词应命中中文"
    assert "混合检索" in hits[0]["text"]

    latin = table.search("LanceDB", query_type="fts").limit(3).to_list()
    assert latin


def test_default_tokenizer_fails_chinese(tmp_settings: Settings, fake_func) -> None:
    """回归测试：默认 simple 分词器对中文召回为 0。"""
    db = connect_db(tmp_settings)
    model = build_chunk_model(fake_func)
    table = db.create_table("simple_probe", schema=model)
    table.add([_row(fake_func, "混合检索系统结合向量召回与关键词匹配")])
    table.create_index(
        "text",
        config=FTS(base_tokenizer="simple", stem=False, remove_stop_words=False),
    )
    assert table.search("混合检索", query_type="fts").limit(3).to_list() == []


def test_fts_sees_appended_rows(tmp_settings: Settings, fake_func) -> None:
    db = connect_db(tmp_settings)
    table = open_table(db, tmp_settings, fake_func)
    table.add([_row(fake_func, "初始文档讲向量检索")])
    table.add([_row(fake_func, "新增文档讲量子纠缠与超导", id_="c2", idx=1)])
    hits = table.search("量子纠缠", query_type="fts").limit(3).to_list()
    assert hits, "FTS 应能查到索引后追加的行"


def test_health_and_stats(tmp_settings: Settings, fake_func) -> None:
    db = connect_db(tmp_settings)
    table = open_table(db, tmp_settings, fake_func)
    table.add([_row(fake_func, "内容")])
    health = health_check(table)
    assert health["chunks"] == 1
    assert health["fts_index"] is True
    assert health["vector_dim"] == 1024
    stats = db_stats(tmp_settings, table)
    assert stats["chunks"] == 1
    assert stats["db_size_mb"] >= 0


def test_ensure_fts_idempotent(tmp_settings: Settings, fake_func) -> None:
    db = connect_db(tmp_settings)
    table = open_table(db, tmp_settings, fake_func)
    before = len(table.list_indices())
    ensure_fts(table)
    assert len(table.list_indices()) == before


def test_optimize(tmp_settings: Settings, fake_func) -> None:
    db = connect_db(tmp_settings)
    table = open_table(db, tmp_settings, fake_func)
    table.add([_row(fake_func, "内容")])
    optimize(table)
    assert table.count_rows() == 1


def test_dimension_mismatch(tmp_settings: Settings, fake_func) -> None:
    db = connect_db(tmp_settings)

    class Bad(LanceModel):
        id: str
        text: str = fake_func.SourceField()
        vector: Vector(8) = fake_func.VectorField()

    db.create_table(tmp_settings.table_name, schema=Bad)
    with pytest.raises(DBSchemaMismatch):
        open_table(db, tmp_settings, fake_func)


def test_ensure_fts_replaces_simple_tokenizer(tmp_settings: Settings, fake_func) -> None:
    db = connect_db(tmp_settings)
    model = build_chunk_model(fake_func)
    table = db.create_table("needs_icu", schema=model)
    table.add([_row(fake_func, "混合检索系统结合向量召回")])
    table.create_index(
        "text",
        config=FTS(base_tokenizer="simple", stem=False, remove_stop_words=False),
    )
    assert table.search("混合检索", query_type="fts").limit(3).to_list() == []
    ensure_fts(table)
    assert table.search("混合检索", query_type="fts").limit(3).to_list()


def test_schema_dynamic_dim(fake_func) -> None:
    model = build_chunk_model(fake_func, dim=768)
    schema = model.to_arrow_schema()
    assert schema.field("vector").type.list_size == 768
