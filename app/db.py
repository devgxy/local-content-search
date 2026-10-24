"""LanceDB 连接、建表、ICU FTS 与健康检查。

主流程：connect_db → open_table（建表/校验 schema）→ ensure_fts
（检查 text_idx 的 ICU 分词器）→ 返回 LanceTable。

Step: STEP-01
Doc: docs/steps/STEP-01-数据层与索引.md §4
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lancedb
import pyarrow as pa
from lancedb.index import FTS
from lancedb.table import LanceTable

from app.config import Settings
from app.schema import build_chunk_model

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
REQUIRED_FIELDS = {
    "id",
    "doc_id",
    "root_id",
    "chunk_index",
    "text",
    "vector",
    "display_text",
    "file_name",
    "file_path",
    "file_type",
}


class DBError(RuntimeError):
    """数据库初始化/访问失败。"""


class DBSchemaMismatch(DBError):
    """现有表结构与当前配置不一致（维度/模型）。"""


def connect_db(settings: Settings) -> lancedb.DBConnection:
    """连接 LanceDB 数据目录（不存在则创建）。

    Args:
        settings: 运行期配置。

    Returns:
        LanceDB 连接。

    Raises:
        DBError: 目录不可创建或连接失败。
    """
    try:
        settings.db_dir.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(str(settings.db_dir))
    except Exception as exc:  # noqa: BLE001
        raise DBError(f"无法连接 LanceDB：{settings.db_dir}：{exc}") from exc


def _table_exists(db: lancedb.DBConnection, name: str) -> bool:
    """判断表是否存在，兼容新旧 lancedb 接口。"""
    try:
        return name in set(db.list_tables().tables)
    except Exception:  # noqa: BLE001
        # 兼容旧接口
        return name in set(db.table_names(limit=1000))


def _vector_dim(schema: pa.Schema) -> int:
    """读取 vector 列的固定维度。"""
    if "vector" not in schema.names:
        raise DBSchemaMismatch("表缺少 vector 字段")
    field_type = schema.field("vector").type
    if not pa.types.is_fixed_size_list(field_type):
        raise DBSchemaMismatch(f"vector 不是 fixed_size_list：{field_type}")
    return int(field_type.list_size)


def _check_schema(table: LanceTable, settings: Settings) -> None:
    """校验字段/维度/模型/provider/prompt，防止混用库。

    Args:
        table: 已打开的表。
        settings: 当前配置。

    Raises:
        DBSchemaMismatch: 任一维度或元数据不一致。
    """
    schema = table.schema
    missing = REQUIRED_FIELDS - set(schema.names)
    if missing:
        raise DBSchemaMismatch(f"表缺少字段：{sorted(missing)}")

    dim = _vector_dim(schema)
    if dim != settings.embed_dim:
        raise DBSchemaMismatch(
            f"向量维度不一致：表={dim}，配置={settings.embed_dim}；换模型需重建库"
        )

    meta = _read_schema_meta(settings)
    if meta is None:
        _write_schema_meta(settings)
        return
    if meta.get("embed_dim") != settings.embed_dim:
        raise DBSchemaMismatch("schema.json 中的 embed_dim 与配置不一致")
    if meta.get("embed_model") != settings.embed_model:
        raise DBSchemaMismatch(
            f"表由模型 {meta.get('embed_model')} 构建，当前配置为 {settings.embed_model}；"
            "请重建库或恢复原模型"
        )
    if meta.get("embed_provider", "bge") != settings.embed_provider:
        raise DBSchemaMismatch("embedding provider 与库不一致，请重建库")
    if meta.get("embed_query_prompt", "") != settings.embed_query_prompt:
        raise DBSchemaMismatch("查询 prompt 与库不一致，请重建库")
    if meta.get("embed_doc_template", "") != settings.embed_doc_template:
        raise DBSchemaMismatch("文档模板与库不一致，请重建库")


def _read_schema_meta(settings: Settings) -> dict | None:
    """读取 data/schema.json；不存在返回 None。"""
    path = settings.schema_meta_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DBSchemaMismatch(f"schema.json 损坏：{exc}") from exc


def _write_schema_meta(settings: Settings) -> None:
    """写入当前模型/维度/prompt 元数据。"""
    payload = {
        "version": SCHEMA_VERSION,
        "embed_dim": settings.embed_dim,
        "embed_model": settings.embed_model,
        "embed_provider": settings.embed_provider,
        "embed_dtype": settings.embed_dtype,
        "embed_query_prompt": settings.embed_query_prompt,
        "embed_doc_template": settings.embed_doc_template,
        "table": settings.table_name,
    }
    path = settings.schema_meta_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def ensure_fts(table: LanceTable) -> None:
    """幂等创建 ICU 分词 FTS 索引。

    不要使用 create_fts_index（0.25 起弃用），也不要传 use_tantivy=True
    （0.40 已移除 Tantivy）。
    """
    for index in table.list_indices():
        if index.index_type != "FTS" or list(index.columns) != ["text"]:
            continue
        details = index.index_details or {}
        if details.get("base_tokenizer") == "icu":
            return
        # 已有 FTS 但分词器不是 ICU：替换为 ICU，避免中文静默零召回。
        break
    try:
        table.create_index(
            "text",
            config=FTS(
                base_tokenizer="icu",
                lower_case=True,
                stem=False,
                remove_stop_words=False,
                ascii_folding=False,
            ),
            replace=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise DBError(f"创建 FTS(ICU) 索引失败：{exc}") from exc


def open_table(
    db: lancedb.DBConnection,
    settings: Settings,
    func,
) -> LanceTable:
    """打开或创建 chunks 表，并保证 FTS 索引存在。"""
    chunk_model = build_chunk_model(func, settings.embed_dim)
    try:
        if _table_exists(db, settings.table_name):
            table = db.open_table(settings.table_name)
            _check_schema(table, settings)
        else:
            table = db.create_table(settings.table_name, schema=chunk_model)
            _write_schema_meta(settings)
            logger.info("已创建表 %s（dim=%s）", settings.table_name, settings.embed_dim)
        ensure_fts(table)
        return table
    except DBSchemaMismatch:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DBError(f"打开/创建表失败：{exc}") from exc


def health_check(table: LanceTable) -> dict:
    """检查表/FTS 是否可用并统计 chunk。

    Args:
        table: chunks 表。

    Returns:
        {ok, chunks, fts_index, ...} 健康信息。
    """
    indices = [{"name": item.name, "type": item.index_type} for item in table.list_indices()]
    try:
        dim = _vector_dim(table.schema)
    except DBSchemaMismatch:
        dim = 0
    return {
        "table": table.name,
        "chunks": table.count_rows(),
        "fts_index": any(item["type"] == "FTS" for item in indices),
        "vector_dim": dim,
        "indices": indices,
    }


def db_stats(settings: Settings, table: LanceTable) -> dict:
    """返回库统计与磁盘大小。

    Args:
        settings: 配置。
        table: chunks 表。

    Returns:
        {chunks, db_size_mb, ...}。
    """
    total_bytes = 0
    if settings.db_dir.exists():
        for path in settings.db_dir.rglob("*"):
            if path.is_file():
                total_bytes += path.stat().st_size
    return {
        "chunks": table.count_rows(),
        "db_size_mb": round(total_bytes / 1_000_000, 1),
        "indices": [
            {"name": item.name, "type": item.index_type} for item in table.list_indices()
        ],
    }


def optimize(table: LanceTable) -> None:
    """压缩/优化 LanceDB 表（维护命令）。"""
    table.optimize()


def table_path_size(path: Path) -> int:
    """递归计算表目录占用字节数。"""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
