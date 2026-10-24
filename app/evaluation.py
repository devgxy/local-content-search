"""检索评测：指标、配置对比、报告。

流程：加载 JSONL → 逐个配置逐 query 调用 SearchService → 计算
Recall/MRR/nDCG/延迟 → 写 Markdown 报告。

Step: STEP-10
Doc: docs/03-评测方案.md · docs/steps/STEP-10-评测与基准.md
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from app.config import Settings
from app.ids import escape_sql_literal, make_doc_id
from app.search import SearchService

logger = logging.getLogger(__name__)

CONFIGS = ("vector", "fts", "hybrid", "hybrid_rerank", "full")


@dataclass(frozen=True)
class ChunkAnchor:
    """内容锚点：重建后仍可按 quote 找回 chunk。

    Attributes:
        path: 文档路径。
        quote: chunk 内的短引用；用于定位。
        heading: 可选标题路径约束。
        doc_hash: 可选文档哈希；变化时标记 stale。
        chunk_index: 可选旧编号；仅作 fallback。
    """

    path: str
    quote: str = ""
    heading: str = ""
    doc_hash: str = ""
    chunk_index: int | None = None


@dataclass
class EvalQuery:
    """一条评测 query。

    Attributes:
        qid: 查询 ID。
        query: 查询文本。
        relevant: 相关文档路径列表（文件级）。
        relevant_ids: 直接给出 doc_id 的标签（公开集用），优先于 relevant。
        relevant_chunks: 相关 chunk 标签，格式 ``path#chunk_index``（固定快照）。
        anchors: 内容锚点列表（动态/重建后仍可解析）。
        scope_doc_ids: 本次评测的 doc_id 白名单（动态选文件）。
        scope_file_paths: 本次评测的文件路径白名单。
        kind: semantic/keyword/mixed，仅用于分析。
    """

    qid: str
    query: str
    relevant: list[str]
    relevant_ids: list[str] = field(default_factory=list)
    relevant_chunks: list[str] = field(default_factory=list)
    anchors: list[ChunkAnchor] = field(default_factory=list)
    scope_doc_ids: list[str] = field(default_factory=list)
    scope_file_paths: list[str] = field(default_factory=list)
    kind: str = "semantic"


@dataclass
class ConfigResult:
    """单个检索配置的评测结果。

    Attributes:
        config: vector/fts/hybrid/hybrid_rerank/full。
        recall5/recall10: Recall@5/10。
        mrr10: MRR@10。
        ndcg10: nDCG@10。
        p50_ms/p95_ms: 查询延迟。
        queries: query 数。
        failures: 未命中/低相关案例。
    """

    config: str
    recall5: float = 0.0
    recall10: float = 0.0
    mrr10: float = 0.0
    ndcg10: float = 0.0
    chunk_recall5: float = 0.0
    chunk_recall10: float = 0.0
    chunk_mrr10: float = 0.0
    chunk_ndcg10: float = 0.0
    chunk_queries: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    queries: int = 0
    failures: list[dict] = field(default_factory=list)


def load_queries(path: Path) -> list[EvalQuery]:
    """读取 JSONL 评测集。

    Args:
        path: 每行一个 query 的 JSONL。

    Returns:
        EvalQuery 列表；空行/# 注释跳过。
    """
    queries = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            queries.append(
                EvalQuery(
                    qid=str(payload.get("qid", len(queries))),
                    query=str(payload["query"]),
                    relevant=[str(item) for item in payload.get("relevant", [])],
                    relevant_ids=[str(item) for item in payload.get("relevant_ids", [])],
                    relevant_chunks=[str(item) for item in payload.get("relevant_chunks", [])],
                    anchors=[
                        ChunkAnchor(
                            path=str(anchor["path"]),
                            quote=str(anchor.get("quote", "")),
                            heading=str(anchor.get("heading", "")),
                            doc_hash=str(anchor.get("doc_hash", "")),
                            chunk_index=(
                                int(anchor["chunk_index"])
                                if anchor.get("chunk_index") is not None
                                else None
                            ),
                        )
                        for anchor in payload.get("relevant_anchors", [])
                    ],
                    scope_doc_ids=[
                        str(item) for item in payload.get("scope_doc_ids", [])
                    ],
                    scope_file_paths=[
                        str(item) for item in payload.get("scope_file_paths", [])
                    ],
                    kind=str(payload.get("type", "semantic")),
                )
            )
    return queries


def _unique(items: list[str]) -> list[str]:
    """保序去重文档 ID。"""
    seen: set[str] = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """计算 Recall@k。"""
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def mrr_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """计算 MRR@k。"""
    for index, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevant: dict[str, int], retrieved: list[str], k: int) -> float:
    """计算 nDCG@k。"""
    if not relevant:
        return 0.0
    dcg = 0.0
    for index, doc_id in enumerate(retrieved[:k], start=1):
        grade = relevant.get(doc_id, 0)
        if grade:
            dcg += (2**grade - 1) / np.log2(index + 1)
    ideal_grades = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / np.log2(index + 1) for index, grade in enumerate(ideal_grades, 1))
    return float(dcg / idcg) if idcg else 0.0


def _doc_ids(paths: list[str]) -> list[str]:
    """把相关文档路径转成 doc_id 集合并去重。"""
    return [make_doc_id(Path(path)) for path in paths]


def _chunk_keys(labels: list[str]) -> set[tuple[str, int]]:
    """把 ``path#chunk_index`` 标签转成可比较的 key。"""
    keys: set[tuple[str, int]] = set()
    for label in labels:
        if "#" not in label:
            continue
        path, index = label.rsplit("#", 1)
        try:
            keys.add((str(Path(path).expanduser().resolve()), int(index)))
        except ValueError:
            continue
    return keys


def _resolve_anchors(
    table,
    manifest,
    anchors: list[ChunkAnchor],
) -> tuple[set[tuple[str, int]], int]:
    """把内容锚点解析成当前 chunk key。

    Args:
        table: LanceDB chunks 表。
        manifest: 可选 Manifest，用于过滤当前 run_id。
        anchors: 内容锚点。

    Returns:
        ``(resolved_keys, missing_count)``。
    """
    keys: set[tuple[str, int]] = set()
    missing = 0
    cache: dict[str, list[dict]] = {}
    for anchor in anchors:
        path = str(Path(anchor.path).expanduser().resolve())
        if path not in cache:
            rows: list[dict] = []
            try:
                rows = (
                    table.search()
                    .where(f"file_path = {escape_sql_literal(path)}")
                    .select(
                        [
                            "chunk_index",
                            "display_text",
                            "text",
                            "heading",
                            "doc_hash",
                            "run_id",
                        ]
                    )
                    .limit(10000)
                    .to_list()
                )
            except Exception:  # noqa: BLE001
                rows = []
            record = manifest.get_file(path) if manifest is not None else None
            if record is not None and record.run_id:
                rows = [row for row in rows if row.get("run_id") == record.run_id]
            cache[path] = rows
        rows = cache[path]
        if anchor.doc_hash:
            rows = [row for row in rows if row.get("doc_hash") == anchor.doc_hash]
        if anchor.heading:
            rows = [row for row in rows if anchor.heading in str(row.get("heading", ""))]
        if anchor.quote:
            quote = re.sub(r"\s+", " ", anchor.quote).strip()
            rows = [
                row
                for row in rows
                if quote
                in re.sub(
                    r"\s+",
                    " ",
                    str(row.get("display_text") or row.get("text") or ""),
                )
            ]
        elif anchor.chunk_index is not None:
            rows = [
                row for row in rows if int(row.get("chunk_index", -1)) == anchor.chunk_index
            ]
        if not rows:
            missing += 1
            continue
        for row in rows:
            keys.add((path, int(row.get("chunk_index", 0))))
    return keys, missing


def evaluate(
    settings: Settings,
    table,
    embedder,
    reranker,
    queries: list[EvalQuery],
    configs: tuple[str, ...] = CONFIGS,
    top_k: int = 20,
    final_k: int = 10,
    manifest=None,
    root_path: str | None = None,
) -> dict:
    """对多种检索配置跑完整评测。

    Args:
        root_path: 可选；只评测该索引根（动态选根）。
    """
    root_id = None
    if root_path:
        if manifest is None:
            raise ValueError("root_path 需要 manifest")
        target = str(Path(root_path).expanduser().resolve())
        matches = [
            record
            for record in manifest.list_roots()
            if str(Path(record.path).resolve()) == target
        ]
        if not matches:
            raise ValueError(f"找不到索引根：{root_path}")
        root_id = matches[0].id
    service = SearchService(settings, table, embedder, reranker, manifest)
    try:
        embedder.warmup()
    except Exception:  # noqa: BLE001
        pass
    try:
        reranker.warmup()
    except Exception:  # noqa: BLE001
        pass
    results: dict[str, ConfigResult] = {}
    resolved_anchors: dict[str, set[tuple[str, int]]] = {}
    anchor_missing = 0
    for item in queries:
        if item.anchors:
            anchors = item.anchors
            if item.scope_doc_ids:
                scope = set(item.scope_doc_ids)
                anchors = [
                    anchor
                    for anchor in anchors
                    if make_doc_id(Path(anchor.path)) in scope
                ]
            if item.scope_file_paths:
                scope_paths = {
                    str(Path(path).expanduser().resolve())
                    for path in item.scope_file_paths
                }
                anchors = [
                    anchor
                    for anchor in anchors
                    if str(Path(anchor.path).expanduser().resolve()) in scope_paths
                ]
            keys, missing = _resolve_anchors(table, manifest, anchors)
            resolved_anchors[item.qid] = keys
            anchor_missing += missing

    for config in configs:
        if config not in CONFIGS:
            raise ValueError(f"未知配置：{config}")
        latencies: list[float] = []
        recalls5: list[float] = []
        recalls10: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        chunk_recalls5: list[float] = []
        chunk_recalls10: list[float] = []
        chunk_mrrs: list[float] = []
        chunk_ndcgs: list[float] = []
        chunk_queries = 0
        failures: list[dict] = []

        for item in queries:
            relevant_ids = item.relevant_ids or _doc_ids(item.relevant)
            scope_ids = set(item.scope_doc_ids)
            if item.scope_file_paths:
                scope_ids.update(_doc_ids(item.scope_file_paths))
            if scope_ids:
                relevant_ids = [doc_id for doc_id in relevant_ids if doc_id in scope_ids]
            relevant_set = set(relevant_ids)
            relevant_grades = {doc_id: 1 for doc_id in relevant_ids}
            result = None
            chunk_retrieved: list[tuple[str, int]] = []
            try:
                result = service.search(
                    item.query,
                    top_k=top_k,
                    final_k=max(final_k, top_k),
                    mode=config,
                    root_id=root_id,
                    doc_ids=item.scope_doc_ids or None,
                    file_paths=item.scope_file_paths or None,
                )
                latencies.append(float(result.took_ms))
                retrieved = _unique([hit.doc_id for hit in result.results])
                chunk_retrieved = _unique(
                    [
                        (str(Path(hit.file_path).expanduser().resolve()), hit.chunk_index)
                        for hit in result.results
                        if hit.chunk_index >= 0
                    ]
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"qid": item.qid, "query": item.query, "error": str(exc)})
                retrieved = []

            recalls5.append(recall_at_k(relevant_set, retrieved, 5))
            recalls10.append(recall_at_k(relevant_set, retrieved, 10))
            mrrs.append(mrr_at_k(relevant_set, retrieved, 10))
            ndcgs.append(ndcg_at_k(relevant_grades, retrieved, 10))
            chunk_relevant: set[tuple[str, int]] | None = None
            if item.anchors:
                chunk_relevant = resolved_anchors.get(item.qid, set())
            elif item.relevant_chunks:
                chunk_relevant = _chunk_keys(item.relevant_chunks)
            if chunk_relevant is not None:
                chunk_queries += 1
                chunk_recalls5.append(recall_at_k(chunk_relevant, chunk_retrieved, 5))
                chunk_recalls10.append(recall_at_k(chunk_relevant, chunk_retrieved, 10))
                chunk_mrrs.append(mrr_at_k(chunk_relevant, chunk_retrieved, 10))
                chunk_ndcgs.append(
                    ndcg_at_k({key: 1 for key in chunk_relevant}, chunk_retrieved, 10)
                )
            if not relevant_set & set(retrieved[:10]):
                failures.append(
                    {
                        "qid": item.qid,
                        "query": item.query,
                        "relevant": item.relevant,
                        "top": [hit.file_path for hit in result.results[:3]] if result else [],
                    }
                )

        results[config] = ConfigResult(
            config=config,
            recall5=float(np.mean(recalls5)) if recalls5 else 0.0,
            recall10=float(np.mean(recalls10)) if recalls10 else 0.0,
            mrr10=float(np.mean(mrrs)) if mrrs else 0.0,
            ndcg10=float(np.mean(ndcgs)) if ndcgs else 0.0,
            chunk_recall5=float(np.mean(chunk_recalls5)) if chunk_recalls5 else 0.0,
            chunk_recall10=float(np.mean(chunk_recalls10)) if chunk_recalls10 else 0.0,
            chunk_mrr10=float(np.mean(chunk_mrrs)) if chunk_mrrs else 0.0,
            chunk_ndcg10=float(np.mean(chunk_ndcgs)) if chunk_ndcgs else 0.0,
            chunk_queries=chunk_queries,
            p50_ms=float(statistics.median(latencies)) if latencies else 0.0,
            p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
            queries=len(queries),
            failures=failures[:10],
        )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "top_k": top_k,
        "final_k": final_k,
        "root_path": root_path,
        "anchor_missing": anchor_missing,
        "results": results,
    }


def write_report(payload: dict, settings: Settings, out: Path) -> None:
    """写 Markdown 评测报告。

    Args:
        payload: evaluate 的结果。
        settings: 配置（写入模型/数据目录）。
        out: 输出路径。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 检索评测报告",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 数据目录：`{settings.data_dir}`",
        f"- 模型：{settings.embed_model} / {settings.rerank_model}",
        f"- 切片：{settings.chunk_size}/{settings.chunk_overlap}；top_k={payload['top_k']}；final_k={payload['final_k']}",
        *(
            [f"- 索引根过滤：`{payload['root_path']}`"]
            if payload.get("root_path")
            else []
        ),
        *(
            [f"- Anchor 未解析：{payload['anchor_missing']}"]
            if payload.get("anchor_missing")
            else []
        ),
        "",
        "| 配置 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 | p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for config, item in payload["results"].items():
        lines.append(
            f"| {config} | {item.recall5:.4f} | {item.recall10:.4f} | {item.mrr10:.4f} | "
            f"{item.ndcg10:.4f} | {item.p50_ms:.0f}ms | {item.p95_ms:.0f}ms |"
        )
    if any(item.chunk_queries for item in payload["results"].values()):
        lines += [
            "",
            "## Chunk 级指标",
            "",
            "| 配置 | Chunk Recall@5 | Chunk Recall@10 | Chunk MRR@10 | Chunk nDCG@10 | Chunk queries |",
            "|---|---|---|---|---|---|",
        ]
        for config, item in payload["results"].items():
            lines.append(
                f"| {config} | {item.chunk_recall5:.4f} | {item.chunk_recall10:.4f} | "
                f"{item.chunk_mrr10:.4f} | {item.chunk_ndcg10:.4f} | {item.chunk_queries} |"
            )

    lines += ["", "## 失败案例（前 10）", ""]
    config, item = next(iter(payload["results"].items()))
    if not item.failures:
        lines.append("无。")
    else:
        for failure in item.failures:
            lines.append(f"- `{failure.get('qid')}` {failure.get('query')}：{failure}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
