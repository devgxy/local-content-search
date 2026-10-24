"""混合检索服务：BM25 + 向量 + RRF + 重排 + 上下文 + 摘要。

主流程：
1. 校验 query/mode/profile，fast 档位强制 hybrid；
2. 按 file_type/root_id 构造 where，FTS 模式跳过 query 编码；
3. LanceDB 返回候选，按 manifest 当前 run_id 过滤旧版本；
4. full/hybrid_rerank 做 CrossEncoder 重排；
5. per_doc_limit 去重、同 section 扩展、抽取摘要、追加图片命中。

Step: STEP-07
Doc: docs/steps/STEP-07-检索与重排.md
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from lancedb.rerankers import RRFReranker
from lancedb.table import LanceTable

from app.config import Settings
from app.ids import escape_sql_literal

logger = logging.getLogger(__name__)

ALLOWED_FILE_TYPES = {"md", "txt", "pdf", "docx", "pptx", "html", "json", "csv", "image"}
SELECT_FIELDS = [
    "id",
    "doc_id",
    "chunk_index",
    "text",
    "display_text",
    "file_name",
    "file_path",
    "file_type",
    "title",
    "heading",
    "page",
    "slide",
    "source_kind",
    "meta_json",
    "run_id",
]


class SearchError(RuntimeError):
    """可映射到 API 错误码的检索异常。

    Args:
        code: 机器可读错误码，如 ``INVALID_QUERY``。
        message: 给用户看的中文说明。
    """

    def __init__(self, code: str, message: str):
        """初始化异常。

        Args:
            code: 机器可读错误码。
            message: 中文错误说明。
        """
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class SearchHit:
    """单条检索结果。

    Attributes:
        chunk_id: chunk 主键。
        doc_id: 文档主键。
        chunk_index: 文档内 chunk 序号；图片为 -1。
        score: 当前模式的主分数（重排/RRF/BM25/距离）。
        rrf_score: RRF 原始分，仅 hybrid/RRF 有意义。
        rerank_score: CrossEncoder 0..1 分，未重排为 None。
        rank_before: 重排前名次，从 1 开始。
        rank_after: 最终展示名次，从 1 开始。
        file_name: 文件名（不含目录）。
        file_path: 绝对路径。
        file_type: md/txt/pdf/docx/pptx/html/image。
        title: 文档标题。
        heading: 命中所在标题路径。
        snippet: section 级上下文，供“命中分块”预览。
        summary: 抽取式摘要，供结果卡片默认展示。
        page: PDF 页码，1 起；无页码为 0。
        slide: PPTX 页码，1 起；无页码为 0。
        source_kind: text/ocr/image 等来源。
        score_kind: rerank/rrf/bm25/distance/image。
    """

    chunk_id: str
    doc_id: str
    chunk_index: int
    score: float
    rrf_score: float
    rerank_score: float | None
    rank_before: int
    rank_after: int
    file_name: str
    file_path: str
    file_type: str
    title: str
    heading: str
    snippet: str
    summary: str
    page: int
    slide: int
    source_kind: str
    score_kind: str = "rrf"
    table_html: str = ""
    table_confidence: float | None = None


@dataclass
class SearchResult:
    """一次检索的聚合结果。

    Attributes:
        query: 规范化后的查询。
        took_ms: 端到端耗时，毫秒。
        candidates: 文本候选 + 图片命中数量。
        results: 最终结果，按 rank_after 排序。
        warnings: 可恢复警告（解析/数据提示）；重排失败不再降级。
        timings: query/search/rerank/post 分段时间。
    """

    query: str
    took_ms: int
    candidates: int
    results: list[SearchHit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def _highlight_terms(query: str) -> list[str]:
    """生成用于高亮和摘要打分的关键词集合。

    Args:
        query: 原始查询，支持中英文混合。

    Returns:
        去重后的词/2-3 字中文 n-gram，最多 20 个；长词优先。
    """
    terms: set[str] = set()
    for word in re.findall(r"[A-Za-z0-9_]+", query):
        if len(word) >= 2:
            terms.add(word.lower())
    chinese = re.sub(r"[^\u4e00-\u9fff]+", " ", query)
    for chunk in chinese.split():
        if len(chunk) >= 2:
            terms.add(chunk)
        for size in (2, 3):
            for index in range(len(chunk) - size + 1):
                terms.add(chunk[index : index + size])
    ordered = sorted(terms, key=lambda item: (-len(item), item))
    return ordered[:20]


class SearchService:
    """混合检索门面。

    负责：查询编码、LanceDB 检索、run_id 过滤、重排、去重、上下文、
    摘要、图片命中合并。不负责 HTTP 序列化（routes_search）和 UI。

    Attributes:
        settings: 运行期配置。
        table: chunks LanceDB 表。
        embedder: 查询编码器。
        reranker: CrossEncoder 重排器。
        manifest: 可选清单，用于按当前 run_id 过滤。
        asset_table: 可选图片向量表。
        image_embedder: 可选图片编码器（检索阶段只用作能力标记）。
    """

    def __init__(
        self,
        settings: Settings,
        table: LanceTable,
        embedder,
        reranker,
        manifest=None,
        asset_table=None,
        image_embedder=None,
    ):
        """保存依赖；构造阶段不加载模型、不访问网络。

        Args:
            settings: 运行期配置。
            table: chunks 表。
            embedder: 查询编码器。
            reranker: 重排器。
            manifest: 可选 Manifest，用于 run_id 过滤。
            asset_table: 可选图片 asset 表。
            image_embedder: 可选图片编码器。
        """
        self.settings = settings
        self.table = table
        self.embedder = embedder
        self.reranker = reranker
        self.manifest = manifest
        self.asset_table = asset_table
        self.image_embedder = image_embedder

    def search(
        self,
        query: str,
        top_k: int | None = None,
        final_k: int | None = None,
        file_types: list[str] | None = None,
        root_id: int | None = None,
        doc_ids: list[str] | None = None,
        file_paths: list[str] | None = None,
        mode: str = "full",
        profile: str | None = None,
        doc_level: bool = False,
    ) -> SearchResult:
        """执行一次检索。

        Args:
            query: 查询文本；空/超过 500 字会拒绝。
            top_k: 初筛数量；默认 settings.top_k。
            final_k: 返回数量；不能大于 top_k。
            file_types: 可选类型过滤。
            root_id: 可选索引根过滤。
            doc_ids: 可选 doc_id 白名单（动态选文件）。
            file_paths: 可选绝对路径白名单（动态选文件）。
            mode: auto/full/hybrid/hybrid_rerank/fts/vector；auto 无 scope 用 full，
                有 doc_ids/file_paths/root_id 时用 hybrid。
            profile: fast/accurate；fast 会关闭重排并把 top_k 限制到 10。

        Returns:
            SearchResult，包含文本命中、图片命中、摘要、上下文和耗时。

        Raises:
            SearchError: 参数非法、检索失败。
        """
        started = time.perf_counter()
        query = (query or "").strip()
        if not query:
            raise SearchError("INVALID_QUERY", "查询不能为空")
        if len(query) > 500:
            raise SearchError("INVALID_QUERY", "查询过长（最多 500 字）")
        if mode not in {"auto", "vector", "fts", "hybrid", "hybrid_rerank", "full"}:
            raise SearchError("INVALID_QUERY", f"未知检索模式：{mode}")
        # 全局检索启用重排；选定文件/根后优先低延迟 hybrid。
        if mode == "auto":
            scoped = bool(doc_ids or file_paths) or root_id is not None
            mode = "hybrid" if scoped else "full"
        # 只有显式 doc_level 或 full 才做文档级限流；hybrid 默认保留
        # chunk 级结果。单文件 scope 的 full 也不强制限流。
        single_file_scope = (
            root_id is None
            and (len(doc_ids or []) + len(file_paths or [])) == 1
        )
        # hybrid 默认保留 chunk 级结果；只有显式 doc_level 或 full 才限流。
        apply_doc_limit = doc_level or (mode == "full" and not single_file_scope)
        profile = profile or self.settings.search_profile
        if profile not in {"fast", "accurate"}:
            raise SearchError("INVALID_QUERY", f"未知检索档位：{profile}")
        if profile == "fast" and mode in {"full", "hybrid", "hybrid_rerank"}:
            mode = "hybrid"
            top_k = min(top_k or self.settings.top_k, 10)

        top_k = top_k or self.settings.top_k
        final_k = final_k or self.settings.final_k
        final_k = min(final_k, top_k)
        where = self._build_where(file_types, root_id, doc_ids, file_paths)
        warnings: list[str] = []

        # FTS 模式不需要 query 向量，避免无谓模型调用。
        t0 = time.perf_counter()
        qvec = None if mode == "fts" else self.embedder.encode_query(query)
        query_ms = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        candidates = self._retrieve(query, qvec, where, top_k, mode)
        # 入库期间可能同时存在新旧 run_id；只保留清单当前版本。
        candidates = self._filter_current(candidates)
        # 版本过滤后立即限制候选，再做上下文和重排；不能把全表有效行送 rerank。
        candidates = candidates[:top_k]
        search_ms = int((time.perf_counter() - t1) * 1000)

        t2 = time.perf_counter()
        rerank_scores = None
        rerank_contexts = None
        if mode in {"hybrid_rerank", "full"} and candidates:
            try:
                from app.reranker import RerankerUnavailable

                rerank_contexts = self._context_maps(
                    [item["doc_id"] for item in candidates]
                )
                rerank_texts = [
                    self._rerank_text(item, rerank_contexts, query)
                    for item in candidates
                ]
                rerank_scores = self.reranker.score(query, rerank_texts)
            except RerankerUnavailable as exc:
                # 显式失败：不允许静默退回 RRF 冒充精排结果。
                raise SearchError("RERANK_FAILED", f"重排失败：{exc}") from exc
        rerank_ms = int((time.perf_counter() - t2) * 1000)

        t3 = time.perf_counter()
        hits = self._postprocess(
            candidates,
            rerank_scores,
            final_k,
            mode,
            query,
            contexts=rerank_contexts,
            apply_doc_limit=True if doc_level else apply_doc_limit,
            doc_level=doc_level,
        )
        image_hits = self._image_hits(
            query,
            final_k,
            hits,
            file_types,
            root_id,
            doc_ids,
            file_paths,
        )
        hits.extend(image_hits)
        post_ms = int((time.perf_counter() - t3) * 1000)

        logger.info(
            "检索完成 query_len=%d mode=%s profile=%s text=%d image=%d total_ms=%d",
            len(query),
            mode,
            profile,
            len(hits) - len(image_hits),
            len(image_hits),
            int((time.perf_counter() - started) * 1000),
        )
        return SearchResult(
            query=query,
            took_ms=int((time.perf_counter() - started) * 1000),
            candidates=len(candidates) + len(image_hits),
            results=hits,
            warnings=warnings,
            timings={
                "query_ms": query_ms,
                "search_ms": search_ms,
                "rerank_ms": rerank_ms,
                "post_ms": post_ms,
            },
        )

    def _image_hits(
        self,
        query: str,
        final_k: int,
        text_hits: list[SearchHit],
        file_types: list[str] | None,
        root_id: int | None,
        doc_ids: list[str] | None = None,
        file_paths: list[str] | None = None,
    ) -> list[SearchHit]:
        """检索图片 asset 表并转换为 SearchHit。

        Args:
            query: 查询文本，用 Gemma 文本塔编码到与图片同一空间。
            final_k: 最多返回的图片数。
            text_hits: 已有文本命中，用于按文件路径去重。
            file_types: 类型过滤；不含 image 时直接跳过。
            root_id: 可选索引根过滤。

        Returns:
            图片命中列表；asset 未启用或编码失败时返回空列表。
        """
        if self.asset_table is None or self.image_embedder is None:
            return []
        if file_types and "image" not in file_types:
            return []
        try:
            qvec = self.embedder.encode_query(query)
            builder = self.asset_table.search(qvec)
            where = self._build_where(file_types, root_id, doc_ids, file_paths)
            if where:
                builder = builder.where(where, prefilter=True)
            # 版本过滤依赖 manifest；先取全量候选，过滤后再截断。
            # 本地规模可接受；大规模时应改为 current-run 预过滤索引。
            rows = builder.limit(max(self.asset_table.count_rows(), final_k)).to_list()
        except Exception as exc:  # noqa: BLE001
            raise SearchError("IMAGE_SEARCH_FAILED", f"图片检索失败：{exc}") from exc
        if self.manifest is not None:
            filtered: list[dict] = []
            for row in rows:
                record = self.manifest.get_file_by_doc_id(str(row.get("doc_id", "")))
                if record is None or not record.run_id:
                    continue
                if row.get("run_id") != record.run_id:
                    continue
                filtered.append(row)
            rows = filtered
        existing = {hit.file_path for hit in text_hits}
        hits: list[SearchHit] = []
        for row in rows:
            if row.get("file_path") in existing:
                continue
            distance = float(row.get("_distance", 1.0))
            hits.append(
                SearchHit(
                    chunk_id=str(row["id"]),
                    doc_id=str(row["doc_id"]),
                    chunk_index=-1,
                    score=1.0 / (1.0 + distance),
                    rrf_score=0.0,
                    rerank_score=None,
                    rank_before=len(hits) + 1,
                    rank_after=len(text_hits) + len(hits) + 1,
                    file_name=str(row.get("file_name", "")),
                    file_path=str(row.get("file_path", "")),
                    file_type="image",
                    title=str(row.get("title", "")),
                    heading="",
                    snippet=str(row.get("text", ""))[: self.settings.context_chars],
                    summary=self._summarize(str(row.get("text", "")), query, str(row.get("text", ""))),
                    page=int(row.get("page", 0)),
                    slide=int(row.get("slide", 0)),
                    source_kind="image",
                    score_kind="image",
                )
            )
            if len(hits) >= final_k:
                break
        return hits

    def _filter_current(self, candidates: list[dict]) -> list[dict]:
        """只返回清单当前 run_id 对应的 chunk，避免入库过程中读到新旧混合版本。

        Args:
            candidates: LanceDB 返回的行，必须包含 doc_id/run_id。

        Returns:
            过滤后的行；manifest 记录没有 run_id 时保留该行。
        """
        if self.manifest is None or not candidates:
            return candidates
        current: dict[str, str] = {}
        for doc_id in dict.fromkeys(item["doc_id"] for item in candidates):
            record = self.manifest.get_file_by_doc_id(doc_id)
            # manifest 存在时，只有明确当前 run 的行才可见；
            # 不接受“没有记录/没有 run”的未提交行。
            if record is not None and record.run_id:
                current[doc_id] = record.run_id
        return [
            item
            for item in candidates
            if item.get("run_id") == current.get(item["doc_id"])
        ]

    def _build_where(
        self,
        file_types: list[str] | None,
        root_id: int | None,
        doc_ids: list[str] | None = None,
        file_paths: list[str] | None = None,
    ) -> str | None:
        """构造 LanceDB SQL 过滤条件。

        Args:
            file_types: 允许的文件类型；非法值会被忽略。
            root_id: 索引根 ID。
            doc_ids: 可选 doc_id 白名单。
            file_paths: 可选文件绝对路径白名单。

        Returns:
            形如 ``file_type IN (...) AND root_id = N`` 的字符串；
            无过滤条件返回 None。
        """
        clauses = []
        if file_types:
            cleaned = [item for item in file_types if item in ALLOWED_FILE_TYPES]
            if cleaned:
                values = ", ".join(escape_sql_literal(item) for item in cleaned)
                clauses.append(f"file_type IN ({values})")
        if root_id is not None:
            clauses.append(f"root_id = {int(root_id)}")
        if doc_ids:
            unique = list(dict.fromkeys(str(item) for item in doc_ids if item))
            if unique:
                values = ", ".join(escape_sql_literal(item) for item in unique[:1000])
                clauses.append(f"doc_id IN ({values})")
        if file_paths:
            normalized = list(
                dict.fromkeys(str(Path(item).expanduser().resolve()) for item in file_paths if item)
            )
            if normalized:
                values = ", ".join(escape_sql_literal(item) for item in normalized[:1000])
                clauses.append(f"file_path IN ({values})")
        return " AND ".join(clauses) if clauses else None

    def _retrieve(
        self,
        query: str,
        qvec,
        where: str | None,
        top_k: int,
        mode: str,
    ) -> list[dict]:
        """执行底层 LanceDB 检索。

        Args:
            query: 原始查询，FTS 使用。
            qvec: 查询向量；FTS 模式为 None。
            where: 可选 SQL 过滤。
            top_k: 初筛数量。
            mode: vector/fts/hybrid/hybrid_rerank/full。

        Returns:
            LanceDB 行字典列表；失败时抛 SearchError。
        """
        if mode == "vector":
            builder = self.table.search(qvec, query_type="vector")
        elif mode == "fts":
            builder = self.table.search(query, query_type="fts")
        else:
            builder = (
                self.table.search(query_type="hybrid")
                .vector(qvec)
                .text(query)
                .rerank(RRFReranker())
            )
        if where:
            builder = builder.where(where, prefilter=True)
        # 版本过滤在检索之后；先取全量候选，过滤后再交给 top_k 截断。
        # 本地规模可接受；大规模时应改为 current-run 预过滤索引。
        total = self.table.count_rows()
        builder = builder.select(SELECT_FIELDS).limit(max(top_k, total))
        try:
            return builder.to_list()
        except Exception as exc:  # noqa: BLE001
            raise SearchError("SEARCH_FAILED", f"检索失败：{exc}") from exc

    def _metadata_boost(self, query: str, candidate: dict) -> float:
        """给查询中显式出现的文件名/路径一个小的排序加成。

        只匹配完整文件名或完整 stem，避免“简历/文档”等泛词误伤。
        动态文件场景下，用户经常直接输入文件名或路径片段。
        """
        normalized = query.casefold().strip()
        if len(normalized) < 4:
            return 0.0
        file_name = str(candidate.get("file_name") or "").casefold()
        stem = Path(file_name).stem
        if len(stem) >= 4 and stem in normalized:
            return 0.05
        if len(file_name) >= 4 and file_name in normalized:
            return 0.05
        file_path = str(candidate.get("file_path") or "").casefold()
        for part in re.split(r"[/\\]", file_path):
            if len(part) >= 4 and part in normalized:
                return 0.05
        return 0.0

    def _postprocess(
        self,
        candidates: list[dict],
        rerank_scores,
        final_k: int,
        mode: str,
        query: str,
        contexts: dict[str, dict[int, dict]] | None = None,
        apply_doc_limit: bool = True,
        doc_level: bool = False,
    ) -> list[SearchHit]:
        """重排/去重/上下文/摘要，生成最终文本命中。

        Args:
            candidates: 经过 run_id 过滤的候选。
            rerank_scores: 可选的 CrossEncoder 分数，长度与 candidates 一致。
            final_k: 最终返回数。
            mode: 检索模式，full 才应用 per_doc_limit。
            query: 用于摘要打分。

        Returns:
            SearchHit 列表，按最终名次排序。
        """
        if not candidates:
            return []

        for candidate in candidates:
            candidate["_metadata_boost"] = self._metadata_boost(query, candidate)
        if rerank_scores is not None:
            for index, (candidate, score) in enumerate(zip(candidates, rerank_scores, strict=True)):
                candidate["_rerank_score"] = float(score)
                candidate["_rank_before"] = index + 1
            order = sorted(
                range(len(candidates)),
                key=lambda i: -(
                    candidates[i]["_rerank_score"]
                    + candidates[i].get("_metadata_boost", 0.0)
                ),
            )
        else:
            for index, candidate in enumerate(candidates):
                candidate["_rerank_score"] = None
                candidate["_rank_before"] = index + 1
            order = sorted(
                range(len(candidates)),
                key=lambda i: (
                    -candidates[i].get("_metadata_boost", 0.0),
                    i,
                ),
            )

        selected: list[dict] = []
        overflow: list[dict] = []
        per_doc: dict[str, int] = {}
        doc_limit = 1 if doc_level else self.settings.per_doc_limit
        for index in order:
            candidate = candidates[index]
            doc_id = candidate["doc_id"]
            if (
                apply_doc_limit
                and per_doc.get(doc_id, 0) >= doc_limit
            ):
                overflow.append(candidate)
                continue
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            selected.append(candidate)
            if len(selected) >= final_k:
                break

        if len(selected) < final_k and not doc_level:
            relaxed = max(self.settings.per_doc_limit, 3)
            for candidate in overflow:
                doc_id = candidate["doc_id"]
                if per_doc.get(doc_id, 0) >= relaxed:
                    continue
                if candidate not in selected:
                    per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
                    selected.append(candidate)
                if len(selected) >= final_k:
                    break

        if contexts is None:
            contexts = self._context_maps([item["doc_id"] for item in selected])
        hits: list[SearchHit] = []
        for rank, candidate in enumerate(selected, start=1):
            if candidate.get("_relevance_score") is not None:
                base_score = float(candidate["_relevance_score"])
                score_kind = "rrf"
            elif candidate.get("_score") is not None:
                base_score = float(candidate["_score"])
                score_kind = "bm25"
            elif candidate.get("_distance") is not None:
                base_score = 1.0 / (1.0 + float(candidate["_distance"]))
                score_kind = "distance"
            else:
                base_score = 0.0
                score_kind = "none"
            rerank_score = candidate.get("_rerank_score")
            snippet = self._build_snippet(candidate, contexts, query)
            summary = self._summarize(candidate.get("display_text", ""), query, snippet)
            table_html = ""
            table_confidence = None
            try:
                meta = json.loads(candidate.get("meta_json") or "{}")
                chunk_meta = meta.get("chunk_meta") if isinstance(meta, dict) else None
                if candidate.get("source_kind") == "table" and isinstance(chunk_meta, dict):
                    raw_html = chunk_meta.get("html")
                    if isinstance(raw_html, str) and raw_html.lstrip().startswith("<table"):
                        table_html = raw_html
                    raw_confidence = chunk_meta.get("confidence")
                    if isinstance(raw_confidence, (int, float)) and math.isfinite(float(raw_confidence)):
                        value = float(raw_confidence)
                        if 0.0 <= value <= 1.0:
                            table_confidence = value
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            hits.append(
                SearchHit(
                    chunk_id=candidate["id"],
                    doc_id=candidate["doc_id"],
                    chunk_index=int(candidate.get("chunk_index") or 0),
                    score=float(rerank_score) if rerank_score is not None else base_score,
                    rrf_score=base_score,
                    rerank_score=float(rerank_score) if rerank_score is not None else None,
                    rank_before=int(candidate.get("_rank_before") or rank),
                    rank_after=rank,
                    file_name=candidate["file_name"],
                    file_path=candidate["file_path"],
                    file_type=candidate["file_type"],
                    title=candidate.get("title", ""),
                    heading=candidate.get("heading", ""),
                    snippet=snippet,
                    summary=summary,
                    page=int(candidate.get("page") or 0),
                    slide=int(candidate.get("slide") or 0),
                    source_kind=candidate.get("source_kind", "text"),
                    score_kind="rerank" if rerank_score is not None else score_kind,
                    table_html=table_html,
                    table_confidence=table_confidence,
                )
            )
        return hits

    def _rerank_text(
        self,
        candidate: dict,
        contexts: dict[str, dict[int, dict]],
        query: str,
    ) -> str:
        """构造重排输入：同 section parent 上下文。

        默认开启；关闭时退回单 chunk text，便于 A/B。
        """
        if not self.settings.rerank_parent_context:
            return str(candidate["text"])
        doc_map = contexts.get(candidate["doc_id"], {})
        parent = [
            item
            for item in doc_map.values()
            if item["heading"] == candidate.get("heading", "")
            and item["page"] == int(candidate.get("page") or 0)
            and item["slide"] == int(candidate.get("slide") or 0)
        ]
        if len(parent) <= 1:
            return str(candidate["text"])
        parent.sort(key=lambda item: item.get("chunk_index", 0))
        body = "\n".join(item["text"] for item in parent)
        body = self._window_around_match(body, query)
        heading = str(candidate.get("heading") or "")
        return f"{heading}\n{body}" if heading else body

    def _context_maps(self, doc_ids: list[str]) -> dict[str, dict[int, str]]:
        """批量读取命中文档的 chunk，避免 N+1 查询。

        Args:
            doc_ids: 文档 ID 列表（可重复）。

        Returns:
            doc_id -> chunk_index -> {text, chunk_index, heading, page, slide}。
        """
        unique = list(dict.fromkeys(doc_ids))
        if not unique:
            return {}
        values = ", ".join(escape_sql_literal(doc_id) for doc_id in unique)
        maps: dict[str, dict[int, str]] = {doc_id: {} for doc_id in unique}
        try:
            rows = (
                self.table.search()
                .where(f"doc_id IN ({values})")
                .select([
                    "doc_id", "chunk_index", "display_text", "heading",
                    "page", "slide", "run_id",
                ])
                .limit(max(self.table.count_rows(), 50000))
                .to_list()
            )
        except Exception as exc:  # noqa: BLE001
            raise SearchError("CONTEXT_FAILED", f"读取上下文失败：{exc}") from exc
        for row in rows:
            if self.manifest is not None:
                record = self.manifest.get_file_by_doc_id(str(row["doc_id"]))
                if record is None or not record.run_id:
                    continue
                if row.get("run_id") != record.run_id:
                    continue
            maps.setdefault(row["doc_id"], {})[int(row["chunk_index"])] = {
                "text": row["display_text"],
                "chunk_index": int(row["chunk_index"]),
                "heading": row.get("heading", ""),
                "page": row.get("page", 0),
                "slide": row.get("slide", 0),
            }
        return maps

    def _build_snippet(
        self,
        candidate: dict,
        contexts: dict[str, dict[int, dict]],
        query: str,
    ) -> str:
        """构造命中上下文。

        优先取同一 heading/page/slide 的整段；没有 section 信息时回退到
        前后相邻 chunk，并在超长时围绕查询词截窗。

        Args:
            candidate: 当前候选行。
            contexts: doc_id -> chunk_index -> 上下文行。
            query: 原始查询。

        Returns:
            最多 context_chars 字的上下文。
        """
        current = candidate["display_text"]
        doc_map = contexts.get(candidate["doc_id"], {})
        index = int(candidate["chunk_index"])
        # P1：用同一 section（heading/page/slide）的整段作为 parent 上下文。
        parent = [
            item
            for item in doc_map.values()
            if item["heading"] == candidate.get("heading", "")
            and item["page"] == int(candidate.get("page") or 0)
            and item["slide"] == int(candidate.get("slide") or 0)
        ]
        if parent:
            parent.sort(key=lambda item: item.get("chunk_index", 0))
            parts = [item["text"] for item in parent]
            return self._window_around_match("\n".join(parts), query)
        budget = max(self.settings.context_chars - len(current), 0)
        if budget <= 0:
            return current[: self.settings.context_chars]
        half = budget // 2
        previous = doc_map.get(index - 1, {}).get("text", "") if doc_map else ""
        following = doc_map.get(index + 1, {}).get("text", "") if doc_map else ""
        if previous and half:
            previous = previous[-half:]
        if following and half:
            following = following[:half]
        parts = [part for part in (previous, current, following) if part]
        return self._window_around_match("\n".join(parts), query)

    def _summarize(self, text: str, query: str, fallback: str) -> str:
        """抽取式摘要：优先包含查询词的句子，避免直接铺满 chunk。

        Args:
            text: 命中 chunk 正文。
            query: 原始查询，用于句子打分。
            fallback: text 为空时的兜底文本。

        Returns:
            去除 Markdown 标记后、最多约 260 字的摘要。
        """
        raw = text or fallback or ""
        raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", raw)
        raw = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", raw)
        raw = re.sub(r"[*_`#>|]+", " ", raw)
        clean = " ".join(raw.split())
        if not clean:
            return ""
        parts = [part.strip(" -·") for part in re.split(r"(?<=[。！？!?；;])|\n+", clean) if part.strip(" -·")]
        if not parts:
            return clean[:240]
        terms = _highlight_terms(query)
        scored: list[tuple[int, int, str]] = []
        for index, part in enumerate(parts):
            hits = sum(1 for term in terms if term in part)
            if hits:
                scored.append((hits, index, part))
        if not scored:
            scored = [(0, index, part) for index, part in enumerate(parts[:2])]
        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = sorted(scored[:2], key=lambda item: item[1])
        summary = " ".join(item[2] for item in chosen)
        return summary[:260] + ("…" if len(summary) > 260 else "")

    def _window_around_match(self, text: str, query: str) -> str:
        """围绕第一个命中词截取上下文窗口。

        Args:
            text: 完整 section 文本。
            query: 原始查询。

        Returns:
            最多 context_chars 字；命中位置前保留约 1/4 窗口。
        """
        limit = self.settings.context_chars
        if len(text) <= limit:
            return text
        positions = []
        if len(query) >= 2:
            positions.append(text.find(query))
        for term in _highlight_terms(query):
            positions.append(text.find(term))
        positions = [pos for pos in positions if pos >= 0]
        if not positions:
            return text[:limit]
        position = min(positions)
        start = max(0, position - limit // 4)
        end = min(len(text), start + limit)
        if end - start < limit:
            start = max(0, end - limit)
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(text) else ""
        return f"{prefix}{text[start:end]}{suffix}"

    @staticmethod
    def highlight_terms(query: str) -> list[str]:
        """兼容旧调用：返回查询高亮词。

        Args:
            query: 原始查询。

        Returns:
            高亮词列表。
        """
        return _highlight_terms(query)
