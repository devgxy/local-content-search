"""搜索路由。"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request

from app.api.errors import AppError
from app.models import SearchHitDTO, SearchRequest, SearchResponse
from app.search import SearchError, SearchService

router = APIRouter(tags=["search"])


@router.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest, request: Request) -> SearchResponse:
    """执行检索并转换为 API 响应。

    Args:
        req: 查询/过滤/模式/档位。
        request: FastAPI 请求，读取 app.state 服务。

    Returns:
        SearchResponse。

    Raises:
        AppError: 参数/检索错误映射为 4xx/5xx。
    """
    state = request.app.state
    service = SearchService(
        state.settings,
        state.table,
        state.embedder,
        state.reranker,
        state.manifest,
        state.asset_table,
        state.image_embedder,
    )
    try:
        result = service.search(
            req.query,
            top_k=req.top_k,
            final_k=req.final_k,
            file_types=req.file_types or None,
            root_id=req.root_id,
            doc_ids=req.doc_ids or None,
            file_paths=req.file_paths or None,
            profile=req.profile,
            mode=req.mode or "auto",
            doc_level=req.doc_level,
        )
    except SearchError as exc:
        raise AppError(exc.code, exc.message, status=400) from exc
    except Exception as exc:  # noqa: BLE001
        raise AppError("SEARCH_FAILED", f"检索失败：{exc}", status=500) from exc

    return SearchResponse(
        query=result.query,
        took_ms=result.took_ms,
        candidates=result.candidates,
        results=[SearchHitDTO(**asdict(hit)) for hit in result.results],
        warnings=result.warnings,
        timings=result.timings,
        highlight_terms=SearchService.highlight_terms(req.query),
    )
