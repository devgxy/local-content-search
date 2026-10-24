"""资料库路由：列表、详情、原始文件预览、打开/重建。

列表只做 SQL 分页；详情按 doc_id 直查 chunks；预览只接受清单中的 doc_id。

Step: STEP-08
Doc: docs/steps/STEP-08-API层.md
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, Response

from app.api.errors import AppError
from app.ids import escape_sql_literal
from app.ingest import IngestService
from app.jobs import BusyError
from app.manifest import FileRecord
from app.models import (
    ChunkDTO,
    DocumentDetail,
    LibraryItem,
    LibraryResponse,
    OpenRequest,
    ReindexRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["library"])


def _to_item(record: FileRecord) -> LibraryItem:
    """把 Manifest FileRecord 转成资料库 DTO。"""
    return LibraryItem(
        doc_id=record.doc_id,
        file_name=Path(record.path).name,
        file_path=record.path,
        file_type=record.file_type,
        size=record.size,
        mtime=record.mtime,
        chunks=record.chunk_count,
        indexed_at=record.last_indexed_at,
        status=record.status,
        last_error=record.last_error,
    )


def _under_roots(path: Path, manifest) -> bool:  # noqa: ANN001
    """判断路径是否位于启用根下。"""
    text = str(path)
    return any(
        root.enabled and (text == root.path or text.startswith(root.path + "/"))
        for root in manifest.list_roots()
    )


def _allowed_path(raw: str, manifest) -> Path:  # noqa: ANN001
    """校验并返回可操作的普通文件路径。

    Args:
        raw: 用户传入路径。
        manifest: 清单。

    Returns:
        解析后的绝对路径。

    Raises:
        AppError: 非普通文件/不在索引范围。
    """
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise AppError("INVALID_PATH", f"不是可打开的普通文件：{path}", status=400)
    if manifest.get_file(str(path)) is not None:
        return path
    if _under_roots(path, manifest):
        return path
    raise AppError("INVALID_PATH", f"路径不在索引范围内：{path}", status=400)


@router.get("/api/library", response_model=LibraryResponse)
def library(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    file_type: str | None = None,
    q: str | None = None,
) -> LibraryResponse:
    """资料库分页列表，支持类型/路径过滤。"""
    state = request.app.state
    records, total = state.manifest.list_files_page(
        page,
        page_size,
        file_type=file_type,
        q=q,
    )
    items = [_to_item(item) for item in records]
    return LibraryResponse(total=total, page=page, page_size=page_size, items=items)


@router.get("/api/documents/{doc_id}", response_model=DocumentDetail)
def document(doc_id: str, request: Request) -> DocumentDetail:
    """读取单文档元数据与全部 chunk。"""
    state = request.app.state
    record = state.manifest.get_file_by_doc_id(doc_id)
    if record is None:
        raise AppError("DOC_NOT_FOUND", "文档不存在", status=404)
    if not record.run_id:
        raise AppError("DOC_NOT_READY", "文档没有已提交版本", status=409)
    try:
        rows = (
            state.table.search()
            .where(f"doc_id = {escape_sql_literal(doc_id)}")
            .select(["id", "chunk_index", "text", "display_text", "heading", "page", "slide", "source_kind", "run_id"])
            .limit(max(state.table.count_rows(), 10000))
            .to_list()
        )
    except Exception as exc:  # noqa: BLE001
        raise AppError("DB_NOT_READY", f"读取 chunks 失败：{exc}", status=500) from exc
    rows = [row for row in rows if row.get("run_id") == record.run_id]
    rows.sort(key=lambda item: item["chunk_index"])
    chunks = [
        ChunkDTO(
            chunk_id=row["id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            display_text=row["display_text"],
            heading=row.get("heading", ""),
            page=row.get("page", 0),
            slide=row.get("slide", 0),
            source_kind=row.get("source_kind", "text"),
        )
        for row in rows
    ]
    return DocumentDetail(item=_to_item(record), chunks=chunks)


@router.get("/api/documents/{doc_id}/preview")
def document_preview(doc_id: str, request: Request, page: int = Query(default=1, ge=1)) -> Response:
    """PDF 页面渲染为 PNG，避免浏览器 PDF 插件不可用导致空白。"""
    state = request.app.state
    record = state.manifest.get_file_by_doc_id(doc_id)
    if record is None:
        raise AppError("DOC_NOT_FOUND", "文档不存在", status=404)
    path = Path(record.path)
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise AppError("PREVIEW_UNAVAILABLE", "该文件不支持页面预览", status=400)
    try:
        import pymupdf

        with pymupdf.open(path) as document:
            if page > document.page_count:
                page = document.page_count
            pixmap = document[page - 1].get_pixmap(dpi=110, alpha=False)
            logger.debug("PDF 页面预览 doc_id=%s page=%d", doc_id, page)
            return Response(content=pixmap.tobytes("png"), media_type="image/png")
    except Exception as exc:  # noqa: BLE001
        raise AppError("PREVIEW_FAILED", f"页面渲染失败：{exc}", status=500) from exc


@router.get("/api/documents/{doc_id}/file")
def document_file(doc_id: str, request: Request) -> FileResponse:
    """只读预览原始文件；路径来自清单，不接受任意路径。"""
    state = request.app.state
    record = state.manifest.get_file_by_doc_id(doc_id)
    if record is None:
        raise AppError("DOC_NOT_FOUND", "文档不存在", status=404)
    path = Path(record.path)
    if not path.is_file():
        raise AppError("FILE_NOT_FOUND", "原始文件不存在", status=404)
    suffix = path.suffix.lower()
    # 浏览器同源执行 HTML/SVG 会带来本地 API 风险；只对可安全内联的
    # PDF/图片 inline，其余一律 attachment + nosniff。
    if suffix in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return FileResponse(
            path,
            filename=path.name,
            content_disposition_type="inline",
            headers={"X-Content-Type-Options": "nosniff"},
        )
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=path.name,
        content_disposition_type="attachment",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.post("/api/open")
def open_path(req: OpenRequest, request: Request) -> dict:
    """用系统默认应用打开/在 Finder 显示文件。"""
    state = request.app.state
    path = _allowed_path(req.path, state.manifest)
    if path.suffix.lower() in {".command", ".app"}:
        raise AppError("INVALID_PATH", "出于安全原因不打开可执行包/脚本", status=400)
    command = ["open", "-R", str(path)] if req.reveal else ["open", str(path)]
    # 不要让 IDE/编辑器继承本项目的 cwd：中文路径经 LaunchServices
    # 传给 JetBrains 时可能被错误解码，IDE 会报 cannot set current directory。
    env = os.environ.copy()
    home = str(Path.home())
    env["PWD"] = home
    logger.info("打开原文件 path=%s reveal=%s", path, req.reveal)
    try:
        subprocess.run(command, check=True, timeout=5, cwd=home, env=env)
    except Exception as exc:  # noqa: BLE001
        raise AppError("OPEN_FAILED", f"无法打开文件：{exc}", status=500) from exc
    return {"ok": True}


@router.post("/api/reindex", status_code=202)
def reindex(req: ReindexRequest, request: Request) -> dict:
    """提交单文件强制重建任务。"""
    state = request.app.state
    path = _allowed_path(req.path, state.manifest)
    service = IngestService(
        state.settings,
        state.table,
        state.manifest,
        state.embedder,
        require_roots=True,
        asset_table=state.asset_table,
        image_embedder=state.image_embedder,
    )

    def run(progress, cancel):  # noqa: ANN001
        """重建任务执行体。"""
        result = service.ingest_paths(
            [path],
            progress=progress,
            cancel=cancel,
            force=True,
        )
        return result.to_dict()

    try:
        job_id = state.jobs.submit_if_idle(
            "reindex",
            run,
            blocked_kinds={"sync", "reindex"},
        )
    except BusyError as exc:
        raise AppError("BUSY", str(exc), status=409) from exc
    logger.info("提交重建 job_id=%s path=%s", job_id, path)
    return {"job_id": job_id}
