"""索引根、扫描、同步与任务路由。

流程：添加/选择根 → 扫描 diff → 用户勾选 → 提交 sync/delete Job。
所有写操作都通过 JobManager 串行，路径在进入 IngestService 前做根校验。

Step: STEP-08
Doc: docs/steps/STEP-08-API层.md
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, Query, Request

from app.api.errors import AppError
from app.ingest import IngestService
from app.jobs import BusyError, JobCancelled
from app.models import RootCreate, RootDTO, ScanRequest, SyncRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sync"])


def _root_dto(record) -> RootDTO:  # noqa: ANN001
    """把 Manifest RootRecord 转成 API DTO。

    Args:
        record: RootRecord。

    Returns:
        RootDTO。
    """
    return RootDTO(
        id=record.id,
        path=record.path,
        enabled=record.enabled,
        created_at=record.created_at,
        doc_count=record.doc_count,
    )


def _under_roots(path: Path, manifest) -> bool:  # noqa: ANN001
    """判断路径是否位于任一启用索引根下。"""
    text = str(path)
    return any(
        root.enabled and (text == root.path or text.startswith(root.path + "/"))
        for root in manifest.list_roots()
    )


@router.post("/api/pick-folder")
def pick_folder() -> dict:
    """调用 macOS 原生目录选择框；仅本地服务使用。

    Returns:
        {"path": 绝对路径}。

    Raises:
        AppError: 取消/无 GUI/目录不存在。
    """
    script = (
        'set chosen to choose folder with prompt "选择要加入索引的文件夹"\n'
        "return POSIX path of chosen"
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise AppError("PICK_FAILED", f"无法打开目录选择框：{exc}", status=500) from exc
    if result.returncode != 0:
        message = (result.stderr or "").strip()
        if "User canceled" in message or "-128" in message:
            raise AppError("CANCELLED", "已取消选择", status=400)
        raise AppError("PICK_FAILED", message or "目录选择失败", status=500)
    path = Path(result.stdout.strip()).expanduser()
    logger.info("原生选择文件夹 path=%s", path)
    if not path.is_dir():
        raise AppError("INVALID_PATH", f"目录不存在：{path}", status=400)
    return {"path": str(path.resolve())}


@router.get("/api/roots", response_model=list[RootDTO])
def list_roots(request: Request) -> list[RootDTO]:
    """列出索引根。"""
    state = request.app.state
    return [_root_dto(item) for item in state.manifest.list_roots()]


@router.post("/api/roots", response_model=RootDTO, status_code=201)
def add_root(req: RootCreate, request: Request) -> RootDTO:
    """添加索引根。"""
    state = request.app.state
    path = Path(req.path).expanduser().resolve()
    if not path.is_dir():
        raise AppError("INVALID_PATH", f"目录不存在：{path}", status=400)
    record = state.manifest.add_root(path)
    logger.info("添加索引根 root_id=%s path=%s", record.id, record.path)
    return _root_dto(record)


@router.delete("/api/roots/{root_id}")
def remove_root(
    root_id: int,
    request: Request,
    delete_files: bool = Query(default=False),
) -> dict:
    """停用或移除索引根并清理索引。"""
    state = request.app.state
    paths = state.manifest.remove_root(root_id, delete_files=delete_files)
    if not delete_files:
        return {"removed": True, "paths": 0, "job_id": None}
    if not paths:
        state.manifest.drop_root(root_id)
        return {"removed": True, "paths": 0, "job_id": None}

    service = IngestService(
        state.settings,
        state.table,
        state.manifest,
        state.embedder,
        asset_table=state.asset_table,
        image_embedder=state.image_embedder,
    )

    def run(progress, cancel):  # noqa: ANN001
        """扫描/同步任务执行体。"""
        try:
            result = service.delete_paths(paths, progress=progress, cancel=cancel)
            if result.get("failed", 0) == 0:
                state.manifest.drop_root(root_id)
            return result
        except JobCancelled:
            raise

    job_id = state.jobs.submit("delete", run)
    logger.info("移除索引根 root_id=%s job_id=%s", root_id, job_id)
    return {"removed": True, "paths": len(paths), "job_id": job_id}


@router.post("/api/scan", status_code=202)
def scan(req: ScanRequest, request: Request) -> dict:
    """提交扫描任务。"""
    state = request.app.state

    def run(progress, cancel):  # noqa: ANN001
        """扫描/同步任务执行体。"""
        diff = state.manifest.scan(
            state.settings,
            root_ids=req.root_ids,
            cancel=cancel,
        )
        return diff.to_dict()

    try:
        job_id = state.jobs.submit_if_idle("scan", run)
    except BusyError as exc:
        raise AppError("BUSY", str(exc), status=409) from exc
    logger.info("提交扫描 job_id=%s", job_id)
    return {"job_id": job_id}


@router.post("/api/sync", status_code=202)
def sync(req: SyncRequest, request: Request) -> dict:
    """提交同步/删除任务。"""
    state = request.app.state
    paths = [Path(item).expanduser().resolve() for item in req.paths]
    # 删除路径不 resolve，保留清单中的符号链接身份。
    deleted = [Path(os.path.abspath(Path(item).expanduser())) for item in req.deleted_paths]

    overlap = {str(item) for item in paths} & {str(item) for item in deleted}
    if overlap:
        raise AppError(
            "INVALID_REQUEST",
            f"同一路径不能同时出现在同步与删除列表：{sorted(overlap)[0]}",
            status=400,
        )

    for path in paths:
        if state.manifest.get_file(str(path)) is None and not _under_roots(path, state.manifest):
            raise AppError("INVALID_PATH", f"路径不在索引范围内：{path}", status=400)
    for path in deleted:
        if not _under_roots(path, state.manifest) and state.manifest.get_file(str(path)) is None:
            raise AppError("INVALID_PATH", f"删除路径不在索引范围内：{path}", status=400)

    if not paths and not (req.delete_missing and deleted):
        raise AppError("INVALID_REQUEST", "没有要同步的文件", status=400)

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
        """扫描/同步任务执行体。"""
        payload = service.ingest_paths(paths, progress=progress, cancel=cancel).to_dict()
        if req.delete_missing and deleted:
            payload["deleted"] = service.delete_paths(
                deleted,
                progress=progress,
                cancel=cancel,
                only_missing=True,
            )
        return payload

    try:
        job_id = state.jobs.submit_if_idle(
            "sync",
            run,
            blocked_kinds={"sync", "reindex"},
        )
    except BusyError as exc:
        raise AppError("BUSY", str(exc), status=409) from exc
    logger.info("提交同步 job_id=%s paths=%d deleted=%d", job_id, len(paths), len(deleted))
    return {"job_id": job_id}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    """查询任务状态。"""
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise AppError("JOB_NOT_FOUND", "任务不存在", status=404)
    return job.to_dict()


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict:
    """请求取消任务。"""
    ok = request.app.state.jobs.cancel(job_id)
    if not ok:
        raise AppError("JOB_NOT_FOUND", "任务不存在或已结束", status=404)
    return {"cancelled": True}
