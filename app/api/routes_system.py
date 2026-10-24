"""系统状态路由。"""

from __future__ import annotations

import logging
import os
import signal
import time

from fastapi import APIRouter, BackgroundTasks, Header, Request

from app import __version__
from app.db import db_stats, health_check
from app.models import HealthResponse, StatsResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


def _shutdown_process() -> None:
    """给响应留出发送时间，再交给 uvicorn 的 SIGTERM 优雅退出。"""
    time.sleep(0.4)
    os.kill(os.getpid(), signal.SIGTERM)


@router.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """健康检查：DB、FTS、模型状态。

    Args:
        request: FastAPI 请求。

    Returns:
        HealthResponse；FTS 缺失时 status=degraded。
    """
    state = request.app.state
    db_health = health_check(state.table)
    models = {
        "embedder": state.embedder.health(),
        "reranker": state.reranker.health(),
    }
    status = "ok" if db_health["fts_index"] else "degraded"
    offline = os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
    return HealthResponse(
        status=status,
        version=__version__,
        models=models,
        db=db_health,
        offline=offline,
    )


@router.post("/api/shutdown")
def shutdown(
    background: BackgroundTasks,
    request: Request,
    x_local_action: str = Header(default="", alias="X-Local-Action"),
) -> dict:
    """本地请求关闭服务，响应后发送 SIGTERM。"""
    if x_local_action != "1":
        from app.api.errors import AppError

        raise AppError("FORBIDDEN", "缺少本地操作标记", status=403)
    callback = getattr(request.app.state, "shutdown_callback", None) or _shutdown_process
    logger.info("收到 Web 关闭服务请求")
    background.add_task(callback)
    return {"ok": True, "message": "服务正在关闭"}


@router.get("/api/stats", response_model=StatsResponse)
def stats(request: Request) -> StatsResponse:
    """清单与数据库统计。

    Args:
        request: FastAPI 请求。

    Returns:
        StatsResponse。
    """
    state = request.app.state
    return StatsResponse(
        manifest=state.manifest.stats(),
        db=db_stats(state.settings, state.table),
    )
