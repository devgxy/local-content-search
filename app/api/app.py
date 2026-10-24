"""FastAPI 应用工厂。

Step: STEP-08
Doc: docs/steps/STEP-08-API层.md §4
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import routes_library, routes_search, routes_sync, routes_system
from app.api.errors import AppError, app_error_handler, generic_error_handler
from app.config import Settings, load_settings
from app.db import connect_db, open_table
from app.embeddings import EmbeddingService
from app.image_index import ImageEmbeddingService, open_asset_table
from app.jobs import JobManager
from app.manifest import Manifest
from app.preflight import run_preflight
from app.reranker import RerankerService

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    preflight: bool = True,
    embedder=None,
    reranker=None,
    table=None,
    manifest: Manifest | None = None,
    jobs: JobManager | None = None,
    shutdown_callback=None,
) -> FastAPI:
    """创建 FastAPI 应用并挂载路由/静态资源。

    Args:
        embedder:
        settings: 可选配置；默认环境变量。
        preflight: 是否执行启动自检。
        embedder/reranker/table/manifest/jobs: 测试注入点。
        shutdown_callback: 测试用关闭回调；None 时向自身发 SIGTERM。

    Returns:
        配置完成的 FastAPI 实例。
    """
    settings = settings or load_settings()
    app = FastAPI(title="本地内容检索", version=__version__)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期：自检 → 打开 DB/模型 → 预热 → yield → 关闭。"""
        if preflight:
            report = run_preflight(settings)
            if not report.ok:
                for item in report.errors:
                    logger.error("自检失败：%s - %s", item.name, item.message)
                raise RuntimeError("启动自检失败")

        settings.ensure_dirs()
        store = manifest or Manifest(settings.manifest_path)
        emb = embedder or EmbeddingService(settings)
        rer = reranker or RerankerService(settings)
        manager = jobs or JobManager()

        db = None
        asset_table = None
        image_embedder = None
        if table is None:
            db = connect_db(settings)
            current_table = open_table(db, settings, emb.func)
            if settings.embed_provider == "gemma" and settings.image_embed_enabled:
                asset_table = open_asset_table(db, settings)
                image_embedder = ImageEmbeddingService(settings)
        else:
            current_table = table

        app.state.settings = settings
        app.state.manifest = store
        app.state.embedder = emb
        app.state.reranker = rer
        app.state.table = current_table
        app.state.asset_table = asset_table
        app.state.image_embedder = image_embedder
        app.state.jobs = manager
        app.state.db = db
        app.state.shutdown_callback = shutdown_callback

        stale = store.recover_indexing()
        if stale:
            logger.warning("发现 %d 个中断文件，已标记为 error", len(stale))

        if preflight:
            # 同步预热：auto 全局可能在健康检查后立即使用 reranker；
            # 后台预热会和首个请求并发初始化 MPS，曾导致进程退出。
            for name, service in (("embedder", emb), ("reranker", rer)):
                try:
                    service.warmup()
                except Exception:  # noqa: BLE001
                    logger.warning("%s 预热失败", name, exc_info=True)

        logger.info("服务启动完成 http://%s:%s", settings.host, settings.port)
        yield
        logger.info("服务开始关闭，取消/等待后台任务")

        manager.shutdown()
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass

    logger.info("创建应用 host=%s port=%s embed=%s", settings.host, settings.port, settings.embed_model)
    app.router.lifespan_context = lifespan
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)

    app.include_router(routes_system.router)
    app.include_router(routes_search.router)
    app.include_router(routes_library.router)
    app.include_router(routes_sync.router)

    if settings.web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            """返回 Web UI 首页。"""
            return FileResponse(settings.web_dir / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def index_missing() -> dict:
            """Web 静态资源缺失时的兜底响应。"""
            return {"name": "本地内容检索", "docs": "/docs", "status": "web assets missing"}

    return app
