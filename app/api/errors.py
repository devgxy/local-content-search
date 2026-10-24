"""API 统一错误。

Step: STEP-08
Doc: docs/steps/STEP-08-API层.md §7
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """可序列化为统一错误结构的应用异常。"""
    def __init__(self, code: str, message: str, hint: str = "", status: int = 400):
        """初始化 AppError。

        Args:
            code: 机器可读错误码。
            message: 用户可读说明。
            status: HTTP 状态码。
            hint: 可选修复建议。
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.status = status


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """AppError 统一 JSON 处理器。"""
    return JSONResponse(
        status_code=exc.status,
        content={
            "detail": {
                "code": exc.code,
                "message": exc.message,
                "hint": exc.hint,
            }
        },
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """未捕获异常兜底：记录日志并返回 500。"""
    logger.exception("未处理异常：%s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "code": "INTERNAL_ERROR",
                "message": "服务内部错误",
                "hint": "查看 data/logs/app.log",
            }
        },
    )
