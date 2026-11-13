"""AppError → HTTP 状态码 + 统一错误体 {code, message, detail?}。

契约：docs/02 §12.1 错误映射；docs/03 §3。
UI 只展示 message；detail 是受控字符串（不含 traceback）。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from hybrid_inpaint.domain.errors import AppError

logger = logging.getLogger("hybrid_inpaint.api")

_HTTP_CODES = {
    401: "unauthorized",
    404: "not_found",
    405: "method_not_allowed",
    413: "resource_limit",
}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "app error code=%s status=%s message=%s detail=%s path=%s",
            exc.code,
            exc.http_status,
            exc.user_message,
            exc.detail,
            request.url.path,
        )
        body: dict = {"code": exc.code, "message": exc.user_message}
        if exc.detail:
            body["detail"] = exc.detail
        return JSONResponse(status_code=exc.http_status, content=body)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "request validation failed",
                "detail": jsonable_encoder(exc.errors()),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": _HTTP_CODES.get(exc.status_code, "http_error"),
                "message": str(exc.detail),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error path=%s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "message": "internal server error"},
        )
