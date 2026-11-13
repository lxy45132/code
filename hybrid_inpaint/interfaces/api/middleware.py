"""RequestID + 访问日志中间件（纯 ASGI 实现）。

不用 BaseHTTPMiddleware：它会包装响应流，干扰 SSE（C7d）的
断开检测与逐事件推送。纯 ASGI 只在 response.start 时追加头与记日志。
"""

from __future__ import annotations

import logging
import re
import time
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("hybrid_inpaint.api.access")

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RequestIDMiddleware:
    """生成/回写 X-Request-ID，并记录 method/path/status/耗时。"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get("x-request-id", "")
        request_id = (
            incoming if _REQUEST_ID_PATTERN.fullmatch(incoming) else uuid4().hex
        )
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append("X-Request-ID", request_id)
                logger.info(
                    "%s %s -> %s %.1fms request_id=%s",
                    scope["method"],
                    scope["path"],
                    message["status"],
                    (time.perf_counter() - started) * 1000.0,
                    request_id,
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)
