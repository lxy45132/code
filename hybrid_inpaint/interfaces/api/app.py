"""create_app(container)：FastAPI 应用工厂（docs/03 §3）。

lifespan 管理异步 Redis 连接（SSE 用，每连接独立协程不占线程池）；
引擎预热已在 build_container 完成（inline）或只在 worker 加载（worker 模式）。
前端 dist 存在时同源托管（IOPaint 模式），不存在时跳过。
"""

from __future__ import annotations

import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.interfaces.api.celery_app import celery_app
from hybrid_inpaint.interfaces.api.dependencies import require_api_key
from hybrid_inpaint.interfaces.api.errors import register_error_handlers
from hybrid_inpaint.interfaces.api.middleware import RequestIDMiddleware
from hybrid_inpaint.interfaces.api.routers import (
    batches,
    health,
    history,
    inpaint,
    models,
    outputs,
    tasks,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Windows 注册表可能把 .js 映射为 text/plain，导致浏览器拒绝执行模块脚本。
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")


def create_app(container: Container) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.aredis = aioredis.from_url(
            container.config.system.redis.url, decode_responses=True
        )
        try:
            yield
        finally:
            await app.state.aredis.aclose()

    api_config = container.config.system.api
    app = FastAPI(title=api_config.title, version="1.0.0", lifespan=lifespan)
    app.state.container = container
    app.state.celery = celery_app

    app.add_middleware(RequestIDMiddleware)
    if api_config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(api_config.cors_origins),
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    register_error_handlers(app)

    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(health.router)  # 探针免鉴权
    protected = APIRouter(dependencies=[Depends(require_api_key)])
    for module in (models, inpaint, outputs, history, batches, tasks):
        protected.include_router(module.router)
    v1.include_router(protected)
    app.include_router(v1)

    dist = _PROJECT_ROOT / "frontend" / "dist"
    if dist.is_dir():
        # IOPaint 模式：同源托管 dist；Vue Router history 模式需 SPA fallback。
        app.mount(
            "/assets", StaticFiles(directory=dist / "assets"), name="assets"
        )

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="not found")
            candidate = (dist / full_path).resolve()
            if full_path and candidate.is_file() and candidate.is_relative_to(dist):
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app
