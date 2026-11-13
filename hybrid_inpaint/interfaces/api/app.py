"""create_app(container)：FastAPI 应用工厂（docs/03 §3）。

lifespan 目前只做占位（引擎预热已在 build_container 完成）；
Redis/Celery 探针与资源释放在 C7d 接入。前端 dist 存在时同源托管
（IOPaint 模式），不存在时跳过（开发期走 Vite dev server + CORS）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.interfaces.api.dependencies import require_api_key
from hybrid_inpaint.interfaces.api.errors import register_error_handlers
from hybrid_inpaint.interfaces.api.middleware import RequestIDMiddleware
from hybrid_inpaint.interfaces.api.routers import (
    health,
    history,
    inpaint,
    models,
    outputs,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def create_app(container: Container) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    api_config = container.config.system.api
    app = FastAPI(title=api_config.title, version="1.0.0", lifespan=lifespan)
    app.state.container = container

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
    for module in (models, inpaint, outputs, history):
        protected.include_router(module.router)
    v1.include_router(protected)
    app.include_router(v1)

    dist = _PROJECT_ROOT / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app
