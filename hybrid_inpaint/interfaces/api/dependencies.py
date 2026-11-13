"""FastAPI Depends 注入：Container 取自 app.state（create_app 装配）。"""

from __future__ import annotations

from fastapi import HTTPException, Request

from hybrid_inpaint.bootstrap import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


def require_api_key(request: Request) -> None:
    """api.api_key 非空时启用 X-API-Key 校验（公网部署预留，docs/03 §10）。"""
    expected = get_container(request).config.system.api.api_key
    if expected is not None and request.headers.get("X-API-Key") != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
