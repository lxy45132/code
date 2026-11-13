"""GET /health：模型/存储/推理模式探针（免鉴权，供部署 healthcheck）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.interfaces.api.dependencies import get_container
from hybrid_inpaint.interfaces.api.schemas.models import HealthResponse, ModelInfo

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(container: Container = Depends(get_container)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=ModelInfo(**container.model_service.health()),
        storage_backend=container.config.system.storage.backend,
        inference_mode=container.config.system.inference.mode,
    )
