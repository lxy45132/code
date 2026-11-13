"""GET /models：活动模型信息（单活动模型，无热切换，docs/02 §9）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.interfaces.api.dependencies import get_container
from hybrid_inpaint.interfaces.api.schemas.models import ModelInfo

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelInfo)
def models(container: Container = Depends(get_container)) -> ModelInfo:
    return ModelInfo(**container.model_service.health())
