"""GET /models：活动模型信息（单活动模型，无热切换，docs/02 §9）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.interfaces.api.caching import get_models_cache, set_models_cache
from hybrid_inpaint.interfaces.api.dependencies import get_container
from hybrid_inpaint.interfaces.api.schemas.models import ModelInfo

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelInfo)
def models(container: Container = Depends(get_container)) -> ModelInfo:
    cached = get_models_cache(container)
    if cached is not None:
        return ModelInfo.model_validate_json(cached)
    info = ModelInfo(**container.model_service.health())
    set_models_cache(container, info.model_dump_json())
    return info
