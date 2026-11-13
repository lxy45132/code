"""GET /outputs/{job_id}：结果 PNG（job_id 不可变，长缓存）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.domain.services import validate_job_id
from hybrid_inpaint.interfaces.api.dependencies import get_container

router = APIRouter(tags=["outputs"])


@router.get("/outputs/{job_id}")
def get_output(job_id: str, container: Container = Depends(get_container)) -> Response:
    validate_job_id(job_id)
    data = container.artifacts.read_output(job_id)
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
