"""POST /inpaint 响应 DTO（docs/02 §9 响应口径）。"""

from __future__ import annotations

from pydantic import BaseModel


class InpaintResponse(BaseModel):
    job_id: str
    output_url: str
    latency_ms: float
    model_version: str
    mask_ratio: float
    input_size: tuple[int, int]  # (width, height)
    warnings: list[str] = []
