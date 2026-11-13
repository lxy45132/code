"""模型信息与健康探针 DTO。"""

from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    model_id: str | None
    model_version: str | None
    ready: bool
    device: str
    last_error: str | None


class HealthResponse(BaseModel):
    status: str
    model: ModelInfo
    storage_backend: str
    inference_mode: str
