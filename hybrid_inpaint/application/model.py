"""ModelService: single active model, load once, no hot swap."""

from __future__ import annotations

from typing import Any

from hybrid_inpaint.domain.errors import (
    AppError,
    InternalError,
    ModelUnavailableError,
)
from hybrid_inpaint.domain.ports import InpaintEngine, ModelRepository


class ModelService:
    def __init__(
        self,
        repository: ModelRepository | None = None,
        *,
        engine: InpaintEngine | None = None,
        model_id: str | None = None,
    ) -> None:
        self._repository = repository
        self._engine = engine
        self._model_id = model_id
        self._last_error: str | None = None

    def load(self, model_id: str) -> InpaintEngine:
        if self._engine is not None:
            if model_id == self._model_id:
                return self._engine
            raise InternalError(
                "a different model is already loaded; hot swap is not supported"
            )
        if self._repository is None:
            raise InternalError("no model repository configured")
        try:
            spec = self._repository.resolve(model_id)
            engine = self._repository.load(spec)
            engine.warmup()
        except AppError as exc:
            self._last_error = exc.user_message
            raise
        except Exception as exc:
            self._last_error = str(exc)
            raise ModelUnavailableError(
                f"cannot load model {model_id!r}", detail=str(exc)
            ) from exc
        self._engine = engine
        self._model_id = model_id
        self._last_error = None
        return engine

    def health(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "model_version": self._engine.version if self._engine else None,
            "ready": self._engine is not None,
            "device": getattr(self._engine, "device", "unknown")
            if self._engine
            else "unknown",
            "last_error": self._last_error,
        }