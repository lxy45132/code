"""FileSystemModelRepository: registered local checkpoints only."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from hybrid_inpaint.domain.entities import ModelSpec
from hybrid_inpaint.domain.errors import ModelUnavailableError, NotFoundError
from hybrid_inpaint.domain.ports import InpaintEngine
from hybrid_inpaint.infrastructure.torch_engine import TorchInpaintEngine


class FileSystemModelRepository:
    def __init__(
        self,
        specs: Sequence[ModelSpec],
        *,
        device: str = "cuda",
        pad_multiple: int = 16,
        tile_size: int = 512,
        tile_overlap: int = 64,
    ) -> None:
        self._specs = {spec.model_id: spec for spec in specs}
        self._device = device
        self._pad_multiple = pad_multiple
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap

    def resolve(self, model_id: str) -> ModelSpec:
        spec = self._specs.get(model_id)
        if spec is None:
            raise NotFoundError(f"unknown model_id: {model_id!r}")
        if not Path(spec.checkpoint).is_file():
            raise ModelUnavailableError(f"checkpoint not found: {spec.checkpoint}")
        return spec

    def load(self, spec: ModelSpec) -> InpaintEngine:
        checkpoint = Path(spec.checkpoint)
        if spec.sha256 is not None:
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != spec.sha256:
                raise ModelUnavailableError(
                    f"checkpoint sha256 mismatch for {spec.model_id!r}"
                )
        return TorchInpaintEngine(
            checkpoint,
            device=self._device,
            pad_multiple=self._pad_multiple,
            tile_size=self._tile_size,
            tile_overlap=self._tile_overlap,
        )