"""InpaintImage use case: validate -> infer -> composite -> persist -> history.

只依赖 domain 端口；不 import torch/PIL/sqlite3，不做直接文件 I/O。
"""

from __future__ import annotations

import time

import numpy as np

from hybrid_inpaint.domain.entities import (
    HistoryRecord,
    Image,
    InpaintRequest,
    InpaintResult,
    Limits,
)
from hybrid_inpaint.domain.errors import (
    InternalError,
    ResourceLimitError,
    ValidationError,
)
from hybrid_inpaint.domain.ports import (
    ArtifactStore,
    Clock,
    HistoryRepository,
    IdGenerator,
    InpaintEngine,
)
from hybrid_inpaint.domain.services import composite, validate_mask


class InpaintImage:
    def __init__(
        self,
        engine: InpaintEngine,
        artifacts: ArtifactStore,
        history: HistoryRepository,
        clock: Clock,
        ids: IdGenerator,
        limits: Limits,
    ) -> None:
        self._engine = engine
        self._artifacts = artifacts
        self._history = history
        self._clock = clock
        self._ids = ids
        self._limits = limits

    def execute(self, request: InpaintRequest) -> InpaintResult:
        image, mask = request.image, request.mask
        if image.data.shape[:2] != mask.data.shape:
            raise ValidationError("image and mask sizes differ")
        if max(image.height, image.width) > self._limits.max_side:
            raise ResourceLimitError("image side exceeds the configured maximum")
        if image.height * image.width > self._limits.max_pixels:
            raise ResourceLimitError("image exceeds the configured pixel budget")
        try:
            validate_mask(mask, allow_empty=True)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        started = time.perf_counter()
        warnings: tuple[str, ...] = ()
        if mask.ratio == 0.0:
            generated = image.data.astype(np.float32) / 255.0
        else:
            generated = self._engine.inpaint(image.data, mask.data)
            warnings = tuple(self._engine.last_warnings)
        output = Image(composite(image.data, mask.data, generated))
        latency_ms = (time.perf_counter() - started) * 1000.0

        job_id = self._ids.new_id()
        paths = self._artifacts.save(job_id, image, mask, output)
        record = HistoryRecord(
            job_id=job_id,
            created_at=self._clock.now(),
            mode=request.mode,
            batch_id=request.batch_id,
            input_path=str(paths.input),
            mask_path=str(paths.mask),
            output_path=str(paths.output),
            model_version=self._engine.version,
            width=image.width,
            height=image.height,
            mask_ratio=mask.ratio,
            latency_ms=latency_ms,
            note=request.note or None,
        )
        try:
            self._history.add(record)
        except Exception as exc:
            self._artifacts.delete(job_id)
            raise InternalError("failed to write history", detail=str(exc)) from exc
        return InpaintResult(
            job_id=job_id,
            image=output,
            latency_ms=latency_ms,
            model_version=self._engine.version,
            mask_ratio=mask.ratio,
            warnings=warnings,
        )