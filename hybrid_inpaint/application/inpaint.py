"""InpaintImage use case: validate -> infer -> composite -> persist -> history.

只依赖 domain 端口；不 import torch/PIL/sqlite3，不做直接文件 I/O。
"""

from __future__ import annotations

import time
from collections.abc import Sequence

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
        self._validate(request)
        started = time.perf_counter()
        if request.mask.ratio == 0.0:
            generated = request.image.data.astype(np.float32) / 255.0
            warnings: tuple[str, ...] = ()
        else:
            generated = self._engine.inpaint(request.image.data, request.mask.data)
            warnings = tuple(self._engine.last_warnings)
        return self._persist(request, generated, warnings, started)

    def execute_many(self, requests: Sequence[InpaintRequest]) -> list[InpaintResult]:
        """批量用例：同尺寸项经 engine.inpaint_many 凑批（03 文档 §4）。

        逐项管线不变量（合成后置校验/历史/artifacts）与 execute 一致；
        零洞项走恒等分支不进引擎。latency 含凑批等待，口径为整批耗时。
        """
        for request in requests:
            self._validate(request)
        started = time.perf_counter()
        results: list[InpaintResult | None] = [None] * len(requests)
        active: list[int] = []
        for index, request in enumerate(requests):
            if request.mask.ratio == 0.0:
                generated = request.image.data.astype(np.float32) / 255.0
                results[index] = self._persist(request, generated, (), started)
            else:
                active.append(index)
        if active:
            generated_list = self._engine.inpaint_many(
                [requests[i].image.data for i in active],
                [requests[i].mask.data for i in active],
            )
            warnings = tuple(self._engine.last_warnings)
            for index, generated in zip(active, generated_list):
                results[index] = self._persist(requests[index], generated, warnings, started)
        return [result for result in results if result is not None]

    def _validate(self, request: InpaintRequest) -> None:
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

    def _persist(
        self,
        request: InpaintRequest,
        generated: np.ndarray,
        warnings: tuple[str, ...],
        started: float,
    ) -> InpaintResult:
        image, mask = request.image, request.mask
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