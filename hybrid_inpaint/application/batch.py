"""BatchInpaint: pair files, run per-item, isolate failures, report."""

from __future__ import annotations

from pathlib import Path

from hybrid_inpaint.application.inpaint import InpaintImage
from hybrid_inpaint.domain.entities import (
    BatchItem,
    BatchReport,
    BatchRequest,
    BatchResult,
    InpaintRequest,
    Limits,
)
from hybrid_inpaint.domain.errors import AppError
from hybrid_inpaint.domain.ports import (
    BatchFileStore,
    CancellationToken,
    IdGenerator,
    ImageCodec,
    ProgressReporter,
)
from hybrid_inpaint.domain.services import pair_files


class BatchInpaint:
    def __init__(
        self,
        inpaint: InpaintImage,
        codec: ImageCodec,
        files: BatchFileStore,
        ids: IdGenerator,
        limits: Limits,
    ) -> None:
        self._inpaint = inpaint
        self._codec = codec
        self._files = files
        self._ids = ids
        self._limits = limits

    def execute(
        self,
        request: BatchRequest,
        reporter: ProgressReporter | None = None,
        token: CancellationToken | None = None,
    ) -> BatchReport:
        pairs = pair_files(
            request.input_dir, request.mask_dir, recursive=request.recursive
        )
        if request.limit is not None:
            pairs = pairs[: request.limit]
        items = [
            BatchItem(
                image_path.relative_to(request.input_dir).as_posix(),
                image_path,
                mask_path,
            )
            for image_path, mask_path in pairs
        ]
        batch_id = self._ids.new_id()
        results: list[BatchResult] = []
        cancelled = False
        if reporter is not None:
            reporter.start(len(items))
        for index, item in enumerate(items):
            if token is not None and token.cancelled:
                cancelled = True
                break
            result = self._process(request, item, batch_id)
            results.append(result)
            if reporter is not None:
                reporter.item(index, item, result)
        report = BatchReport(batch_id, tuple(results), cancelled)
        if reporter is not None:
            reporter.finish(report)
        return report

    def _process(
        self, request: BatchRequest, item: BatchItem, batch_id: str
    ) -> BatchResult:
        relative = item.relative_path
        mask_str = str(item.mask_path) if item.mask_path is not None else None

        def fail(status: str, code: str, message: str) -> BatchResult:
            return BatchResult(relative, mask_str, status, error_code=code, error=message)  # type: ignore[arg-type]

        if item.mask_path is None:
            return fail("missing_mask", "not_found", "no matching mask file")
        output_path = request.output_dir / Path(relative).with_suffix(".png")
        if not request.overwrite and self._files.exists(output_path):
            return BatchResult(
                relative, mask_str, "skipped_exists", output_path=str(output_path)
            )
        try:
            image = self._codec.decode(self._files.read(item.image_path), self._limits)
        except AppError as exc:
            return fail("invalid_image", exc.code, exc.user_message)
        try:
            mask = self._codec.decode_mask(
                self._files.read(item.mask_path),
                height=image.height,
                width=image.width,
            )
        except AppError as exc:
            return fail("invalid_mask", exc.code, exc.user_message)
        try:
            result = self._inpaint.execute(
                InpaintRequest(image, mask, mode="cli", batch_id=batch_id)
            )
            self._files.write(output_path, self._codec.encode_png(result.image))
        except AppError as exc:
            return fail("error", exc.code, exc.user_message)
        return BatchResult(
            relative,
            mask_str,
            "ok",
            output_path=str(output_path),
            latency_ms=result.latency_ms,
            mask_ratio=result.mask_ratio,
            job_id=result.job_id,
        )