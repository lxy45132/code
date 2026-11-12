from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from hybrid_inpaint.application.batch import BatchInpaint
from hybrid_inpaint.application.inpaint import InpaintImage
from hybrid_inpaint.domain.entities import BatchRequest, Image, Limits, Mask
from hybrid_inpaint.infrastructure.fs_batch import FileSystemBatchFiles
from tests.application.fakes import (
    CollectingReporter,
    FakeEngine,
    FixedClock,
    FlagToken,
    InMemoryArtifacts,
    InMemoryHistory,
    SeqIds,
)


class StubCodec:
    """最小 PIL 编解码，仅用于测批量编排；正式适配器见 PillowImageCodec。"""

    def decode(self, data: bytes, limits: Limits) -> Image:
        with PILImage.open(io.BytesIO(data)) as raw:
            return Image(np.asarray(raw.convert("RGB"), dtype=np.uint8))

    def decode_mask(self, data: bytes, *, height: int, width: int) -> Mask:
        with PILImage.open(io.BytesIO(data)) as raw:
            array = np.asarray(raw.convert("L"), dtype=np.uint8)
        mask = array > 0
        if mask.shape != (height, width):
            resized = PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
                (width, height), PILImage.Resampling.NEAREST
            )
            mask = np.asarray(resized) > 0
        return Mask(mask)

    def encode_png(self, image: Image) -> bytes:
        buf = io.BytesIO()
        PILImage.fromarray(image.data).save(buf, format="PNG")
        return buf.getvalue()

    def encode_mask_png(self, mask: Mask) -> bytes:
        buf = io.BytesIO()
        PILImage.fromarray(mask.to_uint8()).save(buf, format="PNG")
        return buf.getvalue()


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


def _write_mask(path: Path, hole: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((8, 8), dtype=np.uint8)
    if hole:
        array[2:6, 2:6] = 255
    PILImage.fromarray(array).save(path)


def _setup(tmp_path: Path):
    engine = FakeEngine()
    history = InMemoryHistory()
    artifacts = InMemoryArtifacts(tmp_path / "artifacts")
    ids = SeqIds()
    inpaint = InpaintImage(engine, artifacts, history, FixedClock(), ids, Limits())
    batch = BatchInpaint(inpaint, StubCodec(), FileSystemBatchFiles(), ids, Limits())
    return batch, history


def _dirs(tmp_path: Path) -> BatchRequest:
    inputs, masks, outputs = tmp_path / "in", tmp_path / "mask", tmp_path / "out"
    _write_image(inputs / "a.png", 10)
    _write_mask(masks / "a.png")
    _write_image(inputs / "b.png", 20)  # 缺掩码
    return BatchRequest(inputs, masks, outputs)


def test_batch_ok_and_missing_mask(tmp_path: Path) -> None:
    batch, history = _setup(tmp_path)
    reporter = CollectingReporter()

    report = batch.execute(_dirs(tmp_path), reporter=reporter)

    statuses = {r.relative_path: r.status for r in report.results}
    assert statuses == {"a.png": "ok", "b.png": "missing_mask"}
    assert report.ok == 1 and report.failed == 1
    assert (tmp_path / "out" / "a.png").is_file()
    assert reporter.total == 2 and len(reporter.items) == 2
    assert len(history.records) == 1
    record = next(iter(history.records.values()))
    assert record.batch_id == report.batch_id


def test_skip_existing_and_overwrite(tmp_path: Path) -> None:
    batch, _ = _setup(tmp_path)
    request = _dirs(tmp_path)
    batch.execute(request)

    report = batch.execute(request)
    assert {r.relative_path: r.status for r in report.results}["a.png"] == "skipped_exists"

    report = batch.execute(
        BatchRequest(
            request.input_dir, request.mask_dir, request.output_dir, overwrite=True
        )
    )
    assert {r.relative_path: r.status for r in report.results}["a.png"] == "ok"


def test_all_hole_mask_is_item_error_not_batch_crash(tmp_path: Path) -> None:
    batch, _ = _setup(tmp_path)
    request = _dirs(tmp_path)
    _write_mask(request.mask_dir / "b.png", hole=False)  # 全黑=零洞 -> ok
    # a.png 换成全洞掩码 -> 用例拒绝，但整批不中断
    PILImage.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(
        request.mask_dir / "a.png"
    )

    report = batch.execute(request)

    statuses = {r.relative_path: r.status for r in report.results}
    assert statuses["a.png"] == "error"
    assert statuses["b.png"] == "ok"
    assert report.results[0].error_code == "validation_error"


def test_limit_and_cancellation(tmp_path: Path) -> None:
    batch, _ = _setup(tmp_path)
    request = _dirs(tmp_path)

    report = batch.execute(
        BatchRequest(request.input_dir, request.mask_dir, request.output_dir, limit=1)
    )
    assert report.total == 1

    report = batch.execute(request, token=FlagToken(cancel_after=1))
    assert report.cancelled is True
    assert report.total == 1