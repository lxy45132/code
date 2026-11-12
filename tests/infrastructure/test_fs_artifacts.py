from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybrid_inpaint.domain.entities import Image, Mask
from hybrid_inpaint.domain.errors import NotFoundError, ValidationError
from hybrid_inpaint.infrastructure.clock import SystemClock, UuidGenerator
from hybrid_inpaint.infrastructure.fs_artifacts import FileSystemArtifactStore
from hybrid_inpaint.infrastructure.pillow_codec import PillowImageCodec


def _store(tmp_path: Path) -> FileSystemArtifactStore:
    return FileSystemArtifactStore(tmp_path / "history", PillowImageCodec())


def test_save_read_delete_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = UuidGenerator().new_id()
    image = Image(np.zeros((4, 4, 3), dtype=np.uint8))
    mask = Mask(np.eye(4, dtype=np.bool_))

    paths = store.save(job_id, image, mask, image)

    assert paths.input.is_file() and paths.mask.is_file() and paths.output.is_file()
    assert store.read_output(job_id)[:8] == b"\x89PNG\r\n\x1a\n"
    store.delete(job_id)
    assert not paths.output.exists()
    store.delete(job_id)  # 重复删除不抛


def test_read_missing_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        _store(tmp_path).read_output(format(9, "032x"))


@pytest.mark.parametrize("bad", ["../evil", "ABC", "/abs/path", "zz" * 16, ""])
def test_job_id_traversal_is_rejected(tmp_path: Path, bad: str) -> None:
    store = _store(tmp_path)
    image = Image(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValidationError):
        store.save(bad, image, Mask(np.zeros((4, 4), dtype=np.bool_)), image)


def test_clock_and_ids() -> None:
    now = SystemClock().now()
    assert now.tzinfo is not None
    first, second = UuidGenerator().new_id(), UuidGenerator().new_id()
    assert first != second and len(first) == 32