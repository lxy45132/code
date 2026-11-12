from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from hybrid_inpaint.domain.entities import ModelSpec
from hybrid_inpaint.domain.errors import ModelUnavailableError, NotFoundError
from hybrid_inpaint.infrastructure.fs_models import FileSystemModelRepository
from tests.infrastructure.test_torch_engine import _tiny_checkpoint


def _repo(tmp_path: Path, spec: ModelSpec) -> FileSystemModelRepository:
    return FileSystemModelRepository(
        (spec,), device="cpu", pad_multiple=16, tile_size=64, tile_overlap=16
    )


def test_resolve_unknown_id(tmp_path: Path) -> None:
    repo = _repo(tmp_path, ModelSpec("main", tmp_path / "x.pt"))
    with pytest.raises(NotFoundError):
        repo.resolve("other")


def test_resolve_missing_file(tmp_path: Path) -> None:
    repo = _repo(tmp_path, ModelSpec("main", tmp_path / "x.pt"))
    with pytest.raises(ModelUnavailableError):
        repo.resolve("main")


def test_load_sha_mismatch(tmp_path: Path) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    repo = _repo(tmp_path, ModelSpec("main", checkpoint, sha256="0" * 64))
    with pytest.raises(ModelUnavailableError):
        repo.load(repo.resolve("main"))


def test_load_ok_with_sha(tmp_path: Path) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    repo = _repo(tmp_path, ModelSpec("main", checkpoint, sha256=digest))
    engine = repo.load(repo.resolve("main"))
    assert re.fullmatch(r"[0-9a-f]{12}-[0-9a-f]{8}", engine.version)