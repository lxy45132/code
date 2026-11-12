"""FileSystemArtifactStore: per-job PNG artifacts with atomic writes."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from hybrid_inpaint.domain.entities import ArtifactPaths, Image, Mask
from hybrid_inpaint.domain.errors import NotFoundError
from hybrid_inpaint.domain.ports import ImageCodec
from hybrid_inpaint.domain.services import validate_job_id
from hybrid_inpaint.infrastructure.fs_batch import atomic_write

logger = logging.getLogger("hybrid_inpaint.artifacts")


class FileSystemArtifactStore:
    def __init__(self, root: Path, codec: ImageCodec) -> None:
        self._root = Path(root)
        self._codec = codec

    def _job_dir(self, job_id: str) -> Path:
        validate_job_id(job_id)  # 拒绝 ..、绝对路径、非 UUID
        return self._root / job_id

    def save(
        self, job_id: str, image: Image, mask: Mask, output: Image
    ) -> ArtifactPaths:
        directory = self._job_dir(job_id)
        paths = ArtifactPaths(
            directory / "input.png", directory / "mask.png", directory / "output.png"
        )
        atomic_write(paths.input, self._codec.encode_png(image))
        atomic_write(paths.mask, self._codec.encode_mask_png(mask))
        atomic_write(paths.output, self._codec.encode_png(output))
        return paths

    def read_output(self, job_id: str) -> bytes:
        path = self._job_dir(job_id) / "output.png"
        if not path.is_file():
            raise NotFoundError(f"no output artifact for job {job_id}")
        return path.read_bytes()

    def delete(self, job_id: str) -> None:
        directory = self._job_dir(job_id)
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("artifact delete failed job_id=%s error=%s", job_id, exc)