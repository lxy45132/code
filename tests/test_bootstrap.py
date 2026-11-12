from __future__ import annotations

from pathlib import Path

import numpy as np

from hybrid_inpaint.bootstrap import build_container
from hybrid_inpaint.domain.entities import Image, InpaintRequest, Mask
from tests.application.fakes import FakeEngine
from tests.infrastructure.test_torch_engine import _tiny_checkpoint


def _config(tmp_path: Path, checkpoint: Path) -> Path:
    path = tmp_path / "system.yaml"
    path.write_text(
        "\n".join(
            [
                "contract_version: 1",
                "experiment_id: bootstrap_test",
                "system:",
                "  model_id: tiny",
                f"  checkpoint: {checkpoint.as_posix()}",
                "  device: cpu",
                f"  history_db: {(tmp_path / 'history.sqlite').as_posix()}",
                f"  output_dir: {(tmp_path / 'history').as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _request() -> InpaintRequest:
    image = Image(np.arange(48, dtype=np.uint8).reshape(4, 4, 3))
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[1:3, 1:3] = True
    return InpaintRequest(image, Mask(mask), mode="cli")


def test_build_container_with_fake_engine(tmp_path: Path) -> None:
    config = _config(tmp_path, tmp_path / "unused.pt")
    container = build_container(config, engine=FakeEngine())

    result = container.inpaint.execute(_request())

    assert container.history.get(result.job_id) is not None
    assert container.artifacts.read_output(result.job_id).startswith(b"\x89PNG")
    assert container.model_service.health()["ready"] is True


def test_build_container_with_real_tiny_engine(tmp_path: Path) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    container = build_container(_config(tmp_path, checkpoint))

    result = container.inpaint.execute(_request())

    assert result.image.data.shape == (4, 4, 3)
    assert container.history.get(result.job_id) is not None
    assert "-" in result.model_version