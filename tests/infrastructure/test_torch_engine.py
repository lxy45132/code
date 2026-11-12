from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from hybrid_inpaint.config import AppConfig, DataConfig, ModelConfig, TrainConfig
from hybrid_inpaint.domain.errors import BusyError, ModelUnavailableError
from hybrid_inpaint.infrastructure.torch_engine import SerializedEngine, TorchInpaintEngine
from hybrid_inpaint.models.unet import UNet
from hybrid_inpaint.training.engine.checkpoint import save_checkpoint


def _tiny_checkpoint(tmp_path: Path) -> Path:
    config = AppConfig(
        experiment_id="engine_test",
        model=ModelConfig(channels=(4, 8, 16, 32), vit_heads=1, vit_depth=0),
        data=DataConfig(dataset="synthetic", image_size=32),
        train=TrainConfig(iterations=2, batch_size=1, warmup_steps=1, device="cpu"),
    )
    model = UNet(
        in_channels=4, out_channels=3, channels=config.model.channels, norm="instance"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    path = tmp_path / "tiny.pt"
    save_checkpoint(
        path,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        iteration=1,
        best_psnr_hole=10.0,
        history=[],
    )
    return path


def _engine(tmp_path: Path, **kwargs) -> TorchInpaintEngine:
    defaults = dict(device="cpu", tile_size=64, tile_overlap=16)
    defaults.update(kwargs)
    return TorchInpaintEngine(_tiny_checkpoint(tmp_path), **defaults)


def test_version_format_and_warmup(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{12}-[0-9a-f]{8}", engine.version)
    engine.warmup()
    assert engine.last_warnings == ()


def test_inpaint_shape_range_and_non_aligned(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    image = np.random.default_rng(0).integers(0, 256, (30, 34, 3), dtype=np.uint8)
    mask = np.zeros((30, 34), dtype=np.bool_)
    mask[10:20, 10:20] = True

    out = engine.inpaint(image, mask)

    assert out.shape == (30, 34, 3)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_large_input_uses_tiling_and_warns(tmp_path: Path) -> None:
    engine = _engine(tmp_path, tile_size=64, tile_overlap=16)
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.bool_)
    mask[30:60, 30:60] = True

    out = engine.inpaint(image, mask)

    assert out.shape == (96, 96, 3)
    assert engine.last_warnings  # 分块警告非空


def test_missing_checkpoint_raises_model_unavailable(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailableError):
        TorchInpaintEngine(tmp_path / "nope.pt", device="cpu")


class _SlowEngine:
    device = "slow"
    last_warnings: tuple[str, ...] = ()
    version = "slow-0"

    def warmup(self) -> None:
        pass

    def inpaint(self, image, mask):
        time.sleep(0.4)
        return np.zeros_like(image, dtype=np.float32)


def test_serialized_engine_busy_timeout() -> None:
    serialized = SerializedEngine(_SlowEngine(), timeout_s=0.05)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.bool_)
    thread = threading.Thread(target=serialized.inpaint, args=(image, mask))
    thread.start()
    time.sleep(0.1)  # 让线程先拿到锁
    with pytest.raises(BusyError):
        serialized.inpaint(image, mask)
    thread.join()
    assert serialized.version == "slow-0"