from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from hybrid_inpaint.config import AppConfig, DataConfig, ModelConfig, TrainConfig
from hybrid_inpaint.models.unet import UNet
from hybrid_inpaint.training.engine.checkpoint import (
    CheckpointError,
    capture_rng_state,
    config_fingerprint,
    environment_info,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from hybrid_inpaint.training.engine.trainer import make_lr_lambda


def _config(tmp_path: Path, *, iterations: int = 4) -> AppConfig:
    return AppConfig(
        experiment_id="checkpoint_test",
        seed=2026,
        model=ModelConfig(channels=(4, 8, 16, 32), vit_heads=1, vit_depth=0),
        data=DataConfig(dataset="synthetic", image_size=32, train_samples=2, val_samples=2),
        train=TrainConfig(
            iterations=iterations,
            batch_size=1,
            warmup_steps=1,
            val_every=2,
            log_every=1,
            device="cpu",
            output_dir=tmp_path / "out",
        ),
    )


def _build(config: AppConfig) -> tuple[UNet, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    model = UNet(
        in_channels=4,
        out_channels=3,
        channels=config.model.channels,
        norm=config.model.norm,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        make_lr_lambda(
            warmup_steps=config.train.warmup_steps,
            iterations=config.train.iterations,
            min_lr_ratio=config.train.min_lr / config.train.lr,
        ),
    )
    return model, optimizer, scheduler


def test_round_trip_weights_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model, optimizer, scheduler = _build(config)
    path = tmp_path / "checkpoints" / "last.pt"

    save_checkpoint(
        path,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        iteration=3,
        best_psnr_hole=12.5,
        history=[{"iteration": 2, "psnr_hole": 12.5}],
        data_identity={"dataset": "synthetic"},
    )

    # The on-disk format must load without arbitrary unpickling.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert raw["schema_version"] == 2
    assert raw["config_fingerprint"] == config_fingerprint(config)

    payload = load_checkpoint(path, expected_config=config)
    assert payload["iteration"] == 3
    assert payload["best_metric"]["psnr_hole"] == pytest.approx(12.5)
    assert payload["data_identity"] == {"dataset": "synthetic"}
    assert set(payload["model"]) == set(model.state_dict())


def test_config_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model, optimizer, scheduler = _build(config)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        iteration=1,
        best_psnr_hole=1.0,
        history=[],
    )

    other = config.model_copy(update={"seed": 2027})
    with pytest.raises(CheckpointError, match="fingerprint"):
        load_checkpoint(path, expected_config=other)


def test_iteration_budget_is_not_part_of_fingerprint(tmp_path: Path) -> None:
    config = _config(tmp_path, iterations=4)
    extended = _config(tmp_path, iterations=8)
    assert config_fingerprint(config) == config_fingerprint(extended)


def test_unsupported_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"schema_version": 99, "contract_version": 1}, path)
    with pytest.raises(CheckpointError, match="schema"):
        load_checkpoint(path)


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "missing.pt")


def test_rng_state_round_trip() -> None:
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    state = capture_rng_state()
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)

    random.random()
    np.random.rand()
    torch.rand(1)

    restore_rng_state(state)
    assert random.random() == pytest.approx(expected_python)
    assert np.random.rand() == pytest.approx(expected_numpy)
    assert torch.equal(torch.rand(1), expected_torch)


def test_environment_info_contains_versions() -> None:
    info = environment_info()
    assert {"python", "torch", "numpy", "cuda_available"} <= set(info)
