from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from hybrid_inpaint.config import AppConfig, DataConfig, LossConfig, ModelConfig, TrainConfig
from hybrid_inpaint.training.data.factory import build_training_datasets
from hybrid_inpaint.training.data.synthetic import SyntheticInpaintDataset
from hybrid_inpaint.training.engine.trainer import Trainer, make_lr_lambda


def _config(
    tmp_path: Path,
    *,
    iterations: int = 40,
    output_name: str = "run",
    **train_overrides: object,
) -> AppConfig:
    train_kwargs: dict[str, object] = dict(
        iterations=iterations,
        batch_size=2,
        lr=3e-3,
        warmup_steps=1,
        min_lr=1e-5,
        precision="fp32",
        val_every=10,
        val_batch_size=2,
        patience=100,
        log_every=100,
        seed=2026,
        device="cpu",
        output_dir=tmp_path / output_name,
        loss=LossConfig(perceptual=0.0),
    )
    train_kwargs.update(train_overrides)
    return AppConfig(
        experiment_id="trainer_test",
        seed=2026,
        model=ModelConfig(channels=(4, 8, 16, 32), vit_heads=1, vit_depth=0),
        data=DataConfig(
            dataset="synthetic",
            image_size=32,
            train_samples=4,
            val_samples=2,
        ),
        train=TrainConfig(**train_kwargs),
    )


def test_lr_schedule_warmup_and_cosine() -> None:
    schedule = make_lr_lambda(warmup_steps=10, iterations=110, min_lr_ratio=0.01)
    assert schedule(0) == pytest.approx(0.1)
    assert schedule(9) == pytest.approx(1.0)
    assert schedule(10) == pytest.approx(1.0)
    assert schedule(110) == pytest.approx(0.01)
    assert schedule(60) < 1.0


def test_training_overfits_small_synthetic_set(tmp_path: Path) -> None:
    config = _config(tmp_path)
    train, val, identity = build_training_datasets(config)
    trainer = Trainer(config, train, val, data_identity=identity)
    result = trainer.fit()

    assert result.final_iteration == config.train.iterations
    assert not result.stopped_early
    assert len(result.history) == 4
    assert math.isfinite(result.best_psnr_hole)

    run_dir = result.run_dir
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "config.resolved.json").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "train.jsonl").read_text(encoding="utf-8").count("\n") == 40

    validation = result.history
    assert validation[-1]["psnr_hole"] > validation[0]["psnr_hole"]
    assert validation[-1]["loss_total"] < validation[0]["loss_total"]


def test_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    # Same iteration budget as the reference run: only the stopping point
    # differs, so the LR schedule and config fingerprint are unchanged.
    short_config = _config(
        tmp_path,
        iterations=8,
        output_name="run_short",
        val_every=2,
    )
    short_train, short_val, short_identity = build_training_datasets(short_config)
    short = Trainer(short_config, short_train, short_val, data_identity=short_identity)
    short.fit(until=4)

    full_config = _config(
        tmp_path,
        iterations=8,
        output_name="run_resumed",
        val_every=2,
    )
    resumed_train, resumed_val, resumed_identity = build_training_datasets(full_config)
    resumed = Trainer(
        full_config,
        resumed_train,
        resumed_val,
        data_identity=resumed_identity,
        resume_from=short.run_dir / "checkpoints" / "last.pt",
    )
    resumed_result = resumed.fit()
    assert resumed_result.start_iteration == 4
    assert resumed_result.final_iteration == 8

    fresh_config = _config(
        tmp_path,
        iterations=8,
        output_name="run_fresh",
        val_every=2,
    )
    fresh_train, fresh_val, fresh_identity = build_training_datasets(fresh_config)
    fresh = Trainer(fresh_config, fresh_train, fresh_val, data_identity=fresh_identity)
    fresh.fit()

    for key, value in fresh.model.state_dict().items():
        assert torch.allclose(value, resumed.model.state_dict()[key], atol=1e-6), key


def test_training_runs_with_bottleneck_vit(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        iterations=2,
        output_name="run_vit",
        batch_size=1,
        val_batch_size=1,
        val_every=1,
        log_every=1,
    ).model_copy(
        update={
            "model": ModelConfig(
                channels=(4, 8, 16, 32),
                vit_heads=4,
                vit_depth=1,
                vit_patch=2,
                vit_mlp_ratio=2.0,
            )
        }
    )
    train, val, identity = build_training_datasets(config)
    trainer = Trainer(config, train, val, data_identity=identity)
    result = trainer.fit()

    assert result.final_iteration == 2
    assert (result.run_dir / "checkpoints" / "last.pt").is_file()
    assert any(
        parameter.requires_grad
        for parameter in trainer.model.bottleneck.parameters()
    )


def test_stage_two_rejected_until_c4(tmp_path: Path) -> None:
    config = _config(tmp_path).model_copy(
        update={
            "train": _config(tmp_path).train.model_copy(update={"stage": "stage2"})
        }
    )
    dataset = SyntheticInpaintDataset(size=32, length=2)
    with pytest.raises(NotImplementedError, match="C4"):
        Trainer(config, dataset, dataset)


def test_configured_perceptual_loss_must_be_wired(tmp_path: Path) -> None:
    config = _config(tmp_path, loss=LossConfig(perceptual=0.1))
    dataset = SyntheticInpaintDataset(size=32, length=2)
    with pytest.raises(ValueError, match="perceptual"):
        Trainer(config, dataset, dataset)


def test_cuda_unavailable_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, device="cuda")
    dataset = SyntheticInpaintDataset(size=32, length=2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        Trainer(config, dataset, dataset)
