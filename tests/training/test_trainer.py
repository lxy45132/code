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


class _FixedLPIPS:
    def __init__(self, value: float = 0.2) -> None:
        self.value = value

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        return self.value


def _stage2_config(
    tmp_path: Path,
    stage1_config: AppConfig,
    *,
    iterations: int,
    output_name: str,
) -> AppConfig:
    train = stage1_config.train.model_copy(
        update={
            "stage": "stage2",
            "iterations": iterations,
            "output_dir": tmp_path / output_name,
        }
    )
    return stage1_config.model_copy(
        update={"experiment_id": "trainer_test_gan", "train": train}
    )


def test_stage_two_training_and_resume(tmp_path: Path) -> None:
    stage1_config = _config(
        tmp_path,
        iterations=2,
        output_name="run_stage1",
        val_every=1,
        log_every=1,
    )
    train, val, identity = build_training_datasets(stage1_config)
    stage1 = Trainer(stage1_config, train, val, data_identity=identity)
    stage1_result = stage1.fit()
    stage1_best = stage1_result.run_dir / "checkpoints" / "best.pt"
    assert stage1_best.is_file()

    stage2_config = _stage2_config(
        tmp_path,
        stage1_config,
        iterations=2,
        output_name="run_stage2",
    )
    stage2_train, stage2_val, stage2_identity = build_training_datasets(stage2_config)
    stage2 = Trainer(
        stage2_config,
        stage2_train,
        stage2_val,
        data_identity=stage2_identity,
        resume_from=stage1_best,
        lpips_metric=_FixedLPIPS(0.2),
    )
    stage2_result = stage2.fit()

    assert stage2_result.start_iteration == 0
    assert stage2_result.final_iteration == 2
    assert stage2_result.best_lpips == pytest.approx(0.2)
    assert all(entry["lpips"] == pytest.approx(0.2) for entry in stage2_result.history)

    checkpoint_dir = stage2_result.run_dir / "checkpoints"
    assert (checkpoint_dir / "last.pt").is_file()
    assert (checkpoint_dir / "best_psnr.pt").is_file()
    assert (checkpoint_dir / "best_lpips.pt").is_file()

    payload = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
    assert payload["stage"] == "stage2"
    assert "model_d" in payload
    assert "optimizer_d" in payload
    assert payload["best_metric"]["lpips"] == pytest.approx(0.2)

    # Extending a stage-two run restores D, both optimisers and the iteration.
    resumed_config = _stage2_config(
        tmp_path,
        stage1_config,
        iterations=4,
        output_name="run_stage2_resumed",
    )
    resumed_train, resumed_val, resumed_identity = build_training_datasets(
        resumed_config
    )
    resumed = Trainer(
        resumed_config,
        resumed_train,
        resumed_val,
        data_identity=resumed_identity,
        resume_from=checkpoint_dir / "last.pt",
        lpips_metric=_FixedLPIPS(0.2),
    )
    resumed_result = resumed.fit()
    assert resumed_result.start_iteration == 2
    assert resumed_result.final_iteration == 4


def test_stage_two_requires_a_stage_one_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path).model_copy(
        update={
            "train": _config(tmp_path).train.model_copy(update={"stage": "stage2"})
        }
    )
    dataset = SyntheticInpaintDataset(size=32, length=2)
    with pytest.raises(ValueError, match="stage-one checkpoint"):
        Trainer(config, dataset, dataset)


def test_cuda_unavailable_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, device="cuda")
    dataset = SyntheticInpaintDataset(size=32, length=2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        Trainer(config, dataset, dataset)
