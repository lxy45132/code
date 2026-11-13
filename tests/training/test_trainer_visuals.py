from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from hybrid_inpaint.config import AppConfig, DataConfig, ModelConfig, TrainConfig
from hybrid_inpaint.domain.entities import Image, Mask, Sample
from hybrid_inpaint.training.engine.trainer import Trainer


def _samples(count: int = 2, size: int = 16) -> list[Sample]:
    samples = []
    for index in range(count):
        image = np.full((size, size, 3), 40 + index * 20, dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.bool_)
        mask[4:12, 4:12] = True
        samples.append(Sample(sample_id=f"s{index}", image=Image(image), mask=Mask(mask)))
    return samples


def _config(tmp_path: Path, visual: int) -> AppConfig:
    return AppConfig(
        experiment_id="visual_test",
        model=ModelConfig(channels=(4, 8, 16, 32), vit_heads=1, vit_depth=0),
        data=DataConfig(dataset="synthetic", image_size=16),
        train=TrainConfig(
            iterations=2,
            batch_size=1,
            warmup_steps=1,
            val_every=1,
            val_batch_size=1,
            log_every=1,
            patience=5,
            precision="fp32",
            device="cpu",
            output_dir=tmp_path / "out",
            val_visual_samples=visual,
            loss={"l1_hole": 1.0, "l1_valid": 0.5, "perceptual": 0.0, "tv": 0.05, "adversarial": 0.1},
        ),
    )


def test_val_visual_samples_rejects_negative() -> None:
    with pytest.raises(PydanticValidationError):
        TrainConfig(val_visual_samples=-1)


def test_fit_saves_comparison_grid_each_validation(tmp_path: Path) -> None:
    samples = _samples()
    trainer = Trainer(_config(tmp_path, visual=2), samples, samples)

    result = trainer.fit()

    grid_dir = result.run_dir / "samples"
    assert (grid_dir / "val_000001.png").is_file()
    assert (grid_dir / "val_000002.png").is_file()
    if importlib.util.find_spec("matplotlib") is not None:
        assert (result.run_dir / "plots" / "train_loss.png").is_file()
        assert (result.run_dir / "plots" / "val_metrics.png").is_file()


def test_fit_with_visual_disabled_writes_no_grids(tmp_path: Path) -> None:
    samples = _samples()
    trainer = Trainer(_config(tmp_path, visual=0), samples, samples)

    result = trainer.fit()

    assert not (result.run_dir / "samples").exists()