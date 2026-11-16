"""Worker-process loading must not change the training trajectory.

``data.num_workers > 0`` loads batches through a DataLoader whose
``_IterationBatchSampler`` replays the deterministic per-iteration indices.
Datasets are pure per-index functions, so these tests pin the contract:
identical batches, identical losses, identical resume behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from hybrid_inpaint.config import (
    AppConfig,
    DataConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
)
from hybrid_inpaint.training.data.factory import build_training_datasets
from hybrid_inpaint.training.engine.trainer import Trainer, _IterationBatchSampler


def _config(
    tmp_path: Path,
    *,
    workers: int,
    iterations: int = 8,
    output_name: str = "run",
    val_every: int = 2,
) -> AppConfig:
    return AppConfig(
        experiment_id="parallel_test",
        seed=2026,
        model=ModelConfig(channels=(4, 8, 16, 32), vit_heads=1, vit_depth=0),
        data=DataConfig(
            dataset="synthetic",
            image_size=32,
            train_samples=4,
            val_samples=2,
            num_workers=workers,
        ),
        train=TrainConfig(
            iterations=iterations,
            batch_size=2,
            lr=3e-3,
            warmup_steps=1,
            min_lr=1e-5,
            precision="fp32",
            val_every=val_every,
            val_batch_size=2,
            patience=100,
            log_every=100,
            seed=2026,
            device="cpu",
            output_dir=tmp_path / output_name,
            loss=LossConfig(perceptual=0.0),
        ),
    )


def _train_entries(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_batch_sampler_replays_trainer_indices(tmp_path: Path) -> None:
    config = _config(tmp_path, workers=0)
    train, val, identity = build_training_datasets(config)
    trainer = Trainer(config, train, val, data_identity=identity)

    for start in (0, 4):
        sampler = _IterationBatchSampler(
            seed=config.train.seed,
            length=len(train),
            batch_size=config.train.batch_size,
            start=start,
            stop=8,
        )
        assert list(sampler) == [
            trainer._batch_indices(iteration) for iteration in range(start, 8)
        ]
        assert len(sampler) == 8 - start


def test_parallel_training_matches_serial_trajectory(tmp_path: Path) -> None:
    serial_config = _config(tmp_path, workers=0, output_name="serial")
    serial_train, serial_val, serial_identity = build_training_datasets(serial_config)
    serial = Trainer(serial_config, serial_train, serial_val, data_identity=serial_identity)
    serial.fit()

    # Two workers keep Windows spawn overhead small while still exercising
    # the multi-process path (fork on Linux, spawn on Windows).
    parallel_config = _config(tmp_path, workers=2, output_name="parallel")
    parallel_train, parallel_val, parallel_identity = build_training_datasets(parallel_config)
    parallel = Trainer(
        parallel_config, parallel_train, parallel_val, data_identity=parallel_identity
    )
    parallel.fit()

    # Identical samples + identical op order => bit-identical losses.
    assert _train_entries(serial.run_dir) == _train_entries(parallel.run_dir)
    for key, value in serial.model.state_dict().items():
        assert torch.equal(value, parallel.model.state_dict()[key]), key
    assert (parallel.run_dir / "checkpoints" / "last.pt").is_file()
    assert (parallel.run_dir / "checkpoints" / "best.pt").is_file()


def test_resume_with_workers_matches_uninterrupted(tmp_path: Path) -> None:
    short_config = _config(tmp_path, workers=2, output_name="short")
    short_train, short_val, short_identity = build_training_datasets(short_config)
    short = Trainer(short_config, short_train, short_val, data_identity=short_identity)
    short.fit(until=4)

    resumed_config = _config(tmp_path, workers=2, output_name="resumed")
    resumed_train, resumed_val, resumed_identity = build_training_datasets(resumed_config)
    resumed = Trainer(
        resumed_config,
        resumed_train,
        resumed_val,
        data_identity=resumed_identity,
        resume_from=short.run_dir / "checkpoints" / "last.pt",
    )
    resumed_result = resumed.fit()
    assert resumed_result.start_iteration == 4
    assert resumed_result.final_iteration == 8

    fresh_config = _config(tmp_path, workers=2, output_name="fresh")
    fresh_train, fresh_val, fresh_identity = build_training_datasets(fresh_config)
    fresh = Trainer(fresh_config, fresh_train, fresh_val, data_identity=fresh_identity)
    fresh.fit()

    # The resumed run only logs iterations 4..7; compare that tail.
    assert _train_entries(resumed.run_dir) == _train_entries(fresh.run_dir)[4:]
    for key, value in fresh.model.state_dict().items():
        assert torch.allclose(value, resumed.model.state_dict()[key], atol=1e-6), key
