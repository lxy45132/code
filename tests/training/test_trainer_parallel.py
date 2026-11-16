"""Threaded sample loading must not change the training trajectory.

``data.num_workers > 0`` fetches batches through a thread pool with one-batch
prefetch.  Datasets are pure per-index functions, so these tests pin the
contract: identical samples, identical losses, identical resume behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hybrid_inpaint.config import (
    AppConfig,
    DataConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
)
from hybrid_inpaint.training.data.factory import build_training_datasets
from hybrid_inpaint.training.engine.trainer import Trainer


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


def test_parallel_fetch_matches_serial_sample_for_sample(tmp_path: Path) -> None:
    config = _config(tmp_path, workers=4)
    train, val, identity = build_training_datasets(config)
    trainer = Trainer(config, train, val, data_identity=identity)
    try:
        for iteration in range(4):
            parallel = trainer._fetch_samples(iteration)
            serial = [train[index] for index in trainer._batch_indices(iteration)]
            assert len(parallel) == len(serial)
            for got, want in zip(parallel, serial):
                assert got.sample_id == want.sample_id
                assert np.array_equal(got.image.data, want.image.data)
                assert np.array_equal(got.mask.data, want.mask.data)
                assert got.meta == want.meta
    finally:
        trainer.close()


def test_parallel_training_matches_serial_trajectory(tmp_path: Path) -> None:
    serial_config = _config(tmp_path, workers=0, output_name="serial")
    serial_train, serial_val, serial_identity = build_training_datasets(serial_config)
    serial = Trainer(serial_config, serial_train, serial_val, data_identity=serial_identity)
    serial.fit()

    parallel_config = _config(tmp_path, workers=4, output_name="parallel")
    parallel_train, parallel_val, parallel_identity = build_training_datasets(parallel_config)
    parallel = Trainer(
        parallel_config, parallel_train, parallel_val, data_identity=parallel_identity
    )
    parallel.fit()

    # Identical samples + identical op order => bit-identical losses.
    assert _train_entries(serial.run_dir) == _train_entries(parallel.run_dir)
    for key, value in serial.model.state_dict().items():
        assert torch.equal(value, parallel.model.state_dict()[key]), key


def test_resume_with_workers_matches_uninterrupted(tmp_path: Path) -> None:
    short_config = _config(tmp_path, workers=4, output_name="short")
    short_train, short_val, short_identity = build_training_datasets(short_config)
    short = Trainer(short_config, short_train, short_val, data_identity=short_identity)
    short.fit(until=4)

    resumed_config = _config(tmp_path, workers=4, output_name="resumed")
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

    fresh_config = _config(tmp_path, workers=4, output_name="fresh")
    fresh_train, fresh_val, fresh_identity = build_training_datasets(fresh_config)
    fresh = Trainer(fresh_config, fresh_train, fresh_val, data_identity=fresh_identity)
    fresh.fit()

    # The resumed run only logs iterations 4..7; compare that tail.
    assert _train_entries(resumed.run_dir) == _train_entries(fresh.run_dir)[4:]
    for key, value in fresh.model.state_dict().items():
        assert torch.allclose(value, resumed.model.state_dict()[key], atol=1e-6), key


def test_close_is_idempotent_and_falls_back_to_serial(tmp_path: Path) -> None:
    config = _config(tmp_path, workers=2)
    train, val, identity = build_training_datasets(config)
    trainer = Trainer(config, train, val, data_identity=identity)
    trainer._fetch_samples(0)
    trainer.close()
    trainer.close()
    # After close the serial path still serves identical samples.
    got = trainer._fetch_samples(1)
    want = [train[index] for index in trainer._batch_indices(1)]
    for sample, expected in zip(got, want):
        assert np.array_equal(sample.image.data, expected.image.data)
