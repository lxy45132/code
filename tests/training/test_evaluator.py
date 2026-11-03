from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hybrid_inpaint.config import (
    AppConfig,
    DataConfig,
    EvalConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
)
from hybrid_inpaint.models.generator import build_generator
from hybrid_inpaint.training.data.manifest import make_entry, write_manifest
from hybrid_inpaint.training.engine.checkpoint import save_checkpoint
from hybrid_inpaint.training.engine.evaluator import (
    evaluate_checkpoint,
    write_evaluation,
)
from hybrid_inpaint.training.engine.trainer import make_lr_lambda

BUCKETS = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6001))
RANGES = ("[0,10%)", "[10,20%)", "[20,30%)", "[30,40%)", "[40,60%]")


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        experiment_id="evaluator_test",
        seed=2026,
        model=ModelConfig(channels=(4, 8, 16, 32), vit_heads=1, vit_depth=0),
        data=DataConfig(
            dataset="places2",
            image_size=64,
            manifest_dir=tmp_path,
            processed_dir=tmp_path,
        ),
        train=TrainConfig(
            iterations=1,
            batch_size=1,
            device="cpu",
            output_dir=tmp_path / "out",
            precision="fp32",
            loss=LossConfig(perceptual=0.0),
        ),
        eval=EvalConfig(batch_size=2, compute_lpips=False),
    )


def _save_checkpoint(tmp_path: Path, config: AppConfig) -> Path:
    model = build_generator(config.model)
    optimizer = torch_optimizer(model)
    scheduler = torch_scheduler(optimizer)
    path = tmp_path / "best.pt"
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
        data_identity={"dataset": "unit"},
    )
    return path


def torch_optimizer(model):
    import torch

    return torch.optim.Adam(model.parameters(), lr=1e-3)


def torch_scheduler(optimizer):
    import torch

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        make_lr_lambda(warmup_steps=1, iterations=1, min_lr_ratio=0.01),
    )


def _mask(path: Path, area: float) -> Path:
    array = np.zeros((256, 256), dtype=np.uint8)
    side = max(1, int(round((area * 256 * 256) ** 0.5)))
    array[100 : 100 + side, 100 : 100 + side] = 255
    Image.fromarray(array, mode="L").save(path)
    return path


def _pairs(tmp_path: Path) -> Path:
    images = []
    for index in range(2):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (256, 256), color=(20 + index, 40, 60)).save(path)
        images.append(path)

    entries = []
    for image_index, image_path in enumerate(images):
        image_entry = make_entry(
            image_path,
            image_id=f"image-{image_index}",
            split="test",
            source="unit-test",
            license="test-only",
            category_id=30,
            category="badlands",
        )
        for bucket, (low, high) in enumerate(BUCKETS):
            mask_path = _mask(tmp_path / f"mask-{image_index}-{bucket}.png", (low + high) / 2)
            mask_entry = make_entry(
                mask_path,
                image_id=f"mask-{image_index}-{bucket}",
                split="test",
                source="unit-test",
                license="test-only",
            )
            entries.append(
                {
                    **image_entry,
                    "image_id": f"image-{image_index}--m{bucket}",
                    "image_source_id": image_entry["image_id"],
                    "mask_relative_path": mask_entry["relative_path"],
                    "mask_sha256": mask_entry["sha256"],
                    "mask_source_id": mask_entry["image_id"],
                    "bucket": bucket,
                    "bucket_range": RANGES[bucket],
                }
            )

    path = tmp_path / "pairs.jsonl"
    write_manifest(
        path,
        entries,
        kind="pairs",
        dataset="places365",
        split="test",
        seed=2026,
        source="unit-test",
        license="test-only",
    )
    return path


def test_evaluate_checkpoint_and_write_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint = _save_checkpoint(tmp_path, config)
    pairs = _pairs(tmp_path)

    result = evaluate_checkpoint(
        config,
        checkpoint,
        pairs,
        dataset="places2",
        device="cpu",
    )

    assert len(result.rows) == 10
    summary = result.summary
    assert summary["pairs"] == 10
    assert summary["images"] == 2
    for key in ("psnr_hole", "psnr_full", "ssim", "mae_hole", "mae_full"):
        assert summary["metrics"][key] is not None
    assert set(summary["by_bucket"]) == {"0", "1", "2", "3", "4"}
    assert summary["by_category"]["badlands"]["images"] == 2

    pairs_path, summary_path = write_evaluation(
        result,
        tmp_path / "eval",
        checkpoint=checkpoint,
        dataset="places2",
        image_size=64,
    )
    assert pairs_path.is_file()
    assert summary_path.is_file()
    assert "pairs.csv" in pairs_path.name


def test_limit_truncates_pairs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint = _save_checkpoint(tmp_path, config)
    pairs = _pairs(tmp_path)
    result = evaluate_checkpoint(
        config,
        checkpoint,
        pairs,
        dataset="places2",
        device="cpu",
        limit=5,
    )
    assert result.summary["pairs"] == 5
    assert result.summary["images"] == 1
