from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from hybrid_inpaint.interfaces.cli.commands import app

runner = CliRunner()


def _write_config(tmp_path: Path, version: int = 1) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(f"contract_version: {version}\n", encoding="utf-8")
    return path


def test_config_validate_ok(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    result = runner.invoke(app, ["config", "validate", "--config", str(path)])

    assert result.exit_code == 0
    assert "CONFIG OK" in result.output


def test_config_validate_rejects_bad_config(tmp_path: Path) -> None:
    path = _write_config(tmp_path, version=2)
    result = runner.invoke(app, ["config", "validate", "--config", str(path)])

    assert result.exit_code == 2
    assert "config error" in result.output.lower()


def test_start_and_worker_are_registered(tmp_path: Path) -> None:
    # C7g 已实现 start/worker；功能测试见 tests/test_cli_web.py
    for command in ("start", "worker"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output


def _write_train_config(
    tmp_path: Path,
    *,
    experiment_id: str = "cli_train",
    output_dir: str = "run",
    vit_depth: int = 0,
    vit_heads: int = 1,
) -> Path:
    config = tmp_path / f"{experiment_id}.yaml"
    config.write_text(
        "\n".join(
            [
                "contract_version: 1",
                f"experiment_id: {experiment_id}",
                "seed: 2026",
                "model:",
                "  name: tiny",
                "  channels: [4, 8, 16, 32]",
                f"  vit_depth: {vit_depth}",
                f"  vit_heads: {vit_heads}",
                "  vit_patch: 2",
                "  vit_mlp_ratio: 2.0",
                "  norm: instance",
                "data:",
                "  dataset: synthetic",
                "  image_size: 32",
                "  train_samples: 2",
                "  val_samples: 2",
                "train:",
                "  stage: stage1",
                "  iterations: 2",
                "  batch_size: 1",
                "  lr: 0.001",
                "  betas: [0.5, 0.999]",
                "  warmup_steps: 1",
                "  min_lr: 0.00001",
                "  precision: fp32",
                "  val_every: 1",
                "  val_batch_size: 1",
                "  patience: 5",
                "  log_every: 1",
                "  device: cpu",
                f"  output_dir: {output_dir}",
                "  loss:",
                "    l1_hole: 1.0",
                "    l1_valid: 0.5",
                "    perceptual: 0.0",
                "    tv: 0.05",
                "    adversarial: 0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def test_train_command_writes_checkpoints(tmp_path: Path) -> None:
    config = _write_train_config(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(config)])

    assert result.exit_code == 0, result.output
    checkpoint_dir = tmp_path / "run" / "cli_train" / "seed_2026" / "checkpoints"
    assert (checkpoint_dir / "last.pt").is_file()
    assert (checkpoint_dir / "best.pt").is_file()


def test_train_command_with_bottleneck_vit(tmp_path: Path) -> None:
    config = _write_train_config(
        tmp_path,
        experiment_id="cli_vit",
        output_dir="run_vit",
        vit_depth=1,
        vit_heads=4,
    )
    result = runner.invoke(app, ["train", "--config", str(config)])

    assert result.exit_code == 0, result.output
    checkpoint_dir = tmp_path / "run_vit" / "cli_vit" / "seed_2026" / "checkpoints"
    assert (checkpoint_dir / "last.pt").is_file()


def test_data_freeze_synthetic(tmp_path: Path) -> None:
    output_dir = tmp_path / "frozen"
    result = runner.invoke(
        app,
        [
            "data",
            "freeze-synthetic",
            "--output-dir",
            str(output_dir),
            "--count",
            "2",
            "--size",
            "64",
            "--seed",
            "2026",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "frozen.json").exists()
    assert len(list((output_dir / "images").glob("*.png"))) == 2
    assert len(list((output_dir / "masks").glob("*.png"))) == 2
