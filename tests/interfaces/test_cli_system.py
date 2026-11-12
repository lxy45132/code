from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from typer.testing import CliRunner

from hybrid_inpaint.interfaces.cli.commands import app
from tests.infrastructure.test_torch_engine import _tiny_checkpoint

runner = CliRunner()


def _setup(tmp_path: Path):
    checkpoint = _tiny_checkpoint(tmp_path)
    config = tmp_path / "system.yaml"
    config.write_text(
        "\n".join(
            [
                "contract_version: 1",
                "experiment_id: cli_system_test",
                "system:",
                "  model_id: tiny",
                f"  checkpoint: {checkpoint.as_posix()}",
                "  device: cpu",
                "  tiling:",
                "    tile_size: 64",
                "    tile_overlap: 16",
                f"  history_db: {(tmp_path / 'history.sqlite').as_posix()}",
                f"  output_dir: {(tmp_path / 'history').as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    inputs, masks = tmp_path / "in", tmp_path / "mask"
    inputs.mkdir()
    masks.mkdir()
    PILImage.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8)).save(inputs / "a.png")
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    PILImage.fromarray(mask).save(masks / "a.png")
    return config, inputs, masks


def test_inpaint_single_writes_output_and_history(tmp_path: Path) -> None:
    config, inputs, masks = _setup(tmp_path)
    output = tmp_path / "out.png"

    result = runner.invoke(
        app,
        ["inpaint", "--image", str(inputs / "a.png"), "--mask", str(masks / "a.png"),
         "--output", str(output), "--config", str(config)],
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert "job_id=" in result.output
    conn = sqlite3.connect(tmp_path / "history.sqlite")
    count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    conn.close()
    assert count == 1


def test_inpaint_all_hole_mask_exits_2(tmp_path: Path) -> None:
    config, inputs, masks = _setup(tmp_path)
    PILImage.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(masks / "a.png")

    result = runner.invoke(
        app,
        ["inpaint", "--image", str(inputs / "a.png"), "--mask", str(masks / "a.png"),
         "--output", str(tmp_path / "out.png"), "--config", str(config)],
    )

    assert result.exit_code == 2
    assert "validation_error" in result.output


def test_run_batch_report_and_exit_codes(tmp_path: Path) -> None:
    config, inputs, masks = _setup(tmp_path)
    PILImage.fromarray(np.full((32, 32, 3), 60, dtype=np.uint8)).save(inputs / "b.png")
    out_dir = tmp_path / "batch_out"
    args = ["run", "--input-dir", str(inputs), "--mask-dir", str(masks),
            "--output-dir", str(out_dir), "--config", str(config)]

    result = runner.invoke(app, args)

    assert result.exit_code == 1  # b.png 缺掩码
    report = out_dir / "report.csv"
    rows = report.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "relative_path,mask_path,status,output_path,latency_ms,mask_ratio,error"
    assert len(rows) == 3
    assert (out_dir / "a.png").is_file()

    # 补上零洞掩码 -> 全成功
    PILImage.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(masks / "b.png")
    result = runner.invoke(app, args + ["--overwrite"])
    assert result.exit_code == 0, result.output


def test_run_limit(tmp_path: Path) -> None:
    config, inputs, masks = _setup(tmp_path)
    out_dir = tmp_path / "batch_out"
    result = runner.invoke(
        app,
        ["run", "--input-dir", str(inputs), "--mask-dir", str(masks),
         "--output-dir", str(out_dir), "--config", str(config), "--limit", "1"],
    )
    assert result.exit_code == 0, result.output
    rows = (out_dir / "report.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2