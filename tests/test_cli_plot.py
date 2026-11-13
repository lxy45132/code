from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hybrid_inpaint.interfaces.cli.commands import app

runner = CliRunner()


def test_plot_command_renders_curves(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"iteration": 0, "total": 0.9, "hole": 0.4, "valid": 0.3, "tv": 0.1})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "val.jsonl").write_text(
        json.dumps({"iteration": 1, "psnr_hole": 11.0, "ssim": 0.6}) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["plot", "--run-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "train_loss.png" in result.output
    assert (tmp_path / "plots" / "train_loss.png").is_file()