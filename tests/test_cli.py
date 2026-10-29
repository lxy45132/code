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


def test_start_is_registered_but_not_implemented(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    result = runner.invoke(app, ["start", "--config", str(path)])

    assert result.exit_code == 3
    assert "not implemented" in result.output.lower()


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
