from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_inpaint.training.engine.plots import plot_run


def _write_run(tmp_path: Path) -> None:
    train_lines = [
        json.dumps(
            {
                "iteration": i,
                "total": 1.0 / (i + 1),
                "hole": 0.5 / (i + 1),
                "valid": 0.3 / (i + 1),
                "tv": 0.1,
            }
        )
        for i in range(5)
    ]
    val_lines = [
        json.dumps(
            {
                "iteration": (i + 1) * 2,
                "psnr_hole": 10.0 + i,
                "ssim": 0.5 + 0.05 * i,
                "lpips": 0.4 - 0.05 * i,
            }
        )
        for i in range(3)
    ]
    (tmp_path / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (tmp_path / "val.jsonl").write_text("\n".join(val_lines) + "\n", encoding="utf-8")


def test_plot_run_writes_loss_and_metric_curves(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    _write_run(tmp_path)

    paths = plot_run(tmp_path)

    names = {path.name for path in paths}
    assert {"train_loss.png", "val_metrics.png"} <= names
    assert all(path.is_file() for path in paths)
    assert all(path.parent == tmp_path / "plots" for path in paths)


def test_plot_run_empty_logs_returns_nothing(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    (tmp_path / "train.jsonl").touch()
    (tmp_path / "val.jsonl").touch()

    assert plot_run(tmp_path) == []


def test_plot_run_writes_dynamics_and_val_loss_curves(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    train_lines = [
        json.dumps(
            {
                "iteration": i,
                "total": 1.0 / (i + 1),
                "hole": 0.5 / (i + 1),
                "valid": 0.3 / (i + 1),
                "tv": 0.1,
                "lr": 1e-4 * (1 - i / 10),
                "grad_norm": 2.0 + 0.1 * i,
            }
        )
        for i in range(6)
    ]
    val_lines = [
        json.dumps(
            {
                "iteration": (i + 1) * 2,
                "psnr_hole": 10.0 + i,
                "psnr_full": 18.0 + i,
                "ssim": 0.5 + 0.05 * i,
                "mae_hole": 0.3 - 0.02 * i,
                "mae_full": 0.1 - 0.005 * i,
                "lpips": 0.4 - 0.05 * i,
                "loss_total": 0.8 - 0.1 * i,
                "loss_hole": 0.4 - 0.05 * i,
                "loss_valid": 0.3 - 0.03 * i,
                "loss_tv": 0.1,
            }
        )
        for i in range(3)
    ]
    (tmp_path / "train.jsonl").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (tmp_path / "val.jsonl").write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    names = {path.name for path in plot_run(tmp_path)}

    assert {"train_dynamics.png", "val_losses.png"} <= names


def test_plot_run_legacy_logs_skip_missing_curves(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"iteration": 0, "total": 0.9}) + "\n", encoding="utf-8"
    )
    (tmp_path / "val.jsonl").write_text(
        json.dumps({"iteration": 1, "psnr_hole": 11.0}) + "\n", encoding="utf-8"
    )

    names = {path.name for path in plot_run(tmp_path)}

    assert "train_dynamics.png" not in names
    assert "val_losses.png" not in names
    assert {"train_loss.png", "val_metrics.png"} <= names


def test_single_point_curves_use_visible_markers(tmp_path: Path, monkeypatch) -> None:
    """单点序列若无 marker 则画不出任何可见像素；短运行必须带 marker。"""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    (tmp_path / "train.jsonl").write_text(
        json.dumps({"iteration": 0, "total": 0.9, "lr": 1e-4, "grad_norm": 2.0}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "val.jsonl").write_text(
        json.dumps(
            {
                "iteration": 1,
                "psnr_hole": 11.0,
                "loss_total": 0.5,
                "loss_hole": 0.2,
                "loss_valid": 0.2,
                "loss_tv": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    captured = []
    real_close = plt.close

    def spy(fig=None, *args, **kwargs):
        if fig is not None:
            captured.append(fig)
        return real_close(fig, *args, **kwargs)

    monkeypatch.setattr(plt, "close", spy)
    plot_run(tmp_path)

    checked = 0
    for fig in captured:
        for ax in fig.axes:
            if ax.get_title() in ("Validation losses", "Training loss"):
                assert ax.lines, f"{ax.get_title()} has no series"
                for line in ax.lines:
                    assert line.get_marker() not in ("", "None", None)
                checked += 1
    assert checked == 2