"""Loss / metric curves rendered from a run's train.jsonl and val.jsonl.

Figures: train_loss.png, train_dynamics.png (lr / grad norm),
val_metrics.png (psnr/ssim/mae/lpips), val_losses.png.  Curves whose keys are
absent from the logs (older runs, stage one without LPIPS, ...) are skipped.

matplotlib is imported lazily so environments without the train extra can
still import the training engine; the CLI reports a clear install hint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TRAIN_KEYS = ("total", "hole", "valid", "tv", "perceptual", "adversarial", "d_loss")
_VAL_KEYS = ("psnr_hole", "psnr_full", "ssim", "mae_hole", "mae_full", "lpips")
_LR_KEYS = ("lr", "lr_d")
_GRAD_KEYS = ("grad_norm", "grad_norm_d")
_VAL_LOSS_KEYS = ("loss_total", "loss_hole", "loss_valid", "loss_tv")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _present_series(
    entries: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[tuple[str, list[float]]]:
    """Only keys present in every entry, so curves never mix partial logs."""
    return [
        (key, [entry[key] for entry in entries])
        for key in keys
        if entries and all(key in entry for entry in entries)
    ]


def _save(fig: Any, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def plot_run(run_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Write the curve PNGs that the logs support; return the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    target = Path(output_dir) if output_dir is not None else run_dir / "plots"
    train = _read_jsonl(run_dir / "train.jsonl")
    val = _read_jsonl(run_dir / "val.jsonl")
    written: list[Path] = []

    if train:
        target.mkdir(parents=True, exist_ok=True)
        iterations = [entry["iteration"] for entry in train]
        marker = "o" if len(iterations) <= 50 else None
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key, values in _present_series(train, _TRAIN_KEYS):
            ax.plot(iterations, values, label=key, linewidth=1.2, marker=marker)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.set_title("Training loss")
        ax.legend()
        ax.grid(alpha=0.3)
        written.append(_save(fig, target / "train_loss.png"))

        lr_series = _present_series(train, _LR_KEYS)
        grad_series = _present_series(train, _GRAD_KEYS)
        if lr_series or grad_series:
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), squeeze=False)
            for ax, series, ylabel in (
                (axes[0, 0], lr_series, "learning rate"),
                (axes[1, 0], grad_series, "grad norm"),
            ):
                for key, values in series:
                    ax.plot(iterations, values, label=key, linewidth=1.2, marker=marker)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
                if series:
                    ax.legend()
            axes[-1, 0].set_xlabel("iteration")
            fig.suptitle("Training dynamics")
            written.append(_save(fig, target / "train_dynamics.png"))

    val_series = _present_series(val, _VAL_KEYS)
    if val_series:
        target.mkdir(parents=True, exist_ok=True)
        iterations = [entry["iteration"] for entry in val]
        fig, axes = plt.subplots(
            len(val_series), 1, figsize=(8, 2.6 * len(val_series)), squeeze=False
        )
        for ax, (key, values) in zip(axes[:, 0], val_series):
            ax.plot(iterations, values, marker="o", linewidth=1.2)
            ax.set_ylabel(key)
            ax.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("iteration")
        fig.suptitle("Validation metrics")
        written.append(_save(fig, target / "val_metrics.png"))

    val_loss_series = _present_series(val, _VAL_LOSS_KEYS)
    if val_loss_series:
        target.mkdir(parents=True, exist_ok=True)
        iterations = [entry["iteration"] for entry in val]
        marker = "o" if len(iterations) <= 50 else None
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key, values in val_loss_series:
            ax.plot(iterations, values, label=key, linewidth=1.2, marker=marker)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.set_title("Validation losses")
        ax.legend()
        ax.grid(alpha=0.3)
        written.append(_save(fig, target / "val_losses.png"))

    return written