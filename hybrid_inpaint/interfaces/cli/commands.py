"""CLI commands.

C0 only implements ``config validate``; the other commands are registered now
so the command surface is stable, but they exit with code 3 until the
corresponding construction step lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from hybrid_inpaint.config import AppConfig, ConfigError, load_config

app = typer.Typer(
    name="hybrid-inpaint",
    no_args_is_help=True,
    add_completion=False,
    help="Hybrid U-Net + ViT + GAN image inpainting system.",
)

config_app = typer.Typer(
    name="config",
    no_args_is_help=True,
    help="Configuration utilities.",
)
app.add_typer(config_app, name="config")

data_app = typer.Typer(
    name="data",
    no_args_is_help=True,
    help="Development data utilities.",
)
app.add_typer(data_app, name="data")


def _load_or_exit(
    config_path: Path,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    try:
        return load_config(config_path, overrides=overrides)
    except ConfigError as exc:
        typer.echo(f"[config error] {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _not_implemented(command: str, step: str) -> None:
    typer.echo(
        f"[{command}] not implemented yet; scheduled for {step}.",
        err=True,
    )
    raise typer.Exit(code=3)


@config_app.command("validate")
def config_validate(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a YAML config file.",
    ),
) -> None:
    """Parse and validate a config file without loading a model."""
    parsed = _load_or_exit(config)
    typer.echo(
        f"CONFIG OK: experiment_id={parsed.experiment_id} "
        f"seed={parsed.seed} contract_version={parsed.contract_version}"
    )
    typer.echo(
        f"  model={parsed.model.name} channels={parsed.model.channels} "
        f"vit_depth={parsed.model.vit_depth} vit_heads={parsed.model.vit_heads}"
    )
    typer.echo(
        f"  data={parsed.data.dataset} root={parsed.data.root} "
        f"image_size={parsed.data.image_size}"
    )
    typer.echo(
        f"  system.device={parsed.system.device} "
        f"checkpoint={parsed.system.checkpoint}"
    )


@data_app.command("freeze-synthetic")
def data_freeze_synthetic(
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False),
    count: int = typer.Option(16, "--count", min=1),
    size: int = typer.Option(256, "--size", min=32),
    seed: int = typer.Option(2026, "--seed", min=0),
) -> None:
    """Write a deterministic synthetic image/mask set and a hashed manifest."""
    from hybrid_inpaint.training.data.synthetic import save_frozen_set

    try:
        manifest = save_frozen_set(output_dir, size=size, count=count, seed=seed)
    except (ValueError, OSError) as exc:
        typer.echo(f"[data error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"frozen {manifest['count']} samples at {output_dir} "
        f"(size={manifest['size']}, seed={manifest['seed']})"
    )


@data_app.command("prepare")
def data_prepare(
    raw_dir: Path = typer.Option(
        Path("data/raw"), "--raw-dir", file_okay=False
    ),
    manifest_dir: Path = typer.Option(
        Path("data/manifests"), "--manifest-dir", file_okay=False
    ),
    processed_dir: Path = typer.Option(
        Path("data/processed"), "--processed-dir", file_okay=False
    ),
    seed: int = typer.Option(2026, "--seed", min=0),
) -> None:
    """Build frozen Places365/CelebA/mask manifests with SHA256 records."""
    from hybrid_inpaint.training.data.prepare import prepare_all

    try:
        summaries = prepare_all(
            raw_dir,
            manifest_dir,
            processed_dir,
            seed=seed,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        typer.echo(f"[data error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    places = summaries["places"]
    celeba = summaries["celeba"]
    masks = summaries["masks"]
    typer.echo(
        f"places: train={places['train']['count']} "
        f"val={places['val']['count']} test={places['test']['count']}"
    )
    typer.echo(
        f"celeba: train={celeba['train']['count']} "
        f"val={celeba['val']['count']} test={celeba['test']['count']}"
    )
    typer.echo(
        f"masks: train={masks['train']['count']} "
        f"val={masks['val']['count']} test={masks['test']['count']} "
        f"buckets={masks['bucket_counts']}"
    )
    typer.echo(
        f"pairs: places_test={summaries['places_pairs']['count']} "
        f"celeba_test={summaries['celeba_pairs']['count']} "
        f"places_val={summaries['places_val_pairs']['count']} "
        f"celeba_val={summaries['celeba_val_pairs']['count']} "
        f"places_stress={summaries['places_stress_pairs']['count']} "
        f"celeba_stress={summaries['celeba_stress_pairs']['count']}"
    )
    typer.echo(
        "manifest sha256: "
        f"places_train={places['train']['sha256']} "
        f"masks_test={masks['test']['sha256']}"
    )


@app.command("train")
def train(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        help="Override train.output_dir (absolute or relative to cwd).",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Override train.device (cuda/cpu/mps).",
    ),
    iterations: int | None = typer.Option(
        None,
        "--iterations",
        min=1,
        help="Override train.iterations.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        min=0,
        help="Override the experiment and training seed.",
    ),
    resume: Path | None = typer.Option(
        None,
        "--resume",
        exists=True,
        dir_okay=False,
        help=(
            "Stage-one best.pt to initialise stage two, or a stage-two "
            "checkpoint to continue."
        ),
    ),
) -> None:
    """Train the generator (stage one or stage two) on the configured dataset."""
    from hybrid_inpaint.training.data.factory import build_training_datasets
    from hybrid_inpaint.training.engine.checkpoint import CheckpointError
    from hybrid_inpaint.training.engine.trainer import Trainer

    overrides: dict[str, Any] = {}
    train_overrides: dict[str, Any] = {}
    if output_dir is not None:
        train_overrides["output_dir"] = str(output_dir.resolve())
    if device is not None:
        train_overrides["device"] = device
    if iterations is not None:
        train_overrides["iterations"] = iterations
    if seed is not None:
        overrides["seed"] = seed
        train_overrides["seed"] = seed
    if train_overrides:
        overrides["train"] = train_overrides

    parsed = _load_or_exit(config, overrides)

    try:
        train_dataset, val_dataset, identity = build_training_datasets(parsed)
        trainer = Trainer(
            parsed,
            train_dataset,
            val_dataset,
            data_identity=identity,
            device=device,
            resume_from=resume,
        )
        result = trainer.fit()
    except (
        CheckpointError,
        ValueError,
        RuntimeError,
        NotImplementedError,
        FloatingPointError,
    ) as exc:
        typer.echo(f"[train error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    lpips_text = (
        f", best LPIPS={result.best_lpips:.4f}"
        if result.best_lpips is not None
        else ""
    )
    typer.echo(
        f"trained {result.start_iteration}->{result.final_iteration} iterations "
        f"at {result.run_dir} "
        f"(stage={result.stage}, best hole PSNR={result.best_psnr_hole:.3f} dB"
        f"{lpips_text}{', early stopped' if result.stopped_early else ''})"
    )
    typer.echo(f"  last={result.run_dir / 'checkpoints' / 'last.pt'}")
    if result.stage == "stage2":
        typer.echo(f"  best_psnr={result.run_dir / 'checkpoints' / 'best_psnr.pt'}")
        if result.best_lpips is not None:
            typer.echo(
                f"  best_lpips={result.run_dir / 'checkpoints' / 'best_lpips.pt'}"
            )
    else:
        typer.echo(f"  best={result.run_dir / 'checkpoints' / 'best.pt'}")


@app.command("evaluate")
def evaluate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", exists=True, dir_okay=False
    ),
    dataset: str = typer.Option(
        ..., "--dataset", help="places2 or celeba_hq"
    ),
    pairs: Path | None = typer.Option(
        None,
        "--pairs",
        exists=True,
        dir_okay=False,
        help="Pair manifest; defaults to <manifest_dir>/<dataset>_test_pairs.jsonl.",
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", file_okay=False
    ),
    device: str | None = typer.Option(None, "--device"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    batch_size: int | None = typer.Option(None, "--batch-size", min=1),
    image_size: int | None = typer.Option(None, "--image-size", min=16),
    tile_size: int | None = typer.Option(None, "--tile-size", min=16),
    tile_overlap: int = typer.Option(64, "--tile-overlap", min=0),
) -> None:
    """Evaluate a checkpoint on frozen image/mask pairs."""
    from hybrid_inpaint.training.engine.checkpoint import CheckpointError
    from hybrid_inpaint.training.engine.evaluator import (
        evaluate_checkpoint,
        write_evaluation,
    )

    if dataset not in ("places2", "celeba_hq"):
        typer.echo("[eval error] --dataset must be places2 or celeba_hq", err=True)
        raise typer.Exit(code=2)
    parsed = _load_or_exit(config)
    prefix = "places365" if dataset == "places2" else "celeba_hq"
    pair_path = pairs or (parsed.data.manifest_dir / f"{prefix}_test_pairs.jsonl")
    target = output_dir or (Path("outputs") / "eval" / checkpoint.stem)

    try:
        result = evaluate_checkpoint(
            parsed,
            checkpoint,
            pair_path,
            dataset=dataset,
            device=device,
            limit=limit,
            batch_size=batch_size,
            image_size=image_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        pairs_path, summary_path, markdown_path = write_evaluation(
            result,
            target,
            checkpoint=checkpoint,
            dataset=dataset,
            image_size=image_size or parsed.data.image_size,
        )
    except (
        CheckpointError,
        ImportError,
        ValueError,
        RuntimeError,
        FileNotFoundError,
        OSError,
    ) as exc:
        typer.echo(f"[eval error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    summary = result.summary
    typer.echo(
        f"evaluated {summary['pairs']} pairs / {summary['images']} images "
        f"({dataset}, image_size={image_size or parsed.data.image_size})"
    )
    for key in ("psnr_hole", "psnr_full", "ssim", "mae_hole", "mae_full", "lpips"):
        stats = summary["metrics"].get(key)
        if stats:
            typer.echo(
                f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f} "
                f"(n={stats['count']})"
            )
    typer.echo(f"  pairs_csv={pairs_path}")
    typer.echo(f"  summary={summary_path}")
    typer.echo(f"  markdown={markdown_path}")


@app.command("start")
def start(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
) -> None:
    """Start the FastAPI + Gradio web application."""
    _load_or_exit(config)
    _not_implemented("start", "C7")


@app.command("gui")
def gui(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
) -> None:
    """Start the Qt desktop application."""
    _load_or_exit(config)
    _not_implemented("gui", "C8")


@app.command("run")
def run(
    input_dir: Path = typer.Option(..., "--input-dir", exists=True, file_okay=False),
    mask_dir: Path = typer.Option(..., "--mask-dir", exists=True, file_okay=False),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False),
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
) -> None:
    """Batch inpaint a directory of images and masks."""
    _load_or_exit(config)
    _not_implemented("run", "C6")


@app.command("inpaint")
def inpaint(
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False),
    mask: Path = typer.Option(..., "--mask", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
) -> None:
    """Inpaint a single image."""
    _load_or_exit(config)
    _not_implemented("inpaint", "C6")
