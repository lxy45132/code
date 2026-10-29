"""CLI commands.

C0 only implements ``config validate``; the other commands are registered now
so the command surface is stable, but they exit with code 3 until the
corresponding construction step lands.
"""

from __future__ import annotations

from pathlib import Path

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


def _load_or_exit(config_path: Path) -> AppConfig:
    try:
        return load_config(config_path)
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
