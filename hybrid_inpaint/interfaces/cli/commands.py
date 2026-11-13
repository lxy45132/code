"""CLI commands.

config/data/train/evaluate/inpaint/run 已实现；start/gui 等待 C7/C8，
在对应施工步骤落地前以退出码 3 提示。
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

db_app = typer.Typer(
    name="db",
    no_args_is_help=True,
    help="Database utilities (init / migrate).",
)
app.add_typer(db_app, name="db")

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


@db_app.command("init")
def db_init(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
) -> None:
    """Create the database and run Alembic migrations to head."""
    import re

    parsed = _load_or_exit(config)
    backend = parsed.system.storage.backend
    project_root = Path(__file__).resolve().parents[3]
    try:
        if backend == "mysql":
            import pymysql
            from pymysql.err import Error as MySqlError

            mysql = parsed.system.storage.mysql
            if not re.fullmatch(r"[A-Za-z0-9_]+", mysql.database):
                typer.echo(
                    f"[db error] database name must be alnum/underscore: "
                    f"{mysql.database}",
                    err=True,
                )
                raise typer.Exit(code=2)
            try:
                connection = pymysql.connect(
                    host=mysql.host,
                    port=mysql.port,
                    user=mysql.user,
                    password=mysql.password,
                )
            except MySqlError as exc:
                typer.echo(
                    f"[db error] cannot reach MySQL at {mysql.host}:{mysql.port}: "
                    f"{exc}",
                    err=True,
                )
                raise typer.Exit(code=2) from exc
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"CREATE DATABASE IF NOT EXISTS `{mysql.database}` "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
            finally:
                connection.close()
        _upgrade_to_head(project_root, parsed)
    except ImportError as exc:
        typer.echo(
            f"[db error] web/db dependencies missing ({exc}); "
            "run: uv sync --extra dev --extra train --extra web",
            err=True,
        )
        raise typer.Exit(code=2) from exc
    except (OSError, RuntimeError) as exc:
        typer.echo(f"[db error] {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"database ready (backend={backend})")


def _upgrade_to_head(project_root: Path, parsed: AppConfig) -> None:
    """Apply schema: Alembic ``upgrade head`` (mysql) or metadata (sqlite)."""
    if parsed.system.storage.backend == "sqlite":
        # Alembic 管的是 MySQL schema；SQLite 后端复用同一 ORM 元数据建表。
        from sqlalchemy import create_engine

        from hybrid_inpaint.infrastructure.sqlalchemy_history import ensure_schema

        db_path = parsed.system.history_db
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_schema(create_engine(f"sqlite:///{db_path}"))
        return
    from alembic import command
    from alembic.config import Config as AlembicConfig

    alembic_ini = project_root / "alembic.ini"
    if not alembic_ini.is_file():
        raise RuntimeError(f"alembic.ini not found at {alembic_ini}")
    alembic_cfg = AlembicConfig(str(alembic_ini))
    alembic_cfg.set_main_option("script_location", str(project_root / "alembic"))
    command.upgrade(alembic_cfg, "head")


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


class _CliProgressReporter:
    def start(self, total: int) -> None:
        typer.echo(f"batch: {total} item(s)")

    def item(self, index: int, item: Any, result: Any) -> None:
        typer.echo(f"  [{index + 1}] {item.relative_path}: {result.status}")

    def finish(self, report: Any) -> None:
        typer.echo(
            f"batch {report.batch_id}: ok={report.ok} failed={report.failed} "
            f"cancelled={report.cancelled}"
        )


def _write_batch_report(path: Path, report: Any) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["relative_path", "mask_path", "status", "output_path",
             "latency_ms", "mask_ratio", "error"]
        )
        for result in report.results:
            writer.writerow(
                [
                    result.relative_path,
                    result.mask_path or "",
                    result.status,
                    result.output_path or "",
                    f"{result.latency_ms:.3f}" if result.latency_ms is not None else "",
                    f"{result.mask_ratio:.6f}" if result.mask_ratio is not None else "",
                    result.error or "",
                ]
            )


@app.command("run")
def run(
    input_dir: Path = typer.Option(..., "--input-dir", exists=True, file_okay=False),
    mask_dir: Path = typer.Option(..., "--mask-dir", exists=True, file_okay=False),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False),
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    recursive: bool = typer.Option(False, "--recursive"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    report: Path | None = typer.Option(None, "--report", dir_okay=False),
    device: str | None = typer.Option(None, "--device"),
) -> None:
    """Batch inpaint a directory of images and masks."""
    from hybrid_inpaint.bootstrap import build_container
    from hybrid_inpaint.domain.entities import BatchRequest
    from hybrid_inpaint.domain.errors import AppError

    overrides = {"system": {"device": device}} if device else None
    try:
        container = build_container(config, overrides)
        batch_report = container.batch.execute(
            BatchRequest(input_dir, mask_dir, output_dir, recursive, overwrite, limit),
            reporter=_CliProgressReporter(),
        )
        report_path = report or (output_dir / "report.csv")
        _write_batch_report(report_path, batch_report)
    except ConfigError as exc:
        typer.echo(f"[config error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except AppError as exc:
        typer.echo(f"[{exc.code}] {exc.user_message}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc
    except OSError as exc:
        typer.echo(f"[io error] {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"report={report_path}")
    raise typer.Exit(code=1 if batch_report.failed else 0)


@app.command("inpaint")
def inpaint(
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False),
    mask: Path = typer.Option(..., "--mask", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", dir_okay=False),
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    device: str | None = typer.Option(None, "--device", help="Override system.device."),
    note: str = typer.Option("", "--note"),
) -> None:
    """Inpaint a single image."""
    from hybrid_inpaint.bootstrap import build_container
    from hybrid_inpaint.domain.entities import InpaintRequest
    from hybrid_inpaint.domain.errors import AppError

    overrides = {"system": {"device": device}} if device else None
    try:
        container = build_container(config, overrides)
        decoded = container.codec.decode(image.read_bytes(), container.limits)
        mask_obj = container.codec.decode_mask(
            mask.read_bytes(), height=decoded.height, width=decoded.width
        )
        result = container.inpaint.execute(
            InpaintRequest(decoded, mask_obj, mode="cli", note=note)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(container.codec.encode_png(result.image))
    except ConfigError as exc:
        typer.echo(f"[config error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except AppError as exc:
        typer.echo(f"[{exc.code}] {exc.user_message}", err=True)
        raise typer.Exit(code=exc.exit_code) from exc
    except OSError as exc:
        typer.echo(f"[io error] {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"job_id={result.job_id} latency_ms={result.latency_ms:.1f} "
        f"model_version={result.model_version} mask_ratio={result.mask_ratio:.4f}"
    )
    typer.echo(f"output={output}")


@app.command("plot")
def plot(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
    output_dir: Path | None = typer.Option(None, "--output-dir", file_okay=False),
) -> None:
    """Render loss/metric curves from a run's train.jsonl / val.jsonl."""
    from hybrid_inpaint.training.engine.plots import plot_run

    try:
        paths = plot_run(run_dir, output_dir)
    except ImportError as exc:
        typer.echo(
            f"[plot error] matplotlib missing ({exc}); run: uv sync --extra dev --extra train",
            err=True,
        )
        raise typer.Exit(code=2) from exc
    except (ValueError, OSError) as exc:
        typer.echo(f"[plot error] {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not paths:
        typer.echo("no train/val logs yet; nothing to plot")
        return
    for path in paths:
        typer.echo(f"plot={path}")
