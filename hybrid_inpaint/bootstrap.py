"""Composition root: config -> adapters -> use cases.

唯一装配点；其他模块不创建全局单例。torch 适配器仅在需要真实
引擎时延迟导入，fake 注入测试不加载 torch。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hybrid_inpaint.application.batch import BatchInpaint
from hybrid_inpaint.application.history import HistoryService
from hybrid_inpaint.application.inpaint import InpaintImage
from hybrid_inpaint.application.model import ModelService
from hybrid_inpaint.config import AppConfig, load_config
from hybrid_inpaint.domain.entities import Limits, ModelSpec
from hybrid_inpaint.domain.ports import (
    ArtifactStore,
    HistoryRepository,
    ImageCodec,
    InpaintEngine,
)
from hybrid_inpaint.infrastructure.clock import SystemClock, UuidGenerator
from hybrid_inpaint.infrastructure.fs_artifacts import FileSystemArtifactStore
from hybrid_inpaint.infrastructure.fs_batch import FileSystemBatchFiles
from hybrid_inpaint.infrastructure.logging import setup_logging
from hybrid_inpaint.infrastructure.pillow_codec import PillowImageCodec
from hybrid_inpaint.infrastructure.sqlite_history import SqliteHistoryRepository


@dataclass
class Container:
    config: AppConfig
    limits: Limits
    codec: ImageCodec
    engine: InpaintEngine
    artifacts: ArtifactStore
    history: HistoryRepository
    inpaint: InpaintImage
    batch: BatchInpaint
    history_service: HistoryService
    model_service: ModelService


def _build_history(config: AppConfig) -> HistoryRepository:
    """Select the history adapter by system.storage.backend (03 文档 §6).

    mysql 后端返回 SQLAlchemy 适配器（建表依赖 alembic upgrade head，
    由 ``db init`` / 部署脚本负责）；sqlite 后端保持原零依赖行为。
    """
    if config.system.storage.backend == "mysql":
        from hybrid_inpaint.infrastructure.sqlalchemy_history import (
            SqlAlchemyHistoryRepository,
            build_mysql_engine,
        )

        mysql = config.system.storage.mysql
        engine = build_mysql_engine(
            mysql.host, mysql.port, mysql.user, mysql.password, mysql.database
        )
        return SqlAlchemyHistoryRepository(
            engine, paths_base=config.system.output_dir.parent
        )
    return SqliteHistoryRepository(
        config.system.history_db, paths_base=config.system.output_dir.parent
    )


def build_container(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    *,
    engine: InpaintEngine | None = None,
    codec: ImageCodec | None = None,
    history: HistoryRepository | None = None,
) -> Container:
    config = load_config(config_path, overrides)
    setup_logging(config.system.history_db.parent / "system.log")
    resolved_codec = codec or PillowImageCodec()
    limits = Limits(
        max_file_mb=config.system.limits.max_file_mb,
        max_side=config.system.limits.max_side,
        max_pixels=config.system.limits.max_pixels,
    )
    clock = SystemClock()
    ids = UuidGenerator()
    artifacts = FileSystemArtifactStore(config.system.output_dir, resolved_codec)
    if history is None:
        history = _build_history(config)
    if engine is None:
        from hybrid_inpaint.infrastructure.fs_models import FileSystemModelRepository
        from hybrid_inpaint.infrastructure.torch_engine import SerializedEngine

        repository = FileSystemModelRepository(
            (ModelSpec(config.system.model_id, config.system.checkpoint),),
            device=config.system.device,
            pad_multiple=config.system.pad_multiple,
            tile_size=config.system.tiling.tile_size,
            tile_overlap=config.system.tiling.tile_overlap,
        )
        engine = SerializedEngine(
            repository.load(repository.resolve(config.system.model_id)),
            timeout_s=config.system.busy_timeout_s,
        )
        engine.warmup()
        model_service = ModelService(
            repository, engine=engine, model_id=config.system.model_id
        )
    else:
        model_service = ModelService(
            None, engine=engine, model_id=config.system.model_id
        )
    inpaint = InpaintImage(engine, artifacts, history, clock, ids, limits)
    batch = BatchInpaint(inpaint, resolved_codec, FileSystemBatchFiles(), ids, limits)
    history_service = HistoryService(history, artifacts)
    return Container(
        config=config,
        limits=limits,
        codec=resolved_codec,
        engine=engine,
        artifacts=artifacts,
        history=history,
        inpaint=inpaint,
        batch=batch,
        history_service=history_service,
        model_service=model_service,
    )