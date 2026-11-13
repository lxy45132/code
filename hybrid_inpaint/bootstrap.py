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
    BatchCancellation,
    BatchStateRepository,
    Cache,
    EventBus,
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
    engine: InpaintEngine | None  # worker 模式的 api 进程不加载模型
    artifacts: ArtifactStore
    history: HistoryRepository
    inpaint: InpaintImage | None
    batch: BatchInpaint | None
    history_service: HistoryService
    model_service: ModelService
    batch_state: BatchStateRepository
    events: EventBus
    cache: Cache
    cancellation: BatchCancellation


def _build_storage(
    config: AppConfig, history: HistoryRepository | None
) -> tuple[HistoryRepository, BatchStateRepository]:
    """按 storage.backend 装配历史 + 批量状态仓储（03 文档 §6）。

    mysql：共用一个 SQLAlchemy 引擎（建表依赖 alembic upgrade head）；
    sqlite：历史用原零依赖适配器，批量表用 SQLAlchemy 同库文件。
    """
    from hybrid_inpaint.infrastructure.sqlalchemy_batch import SqlAlchemyBatchRepository
    from hybrid_inpaint.infrastructure.sqlalchemy_history import ensure_schema

    paths_base = config.system.output_dir.parent
    if config.system.storage.backend == "mysql":
        from hybrid_inpaint.infrastructure.sqlalchemy_history import (
            SqlAlchemyHistoryRepository,
            build_mysql_engine,
        )

        mysql = config.system.storage.mysql
        engine = build_mysql_engine(
            mysql.host, mysql.port, mysql.user, mysql.password, mysql.database
        )
        resolved = history or SqlAlchemyHistoryRepository(engine, paths_base=paths_base)
    else:
        from sqlalchemy import create_engine

        resolved = history or SqliteHistoryRepository(
            config.system.history_db, paths_base=paths_base
        )
        engine = create_engine(
            f"sqlite:///{config.system.history_db.as_posix()}",
            connect_args={"timeout": 30},  # api/worker 跨进程共库的锁等待
        )
        ensure_schema(engine)
    batch_state = SqlAlchemyBatchRepository(engine, paths_base=paths_base)
    return resolved, batch_state


def build_container(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    *,
    engine: InpaintEngine | None = None,
    codec: ImageCodec | None = None,
    history: HistoryRepository | None = None,
    events: EventBus | None = None,
    cache: Cache | None = None,
    cancellation: BatchCancellation | None = None,
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
    resolved_history, batch_state = _build_storage(config, history)

    from hybrid_inpaint.infrastructure.celery_cancel import RedisBatchCancellation
    from hybrid_inpaint.infrastructure.redis_events import (
        RedisCache,
        RedisEventBus,
        build_redis_client,
    )

    redis_client = build_redis_client(config.system.redis.url)  # 懒连接
    resolved_events = events or RedisEventBus(redis_client)
    resolved_cache = cache or RedisCache(redis_client)
    resolved_cancellation = cancellation or RedisBatchCancellation(redis_client)

    if engine is None and config.system.inference.mode == "worker":
        # worker 模式：模型只在 Celery worker 进程加载，api 无状态（03 文档 §4）
        model_service = ModelService(None, engine=None, model_id=config.system.model_id)
    elif engine is None:
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
    inpaint = (
        InpaintImage(engine, artifacts, resolved_history, clock, ids, limits)
        if engine is not None
        else None
    )
    batch = (
        BatchInpaint(inpaint, resolved_codec, FileSystemBatchFiles(), ids, limits)
        if inpaint is not None
        else None
    )
    history_service = HistoryService(resolved_history, artifacts)
    return Container(
        config=config,
        limits=limits,
        codec=resolved_codec,
        engine=engine,
        artifacts=artifacts,
        history=resolved_history,
        inpaint=inpaint,
        batch=batch,
        history_service=history_service,
        model_service=model_service,
        batch_state=batch_state,
        events=resolved_events,
        cache=resolved_cache,
        cancellation=resolved_cancellation,
    )