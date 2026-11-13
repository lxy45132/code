"""Celery 实例：Redis broker/backend，gpu 专队（docs/03 §5）。

Windows 启动方式（C7g 封装为 CLI）：
    celery -A hybrid_inpaint.interfaces.api.celery_app worker --pool=solo -Q gpu -c 1
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

from celery import Celery

from hybrid_inpaint.config import AppConfig, load_config

CONFIG_ENV = "HYBRID_INPAINT_CONFIG"
DEFAULT_CONFIG = "configs/system.yaml"


def redis_db_url(url: str, db: int) -> str:
    """业务库 URL 换 db 索引（broker/backend 与业务分库）。"""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, f"/{db}", parts.query, parts.fragment))


def make_celery(config: AppConfig) -> Celery:
    app = Celery(
        "hybrid_inpaint",
        broker=redis_db_url(config.system.redis.url, config.system.celery.broker_db),
        backend=redis_db_url(config.system.redis.url, config.system.celery.result_db),
        include=[
            "hybrid_inpaint.interfaces.api.tasks.batch_chunk",
            "hybrid_inpaint.interfaces.api.tasks.batch_finalize",
            "hybrid_inpaint.interfaces.api.tasks.inpaint_one",
        ],
    )
    app.conf.update(
        task_default_queue="gpu",
        task_acks_late=True,  # worker 崩溃后任务可重投（幂等兜底）
        worker_prefetch_multiplier=1,  # 单 GPU 串行，不多预取
        task_track_started=True,
        broker_connection_retry_on_startup=True,
        result_expires=86_400,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
    )
    return app


def load_worker_config() -> AppConfig:
    return load_config(os.environ.get(CONFIG_ENV, DEFAULT_CONFIG))


celery_app = make_celery(load_worker_config())
