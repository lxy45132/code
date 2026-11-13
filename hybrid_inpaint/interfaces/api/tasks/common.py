"""worker 进程 Container 单例（懒装配；测试可注入 fake）。

api 进程不 import 本模块的 get_worker_container 运行时路径——
任务只在 Celery worker 内执行（inline 模式 api 直接调用用例）。
"""

from __future__ import annotations

import os

from hybrid_inpaint.bootstrap import Container, build_container
from hybrid_inpaint.interfaces.api.celery_app import CONFIG_ENV, DEFAULT_CONFIG

_container: Container | None = None


def get_worker_container() -> Container:
    global _container
    if _container is None:
        _container = build_container(os.environ.get(CONFIG_ENV, DEFAULT_CONFIG))
    return _container


def set_worker_container(container: Container | None) -> None:
    """测试注入/重置（None 恢复懒装配）。"""
    global _container
    _container = container
