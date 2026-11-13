"""E2E Celery worker：solo pool + gpu 队列，注入 FakeEngine。

启动：uv run --no-sync python -m scripts.e2e_worker
E2E_ITEM_DELAY 控制单项推理耗时（默认 0.8s，取消场景的时间窗口）。
"""

from __future__ import annotations

import os

from hybrid_inpaint.interfaces.api.celery_app import celery_app
from hybrid_inpaint.interfaces.api.tasks.common import set_worker_container
from scripts.e2e_common import build_e2e_container

if __name__ == "__main__":
    delay = float(os.environ.get("E2E_ITEM_DELAY", "0.8"))
    set_worker_container(build_e2e_container(delay))
    celery_app.worker_main(
        argv=[
            "worker",
            "--pool=solo",
            "--queues=gpu",
            "--concurrency=1",
            "--loglevel=WARNING",
        ]
    )
