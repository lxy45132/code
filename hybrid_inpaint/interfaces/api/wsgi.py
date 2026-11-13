"""uvicorn 多 worker 入口（import string 形式，每进程独立装配）。

配置路径经 HYBRID_INPAINT_CONFIG 环境变量传入（start CLI 设置）。
worker 模式多进程要求 mysql backend（SystemConfig 跨字段校验）。
"""

from __future__ import annotations

import os

from hybrid_inpaint.bootstrap import build_container
from hybrid_inpaint.interfaces.api.app import create_app

app = create_app(
    build_container(os.environ.get("HYBRID_INPAINT_CONFIG", "configs/system.yaml"))
)
