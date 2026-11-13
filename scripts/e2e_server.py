"""E2E API 服务：uvicorn :8000，托管 frontend/dist（需先 npm run build）。

启动：uv run --no-sync python -m scripts.e2e_server
前置：Redis 已运行（127.0.0.1:6379）。
"""

from __future__ import annotations

import uvicorn

from hybrid_inpaint.interfaces.api.app import create_app
from scripts.e2e_common import build_e2e_container

container = build_e2e_container()
app = create_app(container)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
