# 多阶段构建：前端 dist + Python 运行时（api 与 worker 共用镜像）。
# 注意：train extra 含 torch（CUDA 轮子体积较大）；compose 演示为 CPU 推理，
# 生产 GPU worker 建议宿主机直跑（hybrid-inpaint worker），见 docs/03 §10。

FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.10-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ENV UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra web --extra train --no-install-project
COPY hybrid_inpaint ./hybrid_inpaint
COPY alembic ./alembic
COPY alembic.ini ./
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --extra web --extra train
COPY --from=frontend /fe/dist ./frontend/dist

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "hybrid-inpaint", "start", "-c", "configs/system.yaml", "--host", "0.0.0.0"]
