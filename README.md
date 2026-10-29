# Hybrid Inpaint

基于 U-Net 与瓶颈 Transformer（ViT）、GAN 两阶段训练的图像修复系统，配套 Qt 桌面端与 Gradio/FastAPI Web 端。

## 当前状态

施工中。当前完成 **C0：工程骨架、配置契约、CLI 骨架、基础测试**。模型、数据、训练、系统模块按 `docs/` 中的施工方案逐步加入。

## 文档

- [总体施工方案](docs/00_总体施工方案.md)
- [模型与训练方案](docs/01_模型与训练方案.md)
- [系统设计方案](docs/02_系统设计方案.md)
- [施工进度](docs/PROGRESS.md)
- [C0 handoff](docs/handoffs/HANDOFF_C0.md)

## 环境

```bash
uv venv --python 3.10 .venv
uv sync --extra dev
uv run --env-file .env pytest
```

> 项目路径含中文（`毕设`），Python 3.10 在 Windows 上读取 editable `.pth` 时会用 GBK 解码而失败。复制 `.env.example` 为 `.env`（已含 `PYTHONUTF8=1`），统一用 `uv run --env-file .env ...` 运行 Python 命令。

GPU 训练依赖（较大）：

```bash
uv sync --extra dev --extra train
```

> `uv sync` 是精确同步：只写 `--extra train` 会移除 pytest 等 dev 依赖。

Web / 桌面依赖：

```bash
uv sync --extra web
uv sync --extra gui
```

## 常用命令

```bash
uv run --env-file .env hybrid-inpaint config validate --config configs/base.yaml
uv run --env-file .env hybrid-inpaint data freeze-synthetic --output-dir data/synthetic/frozen --count 16 --size 256 --seed 2026
uv run --env-file .env hybrid-inpaint start --config configs/system.yaml   # C7 之后可用
uv run --env-file .env hybrid-inpaint gui --config configs/system.yaml     # C8 之后可用
uv run --env-file .env hybrid-inpaint run --input-dir ... --mask-dir ...   # C6 之后可用
```

## 目录

```text
hybrid_inpaint/
  domain/          # 实体、端口、错误、纯函数
  application/     # 用例
  infrastructure/  # torch / pillow / sqlite / 文件系统适配器
  interfaces/      # CLI / FastAPI / Gradio / Qt
  models/          # nn.Module
  training/        # 数据、损失、指标、训练器
  bootstrap.py     # 组合根
  config.py        # pydantic 配置
```
