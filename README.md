# Hybrid Inpaint

基于 U-Net 与瓶颈 Transformer（ViT）、GAN 两阶段训练的图像修复系统，配套 Qt 桌面端与 Gradio/FastAPI Web 端。

## 当前状态

施工中。已完成 **C0（工程骨架、配置契约、CLI、基础测试）**、**C1（合成人脸/场景数据、掩码、数据实体、冻结观测）** 与 **C2（纯 U-Net B0、损失、指标、训练器、checkpoint）**；下一步 C3（瓶颈 ViT、开关、参数与梯度检查）。

项目根目录：`D:\Code\bs\code`（ASCII 路径，不需要 `PYTHONUTF8` 或 `--env-file`）。

## 文档

- [总体施工方案](docs/00_总体施工方案.md)
- [模型与训练方案](docs/01_模型与训练方案.md)
- [系统设计方案](docs/02_系统设计方案.md)
- [施工进度](docs/PROGRESS.md)
- [C0 handoff](docs/handoffs/HANDOFF_C0.md)
- [C1 handoff](docs/handoffs/HANDOFF_C1.md)
- [C2 handoff](docs/handoffs/HANDOFF_C2.md)

## 环境

```bash
uv venv --python 3.10 .venv
uv sync --extra dev
uv run --no-sync pytest
```

GPU 训练依赖（较大，与 dev 一起装）：

```bash
uv sync --extra dev --extra train
```

> `uv sync` 是精确同步：只写 `--extra train` 会移除 pytest 等 dev 依赖。装好额外的 `train`/`web`/`gui` 后，用 `uv run --no-sync ...` 运行命令，避免 `uv run` 再次同步时移除这些可选项。

Web / 桌面依赖：

```bash
uv sync --extra dev --extra web
uv sync --extra dev --extra gui
```

## 常用命令

```bash
uv run --no-sync hybrid-inpaint config validate --config configs/base.yaml
uv run --no-sync hybrid-inpaint data freeze-synthetic --output-dir data/synthetic/frozen --count 16 --size 256 --seed 2026
uv run --no-sync hybrid-inpaint train --config configs/smoke.yaml --iterations 400
uv run --no-sync hybrid-inpaint train --config configs/smoke.yaml --resume outputs/smoke/seed_2026/checkpoints/last.pt
uv run --no-sync hybrid-inpaint start --config configs/system.yaml   # C7 之后可用
uv run --no-sync hybrid-inpaint gui --config configs/system.yaml     # C8 之后可用
uv run --no-sync hybrid-inpaint run --input-dir ... --mask-dir ...   # C6 之后可用
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
