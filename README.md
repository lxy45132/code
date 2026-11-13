# Hybrid Inpaint

基于 U-Net 与瓶颈 Transformer（ViT）、GAN 两阶段训练的图像修复系统，配套 Qt 桌面端与 Gradio/FastAPI Web 端。

## 当前状态

施工中。C0–C6 已完成：算法管线（U-Net + ViT + GAN 两阶段）、T0 真实数据与冻结 manifest、固定评测 CLI、512/1024 分块推理、大洞压力表，以及 C6 系统核心（分层用例、Torch/Pillow/SQLite/文件系统适配器、组合根、`inpaint`/`run` CLI）均已就绪，248 个测试通过；dry-run checkpoint 端到端冒烟通过。训练每次验证自动存“原图|掩码|修复”对比图（`samples/`）并刷新损失/指标曲线（`plots/`）。正式 B0/B1/B2/Main 训练待 24 GB 云卡；C7（FastAPI + Gradio）/C8（Qt）未开始。

项目根目录：`D:\Code\bs\code`（ASCII 路径，不需要 `PYTHONUTF8` 或 `--env-file`）。

## 文档

> docs 目录仅本地维护，不纳入 Git（以下链接在 GitHub 上不可见）。

- [总体施工方案](docs/00_总体施工方案.md)
- [模型与训练方案](docs/01_模型与训练方案.md)
- [系统设计方案](docs/02_系统设计方案.md)
- [施工进度](docs/PROGRESS.md)
- [C0 handoff](docs/handoffs/HANDOFF_C0.md)
- [C1 handoff](docs/handoffs/HANDOFF_C1.md)
- [C2 handoff](docs/handoffs/HANDOFF_C2.md)
- [C3 handoff](docs/handoffs/HANDOFF_C3.md)
- [C4 handoff](docs/handoffs/HANDOFF_C4.md)
- [C5 handoff（进行中）](docs/handoffs/HANDOFF_C5.md)
- [C6 handoff](docs/handoffs/HANDOFF_C6.md)
- [实验矩阵与运行手册](docs/EXPERIMENT_MATRIX.md)
- [给下一个 Agent 的详细交接文档](docs/handoffs/HANDOFF_NEXT_AGENT.md)

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
uv run --no-sync hybrid-inpaint data prepare --raw-dir data/raw --manifest-dir data/manifests --processed-dir data/processed --seed 2026
uv run --no-sync hybrid-inpaint train --config configs/smoke.yaml --iterations 400
uv run --no-sync hybrid-inpaint train --config configs/dryrun_places.yaml
uv run --no-sync hybrid-inpaint evaluate -c configs/dryrun_places.yaml \
  --checkpoint outputs/dryrun_places/seed_2026/checkpoints/best.pt \
  --dataset places2 --limit 10 --output-dir outputs/eval_dryrun
# 高分辨率 / 大洞压力
uv run --no-sync hybrid-inpaint evaluate -c configs/dryrun_celeba.yaml \
  --checkpoint outputs/dryrun_celeba/seed_2026/checkpoints/best.pt \
  --dataset celeba_hq --image-size 512 --tile-size 256 --limit 5 --output-dir outputs/eval_hr
uv run --no-sync hybrid-inpaint evaluate -c configs/dryrun_places.yaml \
  --checkpoint outputs/dryrun_places/seed_2026/checkpoints/best.pt \
  --dataset places2 --pairs data/manifests/places365_stress_pairs.jsonl \
  --limit 5 --output-dir outputs/eval_stress
uv run --no-sync hybrid-inpaint train --config configs/smoke.yaml --resume outputs/smoke/seed_2026/checkpoints/last.pt
# 单图 / 批量修复（C6 已实现；冒烟用 dryrun 配置，正式权重换 configs/system.yaml 的 system.checkpoint）
uv run --no-sync hybrid-inpaint inpaint --image <图> --mask <掩码> --output out.png --config configs/dryrun_places.yaml
uv run --no-sync hybrid-inpaint run --input-dir <图目录> --mask-dir <掩码目录> --output-dir <输出目录> --config configs/dryrun_places.yaml
# 训练曲线（损失/验证指标）：训练中自动刷新，也可事后补画
uv run --no-sync hybrid-inpaint plot --run-dir outputs/smoke/seed_2026
# 阶段二（T0 正式数据就绪后）：从阶段一 best.pt 初始化
# uv run --no-sync hybrid-inpaint train -c configs/experiments/main.yaml \
#   --resume outputs/b1_unet_vit/seed_2026/checkpoints/best.pt
uv run --no-sync hybrid-inpaint start --config configs/system.yaml   # C7 之后可用
uv run --no-sync hybrid-inpaint gui --config configs/system.yaml     # C8 之后可用
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
