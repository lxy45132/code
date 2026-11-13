# Hybrid Inpaint

基于 U-Net 与瓶颈 Transformer（ViT）、GAN 两阶段训练的图像修复系统，配套 Qt 桌面端与 FastAPI + Vue3 Web 端（MySQL/Redis/Celery 批量任务）。

## 当前状态

施工中。C0–C6 已完成：算法管线（U-Net + ViT + GAN 两阶段）、T0 真实数据与冻结 manifest、固定评测 CLI、512/1024 分块推理、大洞压力表，以及 C6 系统核心（分层用例、适配器、组合根、`inpaint`/`run` CLI）。**C7 Web 端已完成 C7a–C7g**：FastAPI + Vue3 三页（单图/批量/历史）、SQLAlchemy + Alembic（MySQL/SQLite 双后端）、Celery + Redis 批量任务（SSE 进度/断线回放/取消/凑批）、Playwright e2e 四场景、docker-compose 编排与 `start`/`worker`/`db init` CLI；333 个后端测试 + 14 个前端 vitest + 4 个 e2e 通过。正式 B0/B1/B2/Main 训练待 24 GB 云卡；C8（Qt）未开始。

项目根目录：`D:\Code\bs\code`（ASCII 路径，不需要 `PYTHONUTF8` 或 `--env-file`）。

## 文档

> docs 目录仅本地维护，不纳入 Git（以下链接在 GitHub 上不可见）。

- [总体施工方案](docs/00_总体施工方案.md)
- [模型与训练方案](docs/01_模型与训练方案.md)
- [系统设计方案](docs/02_系统设计方案.md)
- [Web 端设计方案（C7 施工契约）](docs/03_Web端设计方案.md)
- [施工进度](docs/PROGRESS.md)
- [C0 handoff](docs/handoffs/HANDOFF_C0.md)
- [C1 handoff](docs/handoffs/HANDOFF_C1.md)
- [C2 handoff](docs/handoffs/HANDOFF_C2.md)
- [C3 handoff](docs/handoffs/HANDOFF_C3.md)
- [C4 handoff](docs/handoffs/HANDOFF_C4.md)
- [C5 handoff（进行中）](docs/handoffs/HANDOFF_C5.md)
- [C6 handoff](docs/handoffs/HANDOFF_C6.md)
- [C7 handoff](docs/handoffs/HANDOFF_C7.md)
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
uv sync --extra dev --extra train --extra web   # Web 全栈（含 torch 推理）
uv sync --extra dev --extra gui
```

前端（Node ≥ 20）：

```bash
cd frontend
npm install
npm run dev      # 开发 :5173，代理 /api → 127.0.0.1:8000
npm run build    # 产出 dist，由 FastAPI 同源托管
npm run test     # vitest
npm run lint
npm run test:e2e # Playwright（需 Redis + 后端，自动起测试服务）
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
uv run --no-sync hybrid-inpaint start --config configs/system.yaml   # Web API（托管 frontend/dist）
uv run --no-sync hybrid-inpaint worker --config configs/system.yaml  # Celery worker（批量/worker 模式）
uv run --no-sync hybrid-inpaint db init --config configs/system.yaml # 建库 + Alembic 迁移
uv run --no-sync hybrid-inpaint gui --config configs/system.yaml     # C8 之后可用
```

## Web 端（C7）

- 本地开发：Redis 先起（本机 `D:\Redis-8.6.2-Windows-x64-cygwin-with-Service\redis-server.exe --port 6379`）；`scripts\dev.ps1 api|worker|web|migrate|test|e2e` 一条命令化。
- 默认 SQLite + inline 模式零额外依赖；MySQL 切换：`system.storage.backend: mysql` + `db init`（账号密码经 `HYBRID_INPAINT__SYSTEM__STORAGE__MYSQL__PASSWORD` 等环境变量注入，不入 YAML）。
- 批量任务：Web 上传配对 → Celery chunk 凑批 → SSE 实时进度（Last-Event-ID 断线回放）→ 取消 → report.csv；Windows worker 固定 `--pool=solo -c 1`。
- 容器化演示：`docker compose up --build -d`（mysql + redis + api×2 + worker；`--profile monitor` 加 Flower）；生产 GPU worker 建议宿主机直跑。
- 详细设计与契约变更（AD-05/06/07/11/12）见 [docs/03_Web端设计方案](docs/03_Web端设计方案.md)。

## 目录

```text
hybrid_inpaint/
  domain/          # 实体、端口、错误、纯函数
  application/     # 用例
  infrastructure/  # torch / pillow / sqlite / sqlalchemy / redis / 文件系统适配器
  interfaces/      # CLI / FastAPI（+ Celery 任务）/ Qt
  models/          # nn.Module
  training/        # 数据、损失、指标、训练器
  bootstrap.py     # 组合根
  config.py        # pydantic 配置（YAML < 环境变量 < overrides）
frontend/          # Vue3 + Vite + TS（单图/批量/历史三页）
alembic/           # 数据库迁移
scripts/           # dev.ps1 / e2e 服务
```
