"""E2E 共享装配：FakeEngine + 真实 Redis（db15）+ sqlite（outputs/e2e）。

api 与 worker 两个进程共用同一配置/数据库；Celery 走真实 broker
（configs/system.yaml 的 redis db1/db2），任务在 solo worker 内执行，
完整验证 broker → worker → DB → Stream → SSE → 前端 链路。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from hybrid_inpaint.bootstrap import Container, build_container

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E2E_DIR = PROJECT_ROOT / "outputs" / "e2e"


class E2EEngine:
    """均值填充假引擎；delay_s 模拟推理耗时（取消场景需要时间窗口）。"""

    device = "cpu"

    def __init__(self, delay_s: float = 0.0) -> None:
        self._delay = delay_s
        self._warnings: tuple[str, ...] = ()
        self.calls = 0

    @property
    def version(self) -> str:
        return "e2e-fake-00000000"

    @property
    def last_warnings(self) -> tuple[str, ...]:
        return self._warnings

    def warmup(self) -> None:
        pass

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self._delay > 0:
            time.sleep(self._delay)
        arr = image.astype(np.float32) / 255.0
        visible = ~mask
        fill = (
            arr[visible].mean(axis=0)
            if visible.any()
            else np.full(3, 0.5, dtype=np.float32)
        )
        out = arr.copy()
        out[mask] = fill
        return out

    def inpaint_many(
        self, images: list[np.ndarray], masks: list[np.ndarray]
    ) -> list[np.ndarray]:
        return [self.inpaint(image, mask) for image, mask in zip(images, masks)]


def write_config() -> Path:
    E2E_DIR.mkdir(parents=True, exist_ok=True)
    path = E2E_DIR / "system.yaml"
    path.write_text(
        "\n".join(
            [
                "contract_version: 1",
                "experiment_id: e2e",
                "system:",
                "  model_id: e2e",
                f"  checkpoint: {(E2E_DIR / 'unused.pt').as_posix()}",
                "  device: cpu",
                f"  history_db: {(E2E_DIR / 'history.sqlite').as_posix()}",
                f"  output_dir: {(E2E_DIR / 'history').as_posix()}",
                "  redis:",
                "    url: redis://127.0.0.1:6379/15",
                "  inference:",
                "    chunk_size: 2",
                "  celery:",
                "    retries: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def build_e2e_container(delay_s: float = 0.0) -> Container:
    return build_container(write_config(), engine=E2EEngine(delay_s))
