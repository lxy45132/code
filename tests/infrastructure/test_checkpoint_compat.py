"""旧 checkpoint 向后兼容：内嵌配置含已废弃键（web.mount_gradio）仍可加载。"""

from __future__ import annotations

from pathlib import Path

import torch

from hybrid_inpaint.config import AppConfig, validate_checkpoint_config
from hybrid_inpaint.infrastructure.torch_engine import TorchInpaintEngine
from tests.infrastructure.test_torch_engine import _tiny_checkpoint


def test_validate_checkpoint_config_strips_legacy_key() -> None:
    raw = AppConfig().model_dump()
    raw["system"]["web"]["mount_gradio"] = True  # C6 及以前的废弃键

    parsed = validate_checkpoint_config(raw)

    assert parsed.system.web.workers == 1
    assert raw["system"]["web"]["mount_gradio"] is True  # 原 dict 不被修改（deepcopy）
    assert "mount_gradio" not in parsed.system.web.model_dump()


def test_engine_loads_checkpoint_with_legacy_key(tmp_path: Path) -> None:
    path = _tiny_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"]["system"]["web"]["mount_gradio"] = True
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)

    engine = TorchInpaintEngine(legacy, device="cpu")

    assert engine.version
    engine.warmup()
