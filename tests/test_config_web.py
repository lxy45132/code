from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_inpaint.config import ConfigError, load_config

_HEADER = "contract_version: 1\nexperiment_id: config_web_test"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "system.yaml"
    text = _HEADER + "\n" if not body.strip() else _HEADER + "\nsystem:\n" + body.strip("\n") + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_keep_sqlite_backend_and_inline_mode(tmp_path: Path) -> None:
    path = _write(tmp_path, "")
    parsed = load_config(path)
    assert parsed.system.storage.backend == "sqlite"
    assert parsed.system.inference.mode == "inline"
    assert parsed.system.inference.chunk_size == 8
    assert parsed.system.redis.url == "redis://127.0.0.1:6379/0"
    assert parsed.system.celery.retries == 3
    assert parsed.system.batch.max_items == 32
    assert parsed.system.web.workers == 1
    assert parsed.system.api.api_key is None


def test_mysql_backend_is_accepted(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "\n".join(
            [
                "  storage:",
                "    backend: mysql",
                "    mysql:",
                "      host: 127.0.0.1",
                "      port: 3306",
                "      user: hybrid",
                "      database: hybrid_inpaint",
            ]
        ),
    )
    parsed = load_config(path)
    assert parsed.system.storage.backend == "mysql"
    assert parsed.system.storage.mysql.database == "hybrid_inpaint"


@pytest.mark.parametrize(
    "body",
    [
        "  storage:\n    backend: postgres",  # 非法 backend
        "  redis:\n    url: mysql://127.0.0.1:3306",  # 非 redis 协议
        "  inference:\n    chunk_size: 0",  # 越界
        "  inference:\n    mode: inline3",  # 非法 mode
        "  web:\n    workers: 0",  # 越界
        "  web:\n    port: 70000",  # 越界
        "  batch:\n    max_items: 999",  # 越界
        "  celery:\n    broker_db: 16",  # 越界
        "  celery:\n    retries: -1",  # 负数
        "  api:\n    cors_origins: [\"localhost:5173\"]",  # 非法 origin
        "  busy_timeout_s: -2",  # 低于 -1
        "  storage:\n    mysql:\n      database: ''",  # 空库名
    ],
)
def test_invalid_web_config_is_rejected(tmp_path: Path, body: str) -> None:
    path = _write(tmp_path, body)
    with pytest.raises(ConfigError):
        load_config(path)


def test_busy_timeout_minus_one_means_wait_forever(tmp_path: Path) -> None:
    path = _write(tmp_path, "  busy_timeout_s: -1")
    parsed = load_config(path)
    assert parsed.system.busy_timeout_s == -1


def test_unknown_web_key_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "  inference:\n    batch: 4")
    with pytest.raises(ConfigError):
        load_config(path)
