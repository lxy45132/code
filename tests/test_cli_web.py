"""C7g：start/worker CLI 与环境变量覆盖测试。"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from hybrid_inpaint.config import load_config
from hybrid_inpaint.interfaces.cli.commands import app

runner = CliRunner()


def _config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "system.yaml"
    lines = [
        "contract_version: 1",
        "experiment_id: cli_web_test",
        "system:",
        "  model_id: tiny",
        f"  checkpoint: {(tmp_path / 'unused.pt').as_posix()}",
        "  device: cpu",
        f"  history_db: {(tmp_path / 'history.sqlite').as_posix()}",
        f"  output_dir: {(tmp_path / 'history').as_posix()}",
        "  inference:",
        "    mode: worker",  # api 不加载模型，测试无需 torch
    ]
    if extra:
        lines.append(extra)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_start_single_worker_runs_app_instance(tmp_path: Path, monkeypatch) -> None:
    import uvicorn

    captured: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, **kw: captured.update(target=target, **kw))

    result = runner.invoke(app, ["start", "-c", str(_config(tmp_path)), "--port", "8123"])

    assert result.exit_code == 0, result.output
    assert "api listening" in result.output
    assert captured["port"] == 8123
    assert not isinstance(captured["target"], str)  # 单 worker 直传 app 实例


def test_start_multi_worker_uses_import_string(tmp_path: Path, monkeypatch) -> None:
    import uvicorn

    captured: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda target, **kw: captured.update(target=target, **kw))
    config = _config(
        tmp_path,
        "  storage:\n    backend: mysql\n  web:\n    workers: 2",
    )

    result = runner.invoke(app, ["start", "-c", str(config)])

    assert result.exit_code == 0, result.output
    assert captured["target"] == "hybrid_inpaint.interfaces.api.wsgi:app"
    assert captured["workers"] == 2


def test_start_multi_worker_sqlite_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path, "  web:\n    workers: 2")  # worker 模式 + sqlite
    result = runner.invoke(app, ["start", "-c", str(config)])
    assert result.exit_code != 0


def test_worker_invokes_celery_solo(tmp_path: Path, monkeypatch) -> None:
    from hybrid_inpaint.interfaces.api import celery_app as celery_module

    captured: dict = {}
    monkeypatch.setattr(
        celery_module.celery_app, "worker_main", lambda argv: captured.update(argv=argv)
    )

    result = runner.invoke(app, ["worker", "-c", str(_config(tmp_path))])

    assert result.exit_code == 0, result.output
    argv = captured["argv"]
    assert "--pool=solo" in argv
    assert "--queues=gpu" in argv
    assert "--concurrency=1" in argv
    monkeypatch.delenv("HYBRID_INPAINT_CONFIG", raising=False)


def test_env_override_password_and_port(tmp_path: Path, monkeypatch) -> None:
    path = _config(tmp_path)
    monkeypatch.setenv("HYBRID_INPAINT__SYSTEM__STORAGE__MYSQL__PASSWORD", "s3cret")
    monkeypatch.setenv("HYBRID_INPAINT__SYSTEM__WEB__PORT", "9000")

    parsed = load_config(path)

    assert parsed.system.storage.mysql.password == "s3cret"
    assert parsed.system.web.port == 9000  # JSON 解析为数字


def test_env_override_priority_below_explicit(tmp_path: Path, monkeypatch) -> None:
    path = _config(tmp_path)
    monkeypatch.setenv("HYBRID_INPAINT__SYSTEM__WEB__PORT", "9000")

    parsed = load_config(path, {"system": {"web": {"port": 9500}}})

    assert parsed.system.web.port == 9500
