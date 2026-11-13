"""Alembic environment: resolve the target URL, run migrations.

URL 解析优先级（03 文档 §6）：
1. AlembicConfig attributes（programmatic 调用方注入，如 ``db init``）；
2. ``ALEMBIC_DATABASE_URL`` 环境变量；
3. 项目配置 configs/system.yaml 的 system.storage.mysql。
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from hybrid_inpaint.infrastructure.sqlalchemy_models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_url() -> str:
    injected = context.get_x_argument(as_dictionary=True).get("url")
    if injected:
        return injected
    attributes_url = context.config.get_main_option("sqlalchemy.url")
    if attributes_url:
        return attributes_url
    environment_url = os.environ.get("ALEMBIC_DATABASE_URL")
    if environment_url:
        return environment_url
    from hybrid_inpaint.config import ConfigError, load_config
    from hybrid_inpaint.infrastructure.sqlalchemy_history import mysql_url

    config_path = os.environ.get("HYBRID_INPAINT_CONFIG", "configs/system.yaml")
    try:
        parsed = load_config(config_path)
    except ConfigError as exc:
        raise SystemExit(f"cannot resolve database url from {config_path}: {exc}")
    mysql = parsed.system.storage.mysql
    return mysql_url(
        mysql.host, mysql.port, mysql.user, mysql.password, mysql.database
    )


def run_migrations_offline() -> None:
    url = _resolve_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _resolve_url()
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = url
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
