from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hybrid_inpaint.config import AppConfig, ConfigError, load_config


def write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_minimal_config_uses_defaults(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "contract_version: 1\n")
    config = load_config(path)

    assert isinstance(config, AppConfig)
    assert config.contract_version == 1
    assert config.model.channels == (64, 128, 256, 512)
    assert config.data.image_size == 256
    # Defaults that are not written in the YAML are not anchored to the YAML
    # directory; only explicit path values are resolved.
    assert config.data.root == Path("data/celeba_hq")


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "contract_version: 1\nunknown_key: 1\n")
    with pytest.raises(ConfigError, match="unknown_key"):
        load_config(path)


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
        contract_version: 1
        model:
          not_a_field: 1
        """,
    )
    with pytest.raises(ConfigError, match=r"model\.not_a_field"):
        load_config(path)


def test_invalid_image_size_is_rejected(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
        contract_version: 1
        data:
          image_size: 250
        """,
    )
    with pytest.raises(ConfigError, match="image_size"):
        load_config(path)


def test_zero_vit_heads_is_rejected(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
        contract_version: 1
        model:
          vit_heads: 0
        """,
    )
    with pytest.raises(ConfigError, match="vit_heads"):
        load_config(path)


def test_channels_must_divide_heads(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
        contract_version: 1
        model:
          channels: [64, 128, 256, 510]
          vit_heads: 8
        """,
    )
    with pytest.raises(ConfigError, match="divisible by vit_heads"):
        load_config(path)


def test_train_output_dir_resolves_against_yaml_directory(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
        contract_version: 1
        train:
          output_dir: runs
        """,
    )
    config = load_config(path)
    assert config.train.output_dir == (tmp_path / "runs").resolve()


def test_manifest_dirs_resolve_against_yaml_directory(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
        contract_version: 1
        data:
          manifest_dir: manifests
          processed_dir: processed
        """,
    )
    config = load_config(path)
    assert config.data.manifest_dir == (tmp_path / "manifests").resolve()
    assert config.data.processed_dir == (tmp_path / "processed").resolve()


def test_manifest_limits_must_be_positive(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
        contract_version: 1
        data:
          manifest_train_limit: 0
        """,
    )
    with pytest.raises(ConfigError, match="manifest_train_limit"):
        load_config(path)


def test_zero_train_samples_is_rejected(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
        contract_version: 1
        data:
          train_samples: 0
        """,
    )
    with pytest.raises(ConfigError, match="train_samples"):
        load_config(path)


def test_tile_overlap_must_be_smaller_than_tile(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
        contract_version: 1
        system:
          tiling:
            tile_size: 128
            tile_overlap: 128
        """,
    )
    with pytest.raises(ConfigError, match="tile_overlap"):
        load_config(path)


def test_contract_version_must_be_one(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "contract_version: 2\n")
    with pytest.raises(ConfigError, match="contract_version"):
        load_config(path)


def test_relative_paths_resolve_against_yaml_directory(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
        contract_version: 1
        data:
          root: mydata
          mask_dir: mymasks
        system:
          checkpoint: out/best.pt
          history_db: out/history.sqlite
          output_dir: out/history
        """,
    )
    config = load_config(path)

    assert config.data.root == (tmp_path / "mydata").resolve()
    assert config.data.mask_dir == (tmp_path / "mymasks").resolve()
    assert config.system.checkpoint == (tmp_path / "out/best.pt").resolve()
    assert config.system.history_db == (tmp_path / "out/history.sqlite").resolve()
    assert config.system.output_dir == (tmp_path / "out/history").resolve()


def test_overrides_are_deep_merged(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        """
        contract_version: 1
        model:
          vit_depth: 4
          vit_heads: 8
        """,
    )
    config = load_config(path, overrides={"model": {"vit_depth": 0}})

    assert config.model.vit_depth == 0
    assert config.model.vit_heads == 8


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_non_mapping_root_is_rejected(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_yaml_parse_error_is_wrapped(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "a: [\n")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)
