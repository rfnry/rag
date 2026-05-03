"""The `rfnry-knowledge retrieval init` command must restrict config.toml
permissions to 0o600 so URLs with embedded DB credentials aren't world-readable."""

import os
import stat

import pytest
from click.testing import CliRunner

from rfnry_knowledge.cli.commands.init import init as init_cmd


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr("rfnry_knowledge.cli.commands.init.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("rfnry_knowledge.cli.commands.init.CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr("rfnry_knowledge.cli.commands.init.ENV_FILE", tmp_path / ".env")
    return tmp_path


def test_init_chmods_config_toml_to_0o600(isolated_config) -> None:
    runner = CliRunner()
    result = runner.invoke(init_cmd, [])
    assert result.exit_code == 0, result.output

    config_path = isolated_config / "config.toml"
    assert config_path.exists()
    mode = stat.S_IMODE(os.stat(config_path).st_mode)
    assert mode == 0o600, f"config.toml mode is {oct(mode)}, expected 0o600"


def test_init_chmods_env_to_0o600(isolated_config) -> None:
    runner = CliRunner()
    result = runner.invoke(init_cmd, [])
    assert result.exit_code == 0, result.output

    env_path = isolated_config / ".env"
    mode = stat.S_IMODE(os.stat(env_path).st_mode)
    assert mode == 0o600
