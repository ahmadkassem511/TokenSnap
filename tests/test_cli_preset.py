"""Offline tests for the `tokensnap preset` CLI command."""

import socket

import pytest
from typer.testing import CliRunner

from tokensnap import cli
from tokensnap import config as config_mod
from tokensnap import stats

runner = CliRunner()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(stats, "STATS_DIR", tmp_path)
    monkeypatch.setattr(stats, "STATS_FILE", tmp_path / "stats.json")
    defaults = dict(config_mod.DEFAULTS)
    defaults["port"] = _free_port()
    monkeypatch.setattr(config_mod, "DEFAULTS", defaults)
    yield


@pytest.mark.parametrize(
    "name,keep_messages,selective,compressor_type",
    [
        ("simple", 5, True, "regex"),
        ("balanced", 10, True, "regex"),
        ("complex", 20, False, "regex"),
        ("smart", 25, True, "openrouter"),
        ("maximum", 999, True, "off"),
    ],
)
def test_preset_sets_all_fields(name, keep_messages, selective, compressor_type):
    result = runner.invoke(cli.app, ["preset", name])
    assert result.exit_code == 0
    assert str(keep_messages) in result.stdout
    cfg = config_mod.load()
    assert cfg["keep_messages"] == keep_messages
    assert cfg["selective_compression"] is selective
    assert cfg["compressor_type"] == compressor_type


def test_preset_is_case_insensitive():
    result = runner.invoke(cli.app, ["preset", "COMPLEX"])
    assert result.exit_code == 0
    assert config_mod.load()["keep_messages"] == 20


def test_unknown_preset_errors_without_changing_config():
    before = config_mod.load()["keep_messages"]
    result = runner.invoke(cli.app, ["preset", "extreme"])
    assert result.exit_code == 1
    assert "Unknown preset" in result.stdout
    assert config_mod.load()["keep_messages"] == before


def test_preset_then_status_shows_keep_messages():
    runner.invoke(cli.app, ["preset", "complex"])
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert "keep_messages=20" in result.stdout


def test_smart_preset_warns_when_no_api_key_configured():
    result = runner.invoke(cli.app, ["preset", "smart"])
    assert result.exit_code == 0
    assert "no OpenRouter API key is set" in result.stdout
    assert "openrouter.ai/keys" in result.stdout


def test_smart_preset_no_warning_once_key_is_set():
    config_mod.set_value("openrouter_api_key", "sk-or-test")
    result = runner.invoke(cli.app, ["preset", "smart"])
    assert result.exit_code == 0
    assert "no OpenRouter API key" not in result.stdout
