"""Offline tests for `tokensnap run`: it resolves `claude` past PATH before
launching, errors helpfully when it can't be found, and leaves non-claude
commands untouched. subprocess + proxy startup are mocked so nothing real
spawns.
"""

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
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(stats, "STATS_DIR", tmp_path)
    monkeypatch.setattr(stats, "STATS_FILE", tmp_path / "stats.json")
    defaults = dict(config_mod.DEFAULTS)
    defaults["port"] = _free_port()
    monkeypatch.setattr(config_mod, "DEFAULTS", defaults)
    # Pretend the proxy is already up so `run` never spawns one.
    monkeypatch.setattr(cli.stats, "proxy_running", lambda *a, **k: True)
    yield


def test_run_resolves_claude_and_launches(monkeypatch):
    calls = {}

    def fake_call(cmd, *a, **k):
        calls["cmd"] = cmd
        return 0

    monkeypatch.setattr(cli, "resolve_claude_command",
                        lambda command: ["/abs/bin/claude"] + list(command[1:]))
    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    result = runner.invoke(cli.app, ["run", "claude"])
    assert result.exit_code == 0
    # The resolved absolute path (not bare "claude") is what actually runs.
    assert "/abs/bin/claude" in str(calls["cmd"])


def test_run_errors_when_claude_missing(monkeypatch):
    monkeypatch.setattr(cli, "resolve_claude_command", lambda command: None)
    launched = {"called": False}
    monkeypatch.setattr(cli.subprocess, "call",
                        lambda *a, **k: launched.__setitem__("called", True) or 0)
    result = runner.invoke(cli.app, ["run", "claude"])
    assert result.exit_code == 1
    assert "claude" in result.output.lower()
    assert launched["called"] is False  # never tried to launch anything


def test_run_passes_through_non_claude_command(monkeypatch):
    calls = {}
    monkeypatch.setattr(cli.subprocess, "call",
                        lambda cmd, *a, **k: calls.__setitem__("cmd", cmd) or 0)
    result = runner.invoke(cli.app, ["run", "echo", "hello"])
    # A non-claude command is left as-is by resolve_claude_command.
    assert result.exit_code == 0
    assert "echo" in str(calls["cmd"])
