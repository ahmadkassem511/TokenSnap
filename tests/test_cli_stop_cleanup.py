"""Offline tests for the `tokensnap stop` and `tokensnap cleanup` CLI commands.

Config/stats locations are redirected into tmp_path so these never touch a
real ~/.tokensnap, and no real proxy process is ever spawned - stats.stop_proxy
and stats.proxy_running are exercised through their real (safe) logic in
test_stats.py; here we only need to check the CLI wires them up correctly.
"""

import socket

import pytest
from typer.testing import CliRunner

from tokensnap import cli, config as config_mod, stats

runner = CliRunner()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    config_dir = tmp_path / ".tokensnap"
    monkeypatch.setattr(config_mod, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", config_dir / "config.json")
    monkeypatch.setattr(stats, "STATS_DIR", config_dir)
    monkeypatch.setattr(stats, "STATS_FILE", config_dir / "stats.json")
    # Default to a port nothing listens on, so a real proxy running on the
    # user's machine (port 8889) is never detected — or killed! — by tests.
    defaults = dict(config_mod.DEFAULTS)
    defaults["port"] = _free_port()
    monkeypatch.setattr(config_mod, "DEFAULTS", defaults)
    yield config_dir


class TestStopCommand:
    def test_stop_when_nothing_running(self, isolated_dirs):
        config_mod.set_value("port", str(_free_port()))
        result = runner.invoke(cli.app, ["stop"])
        assert result.exit_code == 0
        assert "No Tokensnap proxy is running" in result.stdout

    def test_stop_reports_success(self, isolated_dirs, monkeypatch):
        config_mod.set_value("port", str(_free_port()))
        monkeypatch.setattr(stats, "stop_proxy", lambda host, port: (True, 4321))
        monkeypatch.setattr(stats, "proxy_running", lambda host, port: False)
        result = runner.invoke(cli.app, ["stop"])
        assert result.exit_code == 0
        assert "stopped" in result.stdout.lower()
        assert "4321" in result.stdout

    def test_stop_reports_failure_if_still_running(self, isolated_dirs, monkeypatch):
        config_mod.set_value("port", str(_free_port()))
        monkeypatch.setattr(stats, "stop_proxy", lambda host, port: (True, 4321))
        monkeypatch.setattr(stats, "proxy_running", lambda host, port: True)
        result = runner.invoke(cli.app, ["stop"])
        assert result.exit_code == 1
        assert "still responding" in result.stdout.lower()


class TestCleanupCommand:
    def test_cleanup_no_directory(self, isolated_dirs):
        assert not isolated_dirs.exists()
        result = runner.invoke(cli.app, ["cleanup", "--force"])
        assert result.exit_code == 0
        assert "Nothing to clean up" in result.stdout

    def test_cleanup_force_removes_directory(self, isolated_dirs):
        isolated_dirs.mkdir(parents=True)
        (isolated_dirs / "config.json").write_text("{}", encoding="utf-8")
        result = runner.invoke(cli.app, ["cleanup", "--force"])
        assert result.exit_code == 0
        assert not isolated_dirs.exists()
        assert "Removed" in result.stdout

    def test_cleanup_without_force_aborts_on_no(self, isolated_dirs):
        isolated_dirs.mkdir(parents=True)
        result = runner.invoke(cli.app, ["cleanup"], input="n\n")
        assert result.exit_code == 1
        assert isolated_dirs.exists()
        assert "Aborted" in result.stdout

    def test_cleanup_without_force_proceeds_on_yes(self, isolated_dirs):
        isolated_dirs.mkdir(parents=True)
        result = runner.invoke(cli.app, ["cleanup"], input="y\n")
        assert result.exit_code == 0
        assert not isolated_dirs.exists()

    def test_cleanup_stops_running_proxy_first(self, isolated_dirs, monkeypatch):
        isolated_dirs.mkdir(parents=True)
        calls = []
        monkeypatch.setattr(stats, "proxy_running", lambda host, port: True)
        monkeypatch.setattr(
            stats, "stop_proxy", lambda host, port: calls.append((host, port)) or (True, 1)
        )
        result = runner.invoke(cli.app, ["cleanup", "--force"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert "stopping it first" in result.stdout.lower()
