"""Offline tests for the Differential Context Engine indicator in
`tokensnap status` and `tokensnap monitor` (via its underlying
`cli._build_dashboard` renderable - `monitor` itself loops forever under
`Live`, so it's never invoked directly in tests).
"""

import io
import socket

import pytest
from rich.console import Console
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


def _render(renderable) -> str:
    """Render a Rich renderable (e.g. a Panel) to plain text for substring
    assertions, independent of markup/color codes."""
    buf = io.StringIO()
    Console(file=buf, width=200).print(renderable)
    return buf.getvalue()


class TestStatusCommand:
    def test_disabled_by_default(self):
        result = runner.invoke(cli.app, ["status"])
        assert result.exit_code == 0
        assert "Differential Context Engine: disabled" in result.stdout

    def test_enabled_shows_tree_size(self):
        config_mod.set_value("context_store_enabled", "true")
        config_mod.set_value("context_tree_size", "33")
        result = runner.invoke(cli.app, ["status"])
        assert result.exit_code == 0
        assert "Differential Context Engine: enabled" in result.stdout
        assert "tree size: 33" in result.stdout

    def test_default_tree_size_shown_when_enabled_without_override(self):
        config_mod.set_value("context_store_enabled", "true")
        result = runner.invoke(cli.app, ["status"])
        assert "tree size: 20" in result.stdout

    def test_shows_enabled_when_adaptive_full_tier_used_it(self):
        # The static override is left at its default (off) - only a request
        # that actually went through the engine happened (as Adaptive
        # Transparency Mode's FULL tier does automatically for long
        # sessions). `status` must reflect that real activity.
        stats.mark_started("127.0.0.1", 8889)
        stats.record_request("/v1/messages", "m", 100, 40, 200, 0.1, context_store=True)
        result = runner.invoke(cli.app, ["status"])
        assert "Differential Context Engine: enabled" in result.stdout
        assert config_mod.load()["context_store_enabled"] is False


class TestMonitorDashboardRenderable:
    """`tokensnap monitor` refreshes `cli._build_dashboard(cfg)` in a Live
    loop; testing that function directly avoids invoking the infinite loop."""

    def test_disabled_by_default(self):
        cfg = config_mod.load()
        text = _render(cli._build_dashboard(cfg))
        assert "Differential Context Engine: disabled" in text

    def test_enabled_shows_tree_size(self):
        config_mod.set_value("context_store_enabled", "true")
        config_mod.set_value("context_tree_size", "42")
        cfg = config_mod.load()
        text = _render(cli._build_dashboard(cfg))
        assert "Differential Context Engine: enabled" in text
        assert "tree size: 42" in text

    def test_shows_enabled_when_adaptive_full_tier_used_it(self):
        stats.mark_started("127.0.0.1", 8889)
        stats.record_request("/v1/messages", "m", 100, 40, 200, 0.1, context_store=True)
        cfg = config_mod.load()
        text = _render(cli._build_dashboard(cfg))
        assert "Differential Context Engine: enabled" in text
        assert cfg["context_store_enabled"] is False

    def test_shown_alongside_keep_messages_and_memory_cards(self):
        # Regression guard: the indicator lives in the same header row as
        # keep_messages / Memory Cards, per the spec ("near where keep_messages
        # and Memory Cards are shown") - verify all three still render together.
        cfg = config_mod.load()
        text = _render(cli._build_dashboard(cfg))
        assert "keep_messages:" in text
        assert "Memory Cards:" in text
        assert "Differential Context Engine:" in text
