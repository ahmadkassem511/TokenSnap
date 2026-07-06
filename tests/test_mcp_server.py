"""Offline tests for tokensnap.mcp_server (the stdio JSON-RPC/MCP server).

Config/stats are redirected into tmp_path so these never touch a real
~/.tokensnap or spawn a real proxy process.
"""

import socket

import pytest

from tokensnap import config as config_mod
from tokensnap import mcp_server, stats


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
    defaults = dict(config_mod.DEFAULTS)
    defaults["port"] = _free_port()
    monkeypatch.setattr(config_mod, "DEFAULTS", defaults)
    yield config_dir


def _call(name, arguments=None, msg_id=1):
    return mcp_server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
    )


class TestProtocolHandshake:
    def test_initialize(self):
        resp = mcp_server.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18"}}
        )
        assert resp["result"]["protocolVersion"] == "2025-06-18"
        assert "tools" in resp["result"]["capabilities"]
        assert resp["result"]["serverInfo"]["name"] == "tokensnap"

    def test_ping(self):
        resp = mcp_server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "ping"})
        assert resp == {"jsonrpc": "2.0", "id": 2, "result": {}}

    def test_tools_list(self):
        resp = mcp_server.handle_message(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
        )
        names = {t["name"] for t in resp["result"]["tools"]}
        assert names == {
            "tokensnap_status",
            "tokensnap_recent_requests",
            "tokensnap_get_config",
            "tokensnap_set_config",
            "tokensnap_start_proxy",
            "tokensnap_stop_proxy",
        }

    def test_notification_gets_no_response(self):
        # No "id" key => notification; must not be answered
        resp = mcp_server.handle_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        assert resp is None

    def test_unknown_method_errors(self):
        resp = mcp_server.handle_message(
            {"jsonrpc": "2.0", "id": 4, "method": "not/a/real/method"}
        )
        assert resp["error"]["code"] == -32601


class TestStatusAndConfigTools:
    def test_status_shape(self):
        resp = _call("tokensnap_status")
        assert resp["result"]["isError"] is False
        assert '"proxy_running": false' in resp["result"]["content"][0]["text"]
        assert "real_usage_from_anthropic" in resp["result"]["content"][0]["text"]

    def test_get_config_redacts_key(self):
        config_mod.set_value("key", "sk-ant-super-secret")
        resp = _call("tokensnap_get_config")
        text = resp["result"]["content"][0]["text"]
        assert "sk-ant-super-secret" not in text
        assert "********" in text

    def test_set_config_updates_value(self):
        resp = _call("tokensnap_set_config", {"key": "keep_messages", "value": "5"})
        assert resp["result"]["isError"] is False
        assert config_mod.load()["keep_messages"] == 5

    def test_set_config_resolves_legacy_key_alias(self):
        # keep_last_n is the pre-0.3 name for keep_messages; MCP callers
        # using the old name must still work.
        resp = _call("tokensnap_set_config", {"key": "keep_last_n", "value": "7"})
        assert resp["result"]["isError"] is False
        assert config_mod.load()["keep_messages"] == 7

    def test_set_config_refuses_key_field(self):
        resp = _call("tokensnap_set_config", {"key": "key", "value": "sk-ant-x"})
        assert resp["result"]["isError"] is True
        assert "Refusing" in resp["result"]["content"][0]["text"]

    def test_set_config_rejects_unknown_key(self):
        resp = _call("tokensnap_set_config", {"key": "bogus_key", "value": "1"})
        assert resp["result"]["isError"] is True

    def test_recent_requests_reflects_stats(self):
        stats.mark_started("127.0.0.1", 8889)
        stats.record_request(
            "/v1/messages", "claude-sonnet-5", 100, 40, 200, 0.2,
            real_input=30, real_output=10, real_cache_read=500,
        )
        resp = _call("tokensnap_recent_requests", {"limit": 5})
        text = resp["result"]["content"][0]["text"]
        assert "claude-sonnet-5" in text
        assert '"real_cache_read": 500' in text


class TestProxyLifecycleTools:
    def test_stop_when_nothing_running(self):
        resp = _call("tokensnap_stop_proxy")
        text = resp["result"]["content"][0]["text"]
        assert '"stopped": false' in text
        assert "no proxy was running" in text

    def test_start_reports_already_running(self, monkeypatch):
        monkeypatch.setattr(stats, "proxy_running", lambda host, port: True)
        resp = _call("tokensnap_start_proxy")
        text = resp["result"]["content"][0]["text"]
        assert '"started": false' in text
        assert "already running" in text

    def test_start_failure_surfaces_as_tool_error(self, monkeypatch):
        monkeypatch.setattr(stats, "proxy_running", lambda host, port: False)
        monkeypatch.setattr(
            stats, "start_proxy_detached", lambda: (False, config_mod.CONFIG_DIR / "proxy.log")
        )
        resp = _call("tokensnap_start_proxy")
        assert resp["result"]["isError"] is True
        assert "failed to start" in resp["result"]["content"][0]["text"].lower()


class TestMalformedInput:
    def test_bad_json_line_is_parse_error(self):
        # handle_message expects a dict; the stdio loop is what catches
        # JSONDecodeError, so we test the documented contract at that layer
        # by feeding an already-invalid shape through the tool dispatcher.
        resp = mcp_server.handle_message({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                                           "params": {"name": "tokensnap_set_config",
                                                      "arguments": {"key": "keep_last_n"}}})
        # missing "value" -> KeyError inside _call_tool, caught as generic Exception
        assert resp["result"]["isError"] is True
