"""Offline tests for tokensnap.proxy.optimize_body: confirms the proxy wires
the configured `keep_messages` value into the compressor (not a hardcoded
constant), that the aggressive-mode threshold is read from config, and that
the Differential Context Engine path activates only when opted in.
"""

import pytest

from tokensnap import config as config_mod
from tokensnap import context_store
from tokensnap.proxy import optimize_body


def _long_history(n_exchanges):
    msgs = []
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": "step %d: edit src/app.py" % i})
        msgs.append({"role": "assistant", "content": "did step %d" % i})
    return msgs


def _cfg(**overrides):
    cfg = dict(config_mod.DEFAULTS)
    cfg["compressor_type"] = "regex"  # keep these tests fully offline
    cfg.update(overrides)
    return cfg


class TestKeepMessagesPlumbing:
    def test_default_keeps_ten_exchanges(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        new_body, meta = optimize_body(body, _cfg())
        assert meta["compressed"] is True
        assert len(new_body["messages"]) == 10 * 2  # 10 exchanges kept verbatim

    def test_custom_keep_messages_changes_kept_count(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        new_body, meta = optimize_body(body, _cfg(keep_messages=3))
        assert len(new_body["messages"]) == 3 * 2

    def test_maximum_preset_value_effectively_disables_compression(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        new_body, meta = optimize_body(body, _cfg(keep_messages=999))
        assert meta["compressed"] is False
        assert len(new_body["messages"]) == len(body["messages"])

    def test_short_history_untouched_regardless_of_keep_messages(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(2)}
        new_body, meta = optimize_body(body, _cfg(keep_messages=1))
        assert meta["compressed"] is False
        assert len(new_body["messages"]) == 4


class TestSelectiveCompressionPlumbing:
    def test_selective_compression_on_by_default_cleans_tool_results(self):
        dump = "\n".join(["log line %d" % i for i in range(200)] + ["error: boom"])
        msgs = _long_history(10)
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": dump}
        ]})
        body = {"model": "claude-sonnet-5", "messages": msgs}
        new_body, _ = optimize_body(body, _cfg())
        kept_tool_result = next(
            m for m in new_body["messages"]
            if isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        )
        text = kept_tool_result["content"][0]["content"]
        assert "tokensnap" in text  # compressed with an omission marker
        assert len(text) < len(dump)

    def test_selective_compression_off_keeps_tool_result_verbatim(self):
        dump = "\n".join(["log line %d" % i for i in range(200)] + ["error: boom"])
        msgs = _long_history(1)
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": dump}
        ]})
        body = {"model": "claude-sonnet-5", "messages": msgs}
        new_body, _ = optimize_body(body, _cfg(selective_compression=False))
        kept_tool_result = next(
            m for m in new_body["messages"]
            if isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        )
        assert kept_tool_result["content"][0]["content"] == dump

    def test_compressor_type_off_disables_truncation_entirely(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        new_body, meta = optimize_body(body, _cfg(compressor_type="off", keep_messages=3))
        assert meta["compressed"] is False
        assert len(new_body["messages"]) == len(body["messages"])


class TestAggressiveThreshold:
    def test_not_aggressive_when_under_threshold(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(3)}
        _, meta = optimize_body(body, _cfg(context_threshold=0.95))
        assert meta["aggressive"] is False

    def test_aggressive_kicks_in_past_threshold(self):
        # A tiny context_threshold guarantees "near the limit" is hit even
        # for a small request, proving the proxy reads context_threshold
        # from config rather than a hardcoded 0.9.
        body = {"model": "claude-sonnet-5", "messages": _long_history(10)}
        _, meta = optimize_body(body, _cfg(context_threshold=0.0001))
        assert meta["aggressive"] is True


class TestContextStoreDisabledByDefault:
    def test_default_path_never_adds_context_tree_or_tool(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        new_body, meta = optimize_body(body, _cfg())
        assert "context_store" not in meta  # classic path taken
        assert "tools" not in new_body
        system = new_body.get("system") or ""
        assert "CONTEXT TREE" not in (system if isinstance(system, str) else str(system))


class TestDifferentialContextPath:
    @pytest.fixture(autouse=True)
    def isolated_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr(context_store, "DB_FILE", tmp_path / "context_store.db")
        yield

    def test_long_history_rebuilt_as_tree_plus_tail(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        new_body, meta = optimize_body(body, _cfg(context_store_enabled=True))
        assert meta["context_store"] is True
        assert meta["compressed"] is True
        # Only the last 2 exchanges remain verbatim.
        assert len(new_body["messages"]) == 4
        assert meta["events_omitted"] == 40 - 4
        # fetch_context tool merged in.
        assert any(t["name"] == "fetch_context" for t in new_body["tools"])
        # Context Tree injected into system.
        system = new_body["system"]
        text = system if isinstance(system, str) else str(system)
        assert "CONTEXT TREE" in text
        # Fewer estimated tokens than the original.
        assert meta["after"] < meta["before"]

    def test_events_are_persisted_to_the_store(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        optimize_body(body, _cfg(context_store_enabled=True))
        # Every non-empty message was mirrored into external memory.
        assert context_store.event_count() == 40

    def test_short_history_not_reconstructed(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(2)}
        new_body, meta = optimize_body(body, _cfg(context_store_enabled=True))
        assert meta["context_store"] is True
        assert meta["compressed"] is False
        assert "tools" not in new_body  # nothing omitted -> no tool advertised
        assert len(new_body["messages"]) == 4

    def test_existing_tools_are_preserved(self):
        body = {
            "model": "claude-sonnet-5",
            "messages": _long_history(20),
            "tools": [{"name": "Bash"}, {"name": "Read"}],
        }
        new_body, _ = optimize_body(body, _cfg(context_store_enabled=True))
        names = [t["name"] for t in new_body["tools"]]
        assert names == ["Bash", "Read", "fetch_context"]
