"""Offline tests for tokensnap.proxy.optimize_body: confirms the proxy wires
the configured `keep_messages` value into the compressor (not a hardcoded
constant), and that the aggressive-mode threshold is read from config.
"""

from tokensnap import config as config_mod
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
