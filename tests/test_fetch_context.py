"""Offline tests for the fetch_context tool cycle: detection, tool_result
construction from the Context Store, message finalization, SSE synthesis, and
the proxy loop driven with a fake upstream session (no network).
"""

import asyncio
import json

import pytest

from tokensnap import context_engine, context_store, fetch_context
from tokensnap.proxy import TokensnapProxy
from tokensnap.usage import UsageAccumulator


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(context_store, "DB_FILE", tmp_path / "context_store.db")
    yield


def _tool_use(name, input_, id_="tu1"):
    return {"type": "tool_use", "id": id_, "name": name, "input": input_}


class TestDetection:
    def test_finds_fetch_context_calls(self):
        content = [
            {"type": "text", "text": "let me check"},
            _tool_use("fetch_context", {"event_ids": ["3", "5"]}, "a"),
        ]
        calls = fetch_context.find_fetch_context_calls(content)
        assert calls == [{"id": "a", "event_ids": ["3", "5"]}]

    def test_coerces_single_string_and_ints(self):
        content = [_tool_use("fetch_context", {"event_ids": 7})]  # not a list
        calls = fetch_context.find_fetch_context_calls(content)
        assert calls[0]["event_ids"] == []  # non-list, non-str -> empty
        content2 = [_tool_use("fetch_context", {"event_ids": "9"})]
        assert fetch_context.find_fetch_context_calls(content2)[0]["event_ids"] == ["9"]

    def test_ignores_non_fetch_tools(self):
        content = [_tool_use("Bash", {"cmd": "ls"})]
        assert fetch_context.find_fetch_context_calls(content) == []

    def test_has_other_tool_calls(self):
        assert fetch_context.has_other_tool_calls([_tool_use("Bash", {})]) is True
        assert fetch_context.has_other_tool_calls(
            [_tool_use("fetch_context", {"event_ids": []})]) is False
        assert fetch_context.has_other_tool_calls([{"type": "text", "text": "x"}]) is False


class TestBuildToolResults:
    def test_resolves_events_from_store(self):
        eid = context_store.store_message("s", 0, "user", "the full detail", "sum", "error")
        calls = [{"id": "tu1", "event_ids": [str(eid)]}]
        results, found = fetch_context.build_tool_results(calls)
        assert found == 1
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "tu1"
        assert "the full detail" in results[0]["content"]

    def test_missing_ids_reported(self):
        calls = [{"id": "tu1", "event_ids": ["999"]}]
        results, found = fetch_context.build_tool_results(calls)
        assert found == 0
        assert "no stored events found" in results[0]["content"]

    def test_large_event_truncated(self):
        big = "x" * 20000
        eid = context_store.store_message("s", 0, "assistant", big, "s", "decision")
        results, _ = fetch_context.build_tool_results([{"id": "t", "event_ids": [str(eid)]}])
        assert "event truncated" in results[0]["content"]
        assert len(results[0]["content"]) < len(big)


class TestFinalizeMessage:
    def test_strips_fetch_context_and_fixes_stop_reason(self):
        msg = {"content": [_tool_use("fetch_context", {"event_ids": ["1"]})],
               "stop_reason": "tool_use"}
        out = fetch_context.finalize_message(msg)
        assert out["content"][0]["type"] == "text"  # substituted fallback
        assert out["stop_reason"] == "end_turn"

    def test_keeps_other_tools_but_drops_fetch(self):
        msg = {"content": [_tool_use("Bash", {"cmd": "ls"}, "b"),
                           _tool_use("fetch_context", {"event_ids": ["1"]}, "f")],
               "stop_reason": "tool_use"}
        out = fetch_context.finalize_message(msg)
        names = [b.get("name") for b in out["content"] if b.get("type") == "tool_use"]
        assert names == ["Bash"]
        assert out["stop_reason"] == "tool_use"  # a real tool remains

    def test_plain_answer_untouched(self):
        msg = {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn"}
        out = fetch_context.finalize_message(msg)
        assert out["content"] == [{"type": "text", "text": "hi"}]
        assert out["stop_reason"] == "end_turn"


class TestSynthesizeSse:
    def _events(self, raw: bytes):
        blocks = [b for b in raw.decode("utf-8").split("\n\n") if b.strip()]
        types = []
        for b in blocks:
            data_line = [ln for ln in b.split("\n") if ln.startswith("data:")][0]
            types.append(json.loads(data_line[len("data:"):].strip())["type"])
        return types, blocks

    def test_text_message_event_sequence(self):
        msg = {"id": "m1", "model": "claude", "role": "assistant",
               "content": [{"type": "text", "text": "hello"}],
               "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 2}}
        raw = fetch_context.synthesize_sse(msg)
        types, _ = self._events(raw)
        assert types == [
            "message_start", "content_block_start", "content_block_delta",
            "content_block_stop", "message_delta", "message_stop",
        ]

    def test_tool_use_uses_input_json_delta(self):
        msg = {"content": [_tool_use("Bash", {"cmd": "ls"})],
               "stop_reason": "tool_use", "usage": {}}
        raw = fetch_context.synthesize_sse(msg).decode("utf-8")
        assert "input_json_delta" in raw
        # The tool input is JSON-serialized into partial_json, so recover it
        # by finding the delta event and parsing its nested string.
        for line in raw.split("\n"):
            if line.startswith("data:") and "input_json_delta" in line:
                delta = json.loads(line[len("data:"):].strip())["delta"]
                assert json.loads(delta["partial_json"]) == {"cmd": "ls"}
                break
        else:
            pytest.fail("no input_json_delta event emitted")

    def test_usage_split_across_start_and_delta(self):
        msg = {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
               "usage": {"input_tokens": 100, "output_tokens": 5}}
        raw = fetch_context.synthesize_sse(msg).decode("utf-8")
        # message_start reports input tokens with output 0; message_delta reports output.
        start = json.loads([ln for ln in raw.split("\n") if ln.startswith("data:")][0][5:])
        assert start["message"]["usage"]["input_tokens"] == 100
        assert start["message"]["usage"]["output_tokens"] == 0


# --------------------------------------------------------------------------
# Loop integration with a fake upstream session (JSON client path).
# --------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status, message, headers=None):
        self.status = status
        self._body = (message if isinstance(message, (bytes, bytearray))
                      else json.dumps(message).encode("utf-8"))
        self.headers = headers or {"Content-Type": "application/json"}

    async def read(self):
        return self._body

    def release(self):
        pass


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []

    async def request(self, method, url, data=None, headers=None):
        self.sent.append(json.loads(data))
        return self._responses.pop(0)


def _proxy_with(responses):
    proxy = TokensnapProxy({"upstream": "http://x", "host": "127.0.0.1", "port": 1})
    proxy.session = _FakeSession(responses)
    return proxy


def _run_loop(proxy, optimized):
    usage = UsageAccumulator()
    meta = {}
    resp, status = asyncio.run(
        proxy._relay_with_fetch_context(None, optimized, {}, "http://x/v1/messages", usage, meta)
    )
    return resp, status, meta, usage


class TestFetchLoop:
    def _base_body(self):
        return {"model": "m", "stream": False,
                "messages": [{"role": "user", "content": "do the thing"}],
                "tools": [context_engine.FETCH_CONTEXT_TOOL]}

    def test_no_fetch_returns_immediately(self):
        final = {"content": [{"type": "text", "text": "done"}],
                 "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 1}}
        proxy = _proxy_with([_FakeResp(200, final)])
        resp, status, meta, usage = _run_loop(proxy, self._base_body())
        assert status == 200
        assert len(proxy.session.sent) == 1
        assert meta["events_fetched"] == 0
        assert json.loads(resp.body)["content"][0]["text"] == "done"
        assert usage.output_tokens == 1

    def test_one_fetch_cycle(self):
        eid = context_store.store_message("s", 0, "user", "SECRET DETAIL", "s", "error")
        fetch_resp = _FakeResp(200, {
            "content": [_tool_use("fetch_context", {"event_ids": [str(eid)]}, "f1")],
            "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 2}})
        final = _FakeResp(200, {
            "content": [{"type": "text", "text": "answer using detail"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 9, "output_tokens": 4}})
        proxy = _proxy_with([fetch_resp, final])
        resp, status, meta, usage = _run_loop(proxy, self._base_body())
        assert status == 200
        assert len(proxy.session.sent) == 2  # fetch, then continuation
        assert meta["events_fetched"] == 1
        # The continuation request carried the assistant turn + a tool_result.
        second = proxy.session.sent[1]
        assert second["messages"][-2]["role"] == "assistant"
        assert second["messages"][-1]["role"] == "user"
        tr = second["messages"][-1]["content"][0]
        assert tr["type"] == "tool_result" and "SECRET DETAIL" in tr["content"]
        # Client sees only the final answer.
        assert json.loads(resp.body)["content"][0]["text"] == "answer using detail"

    def test_mixed_tools_treated_as_final_and_fetch_stripped(self):
        msg = {"content": [_tool_use("Bash", {"cmd": "ls"}, "b"),
                           _tool_use("fetch_context", {"event_ids": ["1"]}, "f")],
               "stop_reason": "tool_use", "usage": {}}
        proxy = _proxy_with([_FakeResp(200, msg)])
        resp, status, meta, _ = _run_loop(proxy, self._base_body())
        assert len(proxy.session.sent) == 1  # no continuation
        names = [b.get("name") for b in json.loads(resp.body)["content"]
                 if b.get("type") == "tool_use"]
        assert names == ["Bash"]  # fetch_context stripped, Bash preserved

    def test_runaway_fetch_stops_at_max(self):
        # Every response asks to fetch again; the loop must bail at the cap.
        responses = [_FakeResp(200, {
            "content": [_tool_use("fetch_context", {"event_ids": []}, "f")],
            "stop_reason": "tool_use", "usage": {}})
            for _ in range(fetch_context.MAX_FETCH_ITERATIONS + 1)]
        proxy = _proxy_with(responses)
        resp, status, meta, _ = _run_loop(proxy, self._base_body())
        assert len(proxy.session.sent) == fetch_context.MAX_FETCH_ITERATIONS + 1
        # Finalized: fetch_context stripped -> fallback text present.
        assert json.loads(resp.body)["content"][0]["type"] == "text"

    def test_upstream_error_passed_through(self):
        err = _FakeResp(429, {"type": "error", "error": {"type": "rate_limit"}})
        proxy = _proxy_with([err])
        resp, status, meta, _ = _run_loop(proxy, self._base_body())
        assert status == 429
        assert json.loads(resp.body)["error"]["type"] == "rate_limit"
