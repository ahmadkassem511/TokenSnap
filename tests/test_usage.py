"""Offline tests for tokensnap.usage.UsageAccumulator."""

import json

from tokensnap.usage import UsageAccumulator


def _sse(event: str, data: dict) -> bytes:
    return ("event: %s\ndata: %s\n\n" % (event, json.dumps(data))).encode("utf-8")


class TestNonStreaming:
    def test_full_body_usage(self):
        body = json.dumps(
            {
                "type": "message",
                "usage": {
                    "input_tokens": 1200,
                    "output_tokens": 350,
                    "cache_read_input_tokens": 8000,
                    "cache_creation_input_tokens": 500,
                },
            }
        ).encode("utf-8")
        u = UsageAccumulator()
        u.feed_full_body(body)
        assert u.input_tokens == 1200
        assert u.output_tokens == 350
        assert u.cache_read_tokens == 8000
        assert u.cache_creation_tokens == 500
        assert u.total == 1200 + 350 + 8000 + 500
        assert u.saw_usage is True

    def test_malformed_body_is_safe(self):
        u = UsageAccumulator()
        u.feed_full_body(b"not json at all")
        u.feed_full_body(b"")
        assert u.total == 0
        assert u.saw_usage is False

    def test_error_body_without_usage(self):
        body = json.dumps(
            {"type": "error", "error": {"type": "authentication_error"}}
        ).encode("utf-8")
        u = UsageAccumulator()
        u.feed_full_body(body)
        assert u.total == 0
        assert u.saw_usage is False


class TestStreaming:
    def test_message_start_then_delta(self):
        u = UsageAccumulator()
        u.feed(
            _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "usage": {
                            "input_tokens": 900,
                            "output_tokens": 1,
                            "cache_read_input_tokens": 12000,
                            "cache_creation_input_tokens": 0,
                        }
                    },
                },
            )
        )
        # output_tokens grows over deltas; the last value wins
        u.feed(_sse("message_delta", {"type": "message_delta", "usage": {"output_tokens": 42}}))
        u.feed(_sse("message_delta", {"type": "message_delta", "usage": {"output_tokens": 128}}))
        assert u.input_tokens == 900
        assert u.output_tokens == 128
        assert u.cache_read_tokens == 12000
        assert u.saw_usage is True

    def test_chunk_split_across_feed_calls(self):
        # An SSE frame split mid-line across two network chunks must still parse
        frame = _sse(
            "message_start",
            {"type": "message_start", "message": {"usage": {"input_tokens": 77, "output_tokens": 1}}},
        )
        u = UsageAccumulator()
        split = len(frame) // 2
        u.feed(frame[:split])
        assert u.input_tokens == 0  # not yet complete
        u.feed(frame[split:])
        assert u.input_tokens == 77

    def test_done_and_blank_lines_ignored(self):
        u = UsageAccumulator()
        u.feed(b"event: ping\ndata: {\"type\":\"ping\"}\n\n")
        u.feed(b"data: [DONE]\n\n")
        u.feed(b"\n\n")
        assert u.total == 0
        assert u.saw_usage is False

    def test_partial_stream_never_raises(self):
        u = UsageAccumulator()
        u.feed(b"data: {broken json")  # incomplete, no newline
        u.feed(b"")
        assert u.total == 0
