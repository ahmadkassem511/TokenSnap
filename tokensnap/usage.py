"""Extract Anthropic's *real* token usage from API responses.

Claude Code reports usage from the `usage` field the API returns, which
Tokensnap's tiktoken estimate can't see: output tokens and prompt-cache
reads/writes. This module reads that field from both response shapes:

- Non-streaming: one JSON body with a top-level `usage` object.
- Streaming (SSE): `message_start` carries input/cache tokens inside
  `message.usage`; `message_delta` carries the running `output_tokens`.

The accumulator is fed raw response bytes (which are always relayed to the
client verbatim regardless) and never raises on malformed input.
"""

import json
from typing import Any, Dict, Optional


class UsageAccumulator:
    """Collects real token usage as response bytes stream through."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
        self.saw_usage = False
        self._buf = b""

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
        )

    def feed(self, chunk: bytes) -> None:
        """Feed a chunk of SSE response bytes (line-buffered internally)."""
        if not chunk:
            return
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._consume_line(line)

    def feed_full_body(self, body: bytes) -> None:
        """Parse a complete non-streaming JSON response body."""
        try:
            obj = json.loads(body)
        except (ValueError, TypeError):
            return
        if isinstance(obj, dict) and isinstance(obj.get("usage"), dict):
            self._apply(obj["usage"])

    def _consume_line(self, line: bytes) -> None:
        line = line.strip()
        if not line.startswith(b"data:"):
            return
        payload = line[len(b"data:"):].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            obj = json.loads(payload.decode("utf-8", "ignore"))
        except ValueError:
            return
        if not isinstance(obj, dict):
            return
        usage = self._usage_for_event(obj)
        if usage:
            self._apply(usage)

    @staticmethod
    def _usage_for_event(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        etype = obj.get("type")
        if etype == "message_start":
            message = obj.get("message")
            if isinstance(message, dict):
                return message.get("usage")
        elif etype == "message_delta":
            return obj.get("usage")
        elif isinstance(obj.get("usage"), dict):
            return obj["usage"]
        return None

    def _apply(self, usage: Dict[str, Any]) -> None:
        if not isinstance(usage, dict):
            return
        # Each field is the latest cumulative value the API reports, so we
        # overwrite rather than add (output_tokens grows across deltas).
        for src, attr in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("cache_read_input_tokens", "cache_read_tokens"),
            ("cache_creation_input_tokens", "cache_creation_tokens"),
        ):
            value = usage.get(src)
            if isinstance(value, int) and value >= 0:
                setattr(self, attr, value)
                self.saw_usage = True
