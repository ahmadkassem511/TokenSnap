"""Approximate token counting for budget control.

Uses tiktoken's cl100k_base encoding (a good approximation for Claude
models). Falls back to a chars/4 estimate when tiktoken is unavailable or
its encoding file can't be downloaded (fully offline environments).
"""

import json
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("tokensnap.tokens")

# None = not loaded yet, False = load failed (use fallback), else the encoder
_encoder: Any = None
_warned = False

DEFAULT_CONTEXT_WINDOW = 200_000
# Rough flat cost for an image block
_IMAGE_TOKENS = 1_500


def _get_encoder() -> Optional[Any]:
    global _encoder, _warned
    if _encoder is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # ImportError, download failure, ...
            _encoder = False
            if not _warned:
                log.warning(
                    "tiktoken unavailable (%s); using chars/4 token estimate", exc
                )
                _warned = True
    return _encoder or None


def count_tokens(text: str) -> int:
    """Token count for a string (approximate for Claude models)."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


def count_message_tokens(
    messages: List[Dict[str, Any]], system: Any = None
) -> int:
    """Approximate token count for a full request payload."""
    total = 0
    if system:
        total += _count_content(system)
    for msg in messages or []:
        total += 4  # per-message framing overhead
        total += _count_content(msg.get("content"))
    return total


def _count_content(content: Any) -> int:
    if isinstance(content, str):
        return count_tokens(content)
    total = 0
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                total += count_tokens(block.get("text") or "")
            elif btype == "tool_result":
                total += _count_content(block.get("content"))
            elif btype == "tool_use":
                total += count_tokens(block.get("name") or "")
                try:
                    total += count_tokens(json.dumps(block.get("input") or {}))
                except (TypeError, ValueError):
                    pass
            elif btype == "image":
                total += _IMAGE_TOKENS
    return total


def context_window_for(model: Optional[str]) -> int:
    """Context window size for a model id. Unknown models get 200k."""
    if not model:
        return DEFAULT_CONTEXT_WINDOW
    m = model.lower()
    if "[1m]" in m or "context-1m" in m or "-1m" in m:
        return 1_000_000
    return DEFAULT_CONTEXT_WINDOW


def near_limit(tokens: int, model: Optional[str], threshold: float = 0.95) -> bool:
    """True when `tokens` is at or past `threshold` of the model's window."""
    return tokens >= context_window_for(model) * threshold
