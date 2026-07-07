"""Optional OpenRouter-backed Memory Card generation.

OpenRouter (https://openrouter.ai) gives free access to several hosted
models (e.g. Meta's Llama 3.1 8B) via an OpenAI-compatible API. When an
`openrouter_api_key` is configured, the compressor asks a free OpenRouter
model to summarize the truncated history into a more accurate Memory Card
than regex extraction alone. Everything here is best-effort: any failure -
no key, network error, rate limit, malformed output - returns None and the
caller falls back to the regex card.

This module only ever uses the *OpenRouter* API key from config, never the
Anthropic API key the proxy is relaying for the user's actual request - the
two must never be mixed.

Uses stdlib urllib (no new dependency, and it keeps this module a drop-in
architectural match for how the old Ollama integration worked). Callers run
in a worker thread (see proxy.optimize_body / compressor.build_memory_card),
so blocking I/O is fine.
"""

import hashlib
import json
import logging
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from tokensnap.utils import message_text

log = logging.getLogger("tokensnap.openrouter")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

_TIMEOUT = 20.0  # free-tier OpenRouter models can be slow under load
# Most recent generated cards, keyed by (model, transcript) hash. Failures
# are cached too so a bad key/model doesn't add a network round trip - or a
# repeated warning - to every single request.
_CARD_CACHE_MAX = 32
# Transcript sent to the model is capped; the newest part of history is the
# most relevant, so we keep the tail.
_MAX_TRANSCRIPT_CHARS = 16_000

_MAX_ITEMS = 10
_MAX_FILES = 25
_MAX_LINE = 200
_TASK_CHARS = 400

# The model is asked for these exact keys; try_generate_card() remaps
# key_files/important_decisions onto the regex card's files_modified/
# decisions fields when merging (see compressor.build_memory_card).
_SYSTEM_PROMPT = (
    "You summarize a conversation between a developer and a coding assistant "
    "into a compact JSON memory card. Respond with ONLY a JSON object with "
    'exactly these keys: "task" (string: one sentence saying what the user '
    'is trying to accomplish overall), "key_files" (array of file path '
    'strings that were created or changed), "important_decisions" (array of '
    "short strings: technical choices that were made, with the reason when "
    'stated), "errors_resolved" (array of short strings formatted as "error '
    '-> fix"). Only include facts stated in the conversation; never invent '
    "anything. Use empty strings/arrays when a section has no facts."
)

_lock = threading.Lock()
_card_cache: "OrderedDict[str, Optional[Dict[str, Any]]]" = OrderedDict()
_MISS = object()  # sentinel: hash not in cache (None is a cached failure)


def reset_caches() -> None:
    """Clear the card cache (used by tests and config reloads)."""
    with _lock:
        _card_cache.clear()


def enabled(cfg: Dict[str, Any]) -> bool:
    """True when compressor_type is "openrouter" and a key is configured."""
    return (
        str(cfg.get("compressor_type", "")).lower() == "openrouter"
        and bool(str(cfg.get("openrouter_api_key", "")).strip())
    )


def _http_post_json(
    url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float
) -> Dict[str, Any]:
    """POST JSON to `url`, return the decoded JSON response body."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def summarize(conversation_text: str, model: str, api_key: str) -> str:
    """Call OpenRouter's chat completions endpoint and return the raw text
    the model produced (expected to be a JSON object, but callers must
    parse defensively - free models don't always obey the schema).

    Raises on network/HTTP/response-shape errors; callers should catch and
    fall back to regex summarization rather than let this propagate.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Summarize this conversation into the JSON memory card:\n\n"
                + conversation_text,
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % api_key,
        # OpenRouter asks integrations to identify themselves; harmless if ignored.
        "HTTP-Referer": "https://github.com/ahmadkassem511/TokenSnap",
        "X-Title": "Tokensnap",
    }
    response = _http_post_json(API_URL, payload, headers, _TIMEOUT)
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter response had no choices: %r" % response)
    return choices[0]["message"]["content"]


def build_transcript(messages: List[Dict[str, Any]]) -> str:
    """Flatten messages into a `role: text` transcript, newest-tail-kept."""
    lines = []
    for msg in messages:
        text = message_text(msg)
        if text:
            lines.append("%s: %s" % (msg.get("role", "?"), text))
    transcript = "\n\n".join(lines)
    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        transcript = "…" + transcript[-_MAX_TRANSCRIPT_CHARS:]
    return transcript


def _clip(text: str, limit: int = _MAX_LINE) -> str:
    text = " ".join(text.split())  # collapse whitespace/newlines
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _sanitize(raw: Any) -> Optional[Dict[str, Any]]:
    """Validate/normalize the model's raw JSON into a safe card fragment,
    remapping its key_files/important_decisions onto the canonical
    files_modified/decisions names the rest of Tokensnap uses.

    Returns None when the output is unusable (wrong shape or no facts).
    """
    if not isinstance(raw, dict):
        return None
    task = raw.get("task")
    card: Dict[str, Any] = {
        "task": _clip(task, _TASK_CHARS) if isinstance(task, str) else ""
    }
    for src_key, dest_key, cap in (
        ("key_files", "files_modified", _MAX_FILES),
        ("important_decisions", "decisions", _MAX_ITEMS),
        ("errors_resolved", "errors_resolved", _MAX_ITEMS),
    ):
        items: List[str] = []
        value = raw.get(src_key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    clipped = _clip(item)
                    if clipped not in items:
                        items.append(clipped)
                if len(items) >= cap:
                    break
        card[dest_key] = items
    if not card["task"] and not any(
        card[k] for k in ("files_modified", "decisions", "errors_resolved")
    ):
        return None
    return card


def try_generate_card(
    messages: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Ask the configured OpenRouter model to summarize `messages`.

    Returns a sanitized dict with keys task / files_modified / decisions /
    errors_resolved, or None when OpenRouter is off, unconfigured, or the
    call failed (caller should use the regex card instead).
    """
    if not enabled(cfg) or not messages:
        return None

    model = str(cfg.get("openrouter_model") or DEFAULT_MODEL)
    api_key = str(cfg.get("openrouter_api_key") or "")
    transcript = build_transcript(messages)
    if not transcript:
        return None

    key = hashlib.sha1(
        (model + "\x00" + transcript).encode("utf-8", "replace")
    ).hexdigest()
    with _lock:
        cached = _card_cache.get(key, _MISS)
        if cached is not _MISS:
            _card_cache.move_to_end(key)
            return cached

    card: Optional[Dict[str, Any]] = None
    try:
        raw_text = summarize(transcript, model, api_key)
        card = _sanitize(json.loads(raw_text))
        if card is None:
            log.info("OpenRouter returned an unusable card; using regex fallback")
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        log.info("OpenRouter generation failed (%s); using regex fallback", exc)

    with _lock:
        _card_cache[key] = card
        while len(_card_cache) > _CARD_CACHE_MAX:
            _card_cache.popitem(last=False)
    return card
