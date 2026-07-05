"""Optional Ollama-backed Memory Card generation.

When a local Ollama server is running, the compressor can ask a local LLM
to summarize the truncated history into a more accurate Memory Card than
regex extraction alone. Everything here is best-effort: any failure —
Ollama not installed, server down, model missing, timeout, malformed
output — returns None and the caller falls back to the regex card.

The conversation transcript is only ever sent to the configured Ollama
URL (localhost by default); it never leaves the machine.

Uses stdlib urllib so Tokensnap gains no new dependencies. Callers run
in a worker thread (see proxy.optimize_body), so blocking I/O is fine.
"""

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from tokensnap.utils import message_text

log = logging.getLogger("tokensnap.ollama")

# How long an availability probe result (up or down) stays valid
_AVAILABILITY_TTL = 60.0
# Availability probes must be fast; generation gets the full ollama_timeout
_PING_TIMEOUT = 2.0
# Most recent generated cards, keyed by transcript hash. Failures are
# cached too so a broken model doesn't add ollama_timeout to every request.
_CARD_CACHE_MAX = 32
# Transcript sent to the local model is capped; the newest part of the
# history is the most relevant, so we keep the tail.
_MAX_TRANSCRIPT_CHARS = 16_000

_MAX_ITEMS = 10
_MAX_FILES = 25
_MAX_LINE = 200
_TASK_CHARS = 400

_SYSTEM_PROMPT = (
    "You summarize a conversation between a developer and a coding assistant "
    "into a compact JSON memory card. Respond with ONLY a JSON object with "
    'exactly these keys: "task" (string: one sentence saying what the user '
    'is trying to accomplish overall), "files_modified" (array of file path '
    'strings that were created or changed), "decisions" (array of short '
    "strings: technical choices that were made, with the reason when stated), "
    '"errors_resolved" (array of short strings formatted as "error -> fix"). '
    "Only include facts stated in the conversation; never invent anything. "
    "Use empty strings/arrays when a section has no facts."
)

_lock = threading.Lock()
_availability: Dict[str, Any] = {"url": None, "checked_at": 0.0, "available": False}
_card_cache: "OrderedDict[str, Optional[Dict[str, Any]]]" = OrderedDict()

_MISS = object()  # sentinel: hash not in cache (None is a cached failure)


def reset_caches() -> None:
    """Clear availability + card caches (used by tests and config reloads)."""
    with _lock:
        _availability.update({"url": None, "checked_at": 0.0, "available": False})
        _card_cache.clear()


def enabled(cfg: Dict[str, Any]) -> bool:
    """True when the LLM compressor is switched on in config."""
    return str(cfg.get("llm_compressor", "off")).lower() in ("auto", "ollama")


def _http_get(url: str, timeout: float) -> int:
    """GET `url`, return the HTTP status. Raises on connection errors."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status


def _http_post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """POST JSON to `url`, return the decoded JSON response body."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def is_available(cfg: Dict[str, Any]) -> bool:
    """True when an Ollama server answers at the configured URL.

    Probes at most once per _AVAILABILITY_TTL seconds; both up and down
    results are cached so an absent Ollama costs (almost) nothing.
    """
    if not enabled(cfg):
        return False
    url = str(cfg.get("ollama_url", "")).rstrip("/")
    if not url:
        return False

    now = time.monotonic()
    with _lock:
        if (
            _availability["url"] == url
            and now - _availability["checked_at"] < _AVAILABILITY_TTL
        ):
            return bool(_availability["available"])

    try:
        ok = 200 <= _http_get(url + "/api/tags", _PING_TIMEOUT) < 300
    except (urllib.error.URLError, OSError, ValueError):
        ok = False

    with _lock:
        _availability.update({"url": url, "checked_at": now, "available": ok})
    if not ok and str(cfg.get("llm_compressor", "")).lower() == "ollama":
        log.warning(
            "llm_compressor is set to 'ollama' but no server answered at %s; "
            "falling back to regex Memory Cards",
            url,
        )
    return ok


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
    """Validate/normalize model output into a safe card fragment.

    Returns None when the output is unusable (wrong shape or no facts)."""
    if not isinstance(raw, dict):
        return None
    task = raw.get("task")
    card: Dict[str, Any] = {
        "task": _clip(task, _TASK_CHARS) if isinstance(task, str) else ""
    }
    for key, cap in (
        ("files_modified", _MAX_FILES),
        ("decisions", _MAX_ITEMS),
        ("errors_resolved", _MAX_ITEMS),
    ):
        items: List[str] = []
        value = raw.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    clipped = _clip(item)
                    if clipped not in items:
                        items.append(clipped)
                if len(items) >= cap:
                    break
        card[key] = items
    if not card["task"] and not any(
        card[k] for k in ("files_modified", "decisions", "errors_resolved")
    ):
        return None
    return card


def try_generate_card(
    messages: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Ask the local Ollama model to summarize `messages` into a card.

    Returns a sanitized dict with keys task / files_modified / decisions /
    errors_resolved, or None when the LLM path is off or failed (caller
    should use the regex card).
    """
    if not enabled(cfg) or not messages:
        return None
    if not is_available(cfg):
        return None

    model = str(cfg.get("ollama_model", "llama3.2"))
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
    url = str(cfg.get("ollama_url", "")).rstrip("/") + "/api/generate"
    timeout = float(cfg.get("ollama_timeout", 10.0))
    try:
        response = _http_post_json(
            url,
            {
                "model": model,
                "system": _SYSTEM_PROMPT,
                "prompt": (
                    "Summarize this conversation into the JSON memory card:"
                    "\n\n" + transcript
                ),
                "format": "json",
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout,
        )
        card = _sanitize(json.loads(response.get("response", "")))
        if card is None:
            log.debug("Ollama returned an unusable card; using regex fallback")
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        log.debug("Ollama generation failed (%s); using regex fallback", exc)

    with _lock:
        _card_cache[key] = card
        while len(_card_cache) > _CARD_CACHE_MAX:
            _card_cache.popitem(last=False)
    return card
