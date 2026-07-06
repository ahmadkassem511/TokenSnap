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
_availability: Dict[str, Any] = {
    "url": None, "checked_at": 0.0, "available": False, "reason": "",
}
_card_cache: "OrderedDict[str, Optional[Dict[str, Any]]]" = OrderedDict()

_MISS = object()  # sentinel: hash not in cache (None is a cached failure)


def reset_caches() -> None:
    """Clear availability + card caches (used by tests and config reloads)."""
    with _lock:
        _availability.update(
            {"url": None, "checked_at": 0.0, "available": False, "reason": ""}
        )
        _card_cache.clear()


def enabled(cfg: Dict[str, Any]) -> bool:
    """True when the LLM compressor is switched on in config."""
    return str(cfg.get("llm_compressor", "off")).lower() in ("auto", "ollama")


def _http_get(url: str, timeout: float) -> Dict[str, Any]:
    """GET `url`, return the decoded JSON body. Raises on connection, HTTP,
    or decode errors."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _http_post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """POST JSON to `url`, return the decoded JSON response body."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _model_pulled(model: str, tags_body: Any) -> bool:
    """True when `model` appears in a /api/tags response. Ollama tags often
    carry a `:tag` suffix (e.g. llama3.2:latest); match with or without it."""
    if not model:
        return False
    models = tags_body.get("models") if isinstance(tags_body, dict) else None
    if not isinstance(models, list):
        return False
    names = [m.get("name") for m in models if isinstance(m, dict)]
    if model in names:
        return True
    base = model.split(":")[0]
    return any(isinstance(n, str) and n.split(":")[0] == base for n in names)


def is_available(cfg: Dict[str, Any]) -> bool:
    """True when an Ollama server is reachable AND the configured
    `ollama_model` is pulled.

    Probes at most once per _AVAILABILITY_TTL seconds; both the "server
    down" and "model missing" outcomes are cached (with a reason string),
    so a broken setup doesn't add latency or repeat warnings on every
    request. Use `status_reason()` for a human-readable explanation.
    """
    if not enabled(cfg):
        return False
    url = str(cfg.get("ollama_url", "")).rstrip("/")
    model = str(cfg.get("ollama_model", "")).strip()
    if not url:
        return False

    cache_key = url + "\x00" + model
    now = time.monotonic()
    with _lock:
        if (
            _availability["url"] == cache_key
            and now - _availability["checked_at"] < _AVAILABILITY_TTL
        ):
            return bool(_availability["available"])

    ok = False
    reason = ""
    try:
        tags = _http_get(url + "/api/tags", _PING_TIMEOUT)
    except (urllib.error.URLError, OSError, ValueError):
        reason = "no Ollama server answered at %s" % url
    else:
        if not model:
            reason = "no ollama_model is configured"
        elif _model_pulled(model, tags):
            ok = True
        else:
            reason = (
                "Ollama is running at %s but model %r is not pulled. "
                "Run `ollama pull %s`, or switch to regex-only cards with "
                '`tokensnap config set ollama_model ""`.' % (url, model, model)
            )

    with _lock:
        _availability.update(
            {"url": cache_key, "checked_at": now, "available": ok, "reason": reason}
        )

    if not ok and reason:
        # "ollama" mode is an explicit user choice, so a missing server/model
        # is worth a louder warning than the silent-by-design "auto" mode.
        level = log.warning if str(cfg.get("llm_compressor", "")).lower() == "ollama" else log.info
        level("Memory Cards: %s; using regex fallback.", reason)
    return ok


def status_reason(cfg: Dict[str, Any]) -> str:
    """One-line human status of the Memory Card generator, for the proxy
    startup log and the stats file (surfaced in `tokensnap status`/`monitor`)."""
    if not enabled(cfg):
        return "regex (llm_compressor=off)"
    if is_available(cfg):
        return "ollama:%s" % cfg.get("ollama_model")
    with _lock:
        reason = _availability.get("reason") or "unavailable"
    return "regex (%s)" % reason


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
