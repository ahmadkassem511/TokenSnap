"""Optional OpenRouter-backed Memory Card generation.

OpenRouter (https://openrouter.ai) gives free access to several hosted
models (e.g. Meta's Llama 3.1 8B) via an OpenAI-compatible API. When an
`openrouter_api_key` is configured, the compressor asks a free OpenRouter
model to summarize the truncated history into a more accurate Memory Card
than regex extraction alone. Everything here is best-effort: any failure -
no key, network error, rate limit, malformed output - returns None and the
caller falls back to the regex card.

Resilience beyond a single call:
- `openrouter_fallback_models` gives an ordered list of backup models tried
  when the primary model fails with a *retryable* error (429/5xx/timeout).
  A non-retryable error (e.g. a bad request) stops the attempt immediately
  rather than burning through the fallback list on something retrying can't
  fix.
- If every attempted model fails, OpenRouter is put into a 60-second
  cooldown: further `try_generate_card()` calls return None immediately
  (falling back to regex) without any network I/O, so a rate-limited or
  misconfigured account doesn't add latency to every single request.
- `X-RateLimit-Remaining` / `X-RateLimit-Reset` response headers are
  captured on every call (success or failure) and mirrored into
  `stats.json`, so `tokensnap openrouter-status` and the dashboard can
  show them without making their own API calls.

This module only ever uses the *OpenRouter* API key from config, never the
Anthropic API key the proxy is relaying for the user's actual request - the
two must never be mixed.

Uses stdlib urllib (no new dependency, and it keeps this module a drop-in
architectural match for how the old Ollama integration worked). Callers run
in a worker thread (see proxy.optimize_body / compressor.build_memory_card),
so blocking I/O - including the retry delay - is fine.
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

# HTTP statuses worth retrying against a fallback model.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# How long OpenRouter calls are skipped after every attempted model fails.
_COOLDOWN_SECONDS = 60.0
# How many recent errors status_snapshot()/`openrouter-status` keep around.
_MAX_RECENT_ERRORS = 10

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

# Resilience state - all guarded by _lock.
_cooldown_until = 0.0  # monotonic timestamp; calls are skipped while now < this
_fallback_active = False  # True once a non-primary model has been attempted
_recent_errors: List[Dict[str, Any]] = []  # [{"ts", "model", "error"}, ...]
_last_rate_limit: Dict[str, Optional[str]] = {"remaining": None, "reset": None}


def reset_caches() -> None:
    """Clear the card cache and all resilience state (used by tests, config
    reloads, and whenever the dashboard saves new OpenRouter settings)."""
    global _cooldown_until, _fallback_active
    with _lock:
        _card_cache.clear()
        _cooldown_until = 0.0
        _fallback_active = False
        _recent_errors.clear()
        _last_rate_limit.update({"remaining": None, "reset": None})


def in_cooldown() -> bool:
    """True while OpenRouter calls are being skipped after a total failure."""
    with _lock:
        return time.monotonic() < _cooldown_until


def status_snapshot() -> Dict[str, Any]:
    """Everything `tokensnap openrouter-status` and the dashboard need, read
    from in-memory state only (no network call)."""
    with _lock:
        now = time.monotonic()
        cooling_down = now < _cooldown_until
        return {
            "rate_limit_remaining": _last_rate_limit["remaining"],
            "rate_limit_reset": _last_rate_limit["reset"],
            "fallback_active": _fallback_active,
            "in_cooldown": cooling_down,
            "cooldown_seconds_left": int(round(_cooldown_until - now)) if cooling_down else 0,
            "recent_errors": list(_recent_errors[-5:]),
        }


def _record_error(model: str, exc: BaseException) -> None:
    with _lock:
        _recent_errors.append({"ts": time.time(), "model": model, "error": str(exc)})
        del _recent_errors[: -_MAX_RECENT_ERRORS]


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRYABLE_STATUS
    if isinstance(exc, urllib.error.URLError):
        return True  # covers connection errors and socket timeouts
    return False


def _capture_rate_limit(headers: Any) -> None:
    """Mirror X-RateLimit-* response headers into memory and stats.json.
    Safe to call with None/empty headers (nothing to record)."""
    if not headers:
        return
    remaining = headers.get("X-RateLimit-Remaining")
    reset = headers.get("X-RateLimit-Reset")
    if remaining is None and reset is None:
        return
    with _lock:
        if remaining is not None:
            _last_rate_limit["remaining"] = remaining
        if reset is not None:
            _last_rate_limit["reset"] = reset
        snapshot = dict(_last_rate_limit)
    try:
        from tokensnap import stats

        stats.set_openrouter_rate_limit(snapshot["remaining"], snapshot["reset"])
    except Exception:  # noqa: BLE001 - stats plumbing must never break a call
        pass


def _sleep(seconds: float) -> None:
    """Thin wrapper so tests can monkeypatch away the real retry delay."""
    time.sleep(seconds)


def enabled(cfg: Dict[str, Any]) -> bool:
    """True when compressor_type is "openrouter" and a key is configured."""
    return (
        str(cfg.get("compressor_type", "")).lower() == "openrouter"
        and bool(str(cfg.get("openrouter_api_key", "")).strip())
    )


def _http_post_json(
    url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float
) -> "tuple[Dict[str, Any], Any]":
    """POST JSON to `url`. Returns (decoded_json_body, response_headers).

    Raises `urllib.error.HTTPError` on 4xx/5xx (with `.headers` still
    attached, so rate-limit headers on a 429 aren't lost) and
    `urllib.error.URLError` on connection failures/timeouts.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8", "replace"))
        return body, resp.headers


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
    try:
        response, resp_headers = _http_post_json(API_URL, payload, headers, _TIMEOUT)
    except urllib.error.HTTPError as exc:
        # Rate-limit headers are present on 429 responses too - capture them
        # before re-raising so callers can report the limit even on failure.
        _capture_rate_limit(getattr(exc, "headers", None))
        raise
    _capture_rate_limit(resp_headers)
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
    """Ask the configured OpenRouter model to summarize `messages`, trying
    fallback models in order on a retryable error (429/5xx/timeout).

    Returns a sanitized dict with keys task / files_modified / decisions /
    errors_resolved, or None when OpenRouter is off, unconfigured, in
    cooldown, or every attempted model failed (caller should use the regex
    card instead).
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

    if in_cooldown():
        return None

    fallback_models = [
        m for m in (cfg.get("openrouter_fallback_models") or []) if m and m != model
    ]
    max_retries = max(0, int(cfg.get("openrouter_max_retries", 1)))
    retry_delay = float(cfg.get("openrouter_retry_delay_seconds", 5))
    candidates = ([model] + fallback_models)[: max_retries + 1]

    card: Optional[Dict[str, Any]] = None
    any_attempt_succeeded = False
    for i, candidate in enumerate(candidates):
        is_last = i == len(candidates) - 1
        if i > 0:
            with _lock:
                global _fallback_active
                _fallback_active = True
        try:
            raw_text = summarize(transcript, candidate, api_key)
            card = _sanitize(json.loads(raw_text))
            any_attempt_succeeded = True
            if card is None:
                log.info(
                    "OpenRouter (%s) returned an unusable card; using regex fallback",
                    candidate,
                )
            break
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            _record_error(candidate, exc)
            retryable = _is_retryable(exc)
            if not retryable or is_last:
                log.info(
                    "OpenRouter model %r failed (%s); using regex fallback",
                    candidate, exc,
                )
                break
            log.info(
                "OpenRouter model %r failed (%s); retrying with %r in %gs",
                candidate, exc, candidates[i + 1], retry_delay,
            )
            _sleep(retry_delay)

    if not any_attempt_succeeded:
        with _lock:
            global _cooldown_until
            _cooldown_until = time.monotonic() + _COOLDOWN_SECONDS
        log.info(
            "All OpenRouter models failed; pausing OpenRouter calls for %ds",
            int(_COOLDOWN_SECONDS),
        )

    with _lock:
        _card_cache[key] = card
        while len(_card_cache) > _CARD_CACHE_MAX:
            _card_cache.popitem(last=False)
    return card
