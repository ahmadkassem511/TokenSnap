"""Read/write ~/.tokensnap/config.json with sane defaults."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger("tokensnap.config")

CONFIG_DIR = Path.home() / ".tokensnap"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    raise ValueError("expected a boolean (true/false), got %r" % value)


def _to_model_list(value: Any) -> list:
    """Accept either a real list (already-loaded JSON) or a CLI string like
    'model-a, model-b' and return a clean list of non-empty model names."""
    if isinstance(value, list):
        items = value
    else:
        items = str(value).split(",")
    return [str(m).strip() for m in items if str(m).strip()]


DEFAULTS: Dict[str, Any] = {
    # Proxy listener
    "host": "127.0.0.1",
    "port": 8889,
    # Real Anthropic API endpoint requests are forwarded to
    "upstream": "https://api.anthropic.com",
    # History compression: keep the last N user/assistant exchanges verbatim
    # before older history is summarized into a Memory Card. 10 is a
    # reasonable default for most projects; bump it for large multi-file
    # projects that need more context, or see `tokensnap preset`.
    "keep_messages": 10,
    # Value keep_messages drops to when nearing the context window
    "aggressive_keep_last_n": 2,
    # Fraction of the model context window that triggers aggressive mode
    "context_threshold": 0.95,
    # Don't compress conversations shorter than this many messages
    "min_messages_to_compress": 8,
    # Per-message noise reduction (assistant messages passed through
    # untouched; user terminal dumps and tool results compressed to their
    # error/warning/status lines) applied before the history is truncated.
    # When False, the legacy uniform keep_messages truncation is used instead.
    "selective_compression": True,
    # Memory Card generator: "openrouter" asks a free OpenRouter model for a
    # high-quality summary; "regex" uses fast rule-based extraction only;
    # "off" disables the Memory Card entirely (full history is kept, no
    # truncation - only per-message noise cleaning still applies).
    "compressor_type": "regex",
    # Get a free key at https://openrouter.ai/keys
    "openrouter_api_key": "",
    "openrouter_model": "meta-llama/llama-3.1-8b-instruct:free",
    # Models tried in order if openrouter_model fails with a retryable error
    # (429/5xx/timeout). Empty means no fallback - a failure just falls back
    # to the regex card, same as before.
    "openrouter_fallback_models": [],
    # Total attempts across openrouter_model + openrouter_fallback_models is
    # capped at 1 + this value.
    "openrouter_max_retries": 1,
    "openrouter_retry_delay_seconds": 5,
    # --- Differential Context Engine ---------------------------------------
    # When True, the proxy mirrors the whole conversation to a local Context
    # Store (~/.tokensnap/context_store.db) and sends the model only the last
    # couple of exchanges plus a compact "Context Tree" (an index of important
    # past events) with a `fetch_context` tool to pull detail back on demand -
    # instead of the Memory Card path. Opt-in: it rewrites the system prompt
    # every turn (which invalidates Anthropic prompt caching), and the
    # fetch_context tool cycle isn't served until Phase 3, so leave it off
    # until you've measured real cache/cost impact for your workload.
    "context_store_enabled": False,
    # How many recent important events the Context Tree summarizes.
    "context_tree_size": 20,
    # --- Project Primer ----------------------------------------------------
    # When True, the proxy injects a compact, auto-generated project overview
    # (languages, framework, folder structure, git branch/last commit, README
    # summary) into the system prompt on the first request of each session, so
    # Claude Code understands the codebase immediately. Generated once per
    # session from the current project directory; costs a small amount of
    # tokens on that first request only.
    "project_primer_enabled": True,
    # Optional stored API key (normally unused: the proxy forwards the
    # key Claude Code already sends in request headers)
    "key": "",
    "log_level": "INFO",
}

# Keys coerced to a specific type when set from the CLI (values arrive as strings)
_TYPES = {
    "port": int,
    "keep_messages": int,
    "aggressive_keep_last_n": int,
    "min_messages_to_compress": int,
    "context_threshold": float,
    "selective_compression": _to_bool,
    "context_store_enabled": _to_bool,
    "context_tree_size": int,
    "project_primer_enabled": _to_bool,
    "openrouter_fallback_models": _to_model_list,
    "openrouter_max_retries": int,
    "openrouter_retry_delay_seconds": float,
}

# Keys restricted to a fixed set of values
_CHOICES = {
    "compressor_type": ("openrouter", "regex", "off"),
}

# Pre-0.3 config key names, kept working so old muscle memory / scripts and
# already-saved config.json files don't silently stop applying.
_ALIASES = {
    "keep_last_n": "keep_messages",
}

# Keys from the removed Ollama integration - dropped on load, never migrated
# to a namesake (there isn't one; OpenRouter has its own model/key concept).
_REMOVED_KEYS = ("ollama_url", "ollama_model", "ollama_timeout")


def resolve_key(key: str) -> str:
    """Map a legacy config key name to its current name (identity if current)."""
    return _ALIASES.get(key, key)


def load() -> Dict[str, Any]:
    """Return config merged over defaults. Never raises on a broken file."""
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # One-time migration: a pre-0.3 file has `keep_last_n` instead of
            # `keep_messages`. Carry the user's configured value forward and
            # drop the old key so it doesn't linger as a stray field.
            for old, new in _ALIASES.items():
                if old in raw:
                    raw.setdefault(new, raw[old])
                    del raw[old]
            # Pre-0.4 files used `llm_compressor` (Ollama-based: auto/ollama/
            # off) instead of `compressor_type`. All three old modes safely
            # map to "regex" - there is no OpenRouter key to carry over, and
            # "regex" preserves the old truncation behavior exactly (unlike
            # the new "off", which disables truncation entirely).
            if "llm_compressor" in raw:
                raw.setdefault("compressor_type", "regex")
                del raw["llm_compressor"]
            for stale in _REMOVED_KEYS:
                raw.pop(stale, None)
            cfg.update(raw)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read %s (%s); using defaults", CONFIG_FILE, exc)
    return cfg


def save(cfg: Dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def set_value(key: str, value: str) -> Any:
    """Set one config key from a CLI string, coercing to the right type."""
    key = resolve_key(key)
    if key not in DEFAULTS:
        raise KeyError(
            "Unknown config key %r. Valid keys: %s" % (key, ", ".join(sorted(DEFAULTS)))
        )
    caster = _TYPES.get(key)
    coerced: Any = value
    if caster is not None:
        coerced = caster(value)
    if key in _CHOICES:
        coerced = str(value).lower()
        if coerced not in _CHOICES[key]:
            raise ValueError(
                "%r must be one of: %s" % (key, ", ".join(_CHOICES[key]))
            )
    cfg = load()
    cfg[key] = coerced
    save(cfg)
    return coerced
