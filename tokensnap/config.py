"""Read/write ~/.tokensnap/config.json with sane defaults."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger("tokensnap.config")

CONFIG_DIR = Path.home() / ".tokensnap"
CONFIG_FILE = CONFIG_DIR / "config.json"

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
    # Memory Card generator: "auto" uses a local Ollama model when one is
    # running and silently falls back to regex; "ollama" is the same but
    # warns when the server is unreachable; "off" is regex-only.
    "llm_compressor": "auto",
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "llama3.2",
    # Seconds to wait for the local model before falling back to regex
    "ollama_timeout": 10.0,
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
    "ollama_timeout": float,
}

# Keys restricted to a fixed set of values
_CHOICES = {
    "llm_compressor": ("auto", "ollama", "off"),
}

# Pre-0.3 config key names, kept working so old muscle memory / scripts and
# already-saved config.json files don't silently stop applying.
_ALIASES = {
    "keep_last_n": "keep_messages",
}


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
