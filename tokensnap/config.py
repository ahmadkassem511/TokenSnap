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
    "keep_last_n": 3,
    # Value keep_last_n drops to when nearing the context window
    "aggressive_keep_last_n": 2,
    # Fraction of the model context window that triggers aggressive mode
    "context_threshold": 0.9,
    # Don't compress conversations shorter than this many messages
    "min_messages_to_compress": 8,
    # Optional stored API key (normally unused: the proxy forwards the
    # key Claude Code already sends in request headers)
    "key": "",
    "log_level": "INFO",
}

# Keys coerced to a specific type when set from the CLI (values arrive as strings)
_TYPES = {
    "port": int,
    "keep_last_n": int,
    "aggressive_keep_last_n": int,
    "min_messages_to_compress": int,
    "context_threshold": float,
}


def load() -> Dict[str, Any]:
    """Return config merged over defaults. Never raises on a broken file."""
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read %s (%s); using defaults", CONFIG_FILE, exc)
    return cfg


def save(cfg: Dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def set_value(key: str, value: str) -> Any:
    """Set one config key from a CLI string, coercing to the right type."""
    if key not in DEFAULTS:
        raise KeyError(
            "Unknown config key %r. Valid keys: %s" % (key, ", ".join(sorted(DEFAULTS)))
        )
    caster = _TYPES.get(key)
    coerced: Any = value
    if caster is not None:
        coerced = caster(value)
    cfg = load()
    cfg[key] = coerced
    save(cfg)
    return coerced
