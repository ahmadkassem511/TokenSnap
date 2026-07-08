"""Tracks the "current project" the proxy tags each request with.

The proxy is a *single long-running process shared across sessions*, so it
can't learn the project from its own (fixed) environment when it was started
earlier for something else - which is why the older env-var-only approach left
everything tagged 'unknown'. Instead, every launch writes the project here and
the proxy reads it *per request*, so switching projects takes effect
immediately, without restarting the proxy.

Semantics: sequential single-project use is exact. For genuinely concurrent
sessions through one shared proxy the most recent launch wins (the proxy has no
per-request signal to tell two live Claude Code sessions apart).

Best-effort like the rest of ~/.tokensnap: a read/write problem degrades to
'unknown', never an error on the proxy's hot path.
"""

import os

from tokensnap import config as config_mod

# Module attribute so tests can redirect it (monkeypatch project.PROJECT_FILE).
PROJECT_FILE = config_mod.CONFIG_DIR / "current_project"


def set_current_project(project: str) -> None:
    """Record the project future requests should be tagged with. Called by
    ``tokensnap run`` and the dashboard's launch button on every launch."""
    try:
        config_mod.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PROJECT_FILE.write_text(str(project or "").strip(), encoding="utf-8")
    except OSError:
        pass


def get_current_project() -> str:
    """The project to tag the next request with.

    Resolution order: the mutable state file (most recent launch) first, then
    the ``TOKENSNAP_PROJECT`` env var (a fallback for a proxy started directly
    with it set, e.g. bare ``tokensnap start``), then 'unknown'.
    """
    try:
        value = PROJECT_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    env = (os.environ.get("TOKENSNAP_PROJECT") or "").strip()
    return env or "unknown"
