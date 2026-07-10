"""The Tokensnap web dashboard - ``tokensnap dashboard``.

A local, single-page control centre for Tokensnap, served on
``http://127.0.0.1:9876``. It's an alternative to the ``tokensnap monitor``
TUI, adding things a terminal can't do well: historical charts, a first-run
setup wizard (compression preset + optional OpenRouter key), and a settings
page.

It runs as its own aiohttp application (the same web stack the proxy already
uses - no new dependency, no Flask), completely independent of the proxy: the
proxy is a separate background process, so starting, stopping, or closing the
dashboard never touches it or the request logging.

Data sources:
  * live status / totals - ``stats.json`` (written by the proxy),
  * historical charts    - ``history.db`` (written by the proxy),
  * live log lines       - ``~/.tokensnap/proxy.log`` (the proxy's own log).

The server binds loopback only. It never receives the Anthropic API key, and
the OpenRouter key it *does* accept (to enable smart Memory Cards) is never
echoed back to the browser once saved - only whether one is set.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from typing import Any, Dict, List

from aiohttp import web

from tokensnap import __version__, context_store, history, openrouter, project, stats
from tokensnap import config as config_mod
from tokensnap import project_primer
from tokensnap.utils import resolve_claude_command

DEFAULT_PORT = 9876

# A completed setup drops this marker so the dashboard opens on the home page
# instead of the wizard next time.
_SETUP_MARKER = config_mod.CONFIG_DIR / ".setup_complete"

# keep_messages values behind the named presets, shared by the wizard + settings.
PRESETS: Dict[str, Dict[str, Any]] = {
    "simple": {"keep_messages": 5, "selective_compression": True, "compressor_type": "regex"},
    "balanced": {"keep_messages": 10, "selective_compression": True, "compressor_type": "regex"},
    "complex": {"keep_messages": 20, "selective_compression": False, "compressor_type": "regex"},
    "smart": {"keep_messages": 25, "selective_compression": True, "compressor_type": "openrouter"},
    "maximum": {"keep_messages": 999, "selective_compression": True, "compressor_type": "off"},
}


def tail_log(n: int = 200) -> List[str]:
    """Return the last ``n`` lines of the proxy's log file (empty if none)."""
    log_path = config_mod.CONFIG_DIR / "proxy.log"
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            return [line.rstrip("\n") for line in f.readlines()[-n:]]
    except OSError:
        return []


def setup_is_complete() -> bool:
    return _SETUP_MARKER.exists()


def mark_setup_complete() -> None:
    config_mod.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _SETUP_MARKER.write_text("ok", encoding="utf-8")


def _public_config() -> Dict[str, Any]:
    """Config values the settings/setup pages need. The OpenRouter key
    itself is never sent to the browser once saved - only whether one is
    set - so it can't leak into page source or browser history."""
    cfg = config_mod.load()
    return {
        "keep_messages": int(cfg["keep_messages"]),
        "selective_compression": bool(cfg["selective_compression"]),
        "compressor_type": cfg["compressor_type"],
        "openrouter_model": cfg["openrouter_model"],
        "openrouter_fallback_models": list(cfg.get("openrouter_fallback_models") or []),
        "openrouter_api_key_set": bool(str(cfg.get("openrouter_api_key", "")).strip()),
        "context_store_enabled": bool(cfg.get("context_store_enabled", False)),
        "context_tree_size": int(cfg.get("context_tree_size", 20)),
        "project_primer_enabled": bool(cfg.get("project_primer_enabled", True)),
        "project_cortex_enabled": bool(cfg.get("project_cortex_enabled", True)),
        "session_bridge_auto_inject": bool(cfg.get("session_bridge_auto_inject", True)),
    }


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------
async def index(request: web.Request) -> web.StreamResponse:
    """Dashboard home - redirects to the wizard on first run."""
    if not setup_is_complete():
        raise web.HTTPFound("/setup")
    return web.Response(text=_dashboard_page(), content_type="text/html")


async def api_stats(request: web.Request) -> web.Response:
    """Live status + running totals, polled by the dashboard every few seconds."""
    cfg = config_mod.load()
    data = stats.load()
    totals = data["totals"]
    running = stats.proxy_running(cfg["host"], int(cfg["port"]))
    before = totals["tokens_before"]
    saved = totals["tokens_saved"]
    pct = 100.0 * saved / before if before else 0.0
    recent = data.get("recent") or []
    active_model = recent[-1]["model"] if recent else "-"
    # Differential Context Engine one-liner, matching the wording used by
    # `tokensnap status` / `tokensnap monitor` so all three read the same.
    ctx_enabled = bool(cfg.get("context_store_enabled", False))
    context_status = (
        "enabled (tree size: %d)" % int(cfg.get("context_tree_size", 20))
        if ctx_enabled else "disabled"
    )
    # Send the tail newest-first so the table renders without extra work client-side.
    recent_rows = [
        {
            "ts": r.get("ts", 0),
            "model": r.get("model", "?"),
            "before": r.get("before", 0),
            "saved": r.get("saved", 0),
            "real_input": r.get("real_input", 0),
            "real_output": r.get("real_output", 0),
            "status": r.get("status", 0),
            "aggressive": bool(r.get("aggressive")),
        }
        for r in reversed(recent[-15:])
    ]
    return web.json_response({
        "running": running,
        "url": "http://%s:%s" % (cfg["host"], cfg["port"]),
        "requests": totals["requests"],
        "tokens_before": before,
        "tokens_after": totals["tokens_after"],
        "tokens_saved": saved,
        "pct": round(pct, 1),
        "real_input": totals["real_input"],
        "real_output": totals["real_output"],
        "real_cache_read": totals["real_cache_read"],
        "real_cache_creation": totals["real_cache_creation"],
        "llm_status": data.get("llm_status") or "not started yet",
        "active_model": active_model,
        "keep_messages": int(cfg["keep_messages"]),
        "compressor_type": cfg["compressor_type"],
        "selective_compression": bool(cfg["selective_compression"]),
        "openrouter_api_key_set": bool(str(cfg.get("openrouter_api_key", "")).strip()),
        "context_store_enabled": ctx_enabled,
        "context_tree_size": int(cfg.get("context_tree_size", 20)),
        "context_status": context_status,
        "recent": recent_rows,
    })


async def api_chart(request: web.Request) -> web.Response:
    """Aggregated saved-token series for the requested period, optionally
    filtered to a single project (``?project=...``)."""
    period = request.query.get("period", "day")
    project = request.query.get("project") or None
    return web.json_response(history.chart_data(period, project=project))


async def api_stats_session(request: web.Request) -> web.Response:
    """Current-session summary from stats.json (resets on proxy restart) - the
    'This Session' level of the dashboard's three-level stats."""
    cfg = config_mod.load()
    data = stats.load()
    totals = data["totals"]
    before = totals["tokens_before"]
    saved = totals["tokens_saved"]
    pct = 100.0 * saved / before if before else 0.0
    return web.json_response({
        "running": stats.proxy_running(cfg["host"], int(cfg["port"])),
        "requests": totals["requests"],
        "tokens_saved": saved,
        "pct": round(pct, 1),
        "real_input": totals["real_input"],
        "real_output": totals["real_output"],
    })


async def api_stats_alltime(request: web.Request) -> web.Response:
    """Cumulative totals from the persistent history DB - the 'All Time' level.
    Survives proxy restarts, unlike the session totals in stats.json."""
    t = history.totals()
    return web.json_response({
        "total_requests": t["requests"],
        "total_est_saved": t["saved"],
        "total_real_in": t["real_in"],
        "total_real_out": t["real_out"],
    })


async def api_stats_projects(request: web.Request) -> web.Response:
    """Per-project all-time breakdown, most tokens-saved first. Each entry also
    carries a short display name (the last path segment) for the UI."""
    projects = []
    for p in history.project_totals():
        projects.append({
            "project": p["project"],
            "name": _project_display_name(p["project"]),
            "requests": p["requests"],
            "saved": p["saved"],
            "real_in": p["real_in"],
            "real_out": p["real_out"],
        })
    return web.json_response({"projects": projects})


def _project_display_name(project: str) -> str:
    """Short, readable label for a project - the last path segment of a
    directory path, or the value itself if it isn't path-like."""
    if not project or project == "unknown":
        return "unknown"
    trimmed = project.replace("\\", "/").rstrip("/")
    tail = trimmed.rsplit("/", 1)[-1]
    return tail or project


async def api_log(request: web.Request) -> web.Response:
    """The tail of the proxy log, for the live log panel."""
    return web.json_response({"lines": tail_log(200)})


async def api_context(request: web.Request) -> web.Response:
    """Differential Context Engine status for the dashboard/settings panel:
    whether it's on, how big the tree is, how many events are mirrored in the
    Context Store, and the tokens/recalls it has accounted for this session."""
    cfg = config_mod.load()
    totals = stats.load()["totals"]
    return web.json_response({
        "enabled": bool(cfg.get("context_store_enabled", False)),
        "tree_size": int(cfg.get("context_tree_size", 20)),
        "events_stored": context_store.event_count(),
        "context_requests": totals.get("context_requests", 0),
        "tokens_saved": totals.get("context_saved", 0),
        "events_fetched": totals.get("context_events_fetched", 0),
    })


async def api_primer(request: web.Request) -> web.Response:
    """Project Primer status for the dashboard/settings panel: whether it's on
    and the most recently generated Project Card (persisted by the proxy)."""
    cfg = config_mod.load()
    return web.json_response({
        "enabled": bool(cfg.get("project_primer_enabled", True)),
        "card": project_primer.load_last_card(),
    })


def _cortex_project_dir() -> str:
    """The project directory Cortex acts on for the dashboard - the currently
    tagged project (falling back to the dashboard's launch dir)."""
    tagged = project.get_current_project()
    if tagged and tagged != "unknown" and os.path.isdir(tagged):
        return tagged
    launch = get_launch_dir()
    return launch if os.path.isdir(launch) else ""


async def api_cortex(request: web.Request) -> web.Response:
    """Project Cortex status for the dashboard: whether it's on, the project's
    DNA (stack, focus, decisions, resolved issues), and the session-bridge count."""
    from tokensnap import project_dna, session_bridge

    cfg = config_mod.load()
    project_dir = _cortex_project_dir()
    # Generate-on-view: ensure_dna scans once (throttled by dna_update_interval)
    # then just reads, so opening the dashboard populates the project's DNA.
    dna = project_dna.ensure_dna(project_dir, cfg) if project_dir else project_dna._empty_dna()
    static = dna.get("static") or {}
    return web.json_response({
        "enabled": bool(cfg.get("project_cortex_enabled", True)),
        "bridge_auto_inject": bool(cfg.get("session_bridge_auto_inject", True)),
        "project_dir": project_dir,
        "project_name": static.get("project_name", ""),
        "language": static.get("language", ""),
        "framework": static.get("framework", ""),
        "key_dependencies": static.get("key_dependencies", []),
        "entry_points": static.get("entry_points", []),
        "focus": dna.get("focus", ""),
        "decisions": [d.get("text", "") for d in dna.get("decisions", [])][-8:],
        "resolved_issues": [r.get("text", "") for r in dna.get("resolved_issues", [])][-8:],
        "session_count": len(session_bridge.list_sessions(project_dir)) if project_dir else 0,
        "updated_at": dna.get("updated_at", 0),
    })


async def api_cortex_focus(request: web.Request) -> web.Response:
    """Set the current project's focus/goal from the dashboard."""
    from tokensnap import project_dna

    project_dir = _cortex_project_dir()
    if not project_dir:
        return web.json_response({"ok": False, "error": "No project directory set."})
    try:
        body = await request.json()
    except (ValueError, TypeError):
        body = {}
    project_dna.set_focus(project_dir, str(body.get("focus", "")))
    return web.json_response({"ok": True, "focus": project_dna.load_dna(project_dir).get("focus", "")})


async def api_cortex_refresh(request: web.Request) -> web.Response:
    """Force a re-scan of the project's static DNA from the dashboard."""
    from tokensnap import project_dna

    project_dir = _cortex_project_dir()
    if not project_dir:
        return web.json_response({"ok": False, "error": "No project directory set."})
    cfg = dict(config_mod.load())
    cfg["dna_update_interval"] = 0  # force rescan
    project_dna.ensure_dna(project_dir, cfg)
    return web.json_response({"ok": True})


async def api_openrouter_status(request: web.Request) -> web.Response:
    """OpenRouter model/fallback/rate-limit status, for the Settings page
    and the Dashboard's Memory Cards indicator."""
    cfg = config_mod.load()
    snap = openrouter.status_snapshot()
    return web.json_response({
        "primary_model": cfg.get("openrouter_model"),
        "fallback_models": list(cfg.get("openrouter_fallback_models") or []),
        "rate_limit_remaining": snap["rate_limit_remaining"],
        "rate_limit_reset": snap["rate_limit_reset"],
        "fallback_active": snap["fallback_active"],
        "in_cooldown": snap["in_cooldown"],
        "cooldown_seconds_left": snap["cooldown_seconds_left"],
        "recent_errors": snap["recent_errors"],
    })


async def setup_page(request: web.Request) -> web.Response:
    return web.Response(text=_setup_page(), content_type="text/html")


async def setup_save(request: web.Request) -> web.Response:
    """Persist the wizard's choices and mark first-run setup complete."""
    saved = await _apply_settings(request)
    mark_setup_complete()
    return web.json_response({"ok": True, "saved": saved})


async def settings_page(request: web.Request) -> web.Response:
    return web.Response(text=_settings_page(), content_type="text/html")


async def settings_save(request: web.Request) -> web.Response:
    saved = await _apply_settings(request)
    return web.json_response({"ok": True, "saved": saved})


async def _apply_settings(request: web.Request) -> Dict[str, Any]:
    """Validate + persist the config keys a POST body carries. Shared by the
    wizard and the settings page. Unknown/invalid keys are ignored."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        body = {}
    if not isinstance(body, dict):
        body = {}

    # A named preset is a shortcut for keep_messages + compressor settings.
    preset = str(body.get("preset", "")).lower()
    if preset in PRESETS:
        for key, value in PRESETS[preset].items():
            body.setdefault(key, value)

    saved: Dict[str, Any] = {}
    for key in ("keep_messages", "selective_compression", "compressor_type",
                "openrouter_model", "context_store_enabled", "context_tree_size",
                "project_primer_enabled", "project_cortex_enabled",
                "session_bridge_auto_inject"):
        if key not in body or body[key] in (None, ""):
            continue
        try:
            saved[key] = config_mod.set_value(key, str(body[key]))
        except (KeyError, ValueError):
            continue  # invalid value - skip it rather than 500

    # openrouter_fallback_models: unlike the keys above, an explicit "" is a
    # meaningful value here (clear the fallback list), not "leave unchanged".
    if "openrouter_fallback_models" in body and body["openrouter_fallback_models"] is not None:
        try:
            saved["openrouter_fallback_models"] = config_mod.set_value(
                "openrouter_fallback_models", str(body["openrouter_fallback_models"])
            )
        except (KeyError, ValueError):
            pass

    # The key is never re-displayed to the browser once saved (see module
    # docstring), so an empty submission means "leave it alone" - only a
    # non-empty value, or an explicit clear_key request, changes it. Either
    # way the response reports only whether a key is now set, never the
    # value itself.
    if body.get("clear_key"):
        config_mod.set_value("openrouter_api_key", "")
        saved["openrouter_api_key_set"] = False
    elif str(body.get("openrouter_api_key") or "").strip():
        config_mod.set_value("openrouter_api_key", str(body["openrouter_api_key"]))
        saved["openrouter_api_key_set"] = True

    # A key/model may have just changed; drop OpenRouter's cached card results.
    openrouter.reset_caches()
    return saved


async def launch(request: web.Request) -> web.Response:
    """Ensure the proxy is running and open a new terminal with Claude Code
    pointed at it. Called by the "Launch Claude Code" buttons on the setup
    wizard and Settings page."""
    ok, message = _launch_claude_terminal()
    return web.json_response({"ok": ok, "message": message})


_CLAUDE_INSTALL_HINT = (
    "Claude Code isn't installed (or not on PATH). Install it from "
    "https://claude.ai/download, or run: npm install -g @anthropic-ai/claude-code"
)


def _proxy_host_port() -> "tuple[str, int]":
    cfg = config_mod.load()
    return cfg["host"], int(cfg["port"])


def _launch_claude_terminal() -> "tuple[bool, str]":
    """Start the proxy (if needed) and open a new OS terminal running
    ``tokensnap run claude`` - which itself sets ANTHROPIC_BASE_URL and
    launches Claude Code once the proxy is confirmed up.

    Uses ``python -m tokensnap`` (always resolvable) so it works whether or
    not the ``tokensnap`` console script is on PATH. Returns (ok, message);
    `message` is shown as both a toast and a persistent on-page result.
    """
    # Resolve claude past PATH (npm global bin / npx) so we don't wrongly
    # report it missing when it's installed but not on PATH. The actual launch
    # is delegated to `tokensnap run claude`, which resolves it again itself.
    if resolve_claude_command(["claude"]) is None:
        return False, _CLAUDE_INSTALL_HINT

    # Tag this session's requests with the selected project folder so the
    # dashboard's per-project stats attribute them correctly. The proxy reads
    # this per request from a state file (authoritative - the full path, so
    # same-named folders in different locations don't collide), so it works
    # even against a proxy that's already running. TOKENSNAP_PROJECT is set too
    # as a fallback for a proxy freshly started from this process's
    # environment; use the short folder name there and don't clobber a value
    # the user set manually.
    project.set_current_project(get_launch_dir())
    os.environ.setdefault("TOKENSNAP_PROJECT", _project_display_name(get_launch_dir()))

    if not stats.proxy_running(*_proxy_host_port()):
        ok, log_path = stats.start_proxy_detached()
        if not ok:
            return False, "Couldn't start the Tokensnap proxy. See log: %s" % log_path

    success_message = (
        "Claude Code launched! A new terminal window should have opened. "
        "You can close this browser tab or keep it open to monitor savings."
    )
    # Open Claude Code in the folder the dashboard is pointed at. Guard against
    # a stale directory (e.g. one picked earlier and since deleted) so we fall
    # back to the default cwd instead of failing the launch outright.
    launch_dir = get_launch_dir()
    cwd = launch_dir if os.path.isdir(launch_dir) else None
    py = sys.executable
    inner = '"%s" -m tokensnap run claude' % py
    try:
        if os.name == "nt":
            # The new console window inherits this cwd, so Claude opens there.
            subprocess.Popen(
                'start "TokenSnap - Claude Code" cmd /k %s' % inner,
                shell=True, cwd=cwd,
            )
            return True, success_message
        if sys.platform == "darwin":
            # Terminal.app's `do script` doesn't inherit our cwd, so cd first.
            script = inner
            if cwd:
                script = "cd '%s' && %s" % (cwd.replace("'", "'\\''"), inner)
            subprocess.Popen(
                ["osascript", "-e",
                 'tell application "Terminal" to do script "%s"' % script]
            )
            return True, success_message
        for term in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm"):
            if shutil.which(term):
                subprocess.Popen([term, "-e", "sh", "-c", inner], cwd=cwd)
                return True, success_message
        return False, (
            "Couldn't find a terminal to open. Run this yourself: "
            "tokensnap run claude"
        )
    except Exception as exc:  # noqa: BLE001
        return False, "Couldn't launch a terminal (%s). Run: tokensnap run claude" % exc


# ---------------------------------------------------------------------------
# Project directory (where "Launch Claude Code" opens Claude)
# ---------------------------------------------------------------------------
# Defaults to the dashboard's startup working directory; the folder picker or
# the manual path box on the dashboard can override it for later launches.
_launch_dir = os.getcwd()


def get_launch_dir() -> str:
    return _launch_dir


def set_launch_dir(path: str) -> bool:
    """Point future launches at `path` if it's an existing directory. Returns
    True on success, False if the path is empty, missing, or not a directory."""
    global _launch_dir
    if not path or not os.path.isdir(path):
        return False
    _launch_dir = os.path.abspath(path)
    return True


# A standalone script so the native folder dialog runs in its own process (its
# own main thread - required by tkinter/Cocoa) instead of blocking the
# dashboard's asyncio event loop or fighting it for the main thread.
_FOLDER_PICKER_SCRIPT = (
    "import sys\n"
    "import tkinter as tk\n"
    "from tkinter import filedialog\n"
    "root = tk.Tk()\n"
    "root.withdraw()\n"
    "root.attributes('-topmost', True)\n"
    "path = filedialog.askdirectory(title='Select a project folder for Claude Code')\n"
    "root.destroy()\n"
    "sys.stdout.write(path or '')\n"
)


def _pick_folder_native(timeout: float = 300.0):
    """Open a native folder picker in a subprocess. Returns (path, error):
      * (path, None)  - a folder was chosen
      * (None, None)  - the user cancelled the dialog
      * (None, error) - the picker couldn't run (headless / no tkinter)
    Blocking; call it via run_in_executor so the event loop stays responsive.
    """
    popen_kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        # Show the Tk dialog without also flashing a console window.
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _FOLDER_PICKER_SCRIPT],
            capture_output=True, text=True, timeout=timeout, **popen_kwargs,
        )
    except subprocess.TimeoutExpired:
        return None, "The folder picker timed out."
    except Exception as exc:  # noqa: BLE001 - any failure -> a clear message, not a 500
        return None, (
            "Couldn't open a folder picker (%s). Type or paste a path instead."
            % exc
        )
    if proc.returncode != 0:
        # tkinter missing, no display, or the dialog couldn't attach to a
        # desktop (e.g. a headless / detached dashboard process).
        detail = (proc.stderr or "").strip().splitlines()
        hint = (" [%s]" % detail[-1]) if detail else ""
        return None, (
            "A native folder picker isn't available here%s - type or paste a "
            "path into the box and click Set instead." % hint
        )
    path = (proc.stdout or "").strip()
    return (path or None), None


async def browse_folder(request: web.Request) -> web.Response:
    """Open a native folder picker and remember the chosen directory for
    subsequent launches. Runs the blocking dialog off the event loop so the
    dashboard stays responsive while it's open.

    Always returns JSON (never a 500), so the client shows a specific reason
    rather than a generic "Folder picker failed" - the picker is optional
    anyway, since a path can be typed into the box instead."""
    try:
        loop = asyncio.get_running_loop()
        path, error = await loop.run_in_executor(None, _pick_folder_native)
    except Exception as exc:  # noqa: BLE001 - degrade to a typed-path fallback
        return web.json_response(
            {"path": None,
             "error": "Folder picker unavailable (%s). Type a path and click Set."
                      % exc}
        )
    if error:
        return web.json_response({"path": None, "error": error})
    if path:
        set_launch_dir(path)
    return web.json_response({"path": path})


async def get_project_dir(request: web.Request) -> web.Response:
    """Return the directory the dashboard will launch Claude Code in."""
    return web.json_response({"path": get_launch_dir()})


async def set_project_dir(request: web.Request) -> web.Response:
    """Set the launch directory from a manually entered path, validating that
    it exists and is a directory on the server's filesystem before accepting."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        body = {}
    path = str((body or {}).get("path") or "").strip()
    if not path:
        return web.json_response({"ok": False, "error": "No path provided."})
    if not os.path.isdir(path):
        return web.json_response(
            {"ok": False, "error": "That path doesn't exist or isn't a folder."}
        )
    set_launch_dir(path)
    return web.json_response({"ok": True, "path": get_launch_dir()})


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------
def build_app() -> web.Application:
    """Construct the dashboard's aiohttp application (routes only, no I/O)."""
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/stats", api_stats)
    app.router.add_get("/api/stats/session", api_stats_session)
    app.router.add_get("/api/stats/alltime", api_stats_alltime)
    app.router.add_get("/api/stats/projects", api_stats_projects)
    app.router.add_get("/api/chart", api_chart)
    app.router.add_get("/api/log", api_log)
    app.router.add_get("/api/openrouter-status", api_openrouter_status)
    app.router.add_get("/api/context", api_context)
    app.router.add_get("/api/primer", api_primer)
    app.router.add_get("/api/cortex", api_cortex)
    app.router.add_post("/api/cortex/focus", api_cortex_focus)
    app.router.add_post("/api/cortex/refresh", api_cortex_refresh)
    app.router.add_get("/setup", setup_page)
    app.router.add_post("/setup/save", setup_save)
    app.router.add_get("/settings", settings_page)
    app.router.add_post("/settings/save", settings_save)
    app.router.add_post("/launch", launch)
    app.router.add_post("/browse-folder", browse_folder)
    app.router.add_get("/get-project-dir", get_project_dir)
    app.router.add_post("/set-project-dir", set_project_dir)
    return app


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Blocking entry point used by ``tokensnap dashboard``."""
    history.init_db()
    app = build_app()
    if open_browser:
        shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        url = "http://%s:%d/" % (shown, port)
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    web.run_app(app, host=host, port=port, print=None, handle_signals=True)


# ---------------------------------------------------------------------------
# HTML (self-contained; only external asset is the Chart.js CDN)
# ---------------------------------------------------------------------------
_CSS = """
:root{
  --bg:#0d1117; --panel:#161b22; --panel2:#1c2230; --border:#2a3242;
  --text:#e6edf3; --muted:#8b949e; --accent:#3fb950; --accent2:#58a6ff;
  --warn:#d29922; --danger:#f85149; --radius:14px;
}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  background:var(--bg);color:var(--text);line-height:1.5}
a{color:var(--accent2);text-decoration:none}
header.topbar{display:flex;align-items:center;gap:20px;padding:16px 28px;
  border-bottom:1px solid var(--border);background:linear-gradient(180deg,#12161f,#0d1117)}
.brand{font-weight:700;font-size:20px;letter-spacing:.3px}
.brand span{color:var(--accent)}
nav.tabs{display:flex;gap:8px;margin-left:auto}
nav.tabs a{padding:8px 16px;border-radius:10px;color:var(--muted);font-weight:500;font-size:14px}
nav.tabs a:hover{background:var(--panel);color:var(--text)}
nav.tabs a.active{background:var(--panel2);color:var(--text)}
.pill{display:inline-flex;align-items:center;gap:8px;padding:5px 13px;border-radius:999px;
  font-size:13px;font-weight:600;border:1px solid var(--border)}
.pill .dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}
.pill.on{color:var(--accent);border-color:rgba(63,185,80,.4)}
.pill.on .dot{background:var(--accent);box-shadow:0 0 10px var(--accent)}
.pill.off{color:var(--danger);border-color:rgba(248,81,73,.4)}
.pill.off .dot{background:var(--danger)}
.pill.warn{color:var(--warn);border-color:rgba(210,153,34,.4)}
.pill.warn .dot{background:var(--warn)}
main{max-width:1180px;margin:0 auto;padding:28px}
.grid{display:grid;gap:18px}
.cards{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
.card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:20px 22px;
  animation:rise .35s ease both}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.card h3{margin:0 0 6px;font-size:12.5px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
.card .big{font-size:30px;font-weight:700;letter-spacing:-.5px}
.card .sub{font-size:12.5px;color:var(--muted);margin-top:4px}
.accent{color:var(--accent)}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:22px;margin-top:20px}
.panel h2{margin:0 0 16px;font-size:16px}
.row{display:flex;gap:18px;flex-wrap:wrap}
.row>*{flex:1;min-width:280px}
.chartbtns{display:flex;gap:8px;margin-bottom:14px}
.chartbtns button,.btn{cursor:pointer;border:1px solid var(--border);background:var(--panel2);color:var(--text);
  padding:8px 15px;border-radius:9px;font-size:13.5px;font-weight:600;transition:.15s}
.chartbtns button:hover,.btn:hover{border-color:var(--accent2)}
.chartbtns button.active{background:var(--accent2);border-color:var(--accent2);color:#0d1117}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#0d1117}
.btn.primary:hover{filter:brightness(1.08)}
.btn.lg{padding:12px 22px;font-size:15px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.5px}
td.r,th.r{text-align:right}
.log{background:#0a0e14;border:1px solid var(--border);border-radius:10px;padding:14px;height:280px;overflow:auto;
  font-family:ui-monospace,'Cascadia Code',Consolas,monospace;font-size:12px;color:#c9d1d9;white-space:pre-wrap}
.empty{color:var(--muted);text-align:center;padding:40px 10px;font-size:14px}
.card.level{cursor:default}
.card.level .big{font-size:26px}
.projscroll{display:flex;gap:14px;overflow-x:auto;padding:4px 2px 10px}
.projcard{flex:0 0 220px;background:var(--panel2);border:1px solid var(--border);border-radius:12px;
  padding:14px 16px;cursor:pointer;transition:.15s}
.projcard:hover{border-color:var(--accent2)}
.projcard.active{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}
.projcard .pname{font-weight:700;font-size:14.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.projcard .ppath{font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px}
.projcard .pnum{font-size:20px;font-weight:700;color:var(--accent);margin-top:10px}
.projcard .pmeta{font-size:11.5px;color:var(--muted);margin-top:2px}
.bar{height:6px;border-radius:999px;background:var(--border);margin-top:10px;overflow:hidden}
.bar>span{display:block;height:100%;background:var(--accent)}
.projfilter{max-width:280px}
label.f{display:block;margin:16px 0 6px;font-weight:600;font-size:14px}
label.f small{color:var(--muted);font-weight:400}
input[type=text],input[type=number],select{width:100%;padding:10px 12px;background:var(--panel2);
  border:1px solid var(--border);border-radius:9px;color:var(--text);font-size:14px}
input:focus,select:focus{outline:none;border-color:var(--accent2)}
.presets{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.presets button{flex:1;min-width:90px}
.hwgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin:8px 0 4px}
.hwtile{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:14px}
.hwtile .k{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.hwtile .v{font-size:20px;font-weight:700;margin-top:3px}
.steps{counter-reset:s;display:flex;flex-direction:column;gap:22px}
.step{display:flex;gap:14px}
.step .n{flex:none;width:30px;height:30px;border-radius:50%;background:var(--panel2);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;font-weight:700;color:var(--accent2)}
.step .body{flex:1}
.step h3{margin:2px 0 8px;font-size:15px}
.toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:var(--panel2);
  border:1px solid var(--accent);color:var(--text);padding:12px 20px;border-radius:10px;font-size:14px;
  opacity:0;pointer-events:none;transition:.25s}
.toast.show{opacity:1;transform:translateX(-50%) translateY(-4px)}
.muted{color:var(--muted)}
"""

_NAV = (
    "<header class='topbar'><div class='brand'>Token<span>Snap</span></div>"
    "<nav class='tabs'>__TABS__</nav>"
    "<span id='statuspill' class='pill'><span class='dot'></span><span id='statustext'>...</span></span>"
    "</header>"
)


def _shell(
    title: str, active: str, body: str, extra_head: str = "", foot_js: str = ""
) -> str:
    """Render the page shell.

    `extra_head` is for content genuinely safe to run before `<body>` exists
    (e.g. the Chart.js CDN `<script src>`, which only defines a global).
    `foot_js` is for page-specific inline scripts that touch the page's own
    DOM (element lookups, event handlers) - these must run after `<main>` is
    parsed, so they're placed at the end of `<body>`, same as `_COMMON_JS`.
    """
    tabs = ""
    for href, label in (("/", "Dashboard"), ("/setup", "Setup"), ("/settings", "Settings")):
        cls = " class='active'" if href == active else ""
        tabs += "<a href='%s'%s>%s</a>" % (href, cls, label)
    nav = _NAV.replace("__TABS__", tabs)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>" + title + "</title><style>" + _CSS + "</style>" + extra_head +
        "</head><body>" + nav + "<main>" + body + "</main>"
        "<div id='toast' class='toast'></div>" + _COMMON_JS + foot_js + "</body></html>"
    )


# Shared JS: keeps the status pill in the header live on every page.
_COMMON_JS = """
<script>
function toast(msg){var t=document.getElementById('toast');t.textContent=msg;
  t.classList.add('show');clearTimeout(t._h);t._h=setTimeout(function(){t.classList.remove('show')},2600);}
function fmt(n){return (n||0).toLocaleString();}
async function refreshPill(){
  try{
    var s=await (await fetch('/api/stats')).json();
    var p=document.getElementById('statuspill'), t=document.getElementById('statustext');
    if(s.running){p.className='pill on';t.textContent='Proxy running';}
    else{p.className='pill off';t.textContent='Proxy stopped';}
    window.__STATS__=s;
    if(window.onStats)window.onStats(s);
  }catch(e){}
}
refreshPill();setInterval(refreshPill,3000);
</script>
"""


def _dashboard_page() -> str:
    body = """
<div class='grid cards'>
  <div class='card level'><h3>This Session</h3><div class='big accent' id='s_saved'>-</div>
    <div class='sub' id='s_sub'>tokens saved &middot; since the proxy last started</div></div>
  <div class='card level'><h3>All Time</h3><div class='big' id='a_saved'>-</div>
    <div class='sub' id='a_sub'>tokens saved across all sessions</div></div>
  <div class='card level'><h3>Total Projects</h3><div class='big' id='p_count'>-</div>
    <div class='sub' id='p_sub'>tracked in your history</div></div>
</div>

<div class='grid cards' style='margin-top:18px'>
  <div class='card'><h3>Real usage (Anthropic)</h3><div class='big' id='c_real'>-</div>
    <div class='sub'>in / out tokens (this session)</div></div>
  <div class='card'><h3>Active model</h3><div class='big' id='c_model' style='font-size:19px'>-</div>
    <div class='sub'>keep_messages = <span id='c_keep'>-</span></div></div>
  <div class='card'><h3>Differential Context Engine</h3><div class='big' id='c_ctx' style='font-size:17px'>-</div>
    <div class='sub' id='c_ctx_sub'>-</div></div>
</div>

<div class='panel'>
  <div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>
    <h2 style='margin:0'>Project stats</h2>
    <span class='sub muted' style='margin-left:auto' id='projhint'></span>
  </div>
  <div class='projscroll' id='projects'></div>
  <div id='projempty' class='empty' style='display:none'>
    No projects tracked yet. Launch Claude Code from this dashboard or via
    <b>tokensnap run claude</b> to start tracking.</div>
</div>

<div class='panel'>
  <div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>
    <h2 style='margin:0'>Tokens saved over time</h2>
    <select id='projfilter' class='projfilter' style='margin-left:auto'>
      <option value=''>All projects</option>
    </select>
    <div class='chartbtns' style='margin:0'>
      <button data-p='day' class='active'>7 days</button>
      <button data-p='week'>8 weeks</button>
      <button data-p='month'>6 months</button>
    </div>
  </div>
  <div style='position:relative;height:300px;margin-top:12px'>
    <canvas id='chart'></canvas>
    <div id='chartempty' class='empty' style='display:none;position:absolute;inset:0'>
      No history yet. Once you run <b>tokensnap run claude</b> through the proxy,
      your savings will appear here.</div>
  </div>
</div>

<div class='row'>
  <div class='panel'>
    <h2>Recent requests</h2>
    <div style='max-height:320px;overflow:auto'>
      <table><thead><tr><th>time</th><th>model</th><th class='r'>est.in</th>
      <th class='r'>saved</th><th class='r'>real.in</th><th class='r'>real.out</th><th class='r'>http</th></tr></thead>
      <tbody id='recent'><tr><td colspan='7' class='empty'>no requests yet</td></tr></tbody></table>
    </div>
  </div>
</div>

<div class='panel'>
  <div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>
    <h2 style='margin:0'>Project directory</h2>
    <button class='btn primary' id='launchbtn' style='margin-left:auto'>Launch Claude Code</button>
  </div>
  <p class='sub muted' style='margin:6px 0 12px'>Claude Code opens in this folder when you launch it from here.
    (Running <b>tokensnap run claude</b> in a terminal still uses that terminal's own folder.)</p>
  <div style='display:flex;gap:10px;flex-wrap:wrap;align-items:center'>
    <input type='text' id='projdir' style='flex:1;min-width:260px' placeholder='/path/to/your/project'>
    <button class='btn' id='browsebtn'>Browse&hellip;</button>
    <button class='btn' id='setdirbtn'>Set</button>
  </div>
  <div class='sub muted' id='projdirmsg' style='margin-top:8px'>Loading&hellip;</div>
</div>

<div class='panel'>
  <div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>
    <h2 style='margin:0'>Project Cortex</h2>
    <span id='cortexpill' class='pill' style='margin-left:auto'><span class='dot'></span><span id='cortextext'>&hellip;</span></span>
  </div>
  <p class='sub muted' style='margin:6px 0 10px'>TokenSnap's persistent second brain for this project: a per-project knowledge
    base (stack, focus, decisions, resolved issues) injected as immutable Core Memory at the start of every session, plus a
    Session Bridge that carries context across restarts. Stored in <b>.tokensnap/</b> inside the project.</p>
  <div style='display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:10px'>
    <input type='text' id='cortexfocus' style='flex:1;min-width:260px' placeholder='Current focus / goal for this project'>
    <button class='btn' id='cortexfocusbtn'>Set focus</button>
    <button class='btn' id='cortexrefreshbtn'>Refresh DNA</button>
  </div>
  <pre class='log' id='cortexcard' style='height:auto;max-height:280px'>Loading&hellip;</pre>
</div>

<div class='panel'>
  <div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>
    <h2 style='margin:0'>Project Primer</h2>
    <span id='primerpill' class='pill' style='margin-left:auto'><span class='dot'></span><span id='primertext'>&hellip;</span></span>
  </div>
  <p class='sub muted' style='margin:6px 0 10px'>On the first request of each session, TokenSnap injects a compact overview of
    your project (language, framework, structure, git state, README) into the system prompt, so Claude understands the
    codebase immediately. Below is the most recently generated Project Card.</p>
  <pre class='log' id='primercard' style='height:auto;max-height:260px'>No card generated yet &mdash; launch Claude Code from a project folder.</pre>
</div>

<div class='panel'>
  <div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>
    <h2 style='margin:0'>Live proxy log</h2>
  </div>
  <div class='log' id='log' style='margin-top:12px'>waiting for the proxy...</div>
  <div class='sub muted' style='margin-top:8px'>Memory Cards:
    <span id='llmpill' class='pill' style='padding:2px 9px;font-size:11px;margin-left:4px'>
      <span class='dot'></span><span id='llm'>-</span>
    </span>
  </div>
</div>
"""
    head_extra = "<script src='https://cdn.jsdelivr.net/npm/chart.js@4'></script>"
    return _shell("TokenSnap Dashboard", "/", body, extra_head=head_extra, foot_js=_DASH_JS)


_DASH_JS = """
<script>
var chart, period='day', orStatus={fallback_active:false,in_cooldown:false};
// Chart project filter, persisted across refreshes ('' = all projects).
var projectFilter='';
try{projectFilter=localStorage.getItem('tsnap_project_filter')||'';}catch(e){}
function esc(s){return String(s).replace(/[&<>]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}
function updateLLMPill(compressorType, keySet){
  var pill=document.getElementById('llmpill');
  // Not selected, or selected but no key configured yet: neutral, not a
  // healthy "on" state - OpenRouter isn't actually doing anything either way.
  if(compressorType!=='openrouter'||!keySet){pill.className='pill';return;}
  if(orStatus.in_cooldown){pill.className='pill off';return;}
  if(orStatus.fallback_active){pill.className='pill warn';return;}
  pill.className='pill on';
}
async function loadORStatus(){
  try{orStatus=await (await fetch('/api/openrouter-status')).json();
    var s=window.__STATS__||{};
    updateLLMPill(s.compressor_type, s.openrouter_api_key_set);}
  catch(e){}
}
window.onStats=function(s){
  // "This Session" level card (volatile stats.json; resets on proxy restart).
  document.getElementById('s_saved').textContent=fmt(s.tokens_saved);
  document.getElementById('s_sub').textContent=
    fmt(s.requests)+' requests \\u00b7 '+(s.pct||0)+'% of context saved';
  document.getElementById('c_real').textContent=fmt(s.real_input)+' / '+fmt(s.real_output);
  document.getElementById('c_model').textContent=s.active_model||'-';
  document.getElementById('c_keep').textContent=s.keep_messages;
  document.getElementById('c_ctx').textContent=s.context_status||'-';
  document.getElementById('c_ctx').className='big'+(s.context_store_enabled?' accent':' muted');
  document.getElementById('c_ctx_sub').textContent=s.context_store_enabled?
    'sending Context Tree + last exchanges':'using Memory Card compression';
  document.getElementById('llm').textContent=s.llm_status;
  updateLLMPill(s.compressor_type, s.openrouter_api_key_set);
  renderRecent(s.recent||[]);
};
// "All Time" level card - persistent totals from the history DB.
async function loadAllTime(){
  try{
    var a=await (await fetch('/api/stats/alltime')).json();
    document.getElementById('a_saved').textContent=fmt(a.total_est_saved);
    document.getElementById('a_sub').textContent=
      fmt(a.total_requests)+' requests \\u00b7 '+fmt(a.total_real_in)+' in / '+fmt(a.total_real_out)+' out';
  }catch(e){}
}
function projShortName(p){var t=String(p||'').replace(/[\\\\/]+$/,'').split(/[\\\\/]/);return t[t.length-1]||p;}
// "Total Projects" card + the horizontal project-stats strip + filter dropdown.
async function loadProjects(){
  try{
    var d=await (await fetch('/api/stats/projects')).json();
    var list=(d.projects||[]).filter(function(p){return p.project && p.project!=='unknown';});
    var unknown=(d.projects||[]).filter(function(p){return !p.project || p.project==='unknown';});
    var shown=list.concat(unknown);
    document.getElementById('p_count').textContent=fmt(list.length);
    document.getElementById('p_sub').textContent=list.length?
      ('top: '+projShortName(list[0].project)):'tracked in your history';
    var box=document.getElementById('projects'), empty=document.getElementById('projempty');
    if(!shown.length){box.innerHTML='';box.style.display='none';empty.style.display='block';
      document.getElementById('projhint').textContent='';syncFilterOptions([]);return;}
    box.style.display='flex';empty.style.display='none';
    document.getElementById('projhint').textContent='click a card to filter the chart';
    var max=Math.max.apply(null,shown.map(function(p){return p.saved||0}).concat([1]));
    box.innerHTML=shown.map(function(p){
      var pct=Math.round(100*(p.saved||0)/max);
      var active=(p.project===projectFilter)?' active':'';
      return "<div class='projcard"+active+"' data-project='"+esc(p.project)+"'>"+
        "<div class='pname'>"+esc(projShortName(p.project))+"</div>"+
        "<div class='ppath'>"+esc(p.project)+"</div>"+
        "<div class='pnum'>"+fmt(p.saved)+"</div>"+
        "<div class='pmeta'>tokens saved \\u00b7 "+fmt(p.requests)+" requests</div>"+
        "<div class='bar'><span style='width:"+pct+"%'></span></div></div>";
    }).join('');
    box.querySelectorAll('.projcard').forEach(function(c){
      c.onclick=function(){setProjectFilter(c.dataset.project===projectFilter?'':c.dataset.project);};
    });
    syncFilterOptions(shown);
  }catch(e){}
}
function syncFilterOptions(projects){
  var sel=document.getElementById('projfilter');
  var opts="<option value=''>All projects</option>"+projects.map(function(p){
    return "<option value='"+esc(p.project)+"'>"+esc(projShortName(p.project))+"</option>";
  }).join('');
  sel.innerHTML=opts;
  sel.value=projectFilter;  // may be '' if the stored project has no rows yet
}
function setProjectFilter(p){
  projectFilter=p||'';
  try{localStorage.setItem('tsnap_project_filter',projectFilter);}catch(e){}
  document.getElementById('projfilter').value=projectFilter;
  document.querySelectorAll('.projcard').forEach(function(c){
    c.classList.toggle('active',c.dataset.project===projectFilter&&projectFilter!=='');});
  loadChart();
}
function renderRecent(rows){
  var tb=document.getElementById('recent');
  if(!rows.length){tb.innerHTML="<tr><td colspan='7' class='empty'>no requests yet</td></tr>";return;}
  tb.innerHTML=rows.map(function(r){
    var t=new Date((r.ts||0)*1000).toLocaleTimeString();
    var m=esc(r.model)+(r.aggressive?" <span class='muted'>[AGG]</span>":"");
    return "<tr><td>"+t+"</td><td>"+m+"</td><td class='r'>"+fmt(r.before)+"</td>"+
      "<td class='r accent'>"+fmt(r.saved)+"</td><td class='r'>"+fmt(r.real_input)+"</td>"+
      "<td class='r'>"+fmt(r.real_output)+"</td><td class='r'>"+r.status+"</td></tr>";
  }).join('');
}
async function loadChart(){
  try{
    var url='/api/chart?period='+period+(projectFilter?'&project='+encodeURIComponent(projectFilter):'');
    var d=await (await fetch(url)).json();
    document.getElementById('chartempty').style.display=d.has_data?'none':'flex';
    var ctx=document.getElementById('chart');
    if(!window.Chart)return;
    var data={labels:d.labels,datasets:[{label:'Tokens saved',data:d.saved,
        backgroundColor:'rgba(63,185,80,.55)',borderColor:'#3fb950',borderWidth:1,borderRadius:6}]};
    if(chart){chart.data=data;chart.update();}
    else{chart=new Chart(ctx,{type:'bar',data:data,options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#8b949e'}}},
      scales:{x:{ticks:{color:'#8b949e'},grid:{color:'#1c2230'}},
              y:{ticks:{color:'#8b949e'},grid:{color:'#1c2230'},beginAtZero:true}}}});}
  }catch(e){}
}
document.querySelectorAll('.chartbtns button').forEach(function(b){
  b.onclick=function(){document.querySelectorAll('.chartbtns button').forEach(function(x){x.classList.remove('active')});
    b.classList.add('active');period=b.dataset.p;loadChart();};});
document.getElementById('projfilter').onchange=function(){setProjectFilter(this.value);};
async function loadLog(){
  try{
    var d=await (await fetch('/api/log')).json();
    var el=document.getElementById('log');
    var atBottom=el.scrollTop+el.clientHeight>=el.scrollHeight-20;
    el.textContent=(d.lines&&d.lines.length)?d.lines.join('\\n'):'(no log output yet)';
    if(atBottom)el.scrollTop=el.scrollHeight;
  }catch(e){}
}
function setProjMsg(msg, ok){
  var el=document.getElementById('projdirmsg');
  el.textContent=msg;el.style.color=ok?'var(--muted)':'var(--danger)';
}
async function loadProjectDir(){
  try{
    var r=await (await fetch('/get-project-dir')).json();
    if(r.path){document.getElementById('projdir').value=r.path;
      setProjMsg('Claude Code will launch in: '+r.path,true);}
  }catch(e){setProjMsg('Could not read the current folder.',false);}
}
// Validate + persist whatever path is in the box. Returns true when the
// server accepts it (path exists and is a directory); disables Launch if not.
async function applyProjectDir(showToast){
  var path=document.getElementById('projdir').value.trim();
  try{
    var r=await (await fetch('/set-project-dir',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({path:path})})).json();
    document.getElementById('launchbtn').disabled=!r.ok;
    if(r.ok){document.getElementById('projdir').value=r.path;
      setProjMsg('Claude Code will launch in: '+r.path,true);
      if(showToast)toast('Project folder set');}
    else{setProjMsg(r.error||'Invalid folder.',false);}
    return !!r.ok;
  }catch(e){setProjMsg('Could not set the folder.',false);return false;}
}
document.getElementById('setdirbtn').onclick=function(){applyProjectDir(true);};
document.getElementById('browsebtn').onclick=async function(){
  var self=this,old=self.textContent;self.disabled=true;self.textContent='Opening\\u2026';
  setProjMsg('A folder picker should have opened - choose a folder.',true);
  try{
    var r=await (await fetch('/browse-folder',{method:'POST'})).json();
    if(r.error){setProjMsg(r.error,false);document.getElementById('projdir').focus();}
    else if(r.path){document.getElementById('projdir').value=r.path;
      document.getElementById('launchbtn').disabled=false;
      setProjMsg('Claude Code will launch in: '+r.path,true);toast('Folder selected');}
    else{setProjMsg('Folder selection cancelled.',true);}
  }catch(e){setProjMsg('Folder picker unavailable - type or paste a path into the box and click Set.',false);
    document.getElementById('projdir').focus();}
  self.disabled=false;self.textContent=old;
};
document.getElementById('launchbtn').onclick=async function(){
  var self=this;self.disabled=true;
  // Make sure the backend launches in whatever path is shown, validating it.
  var ok=await applyProjectDir(false);
  if(!ok){self.disabled=false;toast('Set a valid project folder first');return;}
  try{var r=await (await fetch('/launch',{method:'POST'})).json();toast(r.message||'Launched');}
  catch(e){toast('Launch failed');}
  setTimeout(function(){self.disabled=false;},2500);
};
var PRIMER_ORDER=['project_name','language','framework','key_dependencies',
  'folder_structure','git_branch','last_commit_summary','modified_files','readme_summary'];
async function loadPrimer(){
  try{
    var p=await (await fetch('/api/primer')).json();
    var pill=document.getElementById('primerpill'), t=document.getElementById('primertext');
    pill.className='pill'+(p.enabled?' on':'');t.textContent=p.enabled?'enabled':'disabled';
    var el=document.getElementById('primercard');
    if(p.card){
      var lines=[];
      PRIMER_ORDER.forEach(function(k){
        var v=p.card[k];
        if(v==null||v===''||(Array.isArray(v)&&!v.length))return;
        if(Array.isArray(v))v=v.join(', ');
        lines.push(k+': '+v);
      });
      el.textContent=lines.length?lines.join('\\n'):'(empty card)';
    }else{
      el.textContent='No card generated yet \\u2014 launch Claude Code from a project folder.';
    }
  }catch(e){}
}
var cortexFocusDirty=false;
document.getElementById('cortexfocus').addEventListener('input',function(){cortexFocusDirty=true;});
async function loadCortex(){
  try{
    var c=await (await fetch('/api/cortex')).json();
    var pill=document.getElementById('cortexpill'), t=document.getElementById('cortextext');
    pill.className='pill'+(c.enabled?' on':'');t.textContent=c.enabled?'enabled':'disabled';
    if(!cortexFocusDirty)document.getElementById('cortexfocus').value=c.focus||'';
    var lines=[];
    if(c.project_name)lines.push('project: '+c.project_name);
    if(c.language)lines.push('language: '+c.language+(c.framework?(' / '+c.framework):''));
    if((c.key_dependencies||[]).length)lines.push('dependencies: '+c.key_dependencies.join(', '));
    if((c.entry_points||[]).length)lines.push('entry_points: '+c.entry_points.join(', '));
    lines.push('focus: '+(c.focus||'(none set)'));
    if((c.decisions||[]).length)lines.push('\\ndecisions:\\n  - '+c.decisions.join('\\n  - '));
    if((c.resolved_issues||[]).length)lines.push('\\nresolved issues:\\n  - '+c.resolved_issues.join('\\n  - '));
    lines.push('\\nsaved sessions (bridges): '+(c.session_count||0));
    document.getElementById('cortexcard').textContent=
      c.project_dir?lines.join('\\n'):'No project selected yet \\u2014 set a project folder and launch Claude Code.';
  }catch(e){}
}
document.getElementById('cortexfocusbtn').onclick=async function(){
  var focus=document.getElementById('cortexfocus').value;
  try{var r=await (await fetch('/api/cortex/focus',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({focus:focus})})).json();
    cortexFocusDirty=false;toast(r.ok?'Focus saved':(r.error||'Could not save focus'));loadCortex();}
  catch(e){toast('Could not save focus');}
};
document.getElementById('cortexrefreshbtn').onclick=async function(){
  var self=this;self.disabled=true;
  try{var r=await (await fetch('/api/cortex/refresh',{method:'POST'})).json();
    toast(r.ok?'DNA refreshed':(r.error||'Refresh failed'));loadCortex();}
  catch(e){toast('Refresh failed');}
  self.disabled=false;
};
loadProjectDir();
loadAllTime();setInterval(loadAllTime,15000);
loadProjects();setInterval(loadProjects,15000);
loadPrimer();setInterval(loadPrimer,10000);
loadCortex();setInterval(loadCortex,12000);
setTimeout(loadChart,300);setInterval(loadChart,15000);
loadLog();setInterval(loadLog,3000);
loadORStatus();setInterval(loadORStatus,10000);
</script>
"""


_PRESET_BUTTONS = """
<div class='presets'>
  <button class='btn' data-preset='simple' data-keep='5'>Simple<br><small class='muted'>keep 5</small></button>
  <button class='btn' data-preset='balanced' data-keep='10'>Balanced<br><small class='muted'>keep 10</small></button>
  <button class='btn' data-preset='complex' data-keep='20'>Complex<br><small class='muted'>keep 20</small></button>
  <button class='btn' data-preset='smart' data-keep='25'>Smart<br><small class='muted'>keep 25, OpenRouter</small></button>
  <button class='btn' data-preset='maximum' data-keep='999'>Maximum<br><small class='muted'>keep all</small></button>
</div>
"""


def _setup_page() -> str:
    cfg = json.dumps(_public_config())
    body = ("""
<div class='panel'>
  <h2 style='font-size:20px'>Welcome to TokenSnap</h2>
  <p class='muted' style='margin-top:-6px'>Let's get you set up in two quick steps.</p>
  <div class='steps' style='margin-top:22px'>

    <div class='step'><div class='n'>1</div><div class='body'>
      <h3>Smarter Memory Cards (optional)</h3>
      <p class='muted' style='margin:0 0 8px'>Tokensnap always uses fast rule-based (regex) summarization for
         older conversation history. For a noticeably better summary, add a free
         <a href='https://openrouter.ai/keys' target='_blank'>OpenRouter</a> API key - no local install, no cost.
         Leave this blank to stick with regex-only.</p>
      <label class='f'>OpenRouter API key</label>
      <input type='password' id='apikey' placeholder='sk-or-...'>
      <label class='f'>Model</label>
      <input type='text' id='model' placeholder='meta-llama/llama-3.1-8b-instruct:free'>
    </div></div>

    <div class='step'><div class='n'>2</div><div class='body'>
      <h3>Choose a compression preset</h3>
      <p class='muted' style='margin:0 0 8px'>How much recent conversation to keep verbatim before older history
         is summarized. More context = safer for complex work; less = more savings.</p>
      __PRESET_BUTTONS__
      <label class='f'>keep_messages <small>(exchanges kept verbatim)</small></label>
      <input type='number' id='keep' min='1' value='10'>
    </div></div>
  </div>

  <div id='finishArea' style='display:flex;gap:12px;margin-top:26px'>
    <button class='btn primary lg' id='finish'>Finish setup &rarr;</button>
    <a class='btn lg' href='/'>Skip for now</a>
  </div>

  <div id='launchArea' style='display:none;margin-top:26px;text-align:center'>
    <p class='muted' style='margin-bottom:14px'>Setup saved. Ready to go.</p>
    <button class='btn primary lg' id='launchClaudeBtn'>&#128640; Launch Claude Code with these settings</button>
    <div id='launchResult' class='sub' style='margin-top:14px'></div>
    <div style='margin-top:14px'><a href='/'>Go to dashboard &rarr;</a></div>
  </div>
</div>
<script>window.__TSNAP_CFG__=@@CFG_JSON@@;</script>
"""
        .replace("__PRESET_BUTTONS__", _PRESET_BUTTONS)
        .replace("@@CFG_JSON@@", cfg)
    )
    return _shell("TokenSnap Setup", "/setup", body, foot_js=_SETUP_JS)


_SETUP_JS = """
<script>
var cfg=window.__TSNAP_CFG__||{};
document.getElementById('keep').value=cfg.keep_messages||10;
document.getElementById('model').value=cfg.openrouter_model||'';
document.querySelectorAll('.presets button').forEach(function(b){
  b.onclick=function(){document.getElementById('keep').value=b.dataset.keep;
    document.querySelectorAll('.presets button').forEach(function(x){x.style.borderColor='';});
    b.style.borderColor='var(--accent)';};});
document.getElementById('finish').onclick=async function(){
  var key=document.getElementById('apikey').value.trim();
  var payload={keep_messages:parseInt(document.getElementById('keep').value,10)||10,
    openrouter_model:document.getElementById('model').value.trim(),
    compressor_type: key ? 'openrouter' : 'regex'};
  if(key)payload.openrouter_api_key=key;
  await fetch('/setup/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  toast('Setup saved');
  document.getElementById('finishArea').style.display='none';
  document.getElementById('launchArea').style.display='block';
};
document.getElementById('launchClaudeBtn').onclick=async function(){
  var self=this, resultEl=document.getElementById('launchResult');
  self.disabled=true;
  try{
    var r=await (await fetch('/launch',{method:'POST'})).json();
    resultEl.textContent=r.message||'Launched';
    resultEl.style.color=r.ok?'var(--accent)':'var(--danger)';
    toast(r.ok?'Launched!':'Launch failed');
  }catch(e){
    resultEl.textContent='Launch failed: '+e;
    resultEl.style.color='var(--danger)';
  }
  self.disabled=false;
};
</script>
"""


def _settings_page() -> str:
    cfg = json.dumps(_public_config())
    body = ("""
<div class='panel' style='max-width:640px'>
  <h2 style='font-size:20px'>Settings</h2>
  <p class='muted' style='margin-top:-6px'>Changes are saved immediately and apply to the next request the proxy handles.</p>

  <label class='f'>Compression preset</label>
  __PRESET_BUTTONS__

  <label class='f'>keep_messages <small>(recent exchanges kept verbatim before summarizing)</small></label>
  <input type='number' id='keep' min='1'>

  <label class='f'>Selective compression <small>(clean noise from every message before truncating - recommended)</small></label>
  <select id='selective'>
    <option value='true'>on - assistant messages untouched, terminal dumps/tool output reduced to signal</option>
    <option value='false'>off - legacy uniform truncation only</option>
  </select>

  <label class='f'>Memory Card generator</label>
  <select id='compressor_type'>
    <option value='regex'>regex - fast, rule-based, fully offline (default)</option>
    <option value='openrouter'>openrouter - a free hosted model writes a better summary</option>
    <option value='off'>off - no Memory Card, no truncation (full history every request)</option>
  </select>

  <label class='f'>OpenRouter API key <small id='keystatus'></small></label>
  <input type='password' id='apikey' placeholder='sk-or-... (leave blank to keep the current key)'>
  <div style='margin-top:8px'><a href='https://openrouter.ai/keys' target='_blank' class='muted'>Get a free key &rarr;</a>
    &nbsp;|&nbsp; <a href='#' id='clearkey' class='muted'>Clear saved key</a></div>

  <label class='f'>OpenRouter model</label>
  <input type='text' id='model' placeholder='meta-llama/llama-3.1-8b-instruct:free'>

  <label class='f'>Fallback models <small>(comma-separated - tried in order if the model above hits a rate limit or is down)</small></label>
  <input type='text' id='fallback' placeholder='e.g. qwen/qwen-2.5-7b-instruct:free, mistralai/mistral-7b-instruct:free'>

  <div class='panel' style='margin-top:18px;padding:14px 18px'>
    <div style='font-size:12.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px'>
      OpenRouter status</div>
    <div id='orstatus' class='sub'>Loading...</div>
  </div>

  <div class='panel' style='margin-top:18px;padding:14px 18px'>
    <div style='font-size:12.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px'>
      Differential Context Engine <span class='muted' style='text-transform:none;letter-spacing:0'>&mdash; experimental</span></div>
    <p class='sub' style='margin-top:0'>Mirrors the whole conversation to a local store and sends the model only the last couple of exchanges plus a compact Context Tree, with a <code>fetch_context</code> tool to pull detail back on demand. Maximizes token reduction, but rewrites the system prompt each turn (which invalidates Anthropic prompt caching) and buffers the reply instead of streaming it &mdash; leave off unless you've measured it for your workload.</p>
    <label class='f'>Context Engine</label>
    <select id='ctxenabled'>
      <option value='false'>off &mdash; use Memory Card compression (default)</option>
      <option value='true'>on &mdash; Differential Context Engine</option>
    </select>
    <label class='f'>Context Tree size <small>(recent important events summarized in the tree)</small></label>
    <input type='number' id='ctxtree' min='1'>
    <div id='ctxstats' class='sub' style='margin-top:10px'>Loading&hellip;</div>
  </div>

  <div class='panel' style='margin-top:18px;padding:14px 18px'>
    <div style='font-size:12.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px'>
      Project Primer</div>
    <p class='sub' style='margin-top:0'>Injects a compact project overview (language, framework, folder structure, git
      branch/last commit, README summary) into the system prompt on the first request of each session, so Claude
      understands your codebase immediately. Generated once per session from the current project folder.</p>
    <label class='f'>Project Primer</label>
    <select id='primerenabled'>
      <option value='true'>on &mdash; inject a project overview at session start (default)</option>
      <option value='false'>off &mdash; no project overview</option>
    </select>
  </div>

  <div class='panel' style='margin-top:18px;padding:14px 18px'>
    <div style='font-size:12.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px'>
      Project Cortex</div>
    <p class='sub' style='margin-top:0'>A persistent per-project knowledge base (<b>.tokensnap/project_dna.json</b>): stack,
      current focus, decisions, and resolved issues, injected as immutable Core Memory each session. Supersedes the Project
      Primer when on. Manage the focus and DNA from the dashboard's Project Cortex panel.</p>
    <label class='f'>Project Cortex</label>
    <select id='cortexenabled'>
      <option value='true'>on &mdash; persistent DNA + Core Memory (default)</option>
      <option value='false'>off &mdash; fall back to the one-shot Project Primer</option>
    </select>
    <label class='f'>Session Bridge auto-inject</label>
    <select id='bridgeinject'>
      <option value='true'>on &mdash; resume from the previous session (default)</option>
      <option value='false'>off &mdash; don't inject the previous session summary</option>
    </select>
  </div>

  <div style='display:flex;gap:12px;margin-top:24px;flex-wrap:wrap'>
    <button class='btn primary lg' id='save'>Save settings</button>
    <button class='btn lg' id='launchClaudeBtn'>&#128640; Launch Claude Code with current settings</button>
  </div>
  <div id='launchResult' class='sub' style='margin-top:12px'></div>
</div>
<script>window.__TSNAP_CFG__=@@CFG_JSON@@;</script>
"""
        .replace("__PRESET_BUTTONS__", _PRESET_BUTTONS)
        .replace("@@CFG_JSON@@", cfg)
    )
    return _shell("TokenSnap Settings", "/settings", body, foot_js=_SETTINGS_JS)


_SETTINGS_JS = """
<script>
var cfg=window.__TSNAP_CFG__||{};
var clearKeyRequested=false;
document.getElementById('keep').value=cfg.keep_messages;
document.getElementById('selective').value=cfg.selective_compression?'true':'false';
document.getElementById('compressor_type').value=cfg.compressor_type||'regex';
document.getElementById('model').value=cfg.openrouter_model||'';
document.getElementById('fallback').value=(cfg.openrouter_fallback_models||[]).join(', ');
document.getElementById('keystatus').textContent=cfg.openrouter_api_key_set?'(a key is set)':'(no key set)';
document.getElementById('ctxenabled').value=cfg.context_store_enabled?'true':'false';
document.getElementById('ctxtree').value=cfg.context_tree_size||20;
document.getElementById('primerenabled').value=(cfg.project_primer_enabled===false)?'false':'true';
document.getElementById('cortexenabled').value=(cfg.project_cortex_enabled===false)?'false':'true';
document.getElementById('bridgeinject').value=(cfg.session_bridge_auto_inject===false)?'false':'true';
async function loadCtxStatus(){
  var el=document.getElementById('ctxstats');
  try{
    var c=await (await fetch('/api/context')).json();
    el.innerHTML='Status: <b>'+(c.enabled?'on':'off')+'</b> &middot; '+
      'events mirrored: <b>'+c.events_stored+'</b> &middot; '+
      'requests via engine: <b>'+c.context_requests+'</b> &middot; '+
      'est. tokens saved: <b>'+c.tokens_saved.toLocaleString()+'</b> &middot; '+
      'context recalls served: <b>'+c.events_fetched+'</b>';
  }catch(e){el.textContent='Unavailable';}
}
loadCtxStatus();setInterval(loadCtxStatus,10000);
document.querySelectorAll('.presets button').forEach(function(b){
  b.onclick=function(){document.getElementById('keep').value=b.dataset.keep;
    document.querySelectorAll('.presets button').forEach(function(x){x.style.borderColor='';});
    b.style.borderColor='var(--accent)';};});
document.getElementById('clearkey').onclick=function(e){
  e.preventDefault();clearKeyRequested=true;document.getElementById('apikey').value='';
  document.getElementById('keystatus').textContent='(will be cleared on save)';
};
async function loadORStatus(){
  var el=document.getElementById('orstatus');
  try{
    var s=await (await fetch('/api/openrouter-status')).json();
    var lines=[];
    lines.push('Primary: <b>'+(s.primary_model||'-')+'</b>');
    lines.push('Fallbacks: '+((s.fallback_models&&s.fallback_models.length)?s.fallback_models.join(', '):'(none)'));
    lines.push('Rate limit remaining: '+(s.rate_limit_remaining==null?'unknown':s.rate_limit_remaining));
    lines.push('Rate limit reset: '+(s.rate_limit_reset==null?'unknown':s.rate_limit_reset));
    if(s.in_cooldown)lines.push("<span style='color:var(--danger)'>In cooldown for "+s.cooldown_seconds_left+"s (all models recently failed)</span>");
    else if(s.fallback_active)lines.push("<span style='color:var(--warn)'>Fallback mode active</span>");
    if(s.recent_errors&&s.recent_errors.length){
      lines.push('Recent errors:');
      s.recent_errors.slice(-3).forEach(function(e){lines.push('&nbsp;&nbsp;'+e.model+': '+e.error);});
    }
    el.innerHTML=lines.join('<br>');
  }catch(e){el.textContent='Unavailable';}
}
loadORStatus();setInterval(loadORStatus,10000);
document.getElementById('save').onclick=async function(){
  var payload={keep_messages:parseInt(document.getElementById('keep').value,10)||10,
    selective_compression:document.getElementById('selective').value,
    compressor_type:document.getElementById('compressor_type').value,
    openrouter_model:document.getElementById('model').value.trim(),
    openrouter_fallback_models:document.getElementById('fallback').value.trim(),
    context_store_enabled:document.getElementById('ctxenabled').value,
    context_tree_size:parseInt(document.getElementById('ctxtree').value,10)||20,
    project_primer_enabled:document.getElementById('primerenabled').value,
    project_cortex_enabled:document.getElementById('cortexenabled').value,
    session_bridge_auto_inject:document.getElementById('bridgeinject').value};
  var key=document.getElementById('apikey').value.trim();
  if(clearKeyRequested)payload.clear_key=true;
  else if(key)payload.openrouter_api_key=key;
  var r=await (await fetch('/settings/save',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json();
  toast(r.ok?'Settings saved':'Save failed');
  clearKeyRequested=false;
  loadCtxStatus();
  document.getElementById('keystatus').textContent=(r.saved&&('openrouter_api_key_set' in r.saved))?
    (r.saved.openrouter_api_key_set?'(a key is set)':'(no key set)'):document.getElementById('keystatus').textContent;
  loadORStatus();
};
document.getElementById('launchClaudeBtn').onclick=async function(){
  var self=this, resultEl=document.getElementById('launchResult');
  self.disabled=true;
  try{
    var r=await (await fetch('/launch',{method:'POST'})).json();
    resultEl.textContent=r.message||'Launched';
    resultEl.style.color=r.ok?'var(--accent)':'var(--danger)';
    toast(r.ok?'Launched!':'Launch failed');
  }catch(e){
    resultEl.textContent='Launch failed: '+e;
    resultEl.style.color='var(--danger)';
  }
  self.disabled=false;
};
</script>
"""
