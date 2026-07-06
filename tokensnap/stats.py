"""Shared stats file so `tokensnap status` / `tokensnap monitor` can see
what the proxy process is doing. Written atomically (temp file + replace).

Liveness is determined by actually connecting to the proxy port — reliable
and cross-platform (a pid check with os.kill is unsafe on Windows).

`stop_proxy` additionally needs to *find and terminate* the proxy process,
for which the port-connect check alone isn't enough - that's what the
pid-based helpers below are for.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

STATS_DIR = Path.home() / ".tokensnap"
STATS_FILE = STATS_DIR / "stats.json"

_MAX_RECENT = 50

_EMPTY: Dict[str, Any] = {
    "proxy": {},   # {pid, host, port, started_at}
    # One-line Memory Card generator status (e.g. "ollama:llama3.2" or
    # "regex (model 'llama3.2' not pulled)"), refreshed on each proxy start.
    "llm_status": "",
    "totals": {
        # Tokensnap's own tiktoken estimate of the request body (in/out of
        # the optimizer) - drives the "saved" figure.
        "requests": 0,
        "tokens_before": 0,
        "tokens_after": 0,
        "tokens_saved": 0,
        # Real usage as reported by the Anthropic API `usage` field.
        "real_input": 0,
        "real_output": 0,
        "real_cache_read": 0,
        "real_cache_creation": 0,
    },
    "recent": [],
}


def load() -> Dict[str, Any]:
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = json.loads(json.dumps(_EMPTY))  # deep copy of defaults
            merged.update(data)
            # Backfill totals keys added in newer versions of Tokensnap.
            totals = dict(_EMPTY["totals"])
            totals.update(merged.get("totals") or {})
            merged["totals"] = totals
            return merged
        except (json.JSONDecodeError, OSError):
            pass
    return json.loads(json.dumps(_EMPTY))  # deep copy


def _save(data: Dict[str, Any]) -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(STATS_DIR), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, STATS_FILE)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def mark_started(host: str, port: int) -> None:
    """Record proxy startup and reset session counters."""
    data = load()
    data["proxy"] = {
        "pid": os.getpid(),
        "host": host,
        "port": port,
        "started_at": time.time(),
    }
    data["llm_status"] = ""
    data["totals"] = dict(_EMPTY["totals"])
    data["recent"] = []
    _save(data)


def set_llm_status(status: str) -> None:
    """Record the current Memory Card generator status for status/monitor."""
    data = load()
    data["llm_status"] = status
    _save(data)


def record_request(
    path: str,
    model: Optional[str],
    tokens_before: int,
    tokens_after: int,
    status: int,
    elapsed: float,
    aggressive: bool = False,
    real_input: int = 0,
    real_output: int = 0,
    real_cache_read: int = 0,
    real_cache_creation: int = 0,
) -> None:
    data = load()
    saved = max(0, tokens_before - tokens_after)
    totals = data["totals"]
    totals["requests"] += 1
    totals["tokens_before"] += tokens_before
    totals["tokens_after"] += tokens_after
    totals["tokens_saved"] += saved
    totals["real_input"] += real_input
    totals["real_output"] += real_output
    totals["real_cache_read"] += real_cache_read
    totals["real_cache_creation"] += real_cache_creation
    now = time.time()
    data["recent"].append(
        {
            "ts": now,
            "path": path,
            "model": model or "?",
            "before": tokens_before,
            "after": tokens_after,
            "saved": saved,
            "status": status,
            "elapsed": round(elapsed, 2),
            "aggressive": aggressive,
            "real_input": real_input,
            "real_output": real_output,
            "real_cache_read": real_cache_read,
            "real_cache_creation": real_cache_creation,
        }
    )
    data["recent"] = data["recent"][-_MAX_RECENT:]
    _save(data)

    # Persist to the history DB for the web dashboard's charts. Imported lazily
    # and wrapped so a database problem can never break the proxy's hot path -
    # the stats.json file above is the source of truth for live status either way.
    try:
        from tokensnap import history

        history.log_request(
            model=model,
            est_tokens_in=tokens_before,
            real_tokens_in=real_input,
            real_tokens_out=real_output,
            cache_read=real_cache_read,
            cache_write=real_cache_creation,
            saved=saved,
            http_status=status,
            ts=now,
        )
    except Exception:  # noqa: BLE001 - history is strictly best-effort
        pass


def proxy_running(host: Optional[str] = None, port: Optional[int] = None) -> bool:
    """True when something accepts TCP connections on the proxy address."""
    if host is None or port is None:
        info = load().get("proxy") or {}
        host = host or info.get("host") or "127.0.0.1"
        port = port or info.get("port") or 8889
    try:
        with socket.create_connection((host, int(port)), timeout=0.5):
            return True
    except OSError:
        return False


def mark_stopped() -> None:
    """Clear the proxy heartbeat so `status`/`monitor` show it as stopped."""
    data = load()
    data["proxy"] = {}
    _save(data)


def _pid_alive(pid: Optional[int]) -> bool:
    """Check whether `pid` refers to a live process, without side effects.

    Windows note: os.kill(pid, 0) on Windows terminates the target process
    (see CPython's implementation, which maps any non CTRL_* signal to
    TerminateProcess), so it can't be used as a liveness probe there. We
    instead open the process with a query-only access right.
    """
    if not pid:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, just owned by someone else
    except OSError:
        return False
    return True


def _terminate_pid(pid: int) -> bool:
    """Best-effort process termination. Returns True if a kill was issued
    (not a guarantee the process has exited yet - callers should poll)."""
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
    import signal

    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True  # already gone
    except OSError:
        return False


def find_pid_by_port(host: str, port: int) -> Optional[int]:
    """Best-effort lookup of the pid listening on `port`, used as a
    fallback when the stats file has no (or a stale) pid recorded."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            for line in out.splitlines():
                parts = line.split()
                if (
                    len(parts) >= 5
                    and parts[0].upper() == "TCP"
                    and parts[3].upper() == "LISTENING"
                    and parts[1].endswith(":%d" % port)
                ):
                    return int(parts[-1])
        else:
            out = subprocess.run(
                ["lsof", "-t", "-iTCP:%d" % port, "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if out:
                return int(out.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def start_proxy_detached(timeout: float = 10.0) -> Tuple[bool, Path]:
    """Start `tokensnap start` fully detached from this process, so the
    proxy survives the launching terminal (or MCP client) exiting.

    Returns (ok, log_path). `ok` is True once the proxy accepts connections.
    """
    from tokensnap import config as config_mod

    cfg = config_mod.load()
    host, port = cfg["host"], int(cfg["port"])
    log_path = config_mod.CONFIG_DIR / "proxy.log"
    config_mod.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    popen_kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        popen_kwargs["start_new_session"] = True
    subprocess.Popen(
        [sys.executable, "-m", "tokensnap", "start"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        **popen_kwargs,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proxy_running(host, port):
            return True, log_path
        time.sleep(0.25)
    return False, log_path


def stop_proxy(
    host: Optional[str] = None, port: Optional[int] = None, timeout: float = 5.0
) -> Tuple[bool, Optional[int]]:
    """Stop the running proxy if one is found.

    Returns (attempted, pid): `attempted` is False when no proxy appears to
    be running at all (nothing to do); `pid` is the process id that was
    signalled, if one could be identified. Callers should re-check
    `proxy_running()` afterwards to confirm the shutdown actually completed.
    """
    info = load().get("proxy") or {}
    host = host or info.get("host") or "127.0.0.1"
    port = int(port or info.get("port") or 8889)

    pid = info.get("pid")
    if not _pid_alive(pid):
        pid = find_pid_by_port(host, port)

    if not proxy_running(host, port) and not _pid_alive(pid):
        return False, None

    if pid:
        _terminate_pid(pid)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and proxy_running(host, port):
        time.sleep(0.2)

    mark_stopped()
    return True, pid
