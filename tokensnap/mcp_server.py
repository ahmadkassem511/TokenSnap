"""Minimal MCP (Model Context Protocol) stdio server for Claude Desktop.

Claude Desktop cannot route its chat traffic through the Tokensnap proxy
(the app has no API-endpoint setting and talks to claude.ai's private
backend, not the Anthropic Messages API). What it *can* do is talk to
local MCP servers - this module exposes Tokensnap's status, savings
stats, config, and proxy start/stop as MCP tools so you can manage and
inspect Tokensnap from inside Claude Desktop.

The protocol is newline-delimited JSON-RPC 2.0 over stdio. It is small
enough that implementing it directly beats pulling in an SDK dependency.
"""

import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from tokensnap import __version__, stats
from tokensnap import config as config_mod

PROTOCOL_VERSION = "2025-06-18"

_SENSITIVE_KEYS = {"key"}


def _tool(name: str, description: str, properties: Dict[str, Any],
          required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


TOOLS: List[Dict[str, Any]] = [
    _tool(
        "tokensnap_status",
        "Whether the Tokensnap proxy is running, plus total token savings "
        "and real Anthropic usage counters.",
        {},
    ),
    _tool(
        "tokensnap_recent_requests",
        "The most recent requests handled by the proxy: model, estimated "
        "tokens before/after optimization, and real usage per request.",
        {"limit": {"type": "integer", "description": "Max entries (default 10)"}},
    ),
    _tool(
        "tokensnap_get_config",
        "The effective Tokensnap configuration (secrets redacted).",
        {},
    ),
    _tool(
        "tokensnap_set_config",
        "Set one Tokensnap config key, e.g. keep_last_n=4. "
        "Takes effect on the next proxy start.",
        {
            "key": {"type": "string", "description": "Config key"},
            "value": {"type": "string", "description": "New value"},
        },
        required=["key", "value"],
    ),
    _tool(
        "tokensnap_start_proxy",
        "Start the Tokensnap proxy in the background if it is not running.",
        {},
    ),
    _tool(
        "tokensnap_stop_proxy",
        "Stop the running Tokensnap proxy.",
        {},
    ),
]


def _status_payload() -> Dict[str, Any]:
    cfg = config_mod.load()
    data = stats.load()
    totals = data["totals"]
    running = stats.proxy_running(cfg["host"], int(cfg["port"]))
    before = totals["tokens_before"]
    saved = totals["tokens_saved"]
    return {
        "proxy_running": running,
        "proxy_url": "http://%s:%s" % (cfg["host"], cfg["port"]),
        "requests_handled": totals["requests"],
        "estimated_tokens_before": before,
        "estimated_tokens_after": totals["tokens_after"],
        "estimated_tokens_saved": saved,
        "estimated_savings_percent": round(100.0 * saved / before, 1) if before else 0.0,
        "real_usage_from_anthropic": {
            "input": totals["real_input"],
            "output": totals["real_output"],
            "cache_read": totals["real_cache_read"],
            "cache_creation": totals["real_cache_creation"],
        },
    }


def _recent_payload(limit: int) -> List[Dict[str, Any]]:
    data = stats.load()
    out = []
    for entry in reversed(data["recent"][-max(1, int(limit)):]):
        out.append(
            {
                "time": datetime.fromtimestamp(entry["ts"]).strftime("%H:%M:%S"),
                "model": entry.get("model", "?"),
                "estimated_tokens_before": entry["before"],
                "estimated_tokens_saved": entry["saved"],
                "real_input": entry.get("real_input", 0),
                "real_output": entry.get("real_output", 0),
                "real_cache_read": entry.get("real_cache_read", 0),
                "http_status": entry["status"],
                "aggressive_mode": entry.get("aggressive", False),
            }
        )
    return out


def _config_payload() -> Dict[str, Any]:
    cfg = config_mod.load()
    return {
        k: ("********" if k in _SENSITIVE_KEYS and cfg[k] else v)
        for k, v in sorted(cfg.items())
    }


def _call_tool(name: str, args: Dict[str, Any]) -> Any:
    """Run one tool. Raises ValueError with a user-facing message on bad input."""
    if name == "tokensnap_status":
        return _status_payload()
    if name == "tokensnap_recent_requests":
        return _recent_payload(args.get("limit", 10))
    if name == "tokensnap_get_config":
        return _config_payload()
    if name == "tokensnap_set_config":
        key = str(args["key"])
        if key in _SENSITIVE_KEYS:
            raise ValueError("Refusing to set %r via MCP; use the CLI." % key)
        try:
            coerced = config_mod.set_value(key, str(args["value"]))
        except KeyError as exc:
            raise ValueError(exc.args[0])
        except ValueError:
            raise ValueError("Invalid value %r for key %r" % (args["value"], key))
        return {"set": key, "value": coerced,
                "note": "Takes effect the next time the proxy starts."}
    if name == "tokensnap_start_proxy":
        cfg = config_mod.load()
        if stats.proxy_running(cfg["host"], int(cfg["port"])):
            return {"started": False, "reason": "already running",
                    **_status_payload()}
        ok, log_path = stats.start_proxy_detached()
        if not ok:
            raise ValueError("Proxy failed to start. See log: %s" % log_path)
        return {"started": True, **_status_payload()}
    if name == "tokensnap_stop_proxy":
        cfg = config_mod.load()
        attempted, pid = stats.stop_proxy(cfg["host"], int(cfg["port"]))
        if not attempted:
            return {"stopped": False, "reason": "no proxy was running"}
        still_up = stats.proxy_running(cfg["host"], int(cfg["port"]))
        return {"stopped": not still_up, "pid": pid}
    raise ValueError("Unknown tool: %s" % name)


def handle_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle one JSON-RPC message; return the response dict, or None for
    notifications (which must not be answered)."""
    method = msg.get("method")
    msg_id = msg.get("id")
    is_notification = msg_id is None

    def result(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": payload}

    def error(code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": code, "message": message}}

    if method == "initialize":
        return result(
            {
                "protocolVersion": msg.get("params", {}).get(
                    "protocolVersion", PROTOCOL_VERSION
                ),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "tokensnap", "version": __version__},
            }
        )
    if method == "ping":
        return result({})
    if method == "tools/list":
        return result({"tools": TOOLS})
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name", "")
        args = params.get("arguments") or {}
        try:
            payload = _call_tool(name, args)
            text = json.dumps(payload, indent=2)
            return result({"content": [{"type": "text", "text": text}],
                           "isError": False})
        except ValueError as exc:
            return result({"content": [{"type": "text", "text": str(exc)}],
                           "isError": True})
        except Exception as exc:  # never crash the server on a tool bug
            return result(
                {"content": [{"type": "text",
                              "text": "Internal error: %r" % exc}],
                 "isError": True}
            )
    if is_notification:
        return None  # e.g. notifications/initialized, notifications/cancelled
    return error(-32601, "Method not found: %s" % method)


def serve() -> None:
    """Blocking stdio loop: one JSON-RPC message per line."""
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    for raw in iter(stdin.readline, b""):
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            response: Optional[Dict[str, Any]] = {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
        else:
            response = handle_message(msg)
        if response is not None:
            stdout.write(json.dumps(response).encode("utf-8") + b"\n")
            stdout.flush()
