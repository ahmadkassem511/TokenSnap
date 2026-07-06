"""The Tokensnap web dashboard - ``tokensnap dashboard``.

A local, single-page control centre for Tokensnap, served on
``http://127.0.0.1:9876``. It's an alternative to the ``tokensnap monitor``
TUI, adding things a terminal can't do well: historical charts, a first-run
setup wizard (hardware detection + Ollama model pull), and a settings page.

It runs as its own aiohttp application (the same web stack the proxy already
uses - no new dependency, no Flask), completely independent of the proxy: the
proxy is a separate background process, so starting, stopping, or closing the
dashboard never touches it or the request logging.

Data sources:
  * live status / totals - ``stats.json`` (written by the proxy),
  * historical charts    - ``history.db`` (written by the proxy),
  * live log lines       - ``~/.tokensnap/proxy.log`` (the proxy's own log).

The server binds loopback only and never handles the Anthropic API key.
"""

import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any, Dict, List

from aiohttp import web

from tokensnap import __version__, history, ollama, stats
from tokensnap import config as config_mod

DEFAULT_PORT = 9876

# A completed setup drops this marker so the dashboard opens on the home page
# instead of the wizard next time.
_SETUP_MARKER = config_mod.CONFIG_DIR / ".setup_complete"

# Model names we'll accept for `ollama pull` - passed to a subprocess as an
# argv element (never a shell), and additionally constrained to the characters
# a real Ollama tag uses, so a crafted value can't smuggle anything through.
_MODEL_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,100}$")

# keep_messages values behind the named presets, shared by the wizard + settings.
PRESETS: Dict[str, int] = {
    "simple": 5,
    "balanced": 10,
    "complex": 20,
    "maximum": 999,
}


# ---------------------------------------------------------------------------
# Hardware detection (stdlib only - mirrors the approach used elsewhere)
# ---------------------------------------------------------------------------
def _ram_gb() -> float:
    """Total physical RAM in GB (0.0 if it can't be determined)."""
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            return mem.ullTotalPhys / (1024 ** 3)
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 * 1024)
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:  # noqa: BLE001 - detection is best-effort
        pass
    return 0.0


def hardware_info() -> Dict[str, Any]:
    """Detect CPU cores, RAM, and free disk on the home drive (stdlib only)."""
    try:
        disk_free_gb = shutil.disk_usage(str(Path.home())).free / (1024 ** 3)
    except OSError:
        disk_free_gb = 0.0
    return {
        "os": "%s %s" % (platform.system(), platform.release()),
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": round(_ram_gb(), 1),
        "disk_free_gb": round(disk_free_gb, 1),
    }


def recommend_model(ram_gb: float) -> str:
    """Recommend a Qwen model sized to the machine's RAM."""
    if ram_gb < 8:
        return "qwen2.5:3b"
    if ram_gb <= 16:
        return "qwen2.5:7b"
    return "qwen2.5:14b"


def _valid_model(model: str) -> bool:
    return bool(_MODEL_RE.match(model or ""))


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
    """Config values the settings/setup pages need (never the API key)."""
    cfg = config_mod.load()
    return {
        "keep_messages": int(cfg["keep_messages"]),
        "ollama_model": cfg["ollama_model"],
        "ollama_url": cfg["ollama_url"],
        "llm_compressor": cfg["llm_compressor"],
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
        "ollama_model": cfg["ollama_model"],
        "recent": recent_rows,
    })


async def api_chart(request: web.Request) -> web.Response:
    """Aggregated saved-token series for the requested period."""
    period = request.query.get("period", "day")
    return web.json_response(history.chart_data(period))


async def api_log(request: web.Request) -> web.Response:
    """The tail of the proxy log, for the live log panel."""
    return web.json_response({"lines": tail_log(200)})


async def setup_page(request: web.Request) -> web.Response:
    return web.Response(text=_setup_page(), content_type="text/html")


async def setup_hardware(request: web.Request) -> web.Response:
    """Detected specs plus a RAM-sized model recommendation."""
    hw = hardware_info()
    hw["recommended_model"] = recommend_model(hw["ram_gb"])
    return web.json_response(hw)


async def setup_pull(request: web.Request) -> web.StreamResponse:
    """Stream ``ollama pull <model>`` output to the browser as SSE."""
    model = request.query.get("model", "").strip()
    resp = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
    await resp.prepare(request)

    async def send(line: str) -> None:
        await resp.write(("data: " + line.replace("\r", "").replace("\n", " ") + "\n\n").encode("utf-8"))

    if not _valid_model(model):
        await send("ERROR: invalid model name.")
        await send("[DONE]")
        await resp.write_eof()
        return resp
    if not shutil.which("ollama"):
        await send("ERROR: 'ollama' is not installed or not on PATH.")
        await send("Install it from https://ollama.com, then retry.")
        await send("[DONE]")
        await resp.write_eof()
        return resp

    await send("Pulling %s ..." % model)
    try:
        proc = await asyncio.create_subprocess_exec(
            "ollama", "pull", model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            text = raw.decode("utf-8", "replace").strip()
            if text:
                await send(text)
        await proc.wait()
        if proc.returncode == 0:
            await send("Done - %s is ready." % model)
        else:
            await send("ERROR: ollama pull exited with code %d." % proc.returncode)
    except Exception as exc:  # noqa: BLE001
        await send("ERROR: %s" % exc)
    await send("[DONE]")
    await resp.write_eof()
    return resp


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

    # A named preset is a shortcut for a keep_messages value.
    preset = str(body.get("preset", "")).lower()
    if preset in PRESETS:
        body.setdefault("keep_messages", PRESETS[preset])

    saved: Dict[str, Any] = {}
    for key in ("keep_messages", "ollama_model", "llm_compressor", "ollama_url"):
        if key not in body or body[key] in (None, ""):
            if key == "ollama_model" and body.get(key) == "":
                pass  # allow clearing the model (switches to regex cards)
            else:
                continue
        try:
            saved[key] = config_mod.set_value(key, str(body[key]))
        except (KeyError, ValueError):
            continue  # invalid value - skip it rather than 500
    # A fresh model may now be available; drop Ollama's cached availability probe.
    ollama.reset_caches()
    return saved


async def launch(request: web.Request) -> web.Response:
    """Open a new terminal running ``tokensnap run claude`` (starts the proxy)."""
    ok, message = _launch_claude_terminal()
    return web.json_response({"ok": ok, "message": message})


def _launch_claude_terminal() -> "tuple[bool, str]":
    """Spawn a new OS terminal that runs ``tokensnap run claude``.

    Uses ``python -m tokensnap`` (always resolvable) so it works whether or not
    the ``tokensnap`` console script is on PATH. Returns (ok, human message).
    """
    py = sys.executable
    inner = '"%s" -m tokensnap run claude' % py
    try:
        if os.name == "nt":
            subprocess.Popen(
                'start "TokenSnap - Claude Code" cmd /k %s' % inner, shell=True
            )
            return True, "Launched Claude Code in a new terminal window."
        if sys.platform == "darwin":
            subprocess.Popen(
                ["osascript", "-e",
                 'tell application "Terminal" to do script "%s"' % inner]
            )
            return True, "Launched Claude Code in a new Terminal window."
        for term in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm"):
            if shutil.which(term):
                subprocess.Popen([term, "-e", "sh", "-c", inner])
                return True, "Launched Claude Code in a new terminal window."
        return False, (
            "Couldn't find a terminal to open. Run this yourself: "
            "tokensnap run claude"
        )
    except Exception as exc:  # noqa: BLE001
        return False, "Couldn't launch a terminal (%s). Run: tokensnap run claude" % exc


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------
def build_app() -> web.Application:
    """Construct the dashboard's aiohttp application (routes only, no I/O)."""
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/stats", api_stats)
    app.router.add_get("/api/chart", api_chart)
    app.router.add_get("/api/log", api_log)
    app.router.add_get("/setup", setup_page)
    app.router.add_get("/setup/hardware", setup_hardware)
    app.router.add_get("/setup/pull", setup_pull)
    app.router.add_post("/setup/save", setup_save)
    app.router.add_get("/settings", settings_page)
    app.router.add_post("/settings/save", settings_save)
    app.router.add_post("/launch", launch)
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


def _shell(title: str, active: str, body: str, extra_head: str = "") -> str:
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
        "<div id='toast' class='toast'></div>" + _COMMON_JS + "</body></html>"
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
  <div class='card'><h3>Estimated tokens saved</h3><div class='big accent' id='c_saved'>-</div>
    <div class='sub'><span id='c_pct'>-</span> of request context</div></div>
  <div class='card'><h3>Requests optimized</h3><div class='big' id='c_reqs'>-</div>
    <div class='sub'>since the proxy last started</div></div>
  <div class='card'><h3>Real usage (Anthropic)</h3><div class='big' id='c_real'>-</div>
    <div class='sub'>in / out tokens</div></div>
  <div class='card'><h3>Active model</h3><div class='big' id='c_model' style='font-size:19px'>-</div>
    <div class='sub'>keep_messages = <span id='c_keep'>-</span></div></div>
</div>

<div class='panel'>
  <div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>
    <h2 style='margin:0'>Tokens saved over time</h2>
    <div class='chartbtns' style='margin:0 0 0 auto'>
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
    <h2 style='margin:0'>Live proxy log</h2>
    <button class='btn primary' id='launchbtn' style='margin-left:auto'>Launch Claude Code</button>
  </div>
  <div class='log' id='log' style='margin-top:12px'>waiting for the proxy...</div>
  <div class='sub muted' style='margin-top:8px'>Memory Cards: <span id='llm'>-</span></div>
</div>
"""
    extra = "<script src='https://cdn.jsdelivr.net/npm/chart.js@4'></script>" + _DASH_JS
    return _shell("TokenSnap Dashboard", "/", body, extra)


_DASH_JS = """
<script>
var chart, period='day';
function esc(s){return String(s).replace(/[&<>]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}
window.onStats=function(s){
  document.getElementById('c_saved').textContent=fmt(s.tokens_saved);
  document.getElementById('c_pct').textContent=(s.pct||0)+'%';
  document.getElementById('c_reqs').textContent=fmt(s.requests);
  document.getElementById('c_real').textContent=fmt(s.real_input)+' / '+fmt(s.real_output);
  document.getElementById('c_model').textContent=s.active_model||'-';
  document.getElementById('c_keep').textContent=s.keep_messages;
  document.getElementById('llm').textContent=s.llm_status;
  renderRecent(s.recent||[]);
};
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
    var d=await (await fetch('/api/chart?period='+period)).json();
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
async function loadLog(){
  try{
    var d=await (await fetch('/api/log')).json();
    var el=document.getElementById('log');
    var atBottom=el.scrollTop+el.clientHeight>=el.scrollHeight-20;
    el.textContent=(d.lines&&d.lines.length)?d.lines.join('\\n'):'(no log output yet)';
    if(atBottom)el.scrollTop=el.scrollHeight;
  }catch(e){}
}
document.getElementById('launchbtn').onclick=async function(){
  var self=this;self.disabled=true;
  try{var r=await (await fetch('/launch',{method:'POST'})).json();toast(r.message||'Launched');}
  catch(e){toast('Launch failed');}
  setTimeout(function(){self.disabled=false;},2500);
};
setTimeout(loadChart,300);setInterval(loadChart,15000);
loadLog();setInterval(loadLog,3000);
</script>
"""


def _setup_page() -> str:
    cfg = json.dumps(_public_config())
    body = """
<div class='panel'>
  <h2 style='font-size:20px'>Welcome to TokenSnap</h2>
  <p class='muted' style='margin-top:-6px'>Let's get you set up in three quick steps. Memory Cards use a
     small local model (via Ollama) to summarize old conversation history - all on your machine.</p>
  <div class='steps' style='margin-top:22px'>

    <div class='step'><div class='n'>1</div><div class='body'>
      <h3>Your machine</h3>
      <div class='hwgrid' id='hwgrid'><div class='hwtile'><div class='k'>Detecting...</div><div class='v'>-</div></div></div>
    </div></div>

    <div class='step'><div class='n'>2</div><div class='body'>
      <h3>Pick a local model</h3>
      <p class='muted' style='margin:0 0 8px'>We recommend a Qwen model sized to your RAM. You can change it.</p>
      <input type='text' id='model' placeholder='qwen2.5:7b'>
      <div style='margin-top:10px'><button class='btn' id='pullbtn'>Pull this model with Ollama</button></div>
      <div class='log' id='pulllog' style='height:150px;margin-top:12px;display:none'></div>
    </div></div>

    <div class='step'><div class='n'>3</div><div class='body'>
      <h3>Choose a compression preset</h3>
      <p class='muted' style='margin:0 0 8px'>How much recent conversation to keep verbatim before older history
         is summarized. More context = safer for complex work; less = more savings.</p>
      <div class='presets'>
        <button class='btn' data-preset='simple' data-keep='5'>Simple<br><small class='muted'>keep 5</small></button>
        <button class='btn' data-preset='balanced' data-keep='10'>Balanced<br><small class='muted'>keep 10</small></button>
        <button class='btn' data-preset='complex' data-keep='20'>Complex<br><small class='muted'>keep 20</small></button>
        <button class='btn' data-preset='maximum' data-keep='999'>Maximum<br><small class='muted'>keep all</small></button>
      </div>
      <label class='f'>keep_messages <small>(exchanges kept verbatim)</small></label>
      <input type='number' id='keep' min='1' value='10'>
    </div></div>
  </div>

  <div style='display:flex;gap:12px;margin-top:26px'>
    <button class='btn primary lg' id='finish'>Finish setup &rarr;</button>
    <a class='btn lg' href='/'>Skip for now</a>
  </div>
</div>
<script>window.__CFG__=__CFG__;</script>
""".replace("__CFG__", cfg)
    return _shell("TokenSnap Setup", "/setup", body, _SETUP_JS)


_SETUP_JS = """
<script>
var cfg=window.__CFG__||{};
document.getElementById('keep').value=cfg.keep_messages||10;
document.getElementById('model').value=cfg.ollama_model||'';
async function detectHW(){
  var d=await (await fetch('/setup/hardware')).json();
  var g=document.getElementById('hwgrid');
  g.innerHTML='';
  [['OS',d.os],['CPU cores',d.cpu_cores],['RAM',d.ram_gb+' GB'],['Free disk',d.disk_free_gb+' GB'],
   ['Recommended',d.recommended_model]].forEach(function(p){
    g.innerHTML+="<div class='hwtile'><div class='k'>"+p[0]+"</div><div class='v'>"+p[1]+"</div></div>";});
  if(!document.getElementById('model').value)document.getElementById('model').value=d.recommended_model;
}
detectHW();
document.querySelectorAll('.presets button').forEach(function(b){
  b.onclick=function(){document.getElementById('keep').value=b.dataset.keep;
    document.querySelectorAll('.presets button').forEach(function(x){x.style.borderColor='';});
    b.style.borderColor='var(--accent)';};});
document.getElementById('pullbtn').onclick=function(){
  var model=document.getElementById('model').value.trim();if(!model){toast('Enter a model name');return;}
  var box=document.getElementById('pulllog');box.style.display='block';box.textContent='';
  var es=new EventSource('/setup/pull?model='+encodeURIComponent(model));
  es.onmessage=function(e){
    if(e.data==='[DONE]'){es.close();return;}
    box.textContent+=e.data+'\\n';box.scrollTop=box.scrollHeight;};
  es.onerror=function(){box.textContent+='\\n(connection closed)';es.close();};
};
document.getElementById('finish').onclick=async function(){
  var payload={ollama_model:document.getElementById('model').value.trim(),
    keep_messages:parseInt(document.getElementById('keep').value,10)||10};
  await fetch('/setup/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  toast('Setup saved - opening dashboard');
  setTimeout(function(){location.href='/';},900);
};
</script>
"""


def _settings_page() -> str:
    cfg = json.dumps(_public_config())
    body = """
<div class='panel' style='max-width:640px'>
  <h2 style='font-size:20px'>Settings</h2>
  <p class='muted' style='margin-top:-6px'>Changes are saved immediately and apply to the next request the proxy handles.</p>

  <label class='f'>Compression preset</label>
  <div class='presets'>
    <button class='btn' data-preset='simple' data-keep='5'>Simple<br><small class='muted'>keep 5</small></button>
    <button class='btn' data-preset='balanced' data-keep='10'>Balanced<br><small class='muted'>keep 10</small></button>
    <button class='btn' data-preset='complex' data-keep='20'>Complex<br><small class='muted'>keep 20</small></button>
    <button class='btn' data-preset='maximum' data-keep='999'>Maximum<br><small class='muted'>keep all</small></button>
  </div>

  <label class='f'>keep_messages <small>(recent exchanges kept verbatim before summarizing)</small></label>
  <input type='number' id='keep' min='1'>

  <label class='f'>Memory Card generator</label>
  <select id='llm'>
    <option value='auto'>auto - use Ollama when available, else regex (recommended)</option>
    <option value='ollama'>ollama - require Ollama, warn if unreachable</option>
    <option value='off'>off - regex-only, never call a model</option>
  </select>

  <label class='f'>Ollama model</label>
  <input type='text' id='model' placeholder='qwen2.5:7b'>

  <label class='f'>Ollama URL</label>
  <input type='text' id='url' placeholder='http://127.0.0.1:11434'>

  <div style='margin-top:24px'><button class='btn primary lg' id='save'>Save settings</button></div>
</div>
<script>window.__CFG__=__CFG__;</script>
""".replace("__CFG__", cfg)
    return _shell("TokenSnap Settings", "/settings", body, _SETTINGS_JS)


_SETTINGS_JS = """
<script>
var cfg=window.__CFG__||{};
document.getElementById('keep').value=cfg.keep_messages;
document.getElementById('model').value=cfg.ollama_model||'';
document.getElementById('url').value=cfg.ollama_url||'';
document.getElementById('llm').value=cfg.llm_compressor||'auto';
document.querySelectorAll('.presets button').forEach(function(b){
  b.onclick=function(){document.getElementById('keep').value=b.dataset.keep;
    document.querySelectorAll('.presets button').forEach(function(x){x.style.borderColor='';});
    b.style.borderColor='var(--accent)';};});
document.getElementById('save').onclick=async function(){
  var payload={keep_messages:parseInt(document.getElementById('keep').value,10)||10,
    llm_compressor:document.getElementById('llm').value,
    ollama_model:document.getElementById('model').value.trim(),
    ollama_url:document.getElementById('url').value.trim()};
  var r=await (await fetch('/settings/save',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json();
  toast(r.ok?'Settings saved':'Save failed');
};
</script>
"""
