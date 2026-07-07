"""Tokensnap CLI: start, run, monitor, status, config, stop, cleanup."""

import os
import shutil
import subprocess
import time
from datetime import datetime
from typing import List, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from tokensnap import __version__, stats
from tokensnap import config as config_mod

app = typer.Typer(
    name="tokensnap",
    help="Token-saving proxy for Claude Code. Cuts context bloat 40-70%.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Read/write ~/.tokensnap/config.json", no_args_is_help=True)
app.add_typer(config_app, name="config")

console = Console()


def _base_url(cfg: dict) -> str:
    return "http://%s:%s" % (cfg["host"], cfg["port"])


@app.command()
def start(
    port: Optional[int] = typer.Option(None, help="Port to listen on (default from config: 8889)"),
    host: Optional[str] = typer.Option(None, help="Host to bind (default 127.0.0.1)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Start the proxy in the foreground (Ctrl+C to stop)."""
    from tokensnap.proxy import run_proxy

    cfg = config_mod.load()
    if port:
        cfg["port"] = port
    if host:
        cfg["host"] = host

    console.print(
        Panel.fit(
            "[bold]Tokensnap v%s[/bold]\n\n"
            "Proxy:    [cyan]%s[/cyan]\n"
            "Upstream: %s\n\n"
            "Point Claude Code at the proxy:\n"
            "  [yellow]set ANTHROPIC_BASE_URL=%s[/yellow]   (Windows cmd)\n"
            "  [yellow]$env:ANTHROPIC_BASE_URL=\"%s\"[/yellow]   (PowerShell)\n"
            "  [yellow]export ANTHROPIC_BASE_URL=%s[/yellow]   (bash/zsh)\n\n"
            "...or just run: [green]tokensnap run claude[/green]"
            % (
                __version__,
                _base_url(cfg),
                cfg["upstream"],
                _base_url(cfg),
                _base_url(cfg),
                _base_url(cfg),
            ),
            title="tokensnap start",
            border_style="green",
        )
    )
    run_proxy(host=host, port=port, verbose=verbose)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def run(
    command: List[str] = typer.Argument(
        ..., help="Command to launch through the proxy, e.g. `tokensnap run claude`"
    ),
) -> None:
    """Launch a command (e.g. claude) with ANTHROPIC_BASE_URL pointed at the proxy.

    Starts the proxy in the background first if it isn't already running.
    """
    cfg = config_mod.load()
    base_url = _base_url(cfg)

    if not stats.proxy_running(cfg["host"], int(cfg["port"])):
        console.print("[dim]Proxy not running - starting it in the background...[/dim]")
        ok, log_path = stats.start_proxy_detached()
        if not ok:
            console.print(
                "[red]Proxy failed to start.[/red] See log: %s" % log_path
            )
            raise typer.Exit(code=1)
        console.print("[green]Proxy up[/green] on %s (log: %s)" % (base_url, log_path))

    env = dict(os.environ)
    env["ANTHROPIC_BASE_URL"] = base_url
    console.print(
        "[dim]ANTHROPIC_BASE_URL=%s -> launching:[/dim] [bold]%s[/bold]"
        % (base_url, " ".join(command))
    )
    # shell=True on Windows so .cmd/.bat shims (like `claude`) resolve
    if os.name == "nt":
        code = subprocess.call(subprocess.list2cmdline(command), env=env, shell=True)
    else:
        code = subprocess.call(command, env=env)
    raise typer.Exit(code=code)


def _build_dashboard(cfg: dict) -> Panel:
    data = stats.load()
    totals = data["totals"]
    running = stats.proxy_running(cfg["host"], int(cfg["port"]))

    before = totals["tokens_before"]
    saved = totals["tokens_saved"]
    pct = 100.0 * saved / before if before else 0.0

    header = Table.grid(padding=(0, 3))
    header.add_row(
        "[bold green]● RUNNING[/bold green]" if running else "[bold red]● STOPPED[/bold red]",
        "Requests: [bold]%d[/bold]" % totals["requests"],
        "Est. saved: [bold green]%s[/bold green] ([bold]%.1f%%[/bold])"
        % (format(saved, ","), pct),
    )
    header.add_row(
        "[dim]real usage (from Anthropic):[/dim]",
        "in [bold]%s[/bold]  out [bold]%s[/bold]"
        % (format(totals["real_input"], ","), format(totals["real_output"], ",")),
        "cache read [bold]%s[/bold]  write [bold]%s[/bold]"
        % (
            format(totals["real_cache_read"], ","),
            format(totals["real_cache_creation"], ","),
        ),
    )
    header.add_row(
        "keep_messages: [bold]%d[/bold]" % int(cfg["keep_messages"]),
        "Memory Cards: %s" % (data.get("llm_status") or "not started yet"),
        "",
    )

    table = Table(
        show_header=True, header_style="bold cyan", expand=True, box=None
    )
    table.add_column("time", width=9)
    table.add_column("model", overflow="fold")
    table.add_column("est.in", justify="right")
    table.add_column("saved", justify="right")
    table.add_column("real.in", justify="right")
    table.add_column("real.out", justify="right")
    table.add_column("cache", justify="right")
    table.add_column("http", justify="right")

    for entry in reversed(data["recent"][-15:]):
        b, s = entry["before"], entry["saved"]
        style = "yellow" if entry.get("aggressive") else ""
        table.add_row(
            datetime.fromtimestamp(entry["ts"]).strftime("%H:%M:%S"),
            str(entry["model"]) + (" [AGG]" if entry.get("aggressive") else ""),
            format(b, ","),
            "[green]%s[/green]" % format(s, ","),
            format(entry.get("real_input", 0), ","),
            format(entry.get("real_output", 0), ","),
            format(entry.get("real_cache_read", 0), ","),
            str(entry["status"]),
            style=style,
        )
    if not data["recent"]:
        table.add_row("-", "[dim]no requests yet[/dim]", "-", "-", "-", "-", "-", "-")

    grid = Table.grid()
    grid.add_row(header)
    grid.add_row("")
    grid.add_row(table)
    return Panel(
        grid,
        title="Tokensnap monitor - %s" % _base_url(cfg),
        subtitle="Ctrl+C to exit",
        border_style="cyan",
    )


@app.command()
def dashboard(
    port: int = typer.Option(9876, help="Port for the web dashboard (default 9876)"),
    host: str = typer.Option("127.0.0.1", help="Host to bind (default 127.0.0.1)"),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Don't open a browser automatically"
    ),
) -> None:
    """Launch the web dashboard: live stats, charts, setup wizard, and settings.

    Runs independently of the proxy (which is a separate background process),
    so opening or closing the dashboard never affects request handling. This is
    a richer alternative to the `tokensnap monitor` TUI; both can run at once.
    """
    from tokensnap import webui

    url = "http://%s:%d" % ("127.0.0.1" if host in ("0.0.0.0", "::") else host, port)
    console.print(
        Panel.fit(
            "[bold]Tokensnap dashboard[/bold]\n\n"
            "Open in your browser:\n  [cyan]%s[/cyan]\n\n"
            "Live stats, savings charts, the first-run setup wizard, and settings.\n"
            "[dim]Ctrl+C to stop the dashboard (the proxy keeps running).[/dim]"
            % url,
            title="tokensnap dashboard",
            border_style="green",
        )
    )
    try:
        webui.serve(host=host, port=port, open_browser=not no_browser)
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        console.print(
            "[red]Couldn't start the dashboard on %s[/red] (%s). "
            "Is it already running, or the port in use?" % (url, exc)
        )
        raise typer.Exit(code=1)


@app.command()
def monitor() -> None:
    """Live TUI dashboard: savings, budget, and recent requests."""
    cfg = config_mod.load()
    try:
        with Live(_build_dashboard(cfg), console=console, refresh_per_second=2) as live:
            while True:
                time.sleep(0.5)
                live.update(_build_dashboard(cfg))
    except KeyboardInterrupt:
        pass


@app.command()
def status() -> None:
    """Show whether the proxy is running and total savings."""
    cfg = config_mod.load()
    data = stats.load()
    totals = data["totals"]
    running = stats.proxy_running(cfg["host"], int(cfg["port"]))

    before = totals["tokens_before"]
    saved = totals["tokens_saved"]
    pct = 100.0 * saved / before if before else 0.0

    proxy_info = data.get("proxy") or {}
    since = ""
    if running and proxy_info.get("started_at"):
        since = datetime.fromtimestamp(proxy_info["started_at"]).strftime(
            " (since %Y-%m-%d %H:%M:%S)"
        )
    llm_status = data.get("llm_status") or "not started yet"

    console.print(
        Panel.fit(
            "Proxy:  %s%s\n"
            "URL:    %s\n"
            "Requests handled: %d\n"
            "Compression: keep_messages=%d (last N exchanges kept verbatim)\n"
            "Memory Cards: %s\n"
            "\n[bold]Tokensnap optimization (request body, estimated):[/bold]\n"
            "  before: %s   after: %s\n"
            "  saved:  [bold green]%s[/bold green] ([bold]%.1f%%[/bold])\n"
            "\n[bold]Real usage reported by Anthropic:[/bold]\n"
            "  input: %s   output: %s\n"
            "  cache read: %s   cache write: %s"
            % (
                "[bold green]RUNNING[/bold green]" if running else "[bold red]STOPPED[/bold red]",
                since,
                _base_url(cfg),
                totals["requests"],
                int(cfg["keep_messages"]),
                llm_status,
                format(before, ","),
                format(totals["tokens_after"], ","),
                format(saved, ","),
                pct,
                format(totals["real_input"], ","),
                format(totals["real_output"], ","),
                format(totals["real_cache_read"], ","),
                format(totals["real_cache_creation"], ","),
            ),
            title="tokensnap status",
            border_style="green" if running else "red",
        )
    )


@app.command()
def stop() -> None:
    """Gracefully stop the running proxy, if any."""
    cfg = config_mod.load()
    host, port = cfg["host"], int(cfg["port"])

    attempted, pid = stats.stop_proxy(host, port)
    if not attempted:
        console.print("[yellow]No Tokensnap proxy is running.[/yellow]")
        return

    if stats.proxy_running(host, port):
        console.print(
            "[red]Sent a stop signal but the proxy is still responding on %s.[/red] "
            "You may need to close it manually%s."
            % (_base_url(cfg), " (PID: %d)" % pid if pid else "")
        )
        raise typer.Exit(code=1)

    console.print(
        "[green]Tokensnap proxy stopped.[/green]%s"
        % (" (PID: %d)" % pid if pid else "")
    )


@app.command()
def cleanup(
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip the confirmation prompt"
    ),
) -> None:
    """Stop the proxy (if running) and delete ~/.tokensnap/ entirely."""
    cfg = config_mod.load()
    if stats.proxy_running(cfg["host"], int(cfg["port"])):
        console.print("[dim]Proxy is running - stopping it first...[/dim]")
        stats.stop_proxy(cfg["host"], int(cfg["port"]))

    if not config_mod.CONFIG_DIR.exists():
        console.print(
            "[dim]Nothing to clean up - %s does not exist.[/dim]" % config_mod.CONFIG_DIR
        )
        return

    if not force:
        confirmed = typer.confirm(
            "This will permanently delete %s (config, stats, logs). Continue?"
            % config_mod.CONFIG_DIR,
            default=False,
        )
        if not confirmed:
            console.print("Aborted.")
            raise typer.Exit(code=1)

    shutil.rmtree(config_mod.CONFIG_DIR, ignore_errors=True)
    console.print("[green]Removed %s[/green]" % config_mod.CONFIG_DIR)


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set a config value, e.g. `tokensnap config set keep_messages 15`."""
    key = config_mod.resolve_key(key)
    try:
        coerced = config_mod.set_value(key, value)
    except KeyError as exc:
        console.print("[red]%s[/red]" % exc.args[0])
        raise typer.Exit(code=1)
    except ValueError:
        console.print("[red]Invalid value %r for key %r[/red]" % (value, key))
        raise typer.Exit(code=1)
    shown = "********" if key == "key" and coerced else coerced
    console.print("[green]Set[/green] %s = %s" % (key, shown))


@config_app.command("get")
def config_get(key: str) -> None:
    """Print one config value."""
    key = config_mod.resolve_key(key)
    cfg = config_mod.load()
    if key not in cfg:
        console.print("[red]Unknown key %r[/red]" % key)
        raise typer.Exit(code=1)
    value = "********" if key == "key" and cfg[key] else cfg[key]
    console.print("%s = %s" % (key, value))


@config_app.command("show")
def config_show() -> None:
    """Print the full effective configuration."""
    cfg = config_mod.load()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("key")
    table.add_column("value")
    for k in sorted(cfg):
        v = "********" if k == "key" and cfg[k] else str(cfg[k])
        table.add_row(k, v)
    console.print(table)
    console.print("[dim]file: %s[/dim]" % config_mod.CONFIG_FILE)


_PRESETS = {
    "simple": {"keep_messages": 5, "selective_compression": True, "compressor_type": "regex"},
    "balanced": {"keep_messages": 10, "selective_compression": True, "compressor_type": "regex"},
    "complex": {"keep_messages": 20, "selective_compression": False, "compressor_type": "regex"},
    "smart": {"keep_messages": 25, "selective_compression": True, "compressor_type": "openrouter"},
    "maximum": {"keep_messages": 999, "selective_compression": True, "compressor_type": "off"},
}
_PRESET_HELP = {
    "simple": "quick scripts, single-file tasks",
    "balanced": "the default - suitable for most projects",
    "complex": "large multi-file projects - uniform truncation for maximal safety",
    "smart": "best quality: selective compression + an OpenRouter model writes the Memory Card",
    "maximum": "effectively disables compression (noise cleaning only)",
}


@app.command()
def preset(
    name: str = typer.Argument(
        ..., help="simple | balanced | complex | smart | maximum"
    ),
) -> None:
    """Apply a recommended configuration for your project type.

    More context (higher keep_messages) means Claude keeps more of the
    real conversation and fewer tokens are saved; less context means more
    savings but a higher risk Claude loses track of complex work. `smart`
    additionally turns on selective per-message compression and asks a free
    OpenRouter model to write the Memory Card for older history.
    """
    name = name.lower()
    if name not in _PRESETS:
        console.print(
            "[red]Unknown preset %r.[/red] Choose from: %s"
            % (name, ", ".join(_PRESETS))
        )
        raise typer.Exit(code=1)
    for key, value in _PRESETS[name].items():
        config_mod.set_value(key, str(value))
    console.print(
        "[green]Applied preset[/green] %r (%s): keep_messages=%d"
        % (name, _PRESET_HELP[name], _PRESETS[name]["keep_messages"])
    )
    if name == "smart" and not config_mod.load().get("openrouter_api_key"):
        console.print(
            "[yellow]Note:[/yellow] no OpenRouter API key is set yet, so Memory "
            "Cards will fall back to regex until you add one. Get a free key at "
            "[cyan]https://openrouter.ai/keys[/cyan], then:\n"
            "  tokensnap config set openrouter_api_key <key>"
        )


@app.command()
def mcp() -> None:
    """Run the Tokensnap MCP server on stdio (for Claude Desktop/Code)."""
    from tokensnap import mcp_server

    mcp_server.serve()


@app.command()
def version() -> None:
    """Print the Tokensnap version."""
    console.print("tokensnap %s" % __version__)


if __name__ == "__main__":
    app()
