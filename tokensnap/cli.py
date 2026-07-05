"""Tokensnap CLI: start, run, monitor, status, config, stop, cleanup."""

import os
import shutil
import subprocess
import sys
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
        log_path = config_mod.CONFIG_DIR / "proxy.log"
        config_mod.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            [sys.executable, "-m", "tokensnap", "start"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        for _ in range(40):  # up to ~10s
            if stats.proxy_running(cfg["host"], int(cfg["port"])):
                break
            time.sleep(0.25)
        else:
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
        "Tokens in: [bold]%s[/bold]" % format(before, ","),
        "Saved: [bold green]%s[/bold green] ([bold]%.1f%%[/bold])" % (format(saved, ","), pct),
    )

    table = Table(
        show_header=True, header_style="bold cyan", expand=True, box=None
    )
    table.add_column("time", width=9)
    table.add_column("model", overflow="fold")
    table.add_column("before", justify="right")
    table.add_column("after", justify="right")
    table.add_column("saved", justify="right")
    table.add_column("%", justify="right")
    table.add_column("http", justify="right")

    for entry in reversed(data["recent"][-15:]):
        b, a, s = entry["before"], entry["after"], entry["saved"]
        row_pct = 100.0 * s / b if b else 0.0
        style = "yellow" if entry.get("aggressive") else ""
        table.add_row(
            datetime.fromtimestamp(entry["ts"]).strftime("%H:%M:%S"),
            str(entry["model"]) + (" [AGG]" if entry.get("aggressive") else ""),
            format(b, ","),
            format(a, ","),
            "[green]%s[/green]" % format(s, ","),
            "%.0f%%" % row_pct,
            str(entry["status"]),
            style=style,
        )
    if not data["recent"]:
        table.add_row("-", "[dim]no requests yet[/dim]", "-", "-", "-", "-", "-")

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

    console.print(
        Panel.fit(
            "Proxy:  %s%s\n"
            "URL:    %s\n"
            "Requests handled: %d\n"
            "Tokens before optimization: %s\n"
            "Tokens after optimization:  %s\n"
            "Tokens saved: [bold green]%s[/bold green] ([bold]%.1f%%[/bold])"
            % (
                "[bold green]RUNNING[/bold green]" if running else "[bold red]STOPPED[/bold red]",
                since,
                _base_url(cfg),
                totals["requests"],
                format(before, ","),
                format(totals["tokens_after"], ","),
                format(saved, ","),
                pct,
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
    """Set a config value, e.g. `tokensnap config set keep_last_n 4`."""
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


@app.command()
def version() -> None:
    """Print the Tokensnap version."""
    console.print("tokensnap %s" % __version__)


if __name__ == "__main__":
    app()
