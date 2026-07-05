# Tokensnap

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Tests](https://github.com/ahmadkassem511/TokenSnap/actions/workflows/tests.yml/badge.svg)](https://github.com/ahmadkassem511/TokenSnap/actions/workflows/tests.yml)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](#quickstart)

**Cut your Claude Code token usage by 40–70% — without changing how you work.**

Tokensnap is an intelligent local HTTP proxy that sits between Claude Code and the
Anthropic API. It cleans the junk out of every request before it leaves your machine:
ANSI color codes, redrawn progress bars, duplicated log lines, and — most importantly —
the ever-growing conversation history that silently eats your usage limits.

## Contents

- [The problem](#the-problem)
- [The solution](#the-solution)
- [Quickstart](#quickstart)
- [Commands](#commands)
- [Configuration](#configuration)
- [How Memory Card compression works](#how-memory-card-compression-works)
- [Architecture](#architecture)
- [Safety & scope](#safety--scope)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## The problem

Every message you send in Claude Code re-sends the *entire* conversation: every
terminal dump, every progress bar frame, every file you attached three prompts ago.
A session that starts at 2k tokens per request can quietly balloon to 150k+.
You hit your usage limit not because you asked hard questions, but because of
context bloat you never see.

## The solution

Tokensnap intercepts each API request and applies four optimizations:

1. **ANSI & progress-bar stripping** — color escape codes, spinner frames, and
   `\r`-redrawn progress bars are deleted from terminal output in the context.
2. **Log deduplication** — runs of identical lines (retry storms, repeated
   warnings) collapse into one line plus a repeat count.
3. **Memory Card compression** — long conversation histories are summarized
   into a compact JSON card (task, files modified, decisions, resolved errors)
   injected as a system note. The last **N** exchanges (default 3) are kept
   verbatim, so Claude never loses the thread of what you're doing *right now*.
4. **Budget guard** — token usage is estimated with tiktoken on every request.
   At 90% of the model's context window, Tokensnap automatically gets more
   aggressive: keeps fewer raw messages, drops file contents that appear twice,
   and trims the system prompt.

Responses come back **completely untouched**, including streaming. Your API key
never touches disk — Tokensnap simply forwards the auth headers Claude Code
already sends.

## Quickstart

### 1. Install

**Windows:** double-click `install.bat`
**Linux / macOS:**

```bash
chmod +x install.sh && ./install.sh
```

Or manually, in any Python ≥3.9 environment:

```bash
pip install -e .
```

### 2. Run

The easy way — one command that starts the proxy (if needed) and launches Claude Code through it:

```bash
tokensnap run claude
```

Or manage it yourself:

```bash
# terminal 1: start the proxy
tokensnap start

# terminal 2: point Claude Code at it, then use Claude Code normally
export ANTHROPIC_BASE_URL=http://127.0.0.1:8889     # bash/zsh
set ANTHROPIC_BASE_URL=http://127.0.0.1:8889        # Windows cmd
$env:ANTHROPIC_BASE_URL="http://127.0.0.1:8889"     # PowerShell
claude
```

### 3. Watch the savings

```bash
tokensnap monitor    # live dashboard in a separate terminal
tokensnap status     # one-shot summary
```

## Commands

| Command | What it does |
| --- | --- |
| `tokensnap start` | Start the proxy in the foreground (Ctrl+C to stop). `--port`, `--host`, `--verbose`. |
| `tokensnap run <cmd>` | Ensure the proxy is running, set `ANTHROPIC_BASE_URL`, and launch `<cmd>` (e.g. `claude`) in the same terminal. |
| `tokensnap stop` | Gracefully stop a proxy that's running in the background (e.g. one started via `tokensnap run`). |
| `tokensnap cleanup` | Stop the proxy (if running) and delete `~/.tokensnap/` (config, stats, logs) for a clean slate. |
| `tokensnap monitor` | Live TUI: total savings, per-request table, proxy status. |
| `tokensnap status` | Is the proxy up? How many tokens saved so far? |
| `tokensnap config show` | Print the effective configuration. |
| `tokensnap config set <key> <value>` | Change a setting (see below). |
| `tokensnap config get <key>` | Read one setting. |

### Stopping and resetting

`tokensnap run claude` (and `tokensnap start`) leave the proxy running in the
background after Claude Code exits, so the next `tokensnap run` is instant.
When you're done for the day:

```bash
tokensnap stop
# Tokensnap proxy stopped. (PID: 12345)

tokensnap stop
# No Tokensnap proxy is running.
```

`stop` looks up the proxy's pid from `~/.tokensnap/stats.json`; if that pid is
gone or stale it falls back to finding whatever process is listening on the
configured port. Either way it updates the stats file so `tokensnap status`
immediately reflects the change.

To wipe every trace of Tokensnap (config, stats, logs) and start fresh:

```bash
tokensnap cleanup            # asks for confirmation, stops the proxy first if running
tokensnap cleanup --force    # skip the confirmation prompt
```

## Configuration

Stored in `~/.tokensnap/config.json`. Everything has a sensible default:

| Key | Default | Meaning |
| --- | --- | --- |
| `host` / `port` | `127.0.0.1` / `8889` | Where the proxy listens. |
| `upstream` | `https://api.anthropic.com` | The real API endpoint. |
| `keep_last_n` | `3` | Exchanges kept verbatim when history is compressed. |
| `aggressive_keep_last_n` | `2` | `keep_last_n` when near the context window. |
| `context_threshold` | `0.9` | Fraction of the context window that triggers aggressive mode. |
| `min_messages_to_compress` | `8` | Histories shorter than this are never compressed. |
| `log_level` | `INFO` | Proxy log verbosity. |
| `key` | *(empty)* | Optional stored API key — normally unnecessary; the proxy forwards the key from request headers. |

Example — keep more raw history:

```bash
tokensnap config set keep_last_n 5
```

## How Memory Card compression works

When a request's `messages` array exceeds `min_messages_to_compress`, Tokensnap:

1. Splits the history: everything except the last `keep_last_n` exchanges.
2. Runs rule-based extraction over the old part: file paths touched, lines
   like `Decision: …` / `we will use …`, and error→resolution pairs.
3. Builds a compact JSON card and appends it to the request's system prompt.
4. Sends only the card + the recent exchanges upstream.

The cut point is chosen carefully so the kept history always starts with a
clean user message — tool_use/tool_result pairs are never split, which would
otherwise cause API errors.

## Architecture

```
Claude Code  --ANTHROPIC_BASE_URL-->  Tokensnap proxy (127.0.0.1:8889)  -->  api.anthropic.com
                                            |
                                            |-- cleaner.py       strip ANSI / progress bars / dup lines
                                            |-- compressor.py    build Memory Card, truncate history
                                            |-- token_counter.py tiktoken-based budget check
                                            '-- stats.py         savings + liveness for status/monitor
```

Only `POST /v1/messages` and `/v1/complete` request *bodies* are touched, and
only on the way out. Every response — including SSE streams — is relayed back
byte-for-byte, so Claude Code behaves exactly as if it were talking to
Anthropic directly.

## Safety & scope

- Only `POST /v1/messages` and `/v1/complete` are optimized. **Every other
  request is forwarded byte-for-byte** (including `count_tokens`, models
  listing, etc.).
- Responses — including SSE streams — are relayed verbatim.
- Nothing is sent anywhere except the configured upstream. No telemetry.
- Token counts use tiktoken's `cl100k_base` encoding, a close approximation
  for Claude models; if tiktoken can't load, a chars/4 estimate is used.

## Development

```bash
pip install -e .[dev]
pytest
```

The test suite (cleaner, compressor, token counter, stats, CLI) runs fully
offline — no network access or real API key required.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for how to set up a dev environment, coding conventions, and the PR checklist.

## License

Apache 2.0 — see [LICENSE](LICENSE).
