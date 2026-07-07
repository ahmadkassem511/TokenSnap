#                                                 Tokensnap

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
- [What works with Tokensnap (and what doesn't)](#what-works-with-tokensnap-and-what-doesnt)
- [Commands](#commands)
- [Configuration](#configuration)
- [Tuning for your project type](#tuning-for-your-project-type)
- [How selective compression works](#how-selective-compression-works)
- [Smarter Memory Cards with OpenRouter](#smarter-memory-cards-with-openrouter)
- [Architecture](#architecture)
- [Estimated vs. real tokens](#estimated-vs-real-tokens)
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

Blunt compression makes this worse in a different way: summarize *everything*
uniformly and Claude starts losing the thread on complex work — forgetting a
decision from ten messages ago, or a file it already fixed. Tokensnap's
philosophy is to cut only the noise and leave the substance alone.

## The solution

Tokensnap intercepts each API request and applies these optimizations:

1. **Selective per-message compression** (on by default) — assistant messages
   (Claude's own reasoning) are **never touched**. User messages are left
   intact unless they contain a large terminal/log dump, in which case only
   that dump shrinks to its error/warning/status lines — any surrounding
   prose survives untouched. Tool results are reduced the same way, more
   aggressively, since they're almost always machine noise once the outcome
   is known. See [How selective compression works](#how-selective-compression-works).
2. **ANSI & progress-bar stripping** — color escape codes, spinner frames, and
   `\r`-redrawn progress bars are deleted from terminal output in the context.
3. **Log deduplication** — runs of identical lines (retry storms, repeated
   warnings) collapse into one line plus a repeat count.
4. **Memory Card compression** — history older than the last **N** exchanges
   (`keep_messages`, default 10) is summarized into a compact JSON card
   (task, files touched, decisions, resolved errors) injected as a system
   note, so Claude never loses the thread of what you're doing *right now*.
   Tune `keep_messages` per project with `tokensnap preset` — see
   [Tuning for your project type](#tuning-for-your-project-type). A free
   [OpenRouter](https://openrouter.ai) model can write a noticeably better
   card than the built-in regex extraction; see
   [Smarter Memory Cards with OpenRouter](#smarter-memory-cards-with-openrouter).
5. **Budget guard** — token usage is estimated with tiktoken on every request.
   At 95% of the model's context window, Tokensnap automatically gets more
   aggressive: keeps fewer raw messages, drops file contents that appear twice,
   and trims the system prompt.

Responses come back **completely untouched**, including streaming. Your
Anthropic API key never touches disk — Tokensnap simply forwards the auth
headers Claude Code already sends. If you enable OpenRouter for smarter
Memory Cards, that's a *separate* key you provide, used only for the
summarization call and never mixed with your Anthropic key.

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
tokensnap dashboard  # web UI (http://127.0.0.1:9876): charts, history & settings
tokensnap monitor    # live TUI dashboard in a separate terminal
tokensnap status     # one-shot summary
```

The **web dashboard** (`tokensnap dashboard`) is the richest view: a first-run
setup wizard (compression preset + optional free OpenRouter key), historical
savings charts (7 days / 8 weeks / 6 months), a live log, and a settings page.
It runs independently of the proxy, so opening or closing it never interrupts
request handling — you can leave the proxy running and open the dashboard
whenever you like.

Both views show two sets of numbers:

- **Est. saved** — Tokensnap's own tiktoken estimate of the request body
  before vs. after optimization. This is the token bloat Tokensnap removed.
- **Real usage (from Anthropic)** — the actual `input`, `output`,
  `cache read`, and `cache write` tokens parsed straight from the API
  responses. These match the numbers Claude Code reports, so you can see
  true consumption alongside the savings.

> **Tip:** to confirm Claude Code is actually routed through the proxy,
> keep `tokensnap monitor` open in one terminal and send a prompt in Claude
> Code — a new request row should appear within a second or two. If nothing
> shows up, Claude Code isn't using the proxy (see
> [Estimated vs. real tokens](#estimated-vs-real-tokens)).

## What works with Tokensnap (and what doesn't)

| Client | Supported? | How |
| --- | --- | --- |
| Claude Code (CLI) | ✅ | `tokensnap run claude`, or set `ANTHROPIC_BASE_URL` |
| Claude Code in VS Code / JetBrains | ✅ | Set `ANTHROPIC_BASE_URL` as a persistent user env var (see below) |
| Any Anthropic SDK/API app (Aider, Cline, custom scripts) | ✅ | Point its Anthropic base URL at `http://127.0.0.1:8889` |
| **Claude Desktop / claude.ai (the chat app)** | ❌ | Not possible — see below |

**Why the Claude Desktop chat app can't benefit:** the desktop app doesn't
resend your conversation history from your machine — chats live on Anthropic's
servers and the app only transmits your new message, so there is no
client-side context bloat for a proxy to strip. It also uses claude.ai's
private protocol (not the public `/v1/messages` API) and ignores
`ANTHROPIC_BASE_URL` entirely. For the chat app, the practical levers are:
start new chats instead of continuing very long ones, avoid re-pasting large
documents (use Projects), and move heavy coding work to Claude Code routed
through Tokensnap.

**What Claude Desktop *can* do:** talk to local MCP servers. `tokensnap mcp`
runs Tokensnap as an MCP stdio server exposing `tokensnap_status`,
`tokensnap_recent_requests`, `tokensnap_get_config`, `tokensnap_set_config`,
`tokensnap_start_proxy`, and `tokensnap_stop_proxy` as tools — so you can ask
Claude Desktop things like "how much have I saved with Tokensnap today?" or
have it start the proxy for you. Add it to Claude Desktop's MCP config
(`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "tokensnap": {
      "command": "tokensnap",
      "args": ["mcp"]
    }
  }
}
```

This manages/inspects Tokensnap from the chat app; it does not route the
chat app's own conversation through the proxy (see above for why that's not
possible).

**Routing every Claude Code session automatically** (instead of using
`tokensnap run` each time) — set the variable persistently:

```powershell
# PowerShell (Windows)
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_BASE_URL", "http://127.0.0.1:8889", "User")
```

```bash
# bash/zsh (Linux/macOS) - add to ~/.bashrc or ~/.zshrc
export ANTHROPIC_BASE_URL=http://127.0.0.1:8889
```

> ⚠️ With the variable set persistently, Claude Code **requires** the proxy to
> be running — if it's down you'll get connection errors. Keep
> `tokensnap start` running (or add it to your startup apps), or skip this and
> stick with `tokensnap run claude`. To undo on Windows:
> `[System.Environment]::SetEnvironmentVariable("ANTHROPIC_BASE_URL", $null, "User")`

## Commands

| Command | What it does |
| --- | --- |
| `tokensnap start` | Start the proxy in the foreground (Ctrl+C to stop). `--port`, `--host`, `--verbose`. |
| `tokensnap run <cmd>` | Ensure the proxy is running, set `ANTHROPIC_BASE_URL`, and launch `<cmd>` (e.g. `claude`) in the same terminal. |
| `tokensnap stop` | Gracefully stop a proxy that's running in the background (e.g. one started via `tokensnap run`). |
| `tokensnap cleanup` | Stop the proxy (if running) and delete `~/.tokensnap/` (config, stats, logs) for a clean slate. |
| `tokensnap dashboard` | Web UI at `http://127.0.0.1:9876`: setup wizard, savings charts, live log, and settings. Runs independently of the proxy. `--port`, `--host`, `--no-browser`. |
| `tokensnap monitor` | Live TUI: estimated savings **and real Anthropic usage** (input/output/cache), per-request table, proxy status. |
| `tokensnap status` | Is the proxy up? Shows estimated savings and real token usage so far. |
| `tokensnap config show` | Print the effective configuration. |
| `tokensnap config set <key> <value>` | Change a setting (see below). |
| `tokensnap config get <key>` | Read one setting. |
| `tokensnap preset <name>` | Apply a recommended configuration for your project type: `simple`, `balanced`, `complex`, `smart`, `maximum` — see [Tuning for your project type](#tuning-for-your-project-type). |
| `tokensnap mcp` | Run Tokensnap as an MCP stdio server (status/config/start/stop as tools) — see below. |

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
| `keep_messages` | `10` | Exchanges kept verbatim when history is compressed. Higher = more context, fewer tokens saved. See [Tuning for your project type](#tuning-for-your-project-type). (Pre-0.3 name: `keep_last_n`, still accepted.) |
| `aggressive_keep_last_n` | `2` | `keep_messages` drops to this when near the context window. |
| `context_threshold` | `0.95` | Fraction of the context window that triggers aggressive mode. |
| `min_messages_to_compress` | `8` | Histories shorter than this are never compressed. |
| `selective_compression` | `true` | Clean noise from every message (assistant untouched, terminal dumps/tool output reduced to signal) before truncating. `false` uses legacy uniform truncation only. |
| `compressor_type` | `regex` | Memory Card generator: `regex` (fast, rule-based, offline), `openrouter` (a free hosted model writes a better summary), `off` (no Memory Card at all - full history kept, only noise cleaning applies). |
| `openrouter_api_key` | *(empty)* | Your [OpenRouter](https://openrouter.ai/keys) key (free tier available). Required for `compressor_type=openrouter`. |
| `openrouter_model` | `meta-llama/llama-3.1-8b-instruct:free` | Model used to write Memory Cards when `compressor_type=openrouter`. |
| `log_level` | `INFO` | Proxy log verbosity. |
| `key` | *(empty)* | Optional stored Anthropic API key — normally unnecessary; the proxy forwards the key from request headers. |

Example — keep more raw history:

```bash
tokensnap config set keep_messages 15
```

## Tuning for your project type

`keep_messages` is a trade-off dial, not a "more is always better" setting:

- **Higher `keep_messages`** → Claude keeps more of the actual recent
  conversation in full, so it's less likely to lose track of what it's
  doing on a complex, multi-file task — at the cost of fewer tokens saved,
  since more raw history is sent every request.
- **Lower `keep_messages`** → more aggressive compression, bigger token
  savings — but more of the conversation is replaced by the summarized
  Memory Card, so subtle details (exact variable names, code not captured
  by the regex/LLM summary, etc.) are more likely to get lost on long,
  intricate tasks.

The right value depends on what you're working on, so pick a preset instead
of guessing:

```bash
tokensnap preset simple     # keep_messages=5,   selective on,  regex   - quick scripts, single-file tasks
tokensnap preset balanced   # keep_messages=10,  selective on,  regex   - the default, suitable for most projects
tokensnap preset complex    # keep_messages=20,  selective off, regex  - large multi-file projects, maximal safety
tokensnap preset smart      # keep_messages=25,  selective on,  openrouter - best quality (needs a free API key)
tokensnap preset maximum    # keep_messages=999, selective on,  off     - effectively disables compression
```

If Claude starts "forgetting" earlier decisions or files it already touched
on a big project, that's a sign `keep_messages` is too low for that
project — run `tokensnap preset complex` or `tokensnap preset smart` before
starting your next session. You can always fine-tune further with
`tokensnap config set keep_messages <N>`.

## How selective compression works

Selective compression (`selective_compression = true`, the default) runs
before the history is ever truncated, and treats every message differently
based on its role:

- **Assistant messages are never touched.** Claude's own reasoning and
  responses are the entire point of "reasoning quality" - they're passed
  through byte-for-byte, at every position in the conversation, not just
  the recent tail.
- **User messages** are left alone unless they contain a large terminal/log
  dump (heuristically: over ~500 tokens and shaped like shell output). When
  they do, only that dump shrinks — to its error/warning lines plus a final
  status line — while any surrounding prose ("can you check this error?
  ... what should I do?") survives untouched around it.
- **Tool results** are compressed the same way, more aggressively, since
  they're almost always machine-generated noise once the outcome is known.
  A 200-line build log becomes a couple of error lines plus something like
  `[tokensnap: 180 lines omitted (2 errors, 1 warning)]`.

Set `selective_compression = false` (or use `tokensnap preset complex`) to
go back to the legacy behavior: no per-message cleaning, just uniform
truncation at `keep_messages`.

Separately, once history exceeds `min_messages_to_compress`, Tokensnap:

1. Splits the history: everything except the last `keep_messages` exchanges.
2. Summarizes the older part into a Memory Card — file paths touched, lines
   like `Decision: …` / `we will use …`, and error→resolution pairs (regex),
   or a free hosted model's summary (see below).
3. Injects the card into the request's system prompt and sends only the
   card + the recent exchanges upstream.

The cut point is chosen carefully so the kept history always starts with a
clean user message — tool_use/tool_result pairs are never split, which would
otherwise cause API errors. Setting `compressor_type = off` skips this
truncation step entirely: the full (but noise-cleaned) history is sent every
request.

## Smarter Memory Cards with OpenRouter

Regex extraction is fast and dependency-free, but it can only recognize
patterns it was taught. [OpenRouter](https://openrouter.ai) gives free access
to several hosted models (e.g. Meta's Llama 3.1 8B); point Tokensnap at one
and it writes the Memory Card instead — it understands the conversation, so
the card captures the task, decisions, and error resolutions far more
accurately.

Get a free key at **[openrouter.ai/keys](https://openrouter.ai/keys)** (no
cost, no local install), then:

```bash
tokensnap config set openrouter_api_key <your-key>
tokensnap preset smart          # or: tokensnap config set compressor_type openrouter
tokensnap run claude
```

How it works:

- `compressor_type = openrouter` (set automatically by `tokensnap preset smart`)
  sends the truncated history to `openrouter_model` with a strict JSON-only
  prompt at temperature 0. The output is validated, clipped, and **merged
  over the regex card** — regex-found file paths are always kept, so nothing
  the old extractor caught is ever lost.
- Any hiccup — no key configured, network error, rate limit, malformed
  output — falls back to the regex card. Results (including failures) are
  cached per conversation, so a slow or rate-limited model never taxes every
  request.
- LLM-written cards carry a `"generator": "openrouter:<model>"` field so you
  can tell which path produced them.

**If no key is configured** while `compressor_type=openrouter`, Tokensnap
doesn't just silently degrade — `tokensnap status` / `tokensnap monitor` and
the proxy startup log show exactly what to do:

```
Memory Cards: regex (compressor_type=openrouter but no openrouter_api_key is
set - get a free key at https://openrouter.ai/keys, then `tokensnap config
set openrouter_api_key <key>`)
```

Want a different model, or want it off?

```bash
tokensnap config set openrouter_model qwen/qwen-2.5-7b-instruct:free
tokensnap config set compressor_type regex
```

**Security:** the OpenRouter key is a *separate* credential from your
Anthropic API key and is never mixed with it — the conversation transcript
sent to OpenRouter never carries your Anthropic key, and Anthropic requests
never carry your OpenRouter key. The dashboard never re-displays a saved
key to the browser, only whether one is set.

## Architecture

```
Claude Code  --ANTHROPIC_BASE_URL-->  Tokensnap proxy (127.0.0.1:8889)  -->  api.anthropic.com
                                            |
                                            |-- cleaner.py       strip ANSI / progress bars / dup lines
                                            |-- compressor.py    selective compression, Memory Card, truncation
                                            |-- openrouter.py    optional hosted-LLM card writer (regex fallback)
                                            |-- token_counter.py tiktoken-based budget check
                                            '-- stats.py         savings + liveness for status/monitor
```

Only `POST /v1/messages` and `/v1/complete` request *bodies* are touched, and
only on the way out. Every response — including SSE streams — is relayed back
byte-for-byte, so Claude Code behaves exactly as if it were talking to
Anthropic directly. As responses stream through, Tokensnap reads (but never
alters) the `usage` field to report real token consumption.

## Estimated vs. real tokens

Tokensnap reports two different measurements — don't expect them to be equal:

| | What it measures | Where it comes from |
| --- | --- | --- |
| **Est. saved** | Input request-body tokens removed by cleaning + compression | tiktoken (`cl100k_base`) estimate, computed locally before forwarding |
| **Real usage** | Actual input, output, cache-read and cache-write tokens | The `usage` field in Anthropic's responses (same source Claude Code uses) |

Claude Code caches context aggressively, so a turn that Claude reports as
"40k tokens" is often mostly cheap **cache reads**, not new input. Tokensnap's
"est. saved" only reflects the request body it actually cleaned and
compressed, while "real usage" shows the full picture from Anthropic.

**If real usage stays at 0 while you use Claude Code**, the requests aren't
reaching the proxy. Make sure you launched Claude Code with
`ANTHROPIC_BASE_URL` pointed at the proxy — the simplest way is
`tokensnap run claude`. Running plain `claude` in a terminal that doesn't
have that variable set bypasses Tokensnap entirely.

## Safety & scope

- Only `POST /v1/messages` and `/v1/complete` are optimized. **Every other
  request is forwarded byte-for-byte** (including `count_tokens`, models
  listing, etc.).
- Responses — including SSE streams — are relayed verbatim.
- Nothing is sent anywhere except the configured upstream and (only when
  `compressor_type=openrouter` and a key is set) OpenRouter, for the
  Memory Card summarization call only. No telemetry.
- Token counts use tiktoken's `cl100k_base` encoding, a close approximation
  for Claude models; if tiktoken can't load, a chars/4 estimate is used.

## Development

```bash
pip install -e .[dev]
pytest
```

The test suite (cleaner, compressor, selective compression, OpenRouter
integration, token counter, stats, CLI, web dashboard) runs fully offline —
no network access or real API keys required (the OpenRouter tests mock the
HTTP layer).

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for how to set up a dev environment, coding conventions, and the PR checklist.

## License

Apache 2.0 — see [LICENSE](LICENSE).
