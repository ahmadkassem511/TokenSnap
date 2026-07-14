# TokenSnap — Advanced Guide

This is the technical reference: how TokenSnap works internally, every
config key, every command, and the individual engines Adaptive Transparency
Mode tunes automatically. If you just want to install and use TokenSnap, see
[README.md](README.md) instead — you don't need anything on this page for
day-to-day use.

## Contents

- [How it actually works](#how-it-actually-works)
- [Adaptive Transparency Mode](#adaptive-transparency-mode)
- [What works with TokenSnap (and what doesn't)](#what-works-with-tokensnap-and-what-doesnt)
- [Commands](#commands)
- [Configuration](#configuration)
- [Tuning for your project type](#tuning-for-your-project-type)
- [How selective compression works](#how-selective-compression-works)
- [Smarter Memory Cards with OpenRouter](#smarter-memory-cards-with-openrouter)
- [Differential Context Engine](#differential-context-engine)
- [Project Cortex — a local second brain for your code](#project-cortex--a-local-second-brain-for-your-code)
- [Architecture](#architecture)
- [Estimated vs. real tokens](#estimated-vs-real-tokens)
- [Safety & scope](#safety--scope)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## How it actually works

TokenSnap is a local HTTP proxy that sits between Claude Code and the
Anthropic API. Every message you send in Claude Code re-sends the *entire*
conversation — every terminal dump, every progress-bar frame, every file
attached three prompts ago — so a session that starts at 2k tokens per
request can quietly balloon to 150k+. TokenSnap intercepts each request and
removes the noise before it leaves your machine:

1. **ANSI & progress-bar stripping** — color escape codes, spinner frames,
   and `\r`-redrawn progress bars are deleted from terminal output.
2. **Log deduplication** — runs of identical lines (retry storms, repeated
   warnings) collapse into one line plus a repeat count.
3. **Selective per-message compression** — assistant messages (Claude's own
   reasoning) are **never touched**. User messages are left intact unless
   they contain a large terminal/log dump, in which case only that dump
   shrinks to its error/warning/status lines. Tool results are reduced the
   same way, more aggressively. See
   [How selective compression works](#how-selective-compression-works).
4. **Memory Card compression** — history older than the last **N** exchanges
   is summarized into a compact JSON card (task, files touched, decisions,
   resolved errors) injected as a system note.
5. **Budget guard** — token usage is estimated on every request; near the
   model's context window, TokenSnap automatically gets more aggressive.

Blunt, uniform compression makes long sessions worse in a different way —
summarize *everything* the same way and Claude starts losing the thread,
forgetting a decision from ten messages ago or a file it already fixed.
TokenSnap's philosophy is to cut only the noise and leave the substance
alone — which is also why **when** each of these kicks in is not fixed; see
the next section.

Responses come back **completely untouched**, including streaming. Your
Anthropic API key never touches disk — TokenSnap simply forwards the auth
headers Claude Code already sends.

## Adaptive Transparency Mode

Rather than exposing separate "engines" you have to choose between, TokenSnap
ramps its own behavior up over a session's lifetime:

| Requests | Tier | What happens |
| --- | --- | --- |
| 1–5 | **Transparent** | Pass through untouched (aside from ANSI/noise cleaning) — the first replies feel exactly like talking to Claude directly. |
| 6–15 | **Light** | Selective compression kicks in once history is long enough, plus a tiny one-time project card (~300 tokens) so the model has basic orientation. |
| 16+ | **Full** | The [Differential Context Engine](#differential-context-engine) and [Project Cortex](#project-cortex--a-local-second-brain-for-your-code) — maximum token savings for sessions long enough that it matters. |

A hard safety net runs underneath every tier regardless: if a request is ever
genuinely near the model's context window (e.g. a huge pasted log on message
one), TokenSnap truncates it rather than risk the API rejecting an oversized
request. This is a last resort and essentially never fires in a normal-sized
early session.

Pin one behavior instead of ramping with `compression_level` (Settings →
Compression level, or `tokensnap config set compression_level <value>`):

- `adaptive` (default) — the ramp above.
- `off` — always the Transparent tier (never compress).
- `light` — always the Light tier.
- `full` — always the Full tier, from the first request.

Every session is tracked independently — a brand-new Claude Code conversation
always starts back at the Transparent tier, even if another session on the
same project has already reached Full.

## What works with TokenSnap (and what doesn't)

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
through TokenSnap.

**What Claude Desktop *can* do:** talk to local MCP servers. `tokensnap mcp`
runs TokenSnap as an MCP stdio server exposing `tokensnap_status`,
`tokensnap_recent_requests`, `tokensnap_get_config`, `tokensnap_set_config`,
`tokensnap_start_proxy`, and `tokensnap_stop_proxy` as tools — so you can ask
Claude Desktop things like "how much have I saved with TokenSnap today?" or
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

This manages/inspects TokenSnap from the chat app; it does not route the
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
| `tokensnap dashboard` | Web UI at `http://127.0.0.1:9876`. `--port`, `--host`, `--no-browser`. |
| `tokensnap monitor` | Live TUI: estimated savings **and real Anthropic usage** (input/output/cache), per-request table, proxy status. |
| `tokensnap status` | Is the proxy up? Shows estimated savings and real token usage so far. |
| `tokensnap focus [goal]` | Set (or show) the current project's focus/goal in its [Project Cortex](#project-cortex--a-local-second-brain-for-your-code) DNA. |
| `tokensnap dna [--refresh]` | Show the project's Cortex DNA (stack, focus, decisions, resolved issues); `--refresh` re-scans the static analysis. |
| `tokensnap config show` | Print the effective configuration. |
| `tokensnap config set <key> <value>` | Change a setting (see below). |
| `tokensnap config get <key>` | Read one setting. |
| `tokensnap preset <name>` | Apply a recommended configuration for your project type: `simple`, `balanced`, `complex`, `smart`, `maximum` — see [Tuning for your project type](#tuning-for-your-project-type). |
| `tokensnap openrouter-status` | Show the primary/fallback OpenRouter models, remaining rate limit, whether fallback/cooldown mode is active, and recent errors. |
| `tokensnap mcp` | Run TokenSnap as an MCP stdio server (status/config/start/stop as tools) — see above. |

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

To wipe every trace of TokenSnap (config, stats, logs) and start fresh:

```bash
tokensnap cleanup            # asks for confirmation, stops the proxy first if running
tokensnap cleanup --force    # skip the confirmation prompt
```

## Configuration

Stored in `~/.tokensnap/config.json`. Everything has a sensible default.

The one setting most people ever need:

| Key | Default | Meaning |
| --- | --- | --- |
| `compression_level` | `adaptive` | `adaptive` (ramps up as described in [Adaptive Transparency Mode](#adaptive-transparency-mode)), or pin `off` / `light` / `full`. |

Everything below is what `adaptive` tunes automatically — change these only
if you have a specific reason to override the individual behavior of a tier:

| Key | Default | Meaning |
| --- | --- | --- |
| `host` / `port` | `127.0.0.1` / `8889` | Where the proxy listens. |
| `upstream` | `https://api.anthropic.com` | The real API endpoint. |
| `keep_messages` | `10` | Exchanges kept verbatim when history is compressed (Light tier). Higher = more context, fewer tokens saved. See [Tuning for your project type](#tuning-for-your-project-type). |
| `aggressive_keep_last_n` | `2` | `keep_messages` drops to this when near the context window (the safety net). |
| `context_threshold` | `0.95` | Fraction of the context window that triggers the safety net. |
| `min_messages_to_compress` | `8` | Histories shorter than this are never compressed. |
| `selective_compression` | `true` | Clean noise from every message (assistant untouched, terminal dumps/tool output reduced to signal) before truncating. `false` uses legacy uniform truncation only. |
| `compressor_type` | `regex` | Memory Card generator: `regex` (fast, rule-based, offline), `openrouter` (a free hosted model writes a better summary), `off` (no Memory Card at all). |
| `openrouter_api_key` | *(empty)* | Your [OpenRouter](https://openrouter.ai/keys) key (free tier available). Required for `compressor_type=openrouter`. |
| `openrouter_model` | `meta-llama/llama-3.1-8b-instruct:free` | Model used to write Memory Cards when `compressor_type=openrouter`. |
| `openrouter_fallback_models` | *(empty)* | Comma-separated backup models tried in order if the primary model hits a rate limit or is down (429/5xx/timeout). |
| `openrouter_max_retries` | `1` | Total attempts across the primary model + fallbacks is capped at `1 + this value`. |
| `openrouter_retry_delay_seconds` | `5` | Seconds to wait before trying the next fallback model. |
| `context_store_enabled` | `false` | Force the [Differential Context Engine](#differential-context-engine) on every request, independent of tier (mostly for testing/tuning — the Full tier already enables it automatically once a session earns it). |
| `context_tree_size` | `20` | How many recent important events the Context Tree summarizes. |
| `project_primer_enabled` | `true` | The one-shot project overview card used when Project Cortex is off. |
| `project_cortex_enabled` | `true` | Persistent per-project DNA (stack, focus, decisions, resolved issues) injected as immutable Core Memory at the Full tier. Supersedes the Project Primer when on. |
| `session_bridge_auto_inject` | `true` | Inject the previous session's summary (the Session Bridge) so a new session resumes where the last left off. |
| `dna_update_interval` | `86400` | Minimum seconds between re-scanning the project's static analysis into the DNA. |
| `log_level` | `INFO` | Proxy log verbosity. |
| `key` | *(empty)* | Optional stored Anthropic API key — normally unnecessary; the proxy forwards the key from request headers. |

Example:

```bash
tokensnap config set compression_level full
```

## Tuning for your project type

`keep_messages` is a trade-off dial, not a "more is always better" setting:

- **Higher `keep_messages`** → Claude keeps more of the actual recent
  conversation in full, so it's less likely to lose track of what it's
  doing on a complex, multi-file task — at the cost of fewer tokens saved.
- **Lower `keep_messages`** → more aggressive compression, bigger token
  savings — but more of the conversation is replaced by the summarized
  Memory Card.

Pick a preset instead of guessing:

```bash
tokensnap preset simple     # keep_messages=5,   selective on,  regex   - quick scripts, single-file tasks
tokensnap preset balanced   # keep_messages=10,  selective on,  regex   - the default, suitable for most projects
tokensnap preset complex    # keep_messages=20,  selective off, regex  - large multi-file projects, maximal safety
tokensnap preset smart      # keep_messages=25,  selective on,  openrouter - best quality (needs a free API key)
tokensnap preset maximum    # keep_messages=999, selective on,  off     - effectively disables Memory Card compression
```

If Claude starts "forgetting" earlier decisions or files it already touched
on a big project, that's a sign `keep_messages` is too low for that
project — run `tokensnap preset complex` or `tokensnap preset smart` before
starting your next session.

## How selective compression works

Selective compression (`selective_compression = true`, the default) runs
before the history is ever truncated, and treats every message differently
based on its role:

- **Assistant messages are never touched.** Claude's own reasoning and
  responses are passed through byte-for-byte, at every position in the
  conversation, not just the recent tail.
- **User messages** are left alone unless they contain a large terminal/log
  dump (heuristically: over ~500 tokens and shaped like shell output). When
  they do, only that dump shrinks — to its error/warning lines plus a final
  status line — while any surrounding prose survives untouched around it.
- **Tool results** are compressed the same way, more aggressively, since
  they're almost always machine-generated noise once the outcome is known.
  A 200-line build log becomes a couple of error lines plus something like
  `[tokensnap: 180 lines omitted (2 errors, 1 warning)]`.
- **Reads are the one exception.** A `Read`/`View`/`Open`/`Grep`/`Glob` tool
  result - or a `Bash` call whose entire command is a single unchained read
  (`cat`, `type`, `Get-Content`, `head`, `tail`, `more`, `less` — no `|`,
  `&&`, `;`, `>`/`<`) - is passed through completely untouched, in every
  tier. That result *is* the content (or search matches) Claude asked to
  see; summarizing it would defeat the point of asking in the first place -
  a Grep result, for instance, may contain no error/warning signal words at
  all, so generic compression would gut it rather than trim noise. A command
  that does more than read (`cat file.txt && rm file.txt`) is correctly not
  exempted just because it starts with a read verb; a raw shell `grep`
  (as opposed to the dedicated `Grep` tool) isn't auto-exempted either, since
  it's commonly just one stage of a longer pipeline.

Set `selective_compression = false` (or use `tokensnap preset complex`) to
go back to the legacy behavior: no per-message cleaning, just uniform
truncation at `keep_messages`.

Separately, once history exceeds `min_messages_to_compress`, TokenSnap:

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
to several hosted models (e.g. Meta's Llama 3.1 8B); point TokenSnap at one
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
  over the regex card** — regex-found file paths are always kept.
- Any hiccup — no key configured, network error, rate limit, malformed
  output — falls back to the regex card. Results (including failures) are
  cached per conversation, so a slow or rate-limited model never taxes every
  request.
- LLM-written cards carry a `"generator": "openrouter:<model>"` field so you
  can tell which path produced them.

**If no key is configured** while `compressor_type=openrouter`, TokenSnap
doesn't just silently degrade — `tokensnap status` / `tokensnap monitor` and
the proxy startup log show exactly what to do:

```
Memory Cards: regex (compressor_type=openrouter but no openrouter_api_key is
set - get a free key at https://openrouter.ai/keys, then `tokensnap config
set openrouter_api_key <key>`)
```

### Fallback models and rate-limit resilience

A single free model can hit a 429 (rate limited) or a 503 under load.
Configure backup models to try in order when that happens:

```bash
tokensnap config set openrouter_fallback_models "qwen/qwen-2.5-7b-instruct:free,mistralai/mistral-7b-instruct:free"
```

- On a *retryable* error (429/500/502/503/504, or a timeout), TokenSnap waits
  `openrouter_retry_delay_seconds` (default 5) and tries the next model in
  the list. Total attempts (primary + fallbacks) is capped at
  `1 + openrouter_max_retries` (default 1, i.e. one fallback attempt).
- A *non-retryable* error (e.g. a bad API key) stops immediately.
- If every attempted model fails, TokenSnap enters a 60-second cooldown:
  further requests skip OpenRouter entirely until the cooldown expires.
- `X-RateLimit-Remaining` / `X-RateLimit-Reset` response headers are captured
  on every call and shown by `tokensnap openrouter-status` and the
  dashboard's Advanced section.

**Security:** the OpenRouter key is a *separate* credential from your
Anthropic API key and is never mixed with it. The dashboard never re-displays
a saved key to the browser, only whether one is set.

## Differential Context Engine

Memory Cards still send a *summary of everything* on every request. The
**Differential Context Engine** changes the model entirely: TokenSnap keeps a
full local mirror of the conversation and sends the model only the last few
exchanges plus a compact **Context Tree** — an index of the important past
events — handing it a `fetch_context` tool to pull back the full text of any
event *only when it actually needs it*. This is what the
[Adaptive Transparency Mode](#adaptive-transparency-mode) Full tier turns on
automatically for long sessions; `context_store_enabled` can also force it on
for every request, independent of tier.

### How it works

1. **Mirror.** Every user/assistant message the proxy sees is stored in a
   local SQLite database (`~/.tokensnap/context_store.db`), keyed by its
   position so re-sends of the same conversation never duplicate it. Each
   message is tagged with an `event_type` (`decision`, `error`,
   `file_modification`, `clarification`, `request`, or `other`). Any genuine
   user instruction that doesn't match a more specific category (a plainly
   phrased "read this project and run it") is tagged `request`, not `other` —
   `other` is reserved for assistant chatter and tool-result noise. This
   guarantees **every real ask survives** in the tree, not just things phrased
   as a decision or error; the conversation's very first message additionally
   gets a hard backstop (labeled `"type":"task"`) in the rare case the tree's
   recency limit itself pushes it out.
2. **Reconstruct.** The outgoing request keeps only the last few exchanges
   verbatim (after selective compression). Everything older is replaced by a
   single system block holding the **Context Tree** — a JSON array of
   `{id, summary, type}` for the most recent important events
   (`context_tree_size`, default 20) — plus a `fetch_context` tool definition.
3. **Recall.** If the model calls `fetch_context` with some event ids, the
   *proxy* answers it — it looks the events up in the mirror, feeds their full
   text back as a `tool_result`, and continues the turn upstream itself. Claude
   Code only ever sees the final answer; the recall round-trip is invisible.

```
tree in  ─▶  model asks fetch_context([3,5])  ─▶  proxy serves events 3 & 5
                                                    from the local mirror
                                                        │
   client sees only the final answer  ◀───────────────┘  (continues upstream)
```

### Trade-offs to understand

- **Prompt caching.** The Context Tree is rewritten every turn, which
  invalidates Anthropic's prompt cache. Claude Code leans on that cache
  heavily, so watch **real** input/cache-read tokens (in `tokensnap monitor`
  or the dashboard), not just the local *estimated* savings — on some
  workloads the cache loss can offset the smaller request. This is why it
  only switches on automatically for sessions long enough (16+ requests) that
  the trade-off is worth it.
- **Streaming.** While the fetch loop runs, upstream requests are made
  non-streaming so the proxy can inspect them; the final reply is re-emitted
  to Claude Code as a well-formed event stream, but it arrives once complete
  rather than token-by-token.

## Project Cortex — a local second brain for your code

Compression saves tokens by *forgetting* the old parts of a conversation.
**Project Cortex** is the counterweight: a persistent, per-project knowledge
base that makes sure the things that matter about your project are *never*
forgotten. It's on by default (`project_cortex_enabled`) and activates at the
[Adaptive Transparency Mode](#adaptive-transparency-mode) Full tier.

Everything lives in a `.tokensnap/` folder **inside the project** (git-ignored
by default, since session summaries can be personal), so it's local, private,
and travels with the code.

### Project DNA — the knowledge base

`.tokensnap/project_dna.json` holds two halves:

- **Static analysis** — tech stack, framework, key dependencies, entry points,
  folder map, and git branch/last commit. Scanned from the project and
  refreshed at most once per `dna_update_interval` seconds (default: daily).
- **Living memory** — the **current focus/goal**, plus **decisions** and
  **resolved issues** distilled from your sessions (labelled `Decision:` lines
  and error→fix arcs are captured, even when the fix comes a few messages after
  the error).

A compact rendering of the DNA is injected as an **immutable Core Memory
block** in the system prompt, which compression never touches, so this
context is never truncated away.

### Session Bridge — seamless continuity

When a session ends (the proxy stops), TokenSnap distils it into a summary under
`.tokensnap/sessions/`. The next session for that project automatically picks up
the most recent summary as an optional **Session Bridge** block
(`session_bridge_auto_inject`), so you resume exactly where you left off — even
across restarts. It works **across tools**, too: paste a conversation from
Claude Desktop into a file and import it as a bridge.

### Adaptive compression with context priority

The compressor assigns every message a **value weight** — assistant reasoning >
user > tool result; decisions, explicit `important`/`note` markers, and code
snippets raise it; bulk log/error dumps lower it; older messages decay, but a
message you mark important holds a high floor. When a Memory Card must drop
detail, the **highest-weight decisions are kept first**. (Weighting only
decides *what gets summarised* — it never reorders the messages actually
sent, which would break tool-call pairing.)

### Managing it

- **Dashboard → Advanced → Project Cortex panel:** view the DNA, set the
  current focus, and trigger a DNA refresh.
- **CLI:** `tokensnap focus "add OAuth login"` sets the goal; `tokensnap dna`
  (or `tokensnap dna --refresh`) shows the DNA.

## Architecture

```
Claude Code  --ANTHROPIC_BASE_URL-->  TokenSnap proxy (127.0.0.1:8889)  -->  api.anthropic.com
                                            |
                                            |-- cleaner.py       strip ANSI / progress bars / dup lines
                                            |-- compressor.py    selective compression, Memory Card, truncation
                                            |-- openrouter.py    optional hosted-LLM card writer (regex fallback)
                                            |-- token_counter.py tiktoken-based budget check
                                            |-- context_store.py local conversation mirror (SQLite)
                                            |-- context_engine.py Context Tree builder (Differential Context)
                                            |-- fetch_context.py proxy-side fetch_context tool cycle
                                            |-- project_primer.py one-shot project overview card
                                            |-- project_dna.py    persistent per-project DNA (Cortex)
                                            |-- session_bridge.py cross-session continuity (Cortex)
                                            '-- stats.py         savings + liveness for status/monitor
```

`proxy.py`'s `optimize_body` resolves each request's Adaptive Transparency
Mode tier first, then dispatches to the above accordingly. Only
`POST /v1/messages` and `/v1/complete` request *bodies* are touched, and only
on the way out. Every response — including SSE streams — is relayed back
byte-for-byte (except during the Full tier's `fetch_context` recall cycle,
which re-emits the final message once complete; see the trade-offs above).

## Estimated vs. real tokens

TokenSnap reports two different measurements — don't expect them to be equal:

| | What it measures | Where it comes from |
| --- | --- | --- |
| **Est. saved** | Input request-body tokens removed by cleaning + compression | tiktoken (`cl100k_base`) estimate, computed locally before forwarding |
| **Real usage** | Actual input, output, cache-read and cache-write tokens | The `usage` field in Anthropic's responses (same source Claude Code uses) |

Claude Code caches context aggressively, so a turn that Claude reports as
"40k tokens" is often mostly cheap **cache reads**, not new input. TokenSnap's
"est. saved" only reflects the request body it actually cleaned and
compressed, while "real usage" shows the full picture from Anthropic.

**If real usage stays at 0 while you use Claude Code**, the requests aren't
reaching the proxy. Make sure you launched Claude Code with
`ANTHROPIC_BASE_URL` pointed at the proxy — the simplest way is
`tokensnap run claude`, or the dashboard's Launch button. Running plain
`claude` in a terminal that doesn't have that variable set bypasses TokenSnap
entirely.

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

The test suite runs fully offline — no network access or real API keys
required (the OpenRouter tests mock the HTTP layer).

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for how to set up a dev environment, coding conventions, and the PR checklist.

## License

Apache 2.0 — see [LICENSE](LICENSE).
