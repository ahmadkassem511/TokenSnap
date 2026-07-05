---
name: verify
description: How to run and end-to-end verify the Tokensnap proxy and CLI without a real Anthropic API key or Ollama install.
---

# Verifying Tokensnap

Everything runs offline. The surfaces are (1) the HTTP proxy and (2) the
`tokensnap` CLI.

## Setup

Use the repo venv: `.venv/Scripts/python.exe` (Windows). Package is
imported from the current working directory, so run from the checkout
being verified. Tests: `python -m pytest -q` (fully offline).

## Proxy surface

Launch the real process with an **isolated home** so the user's
`~/.tokensnap/` is never touched (config/stats live under `Path.home()`,
i.e. `USERPROFILE` on Windows, `HOME` elsewhere):

1. Create a temp dir, write `<tmp>/.tokensnap/config.json` with:
   `port` (use a free one, not 8889), `upstream` pointed at a local fake
   Anthropic server, and (for LLM-card testing) `ollama_url` pointed at a
   local fake Ollama server.
2. Fake upstream: `http.server` that records POSTed JSON bodies and
   replies with a JSON message (or `text/event-stream` when the request
   body has `"stream": true` — verifies SSE relay).
3. Fake Ollama: answer `GET /api/tags` with 200 and `POST /api/generate`
   with `{"response": "<json card string>"}`.
4. Start: `python -m tokensnap start` with `USERPROFILE`/`HOME` set to
   the temp dir, cwd = checkout.
5. Drive with `urllib` POSTs to `/v1/messages`. Compression triggers when
   `len(messages) > min_messages_to_compress` (default 8). Inspect what
   the fake upstream received: trimmed `messages`, memory card in
   `system` (`[TOKENSNAP MEMORY CARD]`, `generator` key when the LLM
   path was used).

A ready harness from a previous verification covers: LLM card used,
garbage-LLM regex fallback, short-convo pass-through, SSE relay, GET
pass-through. Rebuild it from this recipe if not present.

## CLI surface

Same isolated `USERPROFILE`/`HOME` trick, then `python -m tokensnap
config show|set|get`, `status`, `version`. Invalid config values must
exit 1 with a red message, not a traceback.

## Gotchas

- Port 8889 can be transiently occupied on dev machines; tests that
  probe `proxy_running` may flake — rerun before blaming the change.
- `tokensnap stop`/`cleanup` kill whatever listens on the configured
  port — never run them against the user's real config during checks.
