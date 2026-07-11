# Contributing to Tokensnap

Thanks for considering a contribution! Tokensnap is a small, focused tool —
the bar for new code is that it must be simple, tested, and offline-testable
wherever possible.

## Setup

```bash
git clone https://github.com/ahmadkassem511/TokenSnap.git
cd TokenSnap
python -m venv .venv
. .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest
```

The suite must run fully offline — no live Anthropic API calls, no real API
keys. If you're adding proxy behavior, prefer testing the pure functions in
`cleaner.py`, `compressor.py`, or `token_counter.py` directly rather than
spinning up the aiohttp server.

## Making a change

1. Open an issue first for anything beyond a small fix, so we can agree on
   the approach before you invest time.
2. Keep changes scoped — this project intentionally avoids feature creep
   (see the MVP philosophy: rule-based heuristics over ML, no telemetry, no
   extra dependencies unless they earn their place).
3. Add or update tests for any behavior change.
4. Update `ADVANCED.md` if you change a command, config key, or default
   (`README.md` is intentionally minimal - see its own note on that).
5. Run `pytest` locally before opening a PR — CI runs the same suite.

## Code style

- No new runtime dependencies without a good reason (currently: `typer`,
  `rich`, `aiohttp`, `tiktoken`).
- Match the existing style: plain functions over classes where possible,
  minimal comments (only where the *why* isn't obvious from the code).
- Python ≥3.9 compatible syntax.

## Reporting bugs / requesting features

Open a [GitHub issue](https://github.com/ahmadkassem511/TokenSnap/issues)
with:
- Your OS and Python version.
- The `tokensnap` version (`tokensnap version`).
- Steps to reproduce, and what you expected vs. what happened.
- Relevant proxy log output (from `tokensnap start -v` or `~/.tokensnap/proxy.log`).

Please don't include real API keys or full conversation contents in bug
reports.
