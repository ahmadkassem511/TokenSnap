# TokenSnap — Make Claude Code last 2-3x longer

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/ahmadkassem511/TokenSnap/actions/workflows/tests.yml/badge.svg)](https://github.com/ahmadkassem511/TokenSnap/actions/workflows/tests.yml)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](#quickstart)

Claude Code hits usage limits too fast because it re-sends your entire
conversation history, including noisy terminal output.

TokenSnap is a local proxy that cleans the noise and intelligently
summarizes old conversation, so you can work longer without hitting limits.

## Quickstart

1. **Windows:** double-click `install.bat`. **macOS/Linux:** run `./install.sh`.
2. The TokenSnap dashboard opens automatically in your browser.
3. Pick your project folder and click **Launch Claude Code**.

That's it — just use Claude Code normally. TokenSnap works quietly in the
background; the dashboard shows how many tokens it's saved you.

Your Anthropic API key never touches disk — TokenSnap simply forwards it
straight through to Anthropic.

Want to know how it works, tune it, or see every setting and command? See
[ADVANCED.md](ADVANCED.md).

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
