#!/usr/bin/env sh
# ============================================================
#  Tokensnap installer for Linux / macOS
#  Checks Python, creates a virtual environment, installs
#  tokensnap, and shows how to get it on PATH.
# ============================================================
set -e
cd "$(dirname "$0")"

echo ""
echo "=== Tokensnap v2 installer ==="
echo ""

# --- 1. Find Python -----------------------------------------
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "[ERROR] Python was not found."
    echo "Install Python 3.9+ first:"
    echo "  macOS:  brew install python"
    echo "  Debian: sudo apt install python3 python3-venv python3-pip"
    echo "  Fedora: sudo dnf install python3"
    echo "Then re-run this installer."
    exit 1
fi

# --- 2. Check version >= 3.9 ---------------------------------
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "[ERROR] Python 3.9 or newer is required. Found: $($PY --version)"
    exit 1
fi
echo "Found $($PY --version)"

# --- 3. Create virtual environment ---------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv ..."
    "$PY" -m venv .venv
fi

# --- 4. Install tokensnap ------------------------------------
echo "Installing tokensnap ..."
./.venv/bin/python -m pip install --upgrade pip --quiet
./.venv/bin/python -m pip install -e .

echo ""
echo "=== Installed successfully! ==="
echo ""
echo "The tokensnap command lives at: $(pwd)/.venv/bin/tokensnap"
echo ""
printf "Create a symlink in ~/.local/bin so 'tokensnap' works everywhere? [y/N] "
read -r ADDPATH
case "$ADDPATH" in
    [Yy]*)
        mkdir -p "$HOME/.local/bin"
        ln -sf "$(pwd)/.venv/bin/tokensnap" "$HOME/.local/bin/tokensnap"
        echo "Linked to ~/.local/bin/tokensnap."
        case ":$PATH:" in
            *":$HOME/.local/bin:"*) ;;
            *) echo "NOTE: add ~/.local/bin to your PATH (e.g. in ~/.bashrc):"
               echo '  export PATH="$HOME/.local/bin:$PATH"' ;;
        esac
        ;;
    *)
        echo "Skipped. Activate the venv to use it:  . .venv/bin/activate"
        ;;
esac

echo ""
echo "Quickstart:"
echo "  tokensnap dashboard        # web UI: setup wizard, charts & settings"
echo "  tokensnap start            # start the proxy"
echo "  tokensnap run claude       # launch Claude Code through the proxy"
echo "  tokensnap monitor          # live savings dashboard (terminal)"
echo "  tokensnap preset smart     # activate intelligent selective compression"
echo ""
echo "For best quality: run 'tokensnap preset smart' once."
echo "Optionally, get a free OpenRouter key (https://openrouter.ai/keys)"
echo "to enable AI-powered Memory Cards: tokensnap config set openrouter_api_key YOUR_KEY"
echo ""