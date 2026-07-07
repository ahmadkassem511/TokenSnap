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
printf "Do you want to open the setup dashboard now? [Y/n] "
read -r OPENDASH
case "$OPENDASH" in
    [Nn]*)
        echo "Skipped. Start it later with: tokensnap dashboard"
        ;;
    *)
        echo "Starting the dashboard in the background - it will open in your browser..."
        nohup ./.venv/bin/tokensnap dashboard >/dev/null 2>&1 &
        disown 2>/dev/null || true
        ;;
esac

echo ""
echo "Creating a desktop shortcut for the dashboard..."
DESKTOP_DIR="$HOME/Desktop"
if [ ! -d "$DESKTOP_DIR" ]; then
    echo "[WARN] No Desktop folder found at $DESKTOP_DIR - skipping shortcut."
else
    TOKENSNAP_BIN="$(pwd)/.venv/bin/tokensnap"
    if [ "$(uname)" = "Darwin" ]; then
        SHORTCUT="$DESKTOP_DIR/TokenSnap Dashboard.command"
        cat > "$SHORTCUT" <<SHORTCUT_EOF
#!/usr/bin/env bash
cd "$(pwd)"
"$TOKENSNAP_BIN" dashboard
SHORTCUT_EOF
        chmod +x "$SHORTCUT"
        echo "Desktop shortcut created: $SHORTCUT"
    else
        SHORTCUT="$DESKTOP_DIR/TokenSnap-Dashboard.desktop"
        cat > "$SHORTCUT" <<SHORTCUT_EOF
[Desktop Entry]
Type=Application
Name=TokenSnap Dashboard
Comment=Open the TokenSnap web dashboard
Exec="$TOKENSNAP_BIN" dashboard
Path=$(pwd)
Terminal=true
Categories=Development;
SHORTCUT_EOF
        chmod +x "$SHORTCUT"
        # Newer GNOME/Nautilus requires marking .desktop launchers as
        # trusted, or it shows an "Untrusted application" warning instead
        # of running. Harmless (and silently skipped) if `gio` isn't
        # installed or the DE doesn't need it.
        gio set "$SHORTCUT" metadata::trusted true >/dev/null 2>&1 || true
        echo "Desktop shortcut created: $SHORTCUT"
    fi
fi

echo ""
if command -v claude >/dev/null 2>&1; then
    echo "Found Claude Code on PATH."
else
    echo "[NOTE] The 'claude' command isn't on your PATH."
    echo "       'tokensnap run claude' will still find it in npm's global bin"
    echo "       or via npx, so you may not need to do anything. If Claude Code"
    echo "       isn't installed yet, install it with:"
    echo "           npm install -g @anthropic-ai/claude-code"
    echo "       or download it from https://claude.ai/download"
fi

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