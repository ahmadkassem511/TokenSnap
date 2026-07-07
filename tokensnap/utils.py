"""Shared helpers: logging setup, Anthropic message-content access, and
locating executables (e.g. `claude`) that may not be on the system PATH.

Anthropic `messages` entries have `content` that is either a plain string or
a list of blocks ({"type": "text"|"tool_use"|"tool_result"|"image", ...}).
These helpers let the rest of the code treat both shapes uniformly.
"""

import logging
import os
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=False, show_path=False)],
        force=True,
    )
    # aiohttp access logs are noisy; we do our own per-request logging
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


def message_text(message: Dict[str, Any]) -> str:
    """All human-readable text in a message, joined with newlines."""
    return "\n".join(_iter_text(message.get("content")))


def _iter_text(content: Any) -> List[str]:
    if isinstance(content, str):
        return [content] if content else []
    parts: List[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif btype == "tool_result":
                parts.extend(_iter_text(block.get("content")))
    return parts


def transform_message_text(
    message: Dict[str, Any], fn: Callable[[str], str]
) -> Dict[str, Any]:
    """Return a copy of `message` with `fn` applied to every text payload
    (top-level strings, text blocks, and text inside tool_result blocks).
    Non-text blocks (tool_use, image) are left untouched."""
    out = dict(message)
    out["content"] = _transform_content(message.get("content"), fn)
    return out


def _transform_content(content: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(content, str):
        return fn(content)
    if isinstance(content, list):
        new_blocks = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text" and isinstance(block.get("text"), str):
                    block = dict(block, text=fn(block["text"]))
                elif btype == "tool_result" and "content" in block:
                    block = dict(
                        block, content=_transform_content(block["content"], fn)
                    )
            new_blocks.append(block)
        return new_blocks
    return content


def system_to_parts(system: Any) -> List[str]:
    """Return the text parts of an Anthropic `system` field, which may be a
    plain string, a list of blocks, or absent. Non-text blocks are ignored."""
    if isinstance(system, str):
        return [system]
    parts: List[str] = []
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return parts


def append_to_system(system: Any, extra: str) -> Any:
    """Append text to a system prompt that may be a string, block list, or absent."""
    if system is None or system == "":
        return extra
    if isinstance(system, str):
        return system + "\n\n" + extra
    if isinstance(system, list):
        return system + [{"type": "text", "text": extra}]
    return system


def is_tool_result_only(message: Dict[str, Any]) -> bool:
    """True when a user message carries nothing but tool_result blocks.

    Such a message must never become the first message of a truncated
    history: the API requires the preceding assistant tool_use turn."""
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


# ---------------------------------------------------------------------------
# Executable resolution
#
# `claude` is typically installed with `npm install -g @anthropic-ai/claude-code`,
# which drops the launcher in npm's global bin directory. On many machines -
# especially Windows - that directory isn't on PATH, so a plain `claude` (or
# shutil.which("claude")) fails even though Claude Code is installed. These
# helpers look past PATH into the well-known npm/node locations so
# `tokensnap run claude` and the dashboard's launch button work anyway.
# ---------------------------------------------------------------------------

CLAUDE_INSTALL_HINT = (
    "Couldn't find the `claude` command. Install Claude Code, then try again:\n"
    "  npm install -g @anthropic-ai/claude-code\n"
    "or download it from https://claude.ai/download"
)


def _npm_global_bin() -> Optional[str]:
    """The directory npm installs global executables into, via `npm root -g`.

    `npm root -g` reports the global *node_modules* path; the launchers live
    beside it (Windows) or in the prefix's `bin` (Unix). Best-effort: returns
    None if npm is absent or the call fails/times out.
    """
    npm = shutil.which("npm")
    if not npm:
        return None
    try:
        result = subprocess.run(
            [npm, "root", "-g"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    root = (result.stdout or "").strip()
    if not root:
        return None
    if os.name == "nt":
        # ...\npm\node_modules  ->  ...\npm  (where claude.cmd lives)
        return os.path.dirname(root)
    # <prefix>/lib/node_modules  ->  <prefix>/bin
    prefix = os.path.dirname(os.path.dirname(root))
    return os.path.join(prefix, "bin")


def _candidate_bin_dirs() -> List[str]:
    """Common install directories that are frequently missing from PATH."""
    dirs: List[str] = []
    if os.name == "nt":
        for var in ("APPDATA", "LOCALAPPDATA"):
            base = os.environ.get(var)
            if base:
                dirs.append(os.path.join(base, "npm"))
        program_files = os.environ.get("ProgramFiles")
        if program_files:
            dirs.append(os.path.join(program_files, "nodejs"))
    else:
        home = os.path.expanduser("~")
        dirs.extend([
            "/usr/local/bin",
            "/usr/bin",
            "/opt/homebrew/bin",
            os.path.join(home, ".npm-global", "bin"),
            os.path.join(home, ".local", "bin"),
            os.path.join(home, "node_modules", ".bin"),
        ])
    npm_bin = _npm_global_bin()
    if npm_bin:
        dirs.append(npm_bin)
    return dirs


def find_executable(name: str) -> Optional[str]:
    """Locate an executable by name, returning its absolute path or None.

    Searches the system PATH first (respecting PATHEXT on Windows via
    shutil.which), then the common install locations in `_candidate_bin_dirs`
    - most importantly npm's global bin directory, so npm-installed CLIs like
    `claude` resolve even when the user's PATH doesn't include it.
    """
    found = shutil.which(name)
    if found:
        return os.path.abspath(found)
    for directory in _candidate_bin_dirs():
        if not directory or not os.path.isdir(directory):
            continue
        # A single-directory search still applies PATHEXT on Windows, so
        # `claude` matches `claude.cmd`/`claude.exe` there.
        hit = shutil.which(name, path=directory)
        if hit:
            return os.path.abspath(hit)
    return None


def resolve_claude_command(command: List[str]) -> Optional[List[str]]:
    """Resolve a `tokensnap run` command for launching Claude Code.

    If the command targets `claude` but it isn't on PATH, this substitutes the
    full path found in npm's global bin, or falls back to `npx claude` (which
    can fetch/run the published package on demand). Commands that don't target
    `claude` are returned unchanged - `tokensnap run` stays fully generic.
    Returns None only when the command targets `claude` and it can't be found
    or run any way.
    """
    if not command:
        return command
    base = os.path.basename(command[0]).lower()
    if base not in ("claude", "claude.exe", "claude.cmd"):
        return command

    resolved = find_executable("claude")
    if resolved:
        return [resolved] + list(command[1:])
    if shutil.which("npx"):
        return ["npx", "claude"] + list(command[1:])
    return None
