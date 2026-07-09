"""Project Primer: a compact, auto-generated overview of the project a Claude
Code session is working in.

On the first request of each session the proxy injects this "Project Card" into
the system prompt, so Claude understands the codebase immediately - its
language/framework, top-level layout, key dependencies, and current git state -
without needing that context to accumulate through conversation.

Design:
  * ``generate_project_card(directory)`` scans a directory and returns a plain
    dict. It is **best-effort and never raises**: every probe (config parsing,
    git, README) is guarded, so a missing tool or unreadable file just leaves
    that field empty. The card is capped well under ~500 tokens by bounding
    every field's size.
  * ``prime_for_session`` de-dupes per session (the proxy is one long-running
    process) so the disk is scanned once per conversation, not per request.
  * The most recent card is persisted to ``~/.tokensnap/last_project_card.json``
    so the *separate* dashboard process can display it.

Offline and dependency-free: stdlib only (``pathlib``/``json``/``subprocess``).
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from tokensnap import config as config_mod

# Persisted so the dashboard (a different process) can show the last card.
LAST_CARD_FILE = config_mod.CONFIG_DIR / "last_project_card.json"

# Directories not worth listing or descending into for an overview.
_IGNORE_DIRS = {
    "node_modules", ".venv", "venv", "env", ".git", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".idea",
    ".vscode", ".next", "target", ".tox", "site-packages", ".gradle", "vendor",
}

# Caps that keep the card comfortably under ~500 tokens (~2000 chars).
_MAX_ENTRIES = 40
_MAX_DEPS = 20
_MAX_MODIFIED = 20
_README_MAX = 300
_COMMIT_MAX = 200
_CARD_CHAR_CAP = 2200

# Framework detection: first dependency name that matches wins.
_JS_FRAMEWORKS = [
    ("next", "Next.js"), ("nuxt", "Nuxt"), ("@angular/core", "Angular"),
    ("@nestjs/core", "NestJS"), ("react", "React"), ("vue", "Vue"),
    ("svelte", "Svelte"), ("express", "Express"), ("fastify", "Fastify"),
    ("koa", "Koa"), ("electron", "Electron"),
]
_PY_FRAMEWORKS = [
    ("django", "Django"), ("fastapi", "FastAPI"), ("flask", "Flask"),
    ("aiohttp", "aiohttp"), ("tornado", "Tornado"), ("pyramid", "Pyramid"),
    ("starlette", "Starlette"),
]

_CARD_KEYS = (
    "project_name", "language", "framework", "key_dependencies",
    "folder_structure", "git_branch", "last_commit_summary",
    "modified_files", "readme_summary",
)

# session_id -> card. Also marks a session as already primed (single-process
# proxy), so we only scan the disk once per conversation.
_primed_cards: Dict[str, Dict[str, Any]] = {}


def _empty_card(name: str = "") -> Dict[str, Any]:
    return {
        "project_name": name,
        "language": "",
        "framework": "",
        "key_dependencies": [],
        "folder_structure": [],
        "git_branch": "",
        "last_commit_summary": "",
        "modified_files": [],
        "readme_summary": "",
    }


def _read_text(path: Path, limit: int = 200_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return ""


def _list_top_level(directory: Path) -> List[str]:
    """Top-level entries (dirs suffixed with '/'), ignoring noise dirs."""
    try:
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        return []
    out: List[str] = []
    for p in entries:
        name = p.name
        if p.is_dir():
            if name in _IGNORE_DIRS:
                continue
            out.append(name + "/")
        else:
            out.append(name)
        if len(out) >= _MAX_ENTRIES:
            break
    return out


def _framework_from(deps: List[str], table) -> str:
    lowered = {d.lower() for d in deps}
    for needle, label in table:
        if needle in lowered:
            return label
    return ""


def _detect_stack(directory: Path) -> Dict[str, Any]:
    """Best-effort (language, framework, key_dependencies) from build files."""
    # --- Node / JS / TS -----------------------------------------------------
    pkg = directory / "package.json"
    if pkg.is_file():
        deps: List[str] = []
        try:
            data = json.loads(_read_text(pkg) or "{}")
            for key in ("dependencies", "devDependencies"):
                section = data.get(key)
                if isinstance(section, dict):
                    deps.extend(section.keys())
        except (ValueError, TypeError):
            data = {}
        is_ts = (directory / "tsconfig.json").is_file() or "typescript" in {
            d.lower() for d in deps
        }
        return {
            "language": "TypeScript" if is_ts else "JavaScript",
            "framework": _framework_from(deps, _JS_FRAMEWORKS),
            "key_dependencies": deps[:_MAX_DEPS],
        }

    # --- Python -------------------------------------------------------------
    py_markers = ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "Pipfile")
    if any((directory / m).is_file() for m in py_markers):
        deps = _python_deps(directory)
        return {
            "language": "Python",
            "framework": _framework_from(deps, _PY_FRAMEWORKS),
            "key_dependencies": deps[:_MAX_DEPS],
        }

    # --- Other single-file ecosystems --------------------------------------
    simple = [
        ("Cargo.toml", "Rust"), ("go.mod", "Go"), ("pom.xml", "Java"),
        ("build.gradle", "Java"), ("build.gradle.kts", "Kotlin"),
        ("Gemfile", "Ruby"), ("composer.json", "PHP"),
        ("CMakeLists.txt", "C/C++"), ("Package.swift", "Swift"),
    ]
    for fname, lang in simple:
        if (directory / fname).is_file():
            return {"language": lang, "framework": "", "key_dependencies": []}
    return {"language": "", "framework": "", "key_dependencies": []}


def _python_deps(directory: Path) -> List[str]:
    """Extract dependency names from requirements.txt / pyproject.toml.

    Deliberately regex/line based (no tomllib) so it works on Python 3.9."""
    names: List[str] = []
    seen = set()

    def add(raw: str) -> None:
        # Strip version specifiers, extras, and comments: 'Django>=4 ; x' -> 'django'.
        name = re.split(r"[<>=!~;\[ ]", raw.strip(), maxsplit=1)[0].strip().lower()
        if name and not name.startswith("#") and name not in seen:
            seen.add(name)
            names.append(name)

    req = directory / "requirements.txt"
    if req.is_file():
        for line in _read_text(req).splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-")):
                add(line)

    pyproject = directory / "pyproject.toml"
    if pyproject.is_file() and len(names) < _MAX_DEPS:
        text = _read_text(pyproject)
        # PEP 621: only the `dependencies = [ ... ]` array, so we don't pick up
        # the project name, version, classifiers, or build-system entries.
        m = re.search(r"(?ms)^\s*dependencies\s*=\s*\[(.*?)\]", text)
        if m:
            for quoted in re.findall(r'["\']([^"\']+)["\']', m.group(1)):
                add(quoted)
        # Poetry: [tool.poetry.dependencies] table of `name = "^1.2"` lines.
        pm = re.search(r"(?ms)\[tool\.poetry\.dependencies\](.*?)(?:^\[|\Z)", text)
        if pm:
            for line in pm.group(1).splitlines():
                key = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*=", line)
                if key and key.group(1).lower() != "python":
                    add(key.group(1))
    return names[:_MAX_DEPS]


def _git_info(directory: Path) -> Dict[str, Any]:
    """Current branch, last commit subject, and modified files (best-effort)."""
    info = {"git_branch": "", "last_commit_summary": "", "modified_files": []}
    if not (directory / ".git").exists():
        return info

    def git(*args: str) -> str:
        # Return raw stdout (no global strip): the leading space in a
        # `git status --porcelain` line is significant (it's a status column).
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(directory), capture_output=True,
                text=True, timeout=5,
            )
            return out.stdout if out.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    info["git_branch"] = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    info["last_commit_summary"] = git("log", "-1", "--pretty=%s").strip()[:_COMMIT_MAX]
    status = git("status", "--porcelain")
    if status.strip():
        files = []
        for line in status.splitlines():
            if not line.strip():
                continue
            # Porcelain v1: two status columns + a space, then the path.
            path = line[3:].strip()
            if path:
                files.append(path)
            if len(files) >= _MAX_MODIFIED:
                break
        info["modified_files"] = files
    return info


def _readme_summary(directory: Path) -> str:
    """First real sentence of README.md, if present (regex, offline)."""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        readme = directory / name
        if readme.is_file():
            text = _read_text(readme, limit=8000)
            for raw in text.splitlines():
                line = raw.strip()
                # Skip headings, badges, blank lines, and HTML comments.
                if not line or line.startswith(("#", "!", "<", "[![", "---", "===")):
                    continue
                # First sentence (up to a period) or the whole line, clipped.
                sentence = re.split(r"(?<=[.!?])\s", line, maxsplit=1)[0].strip()
                summary = sentence or line
                return summary[:_README_MAX]
            break
    return ""


def generate_project_card(directory: str) -> Dict[str, Any]:
    """Scan ``directory`` and return a compact Project Card dict.

    Always returns every key in ``_CARD_KEYS``; missing/unreadable pieces are
    left empty. Never raises - unusable input yields a near-empty card.
    """
    if not directory or not os.path.isdir(directory):
        return _empty_card()
    path = Path(os.path.abspath(directory))
    card = _empty_card(name=path.name or str(path))
    try:
        card["folder_structure"] = _list_top_level(path)
        card.update(_detect_stack(path))
        card.update(_git_info(path))
        card["readme_summary"] = _readme_summary(path)
    except Exception:  # noqa: BLE001 - a primer must never break a request
        pass
    return card


def format_card(card: Dict[str, Any]) -> str:
    """Render a Project Card as a labeled, compact system-prompt block."""
    compact = {k: card.get(k) for k in _CARD_KEYS if card.get(k)}
    payload = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if len(payload) > _CARD_CHAR_CAP:
        payload = payload[:_CARD_CHAR_CAP] + "…"
    return (
        "[TOKENSNAP PROJECT PRIMER]\n"
        "A compact overview of the project you're working in, so you understand "
        "the codebase from the start. This is auto-generated context, not a user "
        "message:\n" + payload
    )


def save_last_card(card: Dict[str, Any]) -> None:
    """Persist the most recent card so the dashboard process can show it."""
    try:
        config_mod.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LAST_CARD_FILE, "w", encoding="utf-8") as f:
            json.dump(card, f)
    except OSError:
        pass


def load_last_card() -> Optional[Dict[str, Any]]:
    """The last generated card, or None if none has been generated/readable."""
    try:
        with open(LAST_CARD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def prime_for_session(
    session_id: str, directory: str, cfg: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Return the Project Card to inject, or None to skip.

    Returns the card only on the **first** request of a session (so the proxy
    injects it once and scans the disk once); later requests of the same
    session return None. Also returns None when ``directory`` isn't a real
    directory (e.g. an untagged 'unknown' session).
    """
    if not directory or not os.path.isdir(directory):
        return None
    if session_id in _primed_cards:
        return None
    card = generate_project_card(directory)
    _primed_cards[session_id] = card
    save_last_card(card)
    return card


def reset_cache() -> None:
    """Forget which sessions have been primed (used by tests)."""
    _primed_cards.clear()
