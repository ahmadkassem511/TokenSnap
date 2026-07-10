"""Project Cortex - Session Bridge: seamless continuity across sessions.

When a session ends, TokenSnap distils it into a compact summary saved under
``<project>/.tokensnap/sessions/``. When a new session for the same project
starts, the most recent summary is injected as an optional "Session Bridge"
system block, so the new session resumes where the last one left off - even
after the proxy restarts or Claude Code exits.

Because a summary is just the existing Memory Card over the whole conversation,
this also works *across tools*: a conversation pasted from Claude Desktop (or
anywhere) can be imported with :func:`import_external` and becomes a bridge the
next session picks up.

Pure, offline, and best-effort: reuses :func:`tokensnap.compressor.build_memory_card`
for the actual summarisation, and never raises on the proxy's hot path.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tokensnap import compressor, project_dna

SESSIONS_DIRNAME = "sessions"

# Keep the last N session summaries per project; older ones are pruned.
_MAX_SESSIONS = 20
# Don't bother bridging a session with almost nothing in it.
_MIN_MESSAGES = 4
_BRIDGE_CHAR_CAP = 2000

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]")


def sessions_dir(project_dir: str) -> Path:
    return Path(project_dir) / project_dna.DNA_DIRNAME / SESSIONS_DIRNAME


def _summarize(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Regex-only (llm_cfg=None): session-end work must be fast and offline.
    card = compressor.build_memory_card(messages, llm_cfg=None)
    # Enrich with the DNA-side extractors so bridges capture the same labelled
    # decisions and cross-message error->fix pairs the DNA does.
    for text in project_dna._extract_labelled_decisions(messages):
        if text not in card["decisions"]:
            card["decisions"].append(text)
    resolved = project_dna._extract_resolved_issues(messages)
    if resolved:
        merged = list(card.get("errors_resolved", []))
        for text in resolved:
            if text not in merged:
                merged.append(text)
        card["errors_resolved"] = merged[:compressor._MAX_ITEMS]
    card["decisions"] = card["decisions"][:compressor._MAX_ITEMS]
    return card


def _prune(directory: Path, keep: Optional[int] = None) -> None:
    # Read the module global at call time (not as a default) so tests can tune it.
    keep = _MAX_SESSIONS if keep is None else keep
    try:
        files = sorted(directory.glob("*.json"))
        for stale in files[:-keep]:
            stale.unlink()
    except OSError:
        pass


def save_session(
    project_dir: str,
    session_id: str,
    messages: List[Dict[str, Any]],
    source: str = "claude-code",
    min_messages: int = _MIN_MESSAGES,
) -> Optional[Path]:
    """Summarise a session and persist it under the project's sessions dir.

    Returns the written path, or None if the project dir is invalid or the
    session is too small to be worth bridging. Never raises."""
    if not project_dir or not Path(project_dir).is_dir():
        return None
    if not messages or len(messages) < min_messages:
        return None
    try:
        now = time.time()
        record = {
            "session_id": session_id,
            "saved_at": now,
            "source": source,
            "n_messages": len(messages),
            "summary": _summarize(messages),
        }
        directory = sessions_dir(project_dir)
        directory.mkdir(parents=True, exist_ok=True)
        safe_id = _SAFE_ID_RE.sub("_", str(session_id))[:24] or "session"
        path = directory / ("%d-%s.json" % (int(now * 1000), safe_id))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        _prune(directory)
        return path
    except OSError:
        return None


def list_sessions(project_dir: str) -> List[Dict[str, Any]]:
    """All saved session summaries for the project, newest first. Never raises;
    unreadable/corrupt files are skipped."""
    directory = sessions_dir(project_dir)
    out: List[Dict[str, Any]] = []
    try:
        paths = list(directory.glob("*.json"))
    except OSError:
        return []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                out.append(data)
        except (OSError, ValueError):
            continue
    out.sort(key=lambda r: float(r.get("saved_at") or 0), reverse=True)
    return out


def latest_session(
    project_dir: str, exclude_session_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """The most recent saved session for the project, skipping
    ``exclude_session_id`` (so a session never bridges to itself)."""
    for record in list_sessions(project_dir):
        if exclude_session_id and record.get("session_id") == exclude_session_id:
            continue
        return record
    return None


def import_external(
    project_dir: str, text: str, label: str = "external"
) -> Optional[Path]:
    """Import a pasted conversation (e.g. from Claude Desktop) as a bridge.

    The raw text is summarised the same way a live session is, and saved with
    a distinct source so the next session can pick it up. Returns the path or
    None. Never raises."""
    if not project_dir or not Path(project_dir).is_dir():
        return None
    if not (text or "").strip():
        return None
    pseudo = [{"role": "user", "content": text}]
    return save_session(
        project_dir,
        session_id="import-%d" % int(time.time()),
        messages=pseudo,
        source=label,
        min_messages=1,  # a pasted transcript is deliberate; always keep it
    )


def format_bridge(record: Dict[str, Any]) -> str:
    """Render a saved session summary as a "Session Bridge" system block.

    Returns "" when the record has nothing useful. Bounded in size."""
    summary = (record or {}).get("summary") or {}
    payload: Dict[str, Any] = {}
    for key in ("task", "decisions", "errors_resolved", "files_modified"):
        if summary.get(key):
            payload[key] = summary[key]
    if not payload:
        return ""

    when = record.get("saved_at")
    ago = ""
    if when:
        mins = max(0, int((time.time() - float(when)) / 60))
        ago = (" from ~%d min ago" % mins) if mins < 240 else " from a previous day"
    source = record.get("source", "claude-code")
    origin = " (imported from %s)" % source if source not in ("claude-code",) else ""

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(body) > _BRIDGE_CHAR_CAP:
        body = body[:_BRIDGE_CHAR_CAP] + "…"
    return (
        "[TOKENSNAP SESSION BRIDGE]\n"
        "A summary of the previous session on this project%s%s, so you can "
        "continue seamlessly. This is context from earlier work, not a new "
        "user request:\n" % (ago, origin) + body
    )
