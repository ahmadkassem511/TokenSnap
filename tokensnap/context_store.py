"""Local mirror of the full conversation, for the Differential Context Engine.

Where ``history.py`` records one row per *optimized request* (for the
dashboard's charts), this module records one row per *message* — a complete,
durable copy of every user/assistant turn the proxy sees. It is the "external
memory" that lets TokenSnap send the model only a compact Context Tree (an
index of important past events) plus the last couple of exchanges, and then
serve the full text of any event back on demand via the ``fetch_context`` tool.

Design notes (shared with ``history.py``):
  * A small SQLite database at ``~/.tokensnap/context_store.db``; stdlib only,
    so TokenSnap gains no new dependency.
  * Writes are **best-effort and never raise** — the proxy calls
    :func:`store_message` on its hot path, so a locked or unwritable database
    must degrade to "no external memory", never a failed request.
  * The proxy process writes while a *separate* dashboard process reads, so we
    open a short-lived connection per call and enable WAL for concurrent
    cross-process access.

Idempotency, because the Anthropic API is stateless:
  Claude Code resends the *entire* conversation on every request, so the proxy
  sees the same early messages over and over as the conversation grows.
  :func:`store_message` therefore **upserts** on the natural key
  ``(session_id, message_index)`` rather than blindly inserting — that keeps
  exactly one row per position, and crucially keeps each event's ``id`` stable
  across re-sends, because ``fetch_context`` references events by that ``id``.
"""

import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

from tokensnap import config as config_mod

log = logging.getLogger("tokensnap.context_store")

# Module attribute so tests can redirect it (monkeypatch context_store.DB_FILE).
DB_FILE = config_mod.CONFIG_DIR / "context_store.db"

# Event categories used to decide what belongs in the Context Tree. Anything
# not "other" is considered important enough to surface (see get_recent_tree).
# "request" marks a genuine user instruction that didn't match a more
# specific category - it must still never be dropped as noise the way
# assistant chatter/tool-result "other" events are.
EVENT_TYPES = ("decision", "error", "file_modification", "clarification", "request", "other")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT    NOT NULL,
    message_index  INTEGER NOT NULL,
    role           TEXT    NOT NULL,
    content        TEXT    NOT NULL,
    summary        TEXT    NOT NULL DEFAULT '',
    event_type     TEXT    NOT NULL DEFAULT 'other',
    created_at     REAL    NOT NULL,
    UNIQUE(session_id, message_index)
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id, message_index);
"""


def _connect() -> sqlite3.Connection:
    """Open a short-lived connection to the store, creating the file/dir.

    WAL + a small busy timeout let the proxy process write while a dashboard
    process reads the same file without either one erroring on a lock.
    """
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    """Create the table if it doesn't exist yet. Safe to call repeatedly."""
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def store_message(
    session_id: str,
    index: int,
    role: str,
    content: str,
    summary: str = "",
    event_type: str = "other",
) -> Optional[int]:
    """Record (or update in place) one message of a conversation.

    Upserts on ``(session_id, message_index)`` so re-sends of the same
    conversation prefix don't duplicate rows and each event keeps a stable
    ``id``. Returns the event's ``id`` (or ``None`` if the write was skipped
    because the DB was unavailable). Best-effort: never raises.
    """
    if event_type not in EVENT_TYPES:
        event_type = "other"
    try:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)  # cheap CREATE IF NOT EXISTS; self-heals a fresh DB
            conn.execute(
                "INSERT INTO events (session_id, message_index, role, content, "
                "summary, event_type, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id, message_index) DO UPDATE SET "
                "role=excluded.role, content=excluded.content, "
                "summary=excluded.summary, event_type=excluded.event_type",
                (str(session_id), int(index), str(role), str(content),
                 str(summary), event_type, time.time()),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id FROM events WHERE session_id = ? AND message_index = ?",
                (str(session_id), int(index)),
            ).fetchone()
            return int(row["id"]) if row else None
        finally:
            conn.close()
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        # The proxy must survive a bad DB (locked, unwritable dir, disk full) -
        # degrade to "no external memory" rather than break the request.
        log.debug("context_store.store_message skipped (%s)", exc)
        return None


def get_event_by_id(event_id: Any) -> Optional[Dict[str, Any]]:
    """Return the full stored event for ``event_id`` (all columns), or None.

    Accepts an int or a numeric string (the Context Tree exposes ids as
    strings), so ``fetch_context`` can pass back whatever the model sends.
    Never raises — an unreadable DB or non-numeric id yields ``None``.
    """
    try:
        eid = int(event_id)
    except (ValueError, TypeError):
        return None
    try:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            row = conn.execute(
                "SELECT id, session_id, message_index, role, content, summary, "
                "event_type, created_at FROM events WHERE id = ?",
                (eid,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        log.debug("context_store.get_event_by_id failed (%s)", exc)
        return None


def get_recent_tree(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return the Context Tree: the most recent *important* events.

    "Important" means ``event_type != 'other'``. Each entry is a compact
    ``{"id", "summary", "type"}`` dict (``id`` as a string for the model to
    quote back to ``fetch_context``). The ``limit`` most recent qualifying
    events are returned oldest→newest, so the tree reads chronologically.
    Never raises — an unreadable DB yields ``[]``.
    """
    try:
        lim = max(1, int(limit))
    except (ValueError, TypeError):
        lim = 20
    try:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            # Take the newest `lim` important events, then flip to chronological.
            rows = conn.execute(
                "SELECT id, summary, event_type FROM events "
                "WHERE session_id = ? AND event_type != 'other' "
                "ORDER BY message_index DESC LIMIT ?",
                (str(session_id), lim),
            ).fetchall()
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        log.debug("context_store.get_recent_tree failed (%s)", exc)
        return []
    return [
        {"id": str(r["id"]), "summary": r["summary"], "type": r["event_type"]}
        for r in reversed(rows)
    ]


def get_first_event(session_id: str) -> Optional[Dict[str, Any]]:
    """Return the earliest-indexed stored event for ``session_id`` - almost
    always the conversation's opening message, i.e. the original task -
    regardless of its ``event_type``.

    Unlike :func:`get_recent_tree`, this never filters by type: the Context
    Tree only surfaces 'important' (non-'other') events, which silently drops
    a plainly-phrased request ("run this tool") that doesn't happen to match
    the decision/error/file-modification/clarification heuristics. The caller
    uses this to guarantee the original task is never entirely lost from the
    tree, the same way the classic Memory Card always captures ``task``
    regardless of phrasing. None if nothing is stored. Never raises.
    """
    try:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            row = conn.execute(
                "SELECT id, summary, event_type FROM events "
                "WHERE session_id = ? ORDER BY message_index ASC LIMIT 1",
                (str(session_id),),
            ).fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        log.debug("context_store.get_first_event failed (%s)", exc)
        return None
    if not row:
        return None
    return {"id": str(row["id"]), "summary": row["summary"], "type": row["event_type"]}


def get_full_history(
    session_id: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Return all stored events for ``session_id``, ordered by message index.

    Includes ``'other'`` events (unlike the Context Tree). With ``limit`` set,
    returns the most recent ``limit`` events, still ordered oldest→newest.
    Never raises — an unreadable DB yields ``[]``.
    """
    try:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            if limit is None:
                rows = conn.execute(
                    "SELECT id, session_id, message_index, role, content, summary, "
                    "event_type, created_at FROM events WHERE session_id = ? "
                    "ORDER BY message_index ASC",
                    (str(session_id),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, session_id, message_index, role, content, summary, "
                    "event_type, created_at FROM events WHERE session_id = ? "
                    "ORDER BY message_index DESC LIMIT ?",
                    (str(session_id), max(0, int(limit))),
                ).fetchall()
                rows = list(reversed(rows))
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        log.debug("context_store.get_full_history failed (%s)", exc)
        return []
    return [dict(r) for r in rows]


def event_count(session_id: Optional[str] = None) -> int:
    """Number of stored events, overall or for one session. Zero on any error.

    Used by the dashboard (Phase 3) to show how much external memory exists.
    """
    try:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            if session_id is None:
                row = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM events WHERE session_id = ?",
                    (str(session_id),),
                ).fetchone()
            return int(row["n"]) if row else 0
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return 0
