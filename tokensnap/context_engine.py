"""Differential Context Engine: turn a full conversation into a compact
Context Tree plus the last few exchanges, backed by the local Context Store.

This module is the glue between :mod:`tokensnap.context_store` (the durable
mirror of every message) and :mod:`tokensnap.proxy` (which rewrites outgoing
requests). It is deliberately pure and offline-testable: every function takes
plain dicts/strings and touches only the Context Store (which tests redirect
to a throwaway DB).

The proxy calls, per request:

    session_id = derive_session_id(system, messages)
    ingest(session_id, messages)                 # mirror everything
    new_messages, new_system, reconstructed, n_omitted = reconstruct(...)
    body["tools"] = merge_fetch_context_tool(body.get("tools"))

`reconstruct` reuses the compressor's proven cut-point and tool-pairing
helpers, so the truncated history it emits is always valid for the API.
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from tokensnap import compressor, context_store
from tokensnap.utils import append_to_system, message_text, system_to_parts

# Number of recent user/assistant exchanges (a pair == 2 messages) kept
# verbatim ahead of the Context Tree. The spec fixes this at "the last 2".
KEEP_EXCHANGES = 2

# One-line event summary length cap (keeps the Context Tree small).
_SUMMARY_MAX = 160

# Assistant/user phrasings that signal a file-modification event. `error` and
# `decision` reuse the compressor's existing regexes so the two modules stay
# consistent about what those words mean.
_FILE_MOD_RE = re.compile(
    r"(?i)\b(i'?ll\s+(?:modify|change|edit|update|add|create|write|refactor|rename)"
    r"|let'?s\s+(?:change|modify|update|add)"
    r"|(?:modifying|editing|updating|creating|writing|refactoring)\s+"
    r"(?:the\s+|a\s+)?[\w./\\-]+\.\w+"
    r"|applying\s+the\s+(?:edit|change)|made\s+the\s+(?:change|edit))\b"
)
_CLARIFY_RE = re.compile(
    r"(?i)\b(to\s+clarify|just\s+to\s+clarify|do\s+you\s+mean|did\s+you\s+mean"
    r"|could\s+you\s+(?:clarify|specify|confirm)|to\s+confirm)\b"
)

# The tool the model uses to pull omitted detail back from external memory.
# The proxy answers these calls itself (Phase 3); until then the tool is only
# advertised when there is actually omitted history to fetch.
FETCH_CONTEXT_TOOL: Dict[str, Any] = {
    "name": "fetch_context",
    "description": (
        "Retrieve the full text of past conversation events from TokenSnap's "
        "external memory. Pass the event IDs (as shown in the Context Tree) you "
        "need details about; returns the full content for those events. Use this "
        "whenever the Context Tree references something you must see in full to "
        "continue the task."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "event_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Event IDs from the Context Tree to fetch in full.",
            }
        },
        "required": ["event_ids"],
    },
}


def derive_session_id(system: Any, messages: List[Dict[str, Any]]) -> str:
    """A stable id for this conversation, derived from its immutable prefix.

    The Anthropic API is stateless, so the same conversation grows by appending
    messages while its system prompt and first user message stay fixed. Hashing
    those yields an id that is stable across every request of one conversation
    yet distinct between conversations.
    """
    parts = system_to_parts(system)
    seed = parts[0] if parts else ""
    for m in messages:
        if m.get("role") == "user":
            seed += "\x00" + message_text(m)
            break
    return hashlib.sha1(seed.encode("utf-8", "replace")).hexdigest()[:16]


def event_type_for(text: str) -> str:
    """Classify a message into a Context-Tree event type by simple heuristics.

    Order is by importance: an error mention wins over a file edit, which wins
    over a decision, which wins over a clarification; anything else is 'other'
    (and thus excluded from the tree)."""
    if compressor._ERROR_RE.search(text):
        return "error"
    if _FILE_MOD_RE.search(text):
        return "file_modification"
    if compressor._DECISION_RE.search(text):
        return "decision"
    if _CLARIFY_RE.search(text):
        return "clarification"
    return "other"


def one_line_summary(text: str, maxlen: int = _SUMMARY_MAX) -> str:
    """First non-empty line of a message, clipped — the tree entry's summary."""
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line if len(line) <= maxlen else line[: maxlen - 1] + "…"
    return ""


def ingest(session_id: str, messages: List[Dict[str, Any]]) -> int:
    """Mirror every message into the Context Store, keyed by its position.

    Stores the message's full (already-cleaned) text so ``fetch_context`` can
    return real detail later. Messages with no extractable text (pure tool_use
    or image turns) are skipped — there's nothing to recall. Positional index
    is the true array position, so re-sends upsert in place. Returns the count
    stored. Best-effort: store_message never raises.
    """
    stored = 0
    for i, m in enumerate(messages):
        text = message_text(m)
        if not text.strip():
            continue
        role = str(m.get("role", "user"))
        context_store.store_message(
            session_id, i, role, text, one_line_summary(text), event_type_for(text)
        )
        stored += 1
    return stored


def build_context_tree_block(tree: List[Dict[str, Any]], n_omitted: int) -> str:
    """The system-prompt block that replaces the omitted history: an index of
    important past events plus how to pull their full text back."""
    tree_json = json.dumps(tree, ensure_ascii=False, separators=(",", ":"))
    return (
        "[TOKENSNAP CONTEXT TREE]\n"
        "The earlier part of this conversation (%d message%s) is stored in "
        "TokenSnap's external memory to save tokens and is not included here in "
        "full. Below is an index of the important past events — the Context "
        "Tree — as a JSON array of {id, summary, type}:\n"
        "%s\n"
        "You have access to a `fetch_context` tool. Use it when you need details "
        "about past events referenced in the Context Tree: it accepts a list of "
        "event IDs and returns the full context for those events. The most "
        "recent messages follow below in full."
    ) % (n_omitted, "" if n_omitted == 1 else "s", tree_json)


def merge_fetch_context_tool(tools: Any) -> List[Dict[str, Any]]:
    """Return the request's ``tools`` list with fetch_context added once."""
    if isinstance(tools, list):
        if any(isinstance(t, dict) and t.get("name") == "fetch_context" for t in tools):
            return tools
        return tools + [FETCH_CONTEXT_TOOL]
    return [FETCH_CONTEXT_TOOL]


def reconstruct(
    messages: List[Dict[str, Any]],
    system: Any,
    session_id: str,
    tree_size: int = 20,
    min_messages: int = 8,
    selective: bool = True,
    keep_exchanges: int = KEEP_EXCHANGES,
) -> Tuple[List[Dict[str, Any]], Any, bool, int]:
    """Rebuild the outgoing message list as Context Tree + recent tail.

    Returns ``(new_messages, new_system, reconstructed, n_omitted)``. When the
    history is too short to have anything worth omitting, returns the messages
    and system unchanged with ``reconstructed=False`` (the caller then skips
    the fetch_context tool). Reuses ``compressor._find_cut_index`` /
    ``_orphaned_results_to_text`` so tool_use/tool_result pairing is preserved.
    ``keep_exchanges`` defaults to the module constant but can be overridden
    (e.g. Adaptive Transparency Mode's FULL tier keeps the last 5).
    """
    if len(messages) <= min_messages:
        return messages, system, False, 0

    cut = compressor._find_cut_index(messages, keep_exchanges)
    if cut <= 0:
        return messages, system, False, 0

    head, tail = messages[:cut], messages[cut:]
    # The tail's first message may carry tool_results whose tool_use turn is in
    # the omitted head; rewrite them to plain text so the history stays valid.
    tail = [compressor._orphaned_results_to_text(tail[0])] + tail[1:]
    if selective:
        tail = [compressor._compress_message_selectively(m) for m in tail]

    tree = context_store.get_recent_tree(session_id, tree_size)
    # Guarantee the original task is never entirely lost. get_recent_tree only
    # surfaces 'important' (non-'other') events, so a plainly-phrased request
    # ("run this tool") that doesn't match the decision/error/file-mod/
    # clarification heuristics would otherwise vanish with no trace once its
    # message falls into the omitted head - unlike the classic Memory Card,
    # which always captures `task` from the first message regardless of
    # phrasing. Prepend it (labeled distinctly) unless it's already present.
    task_event = context_store.get_first_event(session_id)
    if task_event and not any(e["id"] == task_event["id"] for e in tree):
        tree = [dict(task_event, type="task")] + tree
    block = build_context_tree_block(tree, n_omitted=len(head))
    new_system = append_to_system(system, block)
    return tail, new_system, True, len(head)
