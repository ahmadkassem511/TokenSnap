"""History compression: replace old conversation turns with a compact
"Memory Card" — a JSON block capturing the task, files touched, decisions
made, and errors resolved — while keeping the last N exchanges verbatim.

Rule-based (regex) extraction only; no LLM calls. Pure and offline-testable.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from tokensnap import token_counter
from tokensnap.utils import is_tool_result_only, message_text

# File paths worth remembering: something.ext, optionally with directories
_FILE_RE = re.compile(
    r"\b[\w][\w.\-]*(?:[/\\][\w][\w.\-]*)*\."
    r"(?:py|pyi|js|jsx|ts|tsx|json|md|txt|yaml|yml|toml|ini|cfg|html|css|scss"
    r"|sh|bat|ps1|go|rs|java|kt|c|cc|cpp|h|hpp|cs|rb|php|sql|proto|xml|lock)\b"
)
# Lines that record a choice or direction
_DECISION_RE = re.compile(
    r"(?i)\b(decision:|we will|we'll|let's use|i'll use|going with|chose|"
    r"decided to|instead of using|switching to|opted for)\b"
)
# Error mentions and their resolutions
_ERROR_RE = re.compile(r"(?i)\b(error|exception|traceback|failed|failure)\b")
_RESOLVED_RE = re.compile(r"(?i)\b(fixed|resolved|solved|solution|works now|passing now)\b")

_MAX_LINE = 200          # chars kept per extracted line
_MAX_ITEMS = 10          # cap per card section
_MAX_FILES = 25
_TASK_CHARS = 400

MEMORY_CARD_HEADER = "[TOKENSNAP MEMORY CARD]"


def _clip(line: str) -> str:
    line = line.strip()
    return line if len(line) <= _MAX_LINE else line[: _MAX_LINE - 1] + "…"


def build_memory_card(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract key facts from a list of messages into a compact dict."""
    files: List[str] = []
    decisions: List[str] = []
    errors_resolved: List[str] = []
    task = ""

    for msg in messages:
        text = message_text(msg)
        if not text:
            continue
        if not task and msg.get("role") == "user":
            task = _clip(text.split("\n", 1)[0])[:_TASK_CHARS]

        for m in _FILE_RE.finditer(text):
            path = m.group(0)
            if path not in files:
                files.append(path)

        lines = text.split("\n")
        pending_error: Optional[str] = None
        pending_age = 0
        for line in lines:
            if _DECISION_RE.search(line):
                clipped = _clip(line)
                if clipped and clipped not in decisions:
                    decisions.append(clipped)
            if _ERROR_RE.search(line):
                pending_error = _clip(line)
                pending_age = 0
            elif pending_error is not None:
                pending_age += 1
                if _RESOLVED_RE.search(line):
                    entry = "%s -> %s" % (pending_error, _clip(line))
                    if entry not in errors_resolved:
                        errors_resolved.append(entry)
                    pending_error = None
                elif pending_age > 5:
                    pending_error = None

    return {
        "task": task,
        "files_modified": files[:_MAX_FILES],
        "decisions": decisions[:_MAX_ITEMS],
        "errors_resolved": errors_resolved[:_MAX_ITEMS],
        "messages_summarized": len(messages),
    }


def _find_cut_index(messages: List[Dict[str, Any]], keep_last_n: int) -> int:
    """Largest safe index to cut the history at, keeping >= keep_last_n
    exchanges (2*N messages). The first kept message must be a `user`
    message that is not purely tool_result blocks, so the truncated
    history never starts mid tool-call or with an assistant turn."""
    target = len(messages) - keep_last_n * 2
    for i in range(min(target, len(messages) - 1), 0, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not is_tool_result_only(msg):
            return i
    return 0


def compress_messages(
    messages: List[Dict[str, Any]],
    keep_last_n: int = 3,
    min_messages: int = 8,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Compress a long history into (memory_card_text, trimmed_messages).

    Returns (None, messages) unchanged when the history is short or no
    safe cut point exists. The memory card text is meant to be appended
    to the request's `system` prompt by the caller.
    """
    if not messages or len(messages) <= min_messages:
        return None, messages

    cut = _find_cut_index(messages, keep_last_n)
    if cut <= 0:
        return None, messages

    head, tail = messages[:cut], messages[cut:]
    card = build_memory_card(head)
    card["original_tokens"] = token_counter.count_message_tokens(head)

    injection = (
        "%s\n"
        "The first %d messages of this conversation were compressed by "
        "Tokensnap to save tokens. Key facts from the omitted history:\n"
        "%s\n"
        "Continue the task using this context; the most recent messages "
        "follow in full."
    ) % (
        MEMORY_CARD_HEADER,
        len(head),
        json.dumps(card, ensure_ascii=False, separators=(",", ":")),
    )
    return injection, tail
