"""History compression: replace old conversation turns with a compact
"Memory Card" — a JSON block capturing the task, files touched, decisions
made, and errors resolved — while keeping the last N exchanges verbatim.

Extraction is rule-based (regex) by default: pure and offline-testable.
When a config dict is passed via `llm_cfg` and a local Ollama server is
available, the regex card is upgraded with an LLM-written summary (see
tokensnap.ollama); regex remains the fallback on any failure.
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


def build_memory_card(
    messages: List[Dict[str, Any]],
    llm_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract key facts from a list of messages into a compact dict.

    With `llm_cfg` (the Tokensnap config), a local Ollama model is asked
    for a better summary; its output is merged over the regex card (regex
    stays the baseline for file paths, which it extracts reliably).
    """
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

    card = {
        "task": task,
        "files_modified": files[:_MAX_FILES],
        "decisions": decisions[:_MAX_ITEMS],
        "errors_resolved": errors_resolved[:_MAX_ITEMS],
        "messages_summarized": len(messages),
    }

    if llm_cfg is not None:
        from tokensnap import ollama

        llm = ollama.try_generate_card(messages, llm_cfg)
        if llm:
            if llm["task"]:
                card["task"] = llm["task"]
            if llm["decisions"]:
                card["decisions"] = llm["decisions"]
            if llm["errors_resolved"]:
                card["errors_resolved"] = llm["errors_resolved"]
            # Union file lists: regex is reliable, the LLM may add context
            merged = list(card["files_modified"])
            for path in llm["files_modified"]:
                if path not in merged:
                    merged.append(path)
            card["files_modified"] = merged[:_MAX_FILES]
            card["generator"] = "ollama:%s" % llm_cfg.get("ollama_model", "?")

    return card


# Converted tool outputs at the cut boundary are truncated to this length
_ORPHAN_RESULT_MAX_CHARS = 2000


def _find_cut_index(messages: List[Dict[str, Any]], keep_last_n: int) -> int:
    """Largest safe index to cut the history at, keeping >= keep_last_n
    exchanges (2*N messages). The cut must land on a `user` message (the
    API requires histories to start with one). A tool_result-only user
    message is acceptable — the caller converts its orphaned tool_results
    to plain text — which matters for agentic transcripts (Claude Code)
    where nearly every user message is a tool_result."""
    target = len(messages) - keep_last_n * 2
    for i in range(min(target, len(messages) - 1), 0, -1):
        if messages[i].get("role") == "user":
            return i
    return 0


def _orphaned_results_to_text(message: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite a user message whose tool_result blocks would be orphaned by
    the cut (their tool_use turn was compressed away) into plain text, so
    the truncated history is valid for the API."""
    if not is_tool_result_only(message):
        return message
    parts = []
    for block in message.get("content", []):
        text = "\n".join(_iter_text_blocks(block.get("content")))
        if len(text) > _ORPHAN_RESULT_MAX_CHARS:
            text = text[:_ORPHAN_RESULT_MAX_CHARS] + "\n[tokensnap: tool output truncated]"
        parts.append(text)
    joined = "\n".join(p for p in parts if p) or "(no output)"
    return {
        "role": "user",
        "content": "[Output of a tool call from the compressed history]\n" + joined,
    }


def _iter_text_blocks(content: Any) -> List[str]:
    if isinstance(content, str):
        return [content] if content else []
    parts: List[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return parts


def compress_messages(
    messages: List[Dict[str, Any]],
    keep_last_n: int = 3,
    min_messages: int = 8,
    llm_cfg: Optional[Dict[str, Any]] = None,
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
    tail = [_orphaned_results_to_text(tail[0])] + tail[1:]
    card = build_memory_card(head, llm_cfg=llm_cfg)
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
