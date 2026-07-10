"""History compression: replace old conversation turns with a compact
"Memory Card" — a JSON block capturing the task, files touched, decisions
made, and errors resolved — while keeping the last N exchanges verbatim.

Two independent knobs control how aggressive this is:

- `compressor_type` decides how the Memory Card for older history gets
  written: "regex" (fast, rule-based, offline), "openrouter" (a free
  hosted model writes a higher-quality card; regex is still the
  fallback on any failure), or "off" (no card, no truncation at all -
  the full history is kept, only per-message noise cleaning applies).
- `selective_compression` decides whether *every* message first goes
  through per-message noise reduction (see `build_compressed_context`)
  before the keep_messages cut, or whether the legacy uniform
  compression (`compress_messages`) is used instead.

Regex extraction is pure and offline-testable; OpenRouter calls (see
tokensnap.openrouter) run in the same worker thread as the rest of
request optimization, and never touch the Anthropic API key.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from tokensnap import cleaner, token_counter
from tokensnap.utils import is_tool_result_only, message_text, transform_message_text

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

# Adaptive compression - value weighting. Phrases that mark a message as
# high-value and worth preserving even as it ages ("important", explicit
# keep markers, etc.). Kept separate from _DECISION_RE (which is about choices).
_IMPORTANT_RE = re.compile(
    r"(?i)(\b(important|note|remember|caveat|gotcha|warning|must|todo)\b"
    r"|!important|@keep|@important)"
)


def _clip(line: str) -> str:
    line = line.strip()
    return line if len(line) <= _MAX_LINE else line[: _MAX_LINE - 1] + "…"


def message_weight(
    message: Dict[str, Any], index: int = 0, total: int = 1
) -> float:
    """A "how much is this worth keeping" score for adaptive compression.

    Combines four signals the spec calls for:

    * **Role** - assistant reasoning > user > tool result.
    * **Content** - decisions, explicit "important" markers, and code snippets
      raise the weight; log/terminal dumps lower it.
    * **Recency** - newer messages weigh more (older ones decay), via ``index``
      / ``total``.
    * **Marked important** - a message with an explicit importance marker never
      decays below a high floor, so crucial notes are never discarded.

    Pure and side-effect free; callers use it to decide what survives when a
    section must be capped. It never reorders the *sent* messages (that would
    break the API's tool_use/tool_result pairing) - only what gets summarised.
    """
    text = message_text(message)
    role = message.get("role")
    if role == "assistant":
        weight = 3.0
    elif is_tool_result_only(message):
        weight = 1.0
    elif role == "user":
        weight = 2.0
    else:
        weight = 1.5

    important = bool(_IMPORTANT_RE.search(text))
    if _DECISION_RE.search(text) or important:
        weight += 2.0
    if "```" in text:  # a code snippet is worth keeping
        weight += 1.0
    if _looks_like_terminal_dump(text):  # bulk log/trace noise is low value
        weight -= 1.5

    if total > 1:  # recency: 0 for the oldest, +1 for the newest
        weight += index / (total - 1)
    if important:  # explicit markers hold a high floor regardless of age
        weight = max(weight, 4.0)
    return max(0.1, weight)


def build_memory_card(
    messages: List[Dict[str, Any]],
    llm_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract key facts from a list of messages into a compact dict.

    With `llm_cfg` (the Tokensnap config) and `compressor_type=="openrouter"`,
    a free OpenRouter model is asked for a better summary; its output is
    merged over the regex card (regex stays the baseline for file paths,
    which it extracts reliably). Any failure - no key, network error,
    malformed output - silently keeps the regex card.
    """
    files: List[str] = []
    decisions: List[str] = []
    errors_resolved: List[str] = []
    # Value weight of the message each decision came from, so that when there
    # are more decisions than fit, the highest-weight ones are kept (adaptive
    # compression - crucial decisions are never crowded out by trivial ones).
    decision_weight: Dict[str, float] = {}
    task = ""
    total = len(messages)

    for i, msg in enumerate(messages):
        text = message_text(msg)
        if not text:
            continue
        weight = message_weight(msg, i, total)
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
                if clipped:
                    decision_weight[clipped] = max(decision_weight.get(clipped, 0), weight)
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

    # When decisions overflow the cap, keep the highest-weight ones (then
    # restore chronological order among the survivors). Under the cap, order is
    # unchanged - so this only ever changes *which* decisions are dropped.
    if len(decisions) > _MAX_ITEMS:
        keep = set(
            sorted(decisions, key=lambda d: decision_weight.get(d, 0.0), reverse=True)[
                :_MAX_ITEMS
            ]
        )
        kept_decisions = [d for d in decisions if d in keep]
    else:
        kept_decisions = decisions[:_MAX_ITEMS]

    card = {
        "task": task,
        "files_modified": files[:_MAX_FILES],
        "decisions": kept_decisions,
        "errors_resolved": errors_resolved[:_MAX_ITEMS],
        "messages_summarized": len(messages),
    }

    if llm_cfg is not None:
        from tokensnap import openrouter

        llm = openrouter.try_generate_card(messages, llm_cfg)
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
            card["generator"] = "openrouter:%s" % llm_cfg.get("openrouter_model", "?")

    return card


def memory_card_status(cfg: Dict[str, Any]) -> str:
    """One-line human status of the Memory Card generator, for the proxy
    startup log and the stats file (surfaced in `tokensnap status`/`monitor`)."""
    compressor_type = str(cfg.get("compressor_type", "regex")).lower()
    if compressor_type == "off":
        return "off (history kept verbatim up to keep_messages)"
    if compressor_type == "openrouter":
        from tokensnap import openrouter

        if not str(cfg.get("openrouter_api_key", "")).strip():
            return (
                "regex (compressor_type=openrouter but no openrouter_api_key is "
                'set - get a free key at https://openrouter.ai/keys, then '
                "`tokensnap config set openrouter_api_key <key>`)"
            )
        return "openrouter:%s" % cfg.get("openrouter_model", openrouter.DEFAULT_MODEL)
    return "regex"


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
    keep_last_n: int = 10,
    min_messages: int = 8,
    llm_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Compress a long history into (memory_card_text, trimmed_messages).

    Returns (None, messages) unchanged when the history is short or no
    safe cut point exists. The memory card text is meant to be appended
    to the request's `system` prompt by the caller. `keep_last_n` is the
    number of recent exchanges to keep verbatim - the proxy sources this
    from the `keep_messages` config value (see `tokensnap preset`).

    `llm_cfg["compressor_type"] == "off"` disables the Memory Card and the
    truncation itself: the full history is returned unchanged.
    """
    if not messages or len(messages) <= min_messages:
        return None, messages
    if llm_cfg is not None and str(llm_cfg.get("compressor_type", "")).lower() == "off":
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


# ---------------------------------------------------------------------------
# Selective per-message compression.
#
# Philosophy: eliminate only noise, never substance. Assistant messages -
# Claude's own reasoning and responses - are never touched, since preserving
# that reasoning quality is the entire point. User messages are left intact
# unless they contain a large terminal/log dump, in which case only that
# dump is reduced to its error/warning/status lines. Tool results are
# compressed the same way, more aggressively, since they're almost always
# machine-generated noise once the outcome is known.
# ---------------------------------------------------------------------------

# A user or tool_result text block bigger than this is a candidate for
# terminal-dump extraction (spec: ">500 tokens").
_TERMINAL_DUMP_MIN_TOKENS = 500
# Tool results are compressed more eagerly - most of their bulk is noise.
_TOOL_RESULT_MIN_TOKENS = 75
# How many of the final non-empty lines always survive, as the "status".
_STATUS_LINES_KEPT = 2
# Cap on kept error/warning lines - if a dump is nothing but distinct-looking
# errors (e.g. "error at line N" for 200 different N), keeping "all of them"
# because none happen to repeat verbatim isn't noise elimination. First+last
# few are almost always enough to show the pattern.
_MAX_SIGNAL_LINES = 20

# Shell-prompt-ish and error/warning line markers used to recognize a
# terminal dump versus ordinary prose.
_PROMPT_RE = re.compile(
    r"(?m)^(?:\$\s|>\s|#\s|PS [A-Za-z]:\\.*>|[\w.\-]+@[\w.\-]+:.*[$#]\s)"
)
_SIGNAL_RE = re.compile(
    r"(?i)\b(error|exception|traceback|warning|warn|failed|failure|fatal|"
    r"npm (?:warn|err!))\b"
)
_FAIL_RE = re.compile(r"(?i)\b(fail(?:ed|ure)?|error|exception)\b")


def _looks_like_terminal_dump(text: str) -> bool:
    """Heuristic: long text that reads like shell/CLI output rather than
    prose - either it has shell-prompt-style lines, or it's dense with
    error/warning signal words."""
    if token_counter.count_tokens(text) <= _TERMINAL_DUMP_MIN_TOKENS:
        return False
    return bool(_PROMPT_RE.search(text) or _SIGNAL_RE.search(text))


def _extract_signal_lines(text: str) -> str:
    """Clean ANSI/progress-bar/dedup noise, then keep only lines carrying
    real information: errors, warnings, and the final status line(s).
    Everything else becomes a one-line summary of what was omitted.
    """
    cleaned, _ = cleaner.clean_text(text)
    non_empty = [l for l in cleaned.split("\n") if l.strip()]
    if not non_empty:
        return "[tokensnap: empty output]"

    # Index-based selection (not value-based) so a line that repeats
    # verbatim at several positions is judged, and counted, independently
    # each time - matching by line content would either drop all but one
    # copy or resurrect every copy of a merely-common line.
    tail_start = max(0, len(non_empty) - _STATUS_LINES_KEPT)
    signal_indices = [i for i, l in enumerate(non_empty) if _SIGNAL_RE.search(l)]
    # Count every signal line for the summary even if the cap below trims
    # which ones are actually shown.
    n_errors = sum(1 for i in signal_indices if _FAIL_RE.search(non_empty[i]))
    n_warnings = len(signal_indices) - n_errors
    if len(signal_indices) > _MAX_SIGNAL_LINES:
        half = _MAX_SIGNAL_LINES // 2
        signal_indices = signal_indices[:half] + signal_indices[-half:]

    kept_indices = sorted(set(signal_indices) | set(range(tail_start, len(non_empty))))
    kept = [non_empty[i] for i in kept_indices]
    omitted = len(non_empty) - len(kept)

    if not kept:
        return "[tokensnap: %d lines of output cleaned - no errors or warnings detected]" % len(
            non_empty
        )

    summary = "\n".join(kept)
    if omitted > 0:
        summary += "\n[tokensnap: %d lines omitted (%d error%s, %d warning%s)]" % (
            omitted,
            n_errors,
            "" if n_errors == 1 else "s",
            n_warnings,
            "" if n_warnings == 1 else "s",
        )
    return summary


def _compress_user_text(text: str) -> str:
    """Leave prose alone; reduce only paragraphs that look like a large
    terminal dump, preserving any surrounding natural-language text."""
    if not _looks_like_terminal_dump(text):
        return text
    # Split on blank lines so a "please check this:\n\n<dump>\n\nthoughts?"
    # message keeps its prose paragraphs intact and only the dump shrinks.
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for para in paragraphs:
        if _looks_like_terminal_dump(para):
            out.append(_extract_signal_lines(para))
        else:
            out.append(para)
    return "\n\n".join(out)


def _compress_tool_result_text(text: str) -> str:
    """Tool results are compressed unconditionally once they're non-trivial
    in size - almost none of a large tool output is worth its tokens once
    the errors/warnings/final status are captured."""
    if token_counter.count_tokens(text) <= _TOOL_RESULT_MIN_TOKENS:
        return text
    return _extract_signal_lines(text)


def _compress_message_selectively(message: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the selective-compression philosophy to one message:
    assistant messages pass through untouched; user text is scanned for
    terminal dumps; tool_result content is compressed aggressively."""
    role = message.get("role")
    if role == "assistant":
        return message

    content = message.get("content")

    if isinstance(content, str):
        # A plain string message: only "user" messages reach here with a
        # bare string (tool_result content always lives in blocks), so the
        # gentler user-text heuristic applies.
        return transform_message_text(message, _compress_user_text)

    if isinstance(content, list):
        new_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                block = dict(
                    block,
                    content=_transform_tool_result_content(block.get("content")),
                )
            elif isinstance(block, dict) and block.get("type") == "text":
                block = dict(block, text=_compress_user_text(block.get("text", "")))
            new_blocks.append(block)
        return dict(message, content=new_blocks)

    return message


def _transform_tool_result_content(content: Any) -> Any:
    if isinstance(content, str):
        return _compress_tool_result_text(content)
    if isinstance(content, list):
        new_blocks = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                block = dict(block, text=_compress_tool_result_text(block["text"]))
            new_blocks.append(block)
        return new_blocks
    return content


def build_compressed_context(
    messages: List[Dict[str, Any]],
    keep_messages: int = 10,
    cfg: Optional[Dict[str, Any]] = None,
    min_messages: int = 8,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Selective-compression entry point: apply per-message noise reduction
    to the whole history, then truncate the older part into a Memory Card
    exactly like `compress_messages` (same safe cut-point / tool-pairing
    guarantees), unless `compressor_type=="off"`.

    Returns (memory_card_text_or_None, processed_messages).
    """
    cfg = cfg or {}
    messages = [_compress_message_selectively(m) for m in messages]
    return compress_messages(
        messages,
        keep_last_n=keep_messages,
        min_messages=min_messages,
        llm_cfg=cfg,
    )
