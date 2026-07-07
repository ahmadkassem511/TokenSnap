"""The proxy-side half of the fetch_context tool cycle.

When the Differential Context Engine is on, the model is handed a Context Tree
plus a `fetch_context` tool. If it calls that tool, the proxy — not the client
— answers it: it looks the requested events up in the Context Store, feeds
them back as a tool_result, and continues the turn upstream itself, so Claude
Code only ever sees the final answer.

To keep that loop simple and fully offline-testable, the proxy forces upstream
requests to be **non-streaming** while the cycle runs (each response is then a
single JSON message we can inspect). If the client asked for a streaming
response, :func:`synthesize_sse` re-emits the final message as a well-formed
Anthropic event stream. The functions here are pure except for
:func:`build_tool_results`, which reads the Context Store.
"""

import json
from typing import Any, Dict, List, Tuple

from tokensnap import context_store

# Safety valve: how many times we'll answer a fetch_context call and continue
# before giving up and returning whatever the model last produced. Prevents a
# model that keeps fetching from looping forever (and running up cost).
MAX_FETCH_ITERATIONS = 3

TOOL_NAME = "fetch_context"

# A single fetched event can be large; cap what we feed back per event so one
# recall can't blow the window back up.
_MAX_EVENT_CHARS = 8000


def find_fetch_context_calls(content: Any) -> List[Dict[str, Any]]:
    """Return the fetch_context tool_use blocks in an assistant message's
    content, each as ``{"id": tool_use_id, "event_ids": [str, ...]}``."""
    calls: List[Dict[str, Any]] = []
    if not isinstance(content, list):
        return calls
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and block.get("name") == TOOL_NAME:
            raw_ids = (block.get("input") or {}).get("event_ids")
            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            if not isinstance(raw_ids, list):
                raw_ids = []
            calls.append(
                {"id": block.get("id"), "event_ids": [str(x) for x in raw_ids]}
            )
    return calls


def has_other_tool_calls(content: Any) -> bool:
    """True if the message calls any tool other than fetch_context. Such a turn
    can't be answered purely from the Context Store (the client must run the
    real tool), so the loop treats it as final."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict)
        and b.get("type") == "tool_use"
        and b.get("name") != TOOL_NAME
        for b in content
    )


def _format_events(requested_ids: List[str], events: List[Dict[str, Any]]) -> str:
    if not events:
        if not requested_ids:
            return "[tokensnap: no event ids requested]"
        return "[tokensnap: no stored events found for id(s): %s]" % ", ".join(
            requested_ids
        )
    parts = []
    for ev in events:
        body = ev.get("content", "")
        if len(body) > _MAX_EVENT_CHARS:
            body = body[:_MAX_EVENT_CHARS] + "\n[tokensnap: event truncated]"
        parts.append(
            "--- event %s (role=%s, type=%s) ---\n%s"
            % (ev.get("id"), ev.get("role"), ev.get("event_type"), body)
        )
    return "\n\n".join(parts)


def build_tool_results(calls: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Resolve each fetch_context call against the Context Store into a
    tool_result block. Returns ``(tool_result_blocks, events_found)``."""
    results: List[Dict[str, Any]] = []
    found = 0
    for call in calls:
        events = []
        for eid in call["event_ids"]:
            ev = context_store.get_event_by_id(eid)
            if ev:
                events.append(ev)
        found += len(events)
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": _format_events(call["event_ids"], events),
            }
        )
    return results, found


def finalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare the final upstream message for the client: drop any leftover
    fetch_context tool_use blocks (the client can't run that tool) and fix up
    stop_reason so a turn that had only fetch_context reads as a normal end."""
    content = message.get("content") or []
    stripped = [
        b
        for b in content
        if not (
            isinstance(b, dict)
            and b.get("type") == "tool_use"
            and b.get("name") == TOOL_NAME
        )
    ]
    msg = dict(message)
    if not stripped:
        stripped = [
            {"type": "text", "text": "[tokensnap: unable to fetch additional context]"}
        ]
        msg["stop_reason"] = "end_turn"
    elif message.get("stop_reason") == "tool_use" and not any(
        isinstance(b, dict) and b.get("type") == "tool_use" for b in stripped
    ):
        # We removed the only tool_use (a fetch_context) — the turn really ends.
        msg["stop_reason"] = "end_turn"
    msg["content"] = stripped
    return msg


def _sse_event(event_type: str, data: Dict[str, Any]) -> bytes:
    return (
        "event: %s\ndata: %s\n\n"
        % (event_type, json.dumps(data, ensure_ascii=False))
    ).encode("utf-8")


def synthesize_sse(message: Dict[str, Any]) -> bytes:
    """Re-emit a complete Anthropic message as a well-formed SSE byte stream,
    matching the event sequence Claude Code expects (message_start →
    per-block start/delta/stop → message_delta → message_stop)."""
    usage = message.get("usage") or {}
    content = message.get("content") or []

    start_usage = dict(usage)
    start_usage["output_tokens"] = 0
    out = [
        _sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message.get("id", "msg_tokensnap"),
                    "type": "message",
                    "role": message.get("role", "assistant"),
                    "model": message.get("model", ""),
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": start_usage,
                },
            },
        )
    ]

    for i, block in enumerate(content):
        btype = block.get("type") if isinstance(block, dict) else None
        if btype == "text":
            out.append(_sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": i,
                 "content_block": {"type": "text", "text": ""}},
            ))
            text = block.get("text", "")
            if text:
                out.append(_sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": i,
                     "delta": {"type": "text_delta", "text": text}},
                ))
            out.append(_sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": i}))
        elif btype == "tool_use":
            out.append(_sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": i,
                 "content_block": {"type": "tool_use", "id": block.get("id"),
                                   "name": block.get("name"), "input": {}}},
            ))
            out.append(_sse_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": i,
                 "delta": {"type": "input_json_delta",
                           "partial_json": json.dumps(block.get("input", {}),
                                                      ensure_ascii=False)}},
            ))
            out.append(_sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": i}))
        elif btype == "thinking":
            out.append(_sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": i,
                 "content_block": {"type": "thinking", "thinking": ""}},
            ))
            out.append(_sse_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": i,
                 "delta": {"type": "thinking_delta",
                           "thinking": block.get("thinking", "")}},
            ))
            if block.get("signature"):
                out.append(_sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": i,
                     "delta": {"type": "signature_delta",
                               "signature": block["signature"]}},
                ))
            out.append(_sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": i}))
        else:
            # Unknown block shape: emit start/stop with it verbatim so the
            # stream stays well-formed and nothing is silently dropped.
            out.append(_sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": i, "content_block": block},
            ))
            out.append(_sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": i}))

    out.append(_sse_event(
        "message_delta",
        {"type": "message_delta",
         "delta": {"stop_reason": message.get("stop_reason"),
                   "stop_sequence": message.get("stop_sequence")},
         "usage": {"output_tokens": usage.get("output_tokens", 0)}},
    ))
    out.append(_sse_event("message_stop", {"type": "message_stop"}))
    return b"".join(out)
