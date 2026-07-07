"""Shared helpers: logging setup and Anthropic message-content access.

Anthropic `messages` entries have `content` that is either a plain string or
a list of blocks ({"type": "text"|"tool_use"|"tool_result"|"image", ...}).
These helpers let the rest of the code treat both shapes uniformly.
"""

import logging
from typing import Any, Callable, Dict, List

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
