"""The Tokensnap HTTP proxy.

Listens locally, optimizes POST /v1/messages (and /v1/complete) request
bodies — cleaning text, compressing history into a Memory Card — and
forwards everything to the real Anthropic API. Responses (including SSE
streams) are relayed back untouched. The API key travels in the request
headers Claude Code already sends; the proxy never stores it.
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web

from tokensnap import cleaner, compressor, stats, token_counter
from tokensnap.utils import transform_message_text

log = logging.getLogger("tokensnap.proxy")

OPTIMIZED_PATHS = {"/v1/messages", "/v1/complete"}

# Headers never forwarded in either direction
_HOP_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
    # Ask upstream for identity encoding so we can relay bytes verbatim
    "accept-encoding",
}
_RESPONSE_SKIP = _HOP_HEADERS | {"content-encoding"}

# In aggressive mode, text blocks longer than this that repeat verbatim
# earlier in the history get replaced by a stub
_DUP_BLOCK_MIN_CHARS = 500
_AGGRESSIVE_SYSTEM_MAX_CHARS = 6000


def _clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        transform_message_text(m, lambda t: cleaner.clean_text(t)[0])
        for m in messages
    ]


def _system_to_parts(system: Any) -> List[str]:
    if isinstance(system, str):
        return [system]
    parts = []
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return parts


def _append_to_system(system: Any, extra: str) -> Any:
    """Append text to a system prompt that may be a string, block list, or absent."""
    if system is None or system == "":
        return extra
    if isinstance(system, str):
        return system + "\n\n" + extra
    if isinstance(system, list):
        return system + [{"type": "text", "text": extra}]
    return system


def _drop_duplicate_blocks(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace large text payloads that appear verbatim more than once
    (typically re-attached file contents) with a short stub."""
    seen: set = set()

    def dedupe(text: str) -> str:
        if len(text) < _DUP_BLOCK_MIN_CHARS:
            return text
        digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        if digest in seen:
            return "[tokensnap: identical content already appears earlier in this conversation]"
        seen.add(digest)
        return text

    return [transform_message_text(m, dedupe) for m in messages]


def _truncate_system(system: Any) -> Any:
    parts = _system_to_parts(system)
    joined = "\n\n".join(parts)
    if len(joined) <= _AGGRESSIVE_SYSTEM_MAX_CHARS:
        return system
    return (
        joined[:_AGGRESSIVE_SYSTEM_MAX_CHARS]
        + "\n[tokensnap: system prompt truncated to fit the context window]"
    )


def optimize_body(
    body: Dict[str, Any], cfg: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply all optimizations to a /v1/messages request body.

    Returns (new_body, meta) where meta holds token accounting for logging.
    """
    messages = body.get("messages")
    model = body.get("model")
    if not isinstance(messages, list) or not messages:
        return body, {"before": 0, "after": 0, "aggressive": False, "model": model}

    system = body.get("system")
    tokens_before = token_counter.count_message_tokens(messages, system)

    # 1. Clean terminal noise out of every text payload
    messages = _clean_messages(messages)

    # 2. Compress old history into a Memory Card
    card, messages = compressor.compress_messages(
        messages,
        keep_last_n=int(cfg["keep_last_n"]),
        min_messages=int(cfg["min_messages_to_compress"]),
    )
    if card:
        system = _append_to_system(system, card)

    tokens_after = token_counter.count_message_tokens(messages, system)

    # 3. Still near the context window? Get aggressive.
    aggressive = False
    if token_counter.near_limit(tokens_after, model, float(cfg["context_threshold"])):
        aggressive = True
        card, messages = compressor.compress_messages(
            messages,
            keep_last_n=int(cfg["aggressive_keep_last_n"]),
            min_messages=int(cfg["aggressive_keep_last_n"]) * 2,
        )
        if card:
            system = _append_to_system(system, card)
        messages = _drop_duplicate_blocks(messages)
        system = _truncate_system(system)
        tokens_after = token_counter.count_message_tokens(messages, system)

    new_body = dict(body)
    new_body["messages"] = messages
    if system is not None:
        new_body["system"] = system

    meta = {
        "before": tokens_before,
        "after": tokens_after,
        "aggressive": aggressive,
        "model": model,
        "compressed": card is not None,
    }
    return new_body, meta


class TokensnapProxy:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.session: Optional[aiohttp.ClientSession] = None

    def make_app(self) -> web.Application:
        app = web.Application(client_max_size=512 * 1024 * 1024)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def _on_startup(self, app: web.Application) -> None:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        stats.mark_started(self.cfg["host"], int(self.cfg["port"]))
        log.info(
            "[bold green]Tokensnap proxy listening on http://%s:%s[/] "
            "-> forwarding to %s",
            self.cfg["host"],
            self.cfg["port"],
            self.cfg["upstream"],
            extra={"markup": True},
        )

    async def _on_cleanup(self, app: web.Application) -> None:
        if self.session:
            await self.session.close()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        start = time.monotonic()
        raw = await request.read()
        meta: Dict[str, Any] = {}
        body_bytes = raw

        if request.method == "POST" and request.path in OPTIMIZED_PATHS:
            try:
                body = json.loads(raw)
                optimized, meta = optimize_body(body, self.cfg)
                body_bytes = json.dumps(optimized, ensure_ascii=False).encode("utf-8")
            except (json.JSONDecodeError, UnicodeDecodeError):
                log.debug("Non-JSON body on %s; forwarding verbatim", request.path)
        else:
            log.debug("Pass-through: %s %s", request.method, request.path)

        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS
        }
        # Force identity so relayed bytes always match the headers we send back
        headers["Accept-Encoding"] = "identity"
        url = self.cfg["upstream"].rstrip("/") + str(request.rel_url)

        try:
            upstream = await self.session.request(
                request.method, url, data=body_bytes, headers=headers
            )
        except aiohttp.ClientError as exc:
            log.error("Upstream request failed: %s", exc)
            return web.json_response(
                {"type": "error", "error": {"type": "tokensnap_upstream_error",
                                            "message": str(exc)}},
                status=502,
            )

        try:
            response = await self._relay(request, upstream)
        finally:
            upstream.release()

        elapsed = time.monotonic() - start
        if meta.get("before"):
            saved = meta["before"] - meta["after"]
            pct = 100.0 * saved / meta["before"] if meta["before"] else 0.0
            log.info(
                "%s %s [%s] %d -> %d tokens (saved %d, %.1f%%)%s  %.2fs",
                request.method,
                request.path,
                meta.get("model") or "?",
                meta["before"],
                meta["after"],
                saved,
                pct,
                " [AGGRESSIVE]" if meta.get("aggressive") else "",
                elapsed,
            )
            stats.record_request(
                request.path,
                meta.get("model"),
                meta["before"],
                meta["after"],
                upstream.status,
                elapsed,
                aggressive=bool(meta.get("aggressive")),
            )
        else:
            log.info(
                "%s %s -> %d  %.2fs",
                request.method, request.path, upstream.status, elapsed,
            )
        return response

    async def _relay(
        self, request: web.Request, upstream: aiohttp.ClientResponse
    ) -> web.StreamResponse:
        resp_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in _RESPONSE_SKIP
        }
        content_type = upstream.headers.get("Content-Type", "")

        if "text/event-stream" in content_type:
            response = web.StreamResponse(
                status=upstream.status, headers=resp_headers
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response

        body = await upstream.read()
        return web.Response(status=upstream.status, headers=resp_headers, body=body)


def run_proxy(
    host: Optional[str] = None,
    port: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Blocking entry point used by `tokensnap start`."""
    from tokensnap import config as config_mod
    from tokensnap.utils import setup_logging

    cfg = config_mod.load()
    if host:
        cfg["host"] = host
    if port:
        cfg["port"] = port
    setup_logging("DEBUG" if verbose else str(cfg.get("log_level", "INFO")))

    proxy = TokensnapProxy(cfg)
    web.run_app(
        proxy.make_app(),
        host=cfg["host"],
        port=int(cfg["port"]),
        print=None,
        handle_signals=True,
    )
