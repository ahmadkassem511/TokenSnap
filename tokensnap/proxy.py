"""The Tokensnap HTTP proxy.

Listens locally, optimizes POST /v1/messages (and /v1/complete) request
bodies — cleaning text, compressing history into a Memory Card — and
forwards everything to the real Anthropic API. Responses (including SSE
streams) are relayed back untouched. The API key travels in the request
headers Claude Code already sends; the proxy never stores it.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web

from tokensnap import (
    cleaner,
    compressor,
    context_engine,
    fetch_context,
    project,
    stats,
    token_counter,
)
from tokensnap.usage import UsageAccumulator
from tokensnap.utils import (
    append_to_system,
    system_to_parts,
    transform_message_text,
)

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
    parts = system_to_parts(system)
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

    # Differential Context Engine: instead of a Memory Card, mirror the whole
    # conversation to the local Context Store and send only the last couple of
    # exchanges plus a compact Context Tree (+ a fetch_context tool). Opt-in;
    # when disabled the classic clean+compress path below runs unchanged.
    if bool(cfg.get("context_store_enabled", False)):
        return _optimize_differential(body, messages, system, model, tokens_before, cfg)

    # 1. Clean terminal noise out of every text payload
    messages = _clean_messages(messages)

    # 2. Compress old history into a Memory Card. Selective compression
    #    (default) first reduces each message to its signal - assistant
    #    messages untouched, user terminal dumps and tool results trimmed to
    #    errors/warnings/status - before truncating; legacy mode skips that
    #    per-message pass and truncates uniformly, matching pre-0.4 behavior.
    if bool(cfg.get("selective_compression", True)):
        card, messages = compressor.build_compressed_context(
            messages,
            keep_messages=int(cfg["keep_messages"]),
            cfg=cfg,
            min_messages=int(cfg["min_messages_to_compress"]),
        )
    else:
        card, messages = compressor.compress_messages(
            messages,
            keep_last_n=int(cfg["keep_messages"]),
            min_messages=int(cfg["min_messages_to_compress"]),
            llm_cfg=cfg,
        )
    if card:
        system = append_to_system(system, card)

    tokens_after = token_counter.count_message_tokens(messages, system)

    # 3. Still near the context window? Get aggressive. This last-resort
    #    path always uses the uniform legacy truncation (not selective
    #    compression) - it needs a hard, predictable size cap, which
    #    per-message compression doesn't guarantee.
    aggressive = False
    if token_counter.near_limit(tokens_after, model, float(cfg["context_threshold"])):
        aggressive = True
        card, messages = compressor.compress_messages(
            messages,
            keep_last_n=int(cfg["aggressive_keep_last_n"]),
            min_messages=int(cfg["aggressive_keep_last_n"]) * 2,
            llm_cfg=cfg,
        )
        if card:
            system = append_to_system(system, card)
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
        "n_messages": len(body.get("messages") or []),
        "n_kept": len(messages),
    }
    return new_body, meta


def _optimize_differential(
    body: Dict[str, Any],
    messages: List[Dict[str, Any]],
    system: Any,
    model: Any,
    tokens_before: int,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Differential Context Engine path (context_store_enabled).

    Cleans noise, mirrors every message into the Context Store, then rebuilds
    the outgoing request as: the last couple of exchanges (selectively
    compressed) preceded by a single Context Tree system block, with a
    fetch_context tool merged in so the model can pull back omitted detail.
    """
    cleaned = _clean_messages(messages)

    session_id = context_engine.derive_session_id(system, cleaned)
    context_engine.ingest(session_id, cleaned)

    new_messages, new_system, reconstructed, n_omitted = context_engine.reconstruct(
        cleaned,
        system,
        session_id,
        tree_size=int(cfg.get("context_tree_size", 20)),
        min_messages=int(cfg["min_messages_to_compress"]),
        selective=bool(cfg.get("selective_compression", True)),
    )

    new_body = dict(body)
    new_body["messages"] = new_messages
    if new_system is not None:
        new_body["system"] = new_system
    if reconstructed:
        new_body["tools"] = context_engine.merge_fetch_context_tool(
            new_body.get("tools")
        )

    tokens_after = token_counter.count_message_tokens(new_messages, new_system)
    meta = {
        "before": tokens_before,
        "after": tokens_after,
        "aggressive": False,
        "model": model,
        "compressed": reconstructed,
        "n_messages": len(messages),
        "n_kept": len(new_messages),
        "context_store": True,
        "events_omitted": n_omitted,
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
        status = await asyncio.to_thread(compressor.memory_card_status, self.cfg)
        log.info("Memory Cards: %s", status)
        stats.set_llm_status(status)

    async def _on_cleanup(self, app: web.Application) -> None:
        if self.session:
            await self.session.close()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        start = time.monotonic()
        raw = await request.read()
        meta: Dict[str, Any] = {}
        body_bytes = raw
        optimized: Optional[Dict[str, Any]] = None

        if request.method == "POST" and request.path in OPTIMIZED_PATHS:
            try:
                body = json.loads(raw)
                # Worker thread: optimization may block on a local LLM call
                # and must not stall other in-flight requests
                optimized, meta = await asyncio.to_thread(
                    optimize_body, body, self.cfg
                )
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

        usage = UsageAccumulator()

        # Differential Context Engine: run the fetch_context tool loop, which
        # makes its own (possibly multiple) upstream requests and returns the
        # client-facing response plus the final status.
        if optimized is not None and meta.get("context_store") and request.path == "/v1/messages":
            try:
                response, status = await self._relay_with_fetch_context(
                    request, optimized, headers, url, usage, meta
                )
            except aiohttp.ClientError as exc:
                log.error("Upstream request failed: %s", exc)
                return web.json_response(
                    {"type": "error", "error": {"type": "tokensnap_upstream_error",
                                                "message": str(exc)}},
                    status=502,
                )
        else:
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
                response = await self._relay(request, upstream, usage)
                status = upstream.status
            finally:
                upstream.release()

        elapsed = time.monotonic() - start
        if meta.get("before"):
            saved = meta["before"] - meta["after"]
            pct = 100.0 * saved / meta["before"] if meta["before"] else 0.0
            log.info(
                "%s %s [%s] msgs %d->%d, est %d -> %d (saved %d, %.1f%%)%s | "
                "real in=%d out=%d cache_read=%d  %.2fs",
                request.method,
                request.path,
                meta.get("model") or "?",
                meta.get("n_messages", 0),
                meta.get("n_kept", 0),
                meta["before"],
                meta["after"],
                saved,
                pct,
                " [AGGRESSIVE]" if meta.get("aggressive")
                else (" [+%d fetched]" % meta["events_fetched"]
                      if meta.get("events_fetched") else ""),
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_tokens,
                elapsed,
            )
            stats.record_request(
                request.path,
                meta.get("model"),
                meta["before"],
                meta["after"],
                status,
                elapsed,
                aggressive=bool(meta.get("aggressive")),
                real_input=usage.input_tokens,
                real_output=usage.output_tokens,
                real_cache_read=usage.cache_read_tokens,
                real_cache_creation=usage.cache_creation_tokens,
                context_store=bool(meta.get("context_store")),
                events_fetched=int(meta.get("events_fetched") or 0),
                # Tag each row with the current project. Read per request from a
                # small state file (written by `tokensnap run` / the dashboard's
                # launch button), so switching projects works without restarting
                # this long-running proxy. Defaults to 'unknown'.
                project=project.get_current_project(),
            )
        else:
            log.info(
                "%s %s -> %d  %.2fs",
                request.method, request.path, status, elapsed,
            )
        return response

    async def _relay(
        self,
        request: web.Request,
        upstream: aiohttp.ClientResponse,
        usage: UsageAccumulator,
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
                # Relay bytes verbatim, and tee a copy to the usage parser.
                await response.write(chunk)
                usage.feed(chunk)
            await response.write_eof()
            return response

        body = await upstream.read()
        usage.feed_full_body(body)
        return web.Response(status=upstream.status, headers=resp_headers, body=body)

    async def _relay_with_fetch_context(
        self,
        request: web.Request,
        optimized_body: Dict[str, Any],
        headers: Dict[str, str],
        url: str,
        usage: UsageAccumulator,
        meta: Dict[str, Any],
    ) -> Tuple[web.StreamResponse, int]:
        """Drive the fetch_context tool cycle, then emit the final answer.

        Upstream requests are forced non-streaming so each response is a single
        JSON message we can inspect. When the model calls fetch_context, we
        answer it from the Context Store and continue the turn upstream; the
        client only ever sees the final answer (re-emitted as SSE if it asked
        for streaming). Returns (client_response, final_status).
        """
        client_wants_stream = bool(optimized_body.get("stream"))
        work_body = dict(optimized_body)
        work_body["stream"] = False
        events_fetched = 0

        for iteration in range(fetch_context.MAX_FETCH_ITERATIONS + 1):
            data = json.dumps(work_body, ensure_ascii=False).encode("utf-8")
            upstream = await self.session.request(
                "POST", url, data=data, headers=headers
            )
            try:
                raw = await upstream.read()
                status = upstream.status
                resp_headers = {
                    k: v
                    for k, v in upstream.headers.items()
                    if k.lower() not in _RESPONSE_SKIP
                }
            finally:
                upstream.release()

            meta["events_fetched"] = events_fetched

            if status != 200:
                # Pass an upstream error straight back (its own JSON + status).
                return (
                    web.Response(status=status, headers=resp_headers, body=raw),
                    status,
                )
            try:
                message = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return (
                    web.Response(status=status, headers=resp_headers, body=raw),
                    status,
                )

            content = message.get("content") or []
            calls = fetch_context.find_fetch_context_calls(content)
            final = (
                not calls
                or fetch_context.has_other_tool_calls(content)
                or iteration >= fetch_context.MAX_FETCH_ITERATIONS
            )
            if final:
                usage.feed_full_body(raw)
                meta["events_fetched"] = events_fetched
                response = await self._emit_message(
                    request, status, resp_headers,
                    fetch_context.finalize_message(message), client_wants_stream,
                )
                return response, status

            # Answer the fetch_context call(s) from external memory and continue
            # the assistant turn ourselves.
            tool_results, found = fetch_context.build_tool_results(calls)
            events_fetched += found
            log.info("fetch_context: served %d event(s) for %d call(s)",
                     found, len(calls))
            work_body["messages"] = list(work_body["messages"]) + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": tool_results},
            ]

        # The loop always returns inside; this satisfies the type checker.
        return web.json_response(
            {"type": "error", "error": {"type": "tokensnap_error",
                                        "message": "fetch_context loop exhausted"}},
            status=500,
        ), 500

    async def _emit_message(
        self,
        request: web.Request,
        status: int,
        resp_headers: Dict[str, str],
        message: Dict[str, Any],
        client_wants_stream: bool,
    ) -> web.StreamResponse:
        """Send a finished message to the client as SSE (if it requested a
        stream) or as a plain JSON body."""
        out_headers = dict(resp_headers)
        if client_wants_stream:
            out_headers["Content-Type"] = "text/event-stream"
            response = web.StreamResponse(status=status, headers=out_headers)
            await response.prepare(request)
            await response.write(fetch_context.synthesize_sse(message))
            await response.write_eof()
            return response
        out_headers["Content-Type"] = "application/json"
        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        return web.Response(status=status, headers=out_headers, body=body)


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
