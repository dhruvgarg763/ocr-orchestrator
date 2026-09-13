"""Per-request trace context binding.

Deliberately a *pure ASGI* middleware rather than Starlette's
`BaseHTTPMiddleware`, for two reasons:

1. BaseHTTPMiddleware runs the downstream app in a separate asyncio task. Context
   copies into that task, but writes made inside the endpoint do not propagate
   back out - so anything the handler binds is invisible to the middleware.
2. It buffers/interferes with long-lived streaming responses. Module B streams SSE
   for the lifetime of a job, so that is disqualifying.

Pure ASGI middleware runs in the same task as the endpoint: no extra task, no
copy, no streaming interference.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, MutableMapping

from starlette.datastructures import MutableHeaders

from common.logging import get_logger
from common.tracing import new_span_id, new_trace_id, parse_traceparent, set_trace_context

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

log = get_logger("http")


class TraceMiddleware:
    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # lifespan/websocket scopes have no headers; pass them straight through.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw = dict(scope["headers"])  # list[tuple[bytes, bytes]] -> dict
        incoming = raw.get(b"traceparent")
        parsed = parse_traceparent(incoming.decode("latin-1") if incoming else None)

        if parsed:
            # Continue the caller's trace, but start our own span: this hop is
            # a distinct unit of work under the same end-to-end trace.
            trace_id, _parent_span = parsed
            set_trace_context(trace_id, new_span_id())
        else:
            trace_id = new_trace_id()
            set_trace_context(trace_id)

        started = time.perf_counter()
        status_code = 500  # if the app raises before sending, it was a 500

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Echo the id so a client (or the benchmark) can correlate.
                MutableHeaders(scope=message)["x-trace-id"] = trace_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # `finally` so the access log is emitted even on an exception or
            # client disconnect, which is exactly when you need it most.
            log.info(
                "request",
                method=scope.get("method"),
                path=scope.get("path"),
                status=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
