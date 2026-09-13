"""Trace context propagation.

A trace id follows one unit of work (an HTTP request, or one page moving through
the pipeline) across function calls, across `asyncio` tasks, and across process
boundaries via the W3C `traceparent` header.

Why ContextVar and not a global or threading.local():
  - a global is shared by every concurrent task, so 16 pages would overwrite
    each other's id;
  - threading.local() keys off the OS thread, but every coroutine in an asyncio
    program runs on the *same* thread, so they would all share one slot.

ContextVar keys off the logical execution context. `asyncio.create_task()`
copies the current context into the new task, so a child task inherits the
trace id and its own writes stay isolated from siblings.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

# Empty-string defaults so a log call outside any request never raises LookupError.
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_span_id: ContextVar[str] = ContextVar("span_id", default="")

# W3C trace-context: version "-" trace-id "-" parent-id "-" flags
_TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")


def new_trace_id() -> str:
    """128-bit id, 32 lowercase hex chars (W3C trace-id)."""
    return uuid.uuid4().hex


def new_span_id() -> str:
    """64-bit id, 16 lowercase hex chars (W3C span-id)."""
    return uuid.uuid4().hex[:16]


def get_trace_id() -> str:
    return _trace_id.get()


def get_span_id() -> str:
    return _span_id.get()


def set_trace_context(trace_id: str, span_id: str | None = None) -> None:
    """Bind ids to the current context. Visible to this task and any it spawns."""
    _trace_id.set(trace_id)
    _span_id.set(span_id or new_span_id())


def format_traceparent() -> str:
    """Serialise the current context for an outbound HTTP call."""
    trace_id = _trace_id.get() or new_trace_id()
    span_id = _span_id.get() or new_span_id()
    return f"00-{trace_id}-{span_id}-01"


def parse_traceparent(header: str | None) -> tuple[str, str] | None:
    """Parse an inbound header. Returns None if absent or malformed.

    Never trust the wire: a caller could send anything, and an unvalidated id
    would end up in our logs (log-injection risk) and break id-based joins.
    """
    if not header:
        return None
    match = _TRACEPARENT_RE.match(header.strip().lower())
    if not match:
        return None
    return match.group(1), match.group(2)
