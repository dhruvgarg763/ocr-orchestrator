"""Structured JSON logging.

Every log line is one JSON object on stdout, automatically carrying the current
trace_id/span_id. Container stdout is the log transport (12-factor): Docker,
Loki or CloudWatch collect it; the app never manages files or rotation.

Logs from libraries (uvicorn, httpx) go through the stdlib `logging` module,
not structlog. We bridge them with ProcessorFormatter so *every* line is JSON —
a log pipeline that has to parse two different formats is a log pipeline that
silently drops half your data.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable, MutableMapping

import structlog

from common.tracing import get_span_id, get_trace_id

Processor = Callable[[Any, str, MutableMapping[str, Any]], Any]


def _add_trace_context(_logger: Any, _name: str, event: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Stamp the current trace context onto every line, with no caller effort."""
    if trace_id := get_trace_id():
        event["trace_id"] = trace_id
    if span_id := get_span_id():
        event["span_id"] = span_id
    return event


def _service_stamper(service: str) -> Processor:
    """Tag lines with their originating service, since all containers share stdout."""

    def processor(_logger: Any, _name: str, event: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        event["service"] = service
        return event

    return processor


def configure_logging(service: str, level: str = "INFO") -> None:
    """Install JSON logging for both structlog and the stdlib. Idempotent."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Shared by our loggers and by bridged stdlib records, so both get trace ids.
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        _service_stamper(service),
        _add_trace_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        # wrap_for_formatter must be last: it hands the event dict to the
        # stdlib handler below instead of rendering it here.
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            # foreign_pre_chain applies to records from uvicorn/httpx etc.
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]  # replace, so uvicorn's default text handler is gone
    root.setLevel(numeric_level)

    # uvicorn installs its own handlers; clear them so lines aren't emitted twice.
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True

    # Silence uvicorn's access log: TraceMiddleware already emits one `request`
    # event per call with duration and trace context, so uvicorn's version is a
    # duplicate. At 1,000 pages that is thousands of redundant lines competing
    # for the same stdout.
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False


def get_logger(name: str = "") -> Any:
    return structlog.get_logger(name)
