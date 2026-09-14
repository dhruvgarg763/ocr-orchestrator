"""Server-Sent Events: tail one job's results as they land.

SSE over WebSockets because the traffic is one-directional, and SSE keeps
ordinary HTTP infrastructure (status codes, `Last-Event-ID` resume,
proxies, curl) for free. Each page is announced TWICE - `page.partial`
when layout commits (~50ms, `complete:false`) and `page.final` when the
page is terminal (~1.5-3s) - because the 200ms TTFP target is smaller than
a single VLM call, so no amount of concurrency tuning closes that gap; the
two events share `page_index` so a client upgrades in place. Pages are NOT
reordered before streaming - buffering for a slow page would reintroduce
the head-of-line blocking per-page streaming exists to avoid - so
`page_index` is what a client reassembles by. Three identifiers do three
separate jobs: `stream_id` (opaque resume cursor for `Last-Event-ID`),
`seq` (dense per-job counter, so a client can prove it has every event),
`page_index` (the reassembly key).
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request
from redis.exceptions import RedisError
from starlette.responses import StreamingResponse

from app.config import get_settings
from app.queue.results import (
    GAP,
    JOB_COMPLETE,
    STREAM_OPEN,
    ResultReader,
    parse_entry_id,
)
from app.queue.state import PageStateStore
from common.logging import get_logger
from common.tracing import get_trace_id

log = get_logger("sse")

router = APIRouter()

SSE_HEADERS = {
    # A proxy that caches an infinite stream never returns it.
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # nginx buffers proxied responses by default, which would hold our 50ms
    # first event until its buffer filled or the response ended - turning a
    # 200ms budget into a multi-second wait for reasons invisible in our logs.
    "X-Accel-Buffering": "no",
}


class SubscriberSlots:
    """A counted cap on concurrent subscribers.

    A plain counter rather than an asyncio.Semaphore because the required
    behaviour is REFUSE, not wait. A semaphore would park the 65th client on a
    connection that looks healthy and produces nothing - the worst of both,
    since it neither serves the client nor tells it to come back later.

    Acquired in the route and released in the generator's `finally`. That
    ordering is what makes the bound exact: checking the count in the route and
    acquiring in the generator would leave a window in which concurrent
    arrivals all pass the same check - the identical mistake measured at 130%
    overshoot in Step 11's admission control.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self.refused = 0
        self.peak = 0

    def acquire(self) -> bool:
        """Single-threaded event loop, so no lock: there is no await between
        the read and the write, and therefore no point at which another task
        can run."""
        if self.active >= self.limit:
            self.refused += 1
            return False
        self.active += 1
        self.peak = max(self.peak, self.active)
        return True

    def release(self) -> None:
        self.active = max(0, self.active - 1)

    def stats(self) -> dict[str, int]:
        return {
            "active": self.active,
            "limit": self.limit,
            "peak": self.peak,
            "refused": self.refused,
        }


def _frame(event: str, payload: dict[str, Any]) -> str:
    """A locally-synthesised frame, with no `id:`.

    Deliberately no id: `stream.open`, `stream.gap` and heartbeats describe THIS
    connection, not anything that happened to the job. Giving them an id would
    put them in the client's `Last-Event-ID`, and it would then try to resume
    from a cursor that does not exist in the stream.
    """
    body = json.dumps(payload, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"


def _heartbeat() -> str:
    """An SSE comment: a line starting with ':'.

    Every client is required to ignore it, so it needs no schema and cannot be
    mistaken for data. It exists for two reasons, and the second is the real
    one:

      * proxies and load balancers cut idle connections, typically at 60s, and
        a job whose pages are all waiting on a saturated VLM is legitimately
        idle for longer than that.
      * writing to the socket is how a server DISCOVERS a client has gone. A
        reader that only ever blocks on Redis learns nothing about its own
        connection, so an abandoned stream would keep tailing - and keep
        holding a Redis connection - until the job ended.
    """
    return f": heartbeat {int(time.time() * 1000)}\n\n"


def _reader(request: Request) -> ResultReader:
    return request.app.state.result_reader


def _store(request: Request) -> PageStateStore:
    return request.app.state.state_store


def _slots(request: Request) -> SubscriberSlots:
    return request.app.state.sse_slots


@router.get("/jobs/{job_id}/stream")
async def stream_results(
    request: Request,
    job_id: str,
    last_event_id: str | None = Query(
        default=None,
        description="Resume cursor. Same meaning as the Last-Event-ID header, "
        "which browsers send automatically; this is for clients that cannot.",
    ),
) -> StreamingResponse:
    """Tail a job's results until it completes or the client leaves.

    The job is validated BEFORE the StreamingResponse is constructed. Once a
    streaming body has started, status and headers are already on the wire, so
    an unknown job discovered inside the generator could only be reported as an
    error event inside a 200 response - and a client would have to parse the
    body to discover the request failed. Everything that can fail must fail
    while a status code still means something.
    """
    job = await _store(request).get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")

    settings = get_settings()
    total = int(job.get("total_pages", 0))

    # Header first, then query - browsers set the header on reconnect without
    # being asked, so it should win over a stale URL a client built by hand.
    cursor = parse_entry_id(
        request.headers.get("last-event-id")
    ) or parse_entry_id(last_event_id)

    # LAST statement before the return, on purpose. The slot is released in
    # the generator's `finally`, so anything that can raise BETWEEN the
    # acquire and the StreamingResponse would leak one permanently - and a
    # leaked slot never comes back, so the cap would tighten silently until
    # the endpoint stopped serving anyone.
    #
    # Refused BEFORE the body starts, so it is a real 503 with a Retry-After
    # rather than an error event inside a 200. Every subscriber pins a Redis
    # connection for as long as it is blocked in XREAD, so this is a resource
    # bound, not a policy: without it the 65th client waits out the pool
    # timeout and gets a truncated response.
    if not _slots(request).acquire():
        log.warning("sse_refused_at_capacity", job_id=job_id, **_slots(request).stats())
        raise HTTPException(
            status_code=503,
            detail={
                "error": "too_many_subscribers",
                "message": "This replica is serving its maximum number of "
                "streams. Retry, or poll GET /jobs/{job_id} instead.",
                **_slots(request).stats(),
            },
            headers={"Retry-After": str(int(settings.sse_retry_ms / 1000) or 1)},
        )

    return StreamingResponse(
        _events(request, job_id, total=total, cursor=cursor, settings=settings),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def _events(
    request: Request,
    job_id: str,
    *,
    total: int,
    cursor: str | None,
    settings: Any,
) -> AsyncIterator[str]:
    """Owns the subscriber slot and the error policy; delegates the content.

    Split from the body below so that releasing the slot and handling a Redis
    failure are each expressed exactly once, in a `finally` and an `except`,
    rather than at every `return` in a generator that has several.

    The RedisError branch matters more than it looks. A streaming body has
    already sent its 200, so an exception escaping here cannot become a status
    code - uvicorn simply drops the connection, and the client reports
    "incomplete chunked read": a transport-level error for what is actually a
    server-side fault, indistinguishable from a network problem. Catching it
    and closing cleanly turns that into a gap event naming the authoritative
    endpoint, which is something a client can act on.
    """
    slots = _slots(request)
    try:
        async for frame in _frames(
            request, job_id, total=total, cursor=cursor, settings=settings
        ):
            yield frame
    except RedisError as exc:
        log.error("sse_redis_failed", job_id=job_id, error=type(exc).__name__)
        yield _frame(
            GAP,
            {
                "job_id": job_id,
                "message": (
                    "The result stream became unavailable. Re-read "
                    f"GET /jobs/{job_id} for authoritative page state, then "
                    "reconnect."
                ),
                "window_lo": None,
            },
        )
    finally:
        # Runs on every exit, including GeneratorExit when the client
        # disconnects mid-stream - which is the common case and the one a
        # release placed after the loop would miss, leaking a slot per
        # abandoned connection until the cap was permanently exhausted.
        slots.release()


async def _frames(
    request: Request,
    job_id: str,
    *,
    total: int,
    cursor: str | None,
    settings: Any,
) -> AsyncIterator[str]:
    """Catch up from the retained window, then tail live.

    Structured as catch-up-then-tail rather than tail-only because a subscriber
    almost never arrives before the first event. Ingestion returns a job id and
    the client then opens this stream, and by the time it connects the fast
    layout stage has already committed several pages. Starting at `$` - the
    obvious choice, and what most SSE examples do - would silently skip exactly
    those pages, and the client would sit at 3/20 forever waiting for events
    that were published before it arrived.
    """
    reader = _reader(request)
    store = _store(request)
    started = time.monotonic()
    trace_id = get_trace_id()
    sent = 0
    resync = False

    # `retry:` tells a browser's EventSource how long to wait before
    # reconnecting. Sent once, first: the default is 3s, and 50 clients dropped
    # by a restart all returning after the same 3s is the thundering herd from
    # Step 8 rebuilt at the edge.
    yield f"retry: {int(settings.sse_retry_ms)}\n\n"

    if cursor and await reader.gap_after(job_id, cursor):
        # The events between the client's cursor and the oldest surviving entry
        # have been trimmed. XREAD would NOT report this - it returns whatever
        # still exists, so the client would resume cleanly, be permanently
        # missing a run of pages and never be told. Say it out loud instead and
        # name the authoritative source, which is not trimmed.
        resync = True
        cursor = None
        log.warning("sse_gap_detected", job_id=job_id, trace_id=trace_id)

    lo, hi = await reader.window(job_id)
    done, _total = await store.progress(job_id)

    yield _frame(
        STREAM_OPEN,
        {
            "job_id": job_id,
            "total_pages": total,
            # Where the job already is. A client that connects late, or one
            # recovering from a gap, can size its own progress bar from this
            # instead of inferring it from however many events happen to remain.
            "done": done,
            "resumed_from": cursor,
            # The honest statement of what can still be replayed. A client that
            # wants a guarantee rather than a hope compares its own cursor
            # against window_lo.
            "window_lo": lo,
            "window_hi": hi,
            "trace_id": trace_id,
        },
    )

    if resync:
        yield _frame(
            GAP,
            {
                "job_id": job_id,
                "message": (
                    "Events before window_lo were trimmed. Re-read "
                    f"GET /jobs/{job_id} for authoritative page state; page "
                    "state is never trimmed."
                ),
                "window_lo": lo,
            },
        )

    # Replay, then tail from wherever the replay ended. Deriving the tail cursor
    # from the last replayed entry - rather than from `hi` read a moment ago -
    # means an event published between the two reads is picked up by the tail
    # instead of being skipped.
    last_id = cursor or "0"
    complete = False

    for event in await reader.history(
        job_id, after=cursor, count=settings.sse_history_limit
    ):
        yield event.sse()
        sent += 1
        last_id = event.entry_id
        complete = complete or event.event == JOB_COMPLETE

    while not complete:
        if await request.is_disconnected():
            log.info("sse_client_gone", job_id=job_id, sent=sent, trace_id=trace_id)
            return

        if time.monotonic() - started > settings.sse_max_duration_s:
            # A connection is not allowed to live forever. Without this a client
            # that opens a stream for a job that never completes - a page
            # stranded in a dead worker's pending list before Step 14's reaper
            # exists - holds a task slot and a Redis connection indefinitely.
            # Closing is safe precisely because resume works: the client
            # reconnects with its Last-Event-ID and loses nothing.
            log.warning("sse_max_duration", job_id=job_id, sent=sent)
            yield _frame(
                GAP,
                {
                    "job_id": job_id,
                    "message": "Stream duration limit reached; reconnect with "
                    "Last-Event-ID to continue.",
                    "window_lo": last_id,
                },
            )
            return

        events = await reader.tail(
            job_id,
            last_id=last_id,
            block_ms=settings.sse_block_ms,
            count=settings.sse_batch,
        )

        if not events:
            # Idle. Two things are worth doing here and nowhere else, because
            # this is the only point at which we know the pipeline has nothing
            # for us and the cost of an extra round trip is free.
            yield _heartbeat()

            # Termination backstop. `job.complete` is published by whichever
            # worker finishes the last page, so it can be missed: a worker
            # killed between the state commit and the publish, or a stream
            # trimmed before a late subscriber read it. Without this check the
            # client would block on a finished job until the duration limit.
            #
            # Committed state is the authority, not the notification - the same
            # relationship the publisher relies on.
            done, total_now = await store.progress(job_id)
            if total_now and done >= total_now:
                yield _frame(
                    JOB_COMPLETE,
                    {
                        "job_id": job_id,
                        "total_pages": total_now,
                        "done": done,
                        # Named, so a client can tell a normal close from this
                        # one and a log can show how often the event is missed.
                        "derived": True,
                    },
                )
                break
            continue

        for event in events:
            yield event.sse()
            sent += 1
            last_id = event.entry_id
            if event.event == JOB_COMPLETE:
                complete = True

    log.info(
        "sse_stream_closed",
        job_id=job_id,
        sent=sent,
        duration_ms=round((time.monotonic() - started) * 1000, 1),
        trace_id=trace_id,
    )


@router.get("/streams")
async def stream_stats(request: Request) -> dict[str, Any]:
    """Subscriber occupancy and refusals.

    Here for the same reason /admission exists: a cap is only defensible if the
    number of times it fired is visible. A refusal that appears nowhere is
    indistinguishable from a drop.
    """
    return _slots(request).stats()
