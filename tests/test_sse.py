"""The SSE endpoint, driven through a real ASGI stack.

Exercised over HTTP rather than by calling the generator, because half of what
this endpoint has to get right IS the HTTP layer: the status code has to be
decided before the body starts, `Last-Event-ID` arrives as a header, and the
frame terminator is what makes a client emit anything at all.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis

from app.api import stream as stream_api
from app.config import Settings
from app.queue.results import (
    GAP,
    JOB_COMPLETE,
    PAGE_FINAL,
    PAGE_PARTIAL,
    STREAM_OPEN,
    ResultPublisher,
    ResultReader,
)
from app.queue.state import PageStateStore

SUBSCRIBER_LIMIT = 3

# Short block and lifetime so an idle test is a fast test rather than a hung one.
TEST_SETTINGS = Settings(
    sse_block_ms=120,
    sse_batch=64,
    sse_history_limit=512,
    sse_retry_ms=1_500,
    sse_max_duration_s=5.0,
)


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stream_api, "get_settings", lambda: TEST_SETTINGS)


@pytest.fixture
def publisher(redis: Redis) -> ResultPublisher:
    return ResultPublisher(redis, maxlen=256, ttl_s=60)


@pytest_asyncio.fixture
async def client(redis: Redis, store: PageStateStore):
    """A minimal app carrying only what the SSE route reads off app.state.

    Deliberately not the whole application: pulling in the real lifespan would
    make these tests depend on admission control, the adaptive limiter and the
    mock model, none of which this endpoint touches.
    """
    app = FastAPI()
    app.include_router(stream_api.router)
    app.state.state_store = store
    app.state.result_reader = ResultReader(redis)
    app.state.sse_slots = stream_api.SubscriberSlots(SUBSCRIBER_LIMIT)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://sse"
    ) as ac:
        yield ac


# ------------------------------------------------------------------ plumbing


def _parse(raw: str) -> dict[str, Any]:
    """One SSE frame -> a dict. Comments become {"comment": ...}."""
    frame: dict[str, Any] = {}
    for line in raw.splitlines():
        if line.startswith(":"):
            frame["comment"] = line[1:].strip()
        elif line.startswith("id: "):
            frame["id"] = line[4:]
        elif line.startswith("event: "):
            frame["event"] = line[7:]
        elif line.startswith("data: "):
            frame["data"] = json.loads(line[6:])
        elif line.startswith("retry: "):
            frame["retry"] = int(line[7:])
    return frame


async def collect(
    client: AsyncClient,
    job_id: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    max_frames: int = 200,
    timeout: float = 8.0,
) -> tuple[int, list[dict[str, Any]]]:
    """Read frames until the stream closes or `max_frames` is reached.

    `max_frames` exists so a test of a stream that is SUPPOSED to stay open
    (heartbeats) terminates without relying on the timeout.
    """
    frames: list[dict[str, Any]] = []

    async def run() -> int:
        async with client.stream(
            "GET", f"/jobs/{job_id}/stream", headers=headers or {}, params=params or {}
        ) as response:
            if response.status_code != 200:
                await response.aread()
                return response.status_code
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    raw, buffer = buffer.split("\n\n", 1)
                    if raw.strip():
                        frames.append(_parse(raw))
                    if len(frames) >= max_frames:
                        return 200
            return 200

    status = await asyncio.wait_for(run(), timeout=timeout)
    return status, frames


def events(frames: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [f for f in frames if f.get("event") == name]


# ------------------------------------------------------------------- basics


async def test_an_unknown_job_is_404_not_an_error_event(client: AsyncClient) -> None:
    """Validated BEFORE the streaming body starts.

    Once a body is streaming the status is already on the wire, so an unknown
    job discovered inside the generator could only be reported as an error
    event inside a 200 - forcing a client to parse the body to learn its
    request failed.
    """
    status, frames = await collect(client, "no-such-job")

    assert status == 404
    assert frames == []


async def test_the_stream_opens_with_retry_then_a_header_event(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    await store.init_job("j", 3)
    await publisher.publish("j", JOB_COMPLETE, {"total_pages": 3})

    _, frames = await collect(client, "j")

    assert frames[0]["retry"] == 1_500, "a chosen reconnect delay, not the 3s default"
    assert frames[1]["event"] == STREAM_OPEN
    assert frames[1]["data"]["total_pages"] == 3
    assert frames[1]["data"]["window_lo"] is not None


async def test_header_and_heartbeat_frames_carry_no_id(
    client: AsyncClient, store: PageStateStore
) -> None:
    """They describe THIS connection, not the job. Giving them an id would put
    it in the client's Last-Event-ID, and it would then try to resume from a
    cursor that does not exist in the stream."""
    await store.init_job("j", 1)

    _, frames = await collect(client, "j", max_frames=4)

    for frame in frames:
        if frame.get("event") in (STREAM_OPEN, GAP) or "comment" in frame:
            assert "id" not in frame


# --------------------------------------------------------- the two-phase move


async def test_a_page_is_announced_twice_partial_then_final(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    """The whole reason time-to-first-page can beat 200ms.

    A stream emitting one event per finished page cannot put a first byte on
    the wire inside 200ms when a VLM call takes 1.5-3s - the target is smaller
    than one call. Announcing the fast stage separately makes first-byte
    latency a property of the 50ms stage.
    """
    await store.init_job("j", 1)
    await publisher.publish("j", PAGE_PARTIAL, {"page_index": 0, "complete": False})
    await publisher.publish("j", PAGE_FINAL, {"page_index": 0, "complete": True})
    await publisher.publish("j", JOB_COMPLETE, {"total_pages": 1})

    _, frames = await collect(client, "j")

    partial, final = events(frames, PAGE_PARTIAL), events(frames, PAGE_FINAL)
    assert len(partial) == len(final) == 1
    assert partial[0]["data"]["page_index"] == final[0]["data"]["page_index"] == 0
    assert partial[0]["data"]["complete"] is False
    assert final[0]["data"]["complete"] is True
    assert partial[0]["data"]["seq"] < final[0]["data"]["seq"]


async def test_pages_are_delivered_out_of_order_not_buffered(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    """Reordering here would idle the client on the slowest page in the job,
    which is the exact head-of-line blocking per-page streaming avoids."""
    await store.init_job("j", 4)
    for page in (3, 0, 2, 1):
        await publisher.publish("j", PAGE_FINAL, {"page_index": page})
    await publisher.publish("j", JOB_COMPLETE, {"total_pages": 4})

    _, frames = await collect(client, "j")

    assert [f["data"]["page_index"] for f in events(frames, PAGE_FINAL)] == [3, 0, 2, 1]


async def test_every_event_carries_seq_and_stream_id(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    """Three identifiers, three jobs: stream_id resumes, seq proves
    completeness, page_index reassembles."""
    await store.init_job("j", 2)
    for page in range(2):
        await publisher.publish("j", PAGE_FINAL, {"page_index": page})
    await publisher.publish("j", JOB_COMPLETE, {"total_pages": 2})

    _, frames = await collect(client, "j")
    stored = [f for f in frames if "id" in f]

    assert [f["data"]["seq"] for f in stored] == [1, 2, 3]
    assert all(f["data"]["stream_id"] == f["id"] for f in stored)


# ----------------------------------------------------------------- catch-up


async def test_events_published_before_subscribing_are_replayed(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    """Why the generator is catch-up-then-tail rather than tail-only.

    A client cannot subscribe before the first event: ingestion returns a job
    id and only then does it connect, by which time the 50ms layout stage has
    already committed pages. Starting at `$` - what most SSE examples do -
    would skip exactly those, and the client would sit at 0/5 waiting for
    events that were published before it arrived.
    """
    await store.init_job("j", 5)
    for page in range(5):
        await publisher.publish("j", PAGE_FINAL, {"page_index": page})
    await publisher.publish("j", JOB_COMPLETE, {"total_pages": 5})

    _, frames = await collect(client, "j")

    assert len(events(frames, PAGE_FINAL)) == 5


async def test_stream_open_reports_progress_so_a_late_client_can_size_itself(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    from app.queue.state import PageState

    await store.init_job("j", 4)
    for page in range(2):
        await store.transition("j", page, PageState.FAILED)
    await publisher.publish("j", PAGE_FINAL, {"page_index": 0})

    _, frames = await collect(client, "j", max_frames=3)

    assert frames[1]["data"]["done"] == 2


# ------------------------------------------------------------------- resume


async def test_last_event_id_header_resumes_without_redelivering(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    await store.init_job("j", 3)
    first = await publisher.publish("j", PAGE_FINAL, {"page_index": 0})
    await publisher.publish("j", PAGE_FINAL, {"page_index": 1})
    await publisher.publish("j", JOB_COMPLETE, {"total_pages": 3})

    _, frames = await collect(client, "j", headers={"Last-Event-ID": first.entry_id})

    assert [f["data"]["page_index"] for f in events(frames, PAGE_FINAL)] == [1]
    assert frames[1]["data"]["resumed_from"] == first.entry_id


async def test_the_query_parameter_is_an_alternative_to_the_header(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    """Browsers send the header automatically; curl and the benchmark cannot,
    and an endpoint only resumable from a browser is not resumable."""
    await store.init_job("j", 2)
    first = await publisher.publish("j", PAGE_FINAL, {"page_index": 0})
    await publisher.publish("j", PAGE_FINAL, {"page_index": 1})
    await publisher.publish("j", JOB_COMPLETE, {"total_pages": 2})

    _, frames = await collect(client, "j", params={"last_event_id": first.entry_id})

    assert [f["data"]["page_index"] for f in events(frames, PAGE_FINAL)] == [1]


async def test_the_header_wins_over_a_stale_query_parameter(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    """A browser sets the header on reconnect without being asked, so it
    reflects reality; the URL is whatever the client built once."""
    await store.init_job("j", 3)
    a = await publisher.publish("j", PAGE_FINAL, {"page_index": 0})
    b = await publisher.publish("j", PAGE_FINAL, {"page_index": 1})
    await publisher.publish("j", PAGE_FINAL, {"page_index": 2})
    await publisher.publish("j", JOB_COMPLETE, {"total_pages": 3})

    _, frames = await collect(
        client,
        "j",
        headers={"Last-Event-ID": b.entry_id},
        params={"last_event_id": a.entry_id},
    )

    assert [f["data"]["page_index"] for f in events(frames, PAGE_FINAL)] == [2]


async def test_a_malformed_last_event_id_falls_back_to_a_snapshot(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    """An arbitrary header must not be able to produce a 500."""
    await store.init_job("j", 1)
    await publisher.publish("j", PAGE_FINAL, {"page_index": 0})
    await publisher.publish("j", JOB_COMPLETE, {"total_pages": 1})

    status, frames = await collect(client, "j", headers={"Last-Event-ID": "$$$"})

    assert status == 200
    assert len(events(frames, PAGE_FINAL)) == 1
    assert frames[1]["data"]["resumed_from"] is None


async def test_a_trimmed_cursor_produces_an_explicit_gap_event(
    client: AsyncClient, redis: Redis, store: PageStateStore
) -> None:
    """The failure XREAD will not report.

    Resuming from a trimmed id succeeds and silently omits everything between
    the cursor and the retained window, so the client is permanently short a
    run of pages and is never told. The gap event names the authoritative
    source instead, which - unlike this stream - is never trimmed.
    """
    await store.init_job("j", 1)
    tight = ResultPublisher(redis, maxlen=5, ttl_s=60)
    stale = await tight.publish("j", PAGE_PARTIAL, {"page_index": 0})
    for page in range(1, 200):
        await tight.publish("j", PAGE_FINAL, {"page_index": page})
    await tight.publish("j", JOB_COMPLETE, {"total_pages": 1})

    _, frames = await collect(client, "j", headers={"Last-Event-ID": stale.entry_id})

    (gap,) = events(frames, GAP)
    assert "trimmed" in gap["data"]["message"]
    assert "GET /jobs/j" in gap["data"]["message"], "must name where to recover from"
    assert frames[1]["data"]["resumed_from"] is None, "the bad cursor is discarded"


# -------------------------------------------------------------- termination


async def test_job_complete_closes_the_stream(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    """The response must END, not merely stop emitting - a client waiting on a
    connection that never closes cannot tell 'finished' from 'stalled'."""
    await store.init_job("j", 1)
    await publisher.publish("j", PAGE_FINAL, {"page_index": 0})
    await publisher.publish("j", JOB_COMPLETE, {"total_pages": 1, "done": 1})

    # No max_frames: if the stream did not close, this hits the timeout.
    _, frames = await collect(client, "j", max_frames=10_000, timeout=6.0)

    assert frames[-1]["event"] == JOB_COMPLETE
    assert frames[-1]["data"].get("derived") is None, "the real event, not the backstop"


async def test_a_finished_job_terminates_even_if_job_complete_was_lost(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    """The termination backstop.

    `job.complete` is published by whichever worker finishes the last page, so
    it can be missed - a worker killed between the state commit and the
    publish, or a stream trimmed before a late subscriber read it. Without this
    check a client would block on an already-finished job until the duration
    limit. Committed state is the authority; the notification is not.
    """
    await store.init_job("j", 2)
    from app.queue.state import PageState

    for page in range(2):
        await store.transition("j", page, PageState.FAILED)
    # Page events exist, but the job.complete that should follow them does not.
    await publisher.publish("j", PAGE_FINAL, {"page_index": 0})
    await publisher.publish("j", PAGE_FINAL, {"page_index": 1})

    _, frames = await collect(client, "j", max_frames=10_000, timeout=6.0)

    assert frames[-1]["event"] == JOB_COMPLETE
    assert frames[-1]["data"]["derived"] is True, "flagged, so a log can count misses"
    assert frames[-1]["data"]["done"] == 2


async def test_an_unfinished_job_heartbeats_instead_of_closing(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    """A job whose pages are all waiting on a saturated VLM is legitimately
    idle for minutes. Heartbeats keep proxies from cutting the connection -
    and writing to the socket is the only way the server discovers that the
    client has gone."""
    await store.init_job("j", 10)
    await publisher.publish("j", PAGE_PARTIAL, {"page_index": 0})

    _, frames = await collect(client, "j", max_frames=5, timeout=6.0)

    assert sum(1 for f in frames if "comment" in f) >= 2
    assert not events(frames, JOB_COMPLETE)


async def test_the_stream_has_a_hard_lifetime(
    client: AsyncClient, store: PageStateStore
) -> None:
    """A stream is not allowed to live forever: a job that never completes -
    a page stranded in a dead worker's pending list - would otherwise pin a
    connection and a Redis socket indefinitely. Safe only because resume
    works, so the client reconnects and loses nothing."""
    await store.init_job("j", 10)

    _, frames = await collect(client, "j", max_frames=10_000, timeout=15.0)

    (gap,) = events(frames, GAP)
    assert "reconnect" in gap["data"]["message"]


# ------------------------------------------------------------ subscriber cap


async def test_the_subscriber_cap_refuses_with_503_not_a_hang(
    client: AsyncClient, store: PageStateStore
) -> None:
    """A subscriber pins a Redis connection for as long as it is blocked in
    XREAD, so this is a resource bound rather than a policy.

    Refused BEFORE the body starts, so it is a real status code with a
    Retry-After. Left uncapped the extra client would instead wait out the pool
    timeout and receive a truncated response - an exhaustion reported as a
    transport error, which is the least actionable outcome available.
    """
    await store.init_job("j", 10)
    app = client._transport.app  # type: ignore[attr-defined]
    app.state.sse_slots.active = SUBSCRIBER_LIMIT

    response = await client.get("/jobs/j/stream")

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.json()["detail"]["error"] == "too_many_subscribers"
    assert app.state.sse_slots.refused == 1


async def test_a_slot_is_released_when_the_stream_ends(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    await store.init_job("j", 1)
    await publisher.publish("j", PAGE_FINAL, {"page_index": 0})
    await publisher.publish("j", JOB_COMPLETE, {"total_pages": 1})
    slots = client._transport.app.state.sse_slots  # type: ignore[attr-defined]

    for _ in range(SUBSCRIBER_LIMIT + 2):
        await collect(client, "j", max_frames=10_000, timeout=6.0)

    assert slots.active == 0, "slots leaked; the cap would exhaust permanently"
    assert slots.refused == 0
    assert slots.peak == 1


async def test_a_slot_is_released_when_the_client_disconnects_early(
    client: AsyncClient, store: PageStateStore, publisher: ResultPublisher
) -> None:
    """The COMMON case, and the one a release placed after the loop would miss.

    A client that closes mid-stream raises GeneratorExit inside the generator
    rather than letting it run to completion, so the release has to sit in a
    `finally`. Otherwise every abandoned connection leaks a slot and the cap
    exhausts permanently - a leak that only appears under real client
    behaviour, never in a test that reads to the end.
    """
    await store.init_job("j", 50)
    await publisher.publish("j", PAGE_PARTIAL, {"page_index": 0})
    slots = client._transport.app.state.sse_slots  # type: ignore[attr-defined]

    for _ in range(SUBSCRIBER_LIMIT + 2):
        # max_frames well short of the job: this abandons the stream.
        await collect(client, "j", max_frames=3, timeout=6.0)

    assert slots.active == 0
    assert slots.refused == 0


async def test_the_stats_endpoint_reports_occupancy(
    client: AsyncClient, store: PageStateStore
) -> None:
    """A cap is only defensible if the number of times it fired is visible."""
    await store.init_job("j", 1)
    client._transport.app.state.sse_slots.active = SUBSCRIBER_LIMIT  # type: ignore

    await client.get("/jobs/j/stream")
    stats = (await client.get("/streams")).json()

    assert stats == {
        "active": SUBSCRIBER_LIMIT,
        "limit": SUBSCRIBER_LIMIT,
        "peak": 0,
        "refused": 1,
    }


def test_the_cap_is_exact_because_it_is_checked_and_taken_together() -> None:
    """Acquire is one atomic step on a single-threaded loop.

    Checking the count in the route and taking the slot in the generator would
    leave a window where concurrent arrivals all pass the same check - the
    identical mistake measured at 130% overshoot in Step 11's admission
    control. There is no await inside acquire(), so no task can interleave.
    """
    slots = stream_api.SubscriberSlots(4)

    taken = [slots.acquire() for _ in range(10)]

    assert taken.count(True) == 4
    assert slots.active == 4
    assert slots.refused == 6
