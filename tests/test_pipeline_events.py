"""What the pipeline actually publishes, and what it must NOT publish twice.

Separate from test_result_stream.py, which tests the stream primitive, and from
test_sse.py, which tests the wire. This is the join: the rule that an event is
emitted if and only if THIS caller performed the state transition.

That rule is what makes at-least-once delivery survivable on the notification
path. The state CAS already makes a redelivered page a no-op; gating the event
on the same CAS makes the notification a no-op too. Without it a page handed to
two workers produces two `page.final` events, a client counting events sees 21
for a 20-page job, and it can never reconcile with `total_pages`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from redis.asyncio import Redis

from app.queue.results import (
    JOB_COMPLETE,
    PAGE_FINAL,
    PAGE_PARTIAL,
    ResultPublisher,
    ResultReader,
)
from app.queue.state import PageState, PageStateStore
from app.ratelimit.breaker import CircuitOpen
from app.worker.pipeline import FALLBACK_CONFIDENCE, Outcome, process_page

JOB = "ev-job"


class FakeModelClient:
    """`**_` absorbs kwargs the real client gains over time - the same guard as
    in test_pipeline.py, for the same reason it was needed three times."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()

    async def layout(self, job_id: str, page_index: int, **_: object) -> dict[str, Any]:
        if "layout" in self.fail_on:
            raise RuntimeError("layout exploded")
        return {"model": "fast-layout", "boxes": [{"x": 1, "y": 2, "w": 3, "h": 4}]}

    async def vlm(self, job_id: str, page_index: int, **_: object) -> dict[str, Any]:
        if "vlm" in self.fail_on:
            raise RuntimeError("vlm exploded")
        if "vlm_open" in self.fail_on:
            raise CircuitOpen("vlm", retry_after_s=5.0)
        return {"model": "vlm", "text": "hello", "confidence": 0.97}


@pytest.fixture
def publisher(redis: Redis) -> ResultPublisher:
    return ResultPublisher(redis, maxlen=256, ttl_s=60)


@pytest.fixture
def reader(redis: Redis) -> ResultReader:
    return ResultReader(redis)


async def run(
    store: PageStateStore,
    publisher: ResultPublisher,
    *,
    page: int = 0,
    client: FakeModelClient | None = None,
    final_attempt: bool = False,
):
    return await process_page(
        JOB,
        page,
        store=store,
        client=client or FakeModelClient(),
        results=publisher,
        final_attempt=final_attempt,
    )


# --------------------------------------------------------------- happy path


async def test_one_page_produces_partial_then_final(
    store: PageStateStore, publisher: ResultPublisher, reader: ResultReader
) -> None:
    """Exactly two events, in that order, for the same page_index."""
    await store.init_job(JOB, 1)
    await run(store, publisher)

    history = await reader.history(JOB)

    assert [e.event for e in history] == [PAGE_PARTIAL, PAGE_FINAL, JOB_COMPLETE]
    assert [e.data["page_index"] for e in history[:2]] == [0, 0]
    assert history[0].data["complete"] is False
    assert history[1].data["complete"] is True


async def test_the_partial_is_published_before_the_vlm_is_called(
    store: PageStateStore, publisher: ResultPublisher, reader: ResultReader
) -> None:
    """The time-to-first-page claim, tested rather than asserted in a comment.

    The partial must be observable while the heavy stage is still running. If
    it were published after, first-byte latency would be bounded below by a
    1.5-3s VLM call and the 200ms target would be unreachable by construction.
    """
    await store.init_job(JOB, 1)
    seen_during_vlm: list[str] = []

    class SlowVlm(FakeModelClient):
        async def vlm(self, job_id: str, page_index: int, **_: object) -> dict[str, Any]:
            seen_during_vlm.extend(e.event for e in await reader.history(JOB))
            return {"model": "vlm", "confidence": 0.9}

    await run(store, publisher, client=SlowVlm())

    assert seen_during_vlm == [PAGE_PARTIAL], "the partial had not been published yet"


async def test_the_partial_carries_the_layout_result_itself(
    store: PageStateStore, publisher: ResultPublisher, reader: ResultReader
) -> None:
    """Not just a notification. Carrying the payload is what lets a client
    render at 50ms without a REST round trip per page - which at 1,000 pages
    would be 1,000 extra requests against the service it is streaming from."""
    await store.init_job(JOB, 1)
    await run(store, publisher)

    (partial,) = [e for e in await reader.history(JOB) if e.event == PAGE_PARTIAL]

    assert partial.data["layout"]["boxes"] == [{"x": 1, "y": 2, "w": 3, "h": 4}]


async def test_the_final_carries_confidence_and_the_vlm_result(
    store: PageStateStore, publisher: ResultPublisher, reader: ResultReader
) -> None:
    await store.init_job(JOB, 1)
    await run(store, publisher)

    (final,) = [e for e in await reader.history(JOB) if e.event == PAGE_FINAL]

    assert final.data["state"] == PageState.DONE.value
    assert final.data["confidence"] == 0.97
    assert final.data["degraded"] is False
    assert final.data["vlm"]["text"] == "hello"


# -------------------------------------------------------------- idempotency


async def test_a_redelivered_page_does_not_publish_a_second_final(
    store: PageStateStore, publisher: ResultPublisher, reader: ResultReader
) -> None:
    """THE rule. At-least-once delivery means this WILL happen.

    A worker that finishes a page and dies before XACK has its task
    redelivered. The state CAS makes the reprocessing a no-op; this asserts the
    notification is a no-op too, because a client that counts events against
    total_pages would otherwise never see the two agree.
    """
    await store.init_job(JOB, 1)
    await run(store, publisher)
    result = await run(store, publisher)

    assert result.outcome is Outcome.ALREADY_TERMINAL
    assert [e.event for e in await reader.history(JOB)] == [
        PAGE_PARTIAL,
        PAGE_FINAL,
        JOB_COMPLETE,
    ]


async def test_a_resumed_page_does_not_republish_its_partial(
    store: PageStateStore, publisher: ResultPublisher, reader: ResultReader
) -> None:
    """A page recovered mid-flight re-runs only the uncommitted stage, so the
    already-committed layout must not be announced again."""
    await store.init_job(JOB, 1)
    # First attempt banks layout, then the VLM stage explodes into FAILED.
    await run(store, publisher, client=FakeModelClient(fail_on={"vlm"}))
    before = [e.event for e in await reader.history(JOB)]

    # Reset to the checkpoint the reaper would restore, then re-run.
    await store.transition(JOB, 0, PageState.LAYOUT_DONE, allowed_from=[PageState.FAILED])
    await run(store, publisher)

    after = [e.event for e in await reader.history(JOB)]

    assert before.count(PAGE_PARTIAL) == 1
    assert after.count(PAGE_PARTIAL) == 1, "layout was re-announced without re-running"


async def test_job_complete_is_published_exactly_once_under_concurrency(
    store: PageStateStore, publisher: ResultPublisher, reader: ResultReader
) -> None:
    """Uniqueness with no lock, no flag and no coordination.

    HINCRBY is atomic and Redis is single threaded, so of N workers finishing
    the last N pages exactly one receives a return value equal to total_pages -
    and `total_pages` is read inside the same script, so the comparison cannot
    be raced either. Read outside it, two workers could both see done == total
    and publish twice, or neither could and the stream would never close.
    """
    await store.init_job(JOB, 12)

    await asyncio.gather(*(run(store, publisher, page=p) for p in range(12)))

    history = await reader.history(JOB, count=1_000)
    completes = [e for e in history if e.event == JOB_COMPLETE]

    assert len(completes) == 1
    assert completes[0].data == {
        **completes[0].data,
        "total_pages": 12,
        "done": 12,
    }
    assert history[-1].event == JOB_COMPLETE, "it must be the LAST event"
    assert sum(1 for e in history if e.event == PAGE_FINAL) == 12


# ----------------------------------------------------------- degraded, failed


async def test_a_degraded_page_still_gets_a_final(
    store: PageStateStore, publisher: ResultPublisher, reader: ResultReader
) -> None:
    """A degraded page is terminal and carries output, so the client must stop
    waiting for an upgrade that will never come. The `degraded` flag and the
    low confidence are the difference - not a separate event type, which would
    make every client handle three terminal cases instead of one."""
    await store.init_job(JOB, 1)

    result = await run(
        store,
        publisher,
        client=FakeModelClient(fail_on={"vlm_open"}),
        final_attempt=True,
    )

    assert result.outcome is Outcome.DEGRADED
    (final,) = [e for e in await reader.history(JOB) if e.event == PAGE_FINAL]
    assert final.data["state"] == PageState.FALLBACK_DONE.value
    assert final.data["degraded"] is True
    assert final.data["confidence"] == FALLBACK_CONFIDENCE
    assert final.data["complete"] is True


async def test_a_failed_page_is_announced_rather_than_silently_dropped(
    store: PageStateStore, publisher: ResultPublisher, reader: ResultReader
) -> None:
    """Silence would be the worst outcome for a subscriber: the job never
    reaches done == total from its point of view, so it waits for an event that
    cannot arrive. "0% unhandled" has to mean VISIBLE, not merely counted."""
    await store.init_job(JOB, 1)

    result = await run(store, publisher, client=FakeModelClient(fail_on={"layout"}))

    assert result.outcome is Outcome.FAILED
    history = await reader.history(JOB)
    assert [e.event for e in history] == [PAGE_FINAL, JOB_COMPLETE]
    assert history[0].data["state"] == PageState.FAILED.value
    assert "layout exploded" in history[0].data["error"]


async def test_a_saturated_page_publishes_nothing(
    store: PageStateStore, publisher: ResultPublisher, reader: ResultReader
) -> None:
    """It will be retried at full fidelity, so announcing it would tell the
    client something that is not yet true about a page that has lost nothing."""
    await store.init_job(JOB, 1)

    result = await run(
        store, publisher, client=FakeModelClient(fail_on={"vlm_open"})
    )

    assert result.outcome is Outcome.SATURATED
    # The partial is legitimate: layout really did commit. The absence of a
    # final is the point.
    assert [e.event for e in await reader.history(JOB)] == [PAGE_PARTIAL]


# ------------------------------------------------------------------- safety


async def test_the_pipeline_works_with_no_publisher_at_all(
    store: PageStateStore, reader: ResultReader
) -> None:
    """Streaming is an addition, not a dependency. A worker built without a
    publisher - as every test written before this step does - must still
    process pages."""
    await store.init_job(JOB, 1)

    result = await process_page(
        JOB, 0, store=store, client=FakeModelClient(), results=None
    )

    assert result.outcome is Outcome.COMPLETED
    assert await reader.history(JOB) == []


async def test_a_broken_publisher_cannot_fail_a_page(
    store: PageStateStore,
) -> None:
    """The observability path must never break the thing it observes.

    A Redis hiccup while publishing would otherwise turn a page that was
    processed perfectly - and whose result is already durably committed - into
    a FAILED page. Losing a notification is recoverable; losing the page is not.
    """
    await store.init_job(JOB, 1)

    class Broken(ResultPublisher):
        async def publish(self, *a: object, **k: object):  # type: ignore[override]
            raise ConnectionError("redis is on fire")

    result = await process_page(
        JOB,
        0,
        store=store,
        client=FakeModelClient(),
        results=Broken.__new__(Broken),
    )

    assert result.outcome is Outcome.COMPLETED
    assert (await store.get_page(JOB, 0))["state"] == PageState.DONE.value
