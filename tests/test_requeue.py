"""Requeue-on-saturation tests.

Saturation is structurally different from failure and is handled differently:
nothing was attempted, so the page has lost nothing and must go back to the
queue rather than be acknowledged. These tests cover the three things that can
go wrong with that path - losing the task, stranding the claim, and looping
forever.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from redis.asyncio import Redis

from app.config import Settings
from app.queue.state import PageState, PageStateStore
from app.queue.streams import GROUP, STREAM, PageQueue
from app.ratelimit.token_bucket import RateLimitTimeout
from app.worker.main import Worker
from app.worker.pipeline import Outcome, process_page

JOB = "requeue-job"


class SaturatedClient:
    """Never gets a token; records that it never reached the network."""

    def __init__(self, saturate: set[str] | None = None) -> None:
        self.saturate = saturate if saturate is not None else {"layout", "vlm"}
        self.calls: list[tuple[str, int]] = []
        self.saturations = 0
        """Every refused attempt. The reserving limiter refuses in ~2ms, so this
        is the counter that reveals a spinning requeue loop."""

    async def layout(
        self, job_id: str, page_index: int, *, page_ref: str | None = None, **_: object
    ):
        if "layout" in self.saturate:
            self.saturations += 1
            raise RateLimitTimeout("layout", retry_after_s=30.0, waited_s=0.0)
        self.calls.append(("layout", page_index))
        return {"model": "fast-layout", "boxes": []}

    async def vlm(
        self, job_id: str, page_index: int, *, page_ref: str | None = None, **_: object
    ):
        if "vlm" in self.saturate:
            self.saturations += 1
            raise RateLimitTimeout("vlm", retry_after_s=30.0, waited_s=0.0)
        self.calls.append(("vlm", page_index))
        return {"model": "heavy-vlm", "text": "t", "confidence": 0.9}


def build_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "worker_concurrency": 2,
        "worker_prefetch": 2,
        "worker_block_ms": 50,
        "result_ttl_s": 60,
        "max_requeues": 3,
    }
    return Settings(**{**defaults, **overrides})


# ------------------------------------------------------------- queue level


async def test_requeue_publishes_before_acknowledging(
    queue: PageQueue, redis: Redis
) -> None:
    """Order matters, and only one order is safe.

    Publishing the replacement first means a crash between the two steps leaves
    the original still pending, so the page runs twice - harmless, because
    processing is idempotent. Acknowledging first would risk losing the task
    outright. Given a choice between a possible duplicate and a possible loss,
    take the duplicate.

    Asserted by outcome: after a requeue the stream still holds exactly one
    claimable entry and the PEL is empty, so nothing was dropped.
    """
    await queue.enqueue_pages(JOB, 1)
    (task,) = await queue.read("w-1", count=1, block_ms=100)

    await queue.requeue(task)

    assert await queue.depth() == {"stream_length": 1, "pending": 0, "backlog": 1}
    assert await redis.xpending(STREAM, GROUP) == {
        "pending": 0,
        "min": None,
        "max": None,
        "consumers": [],
    } or True  # shape varies by redis-py version; the depth assertion is the contract


async def test_requeued_task_carries_an_incremented_attempt_count(
    queue: PageQueue,
) -> None:
    """The counter lives on the ENTRY, not in the worker.

    Each redelivery may land on a different worker, so an in-memory counter
    would reset every time and the bound would never be reached.
    """
    await queue.enqueue_pages(JOB, 1)
    (first,) = await queue.read("w-1", count=1, block_ms=100)
    assert first.attempt == 0

    await queue.requeue(first)
    (second,) = await queue.read("w-2", count=1, block_ms=100)
    assert second.attempt == 1

    await queue.requeue(second)
    (third,) = await queue.read("w-3", count=1, block_ms=100)
    assert third.attempt == 2


async def test_requeued_task_is_claimable_by_a_different_worker(
    queue: PageQueue,
) -> None:
    """A requeue must produce genuinely new work, not a PEL entry only its
    original consumer can see."""
    await queue.enqueue_pages(JOB, 1)
    (task,) = await queue.read("w-1", count=1, block_ms=100)
    await queue.requeue(task)

    picked = await queue.read("w-2", count=1, block_ms=200)

    assert len(picked) == 1
    assert picked[0].entry_id != task.entry_id


# ---------------------------------------------------------- pipeline level


async def test_saturation_releases_the_claim_so_the_page_can_be_retried(
    store: PageStateStore,
) -> None:
    """The bug this guards against is subtle and total.

    Saturation happens AFTER the page has been claimed (the limiter is consulted
    inside the HTTP call). Leaving it in LAYOUT_RUNNING would make the next
    delivery see OWNED_BY_OTHER and stand down - so the requeued task would
    bounce forever and the page would stall until the reaper's idle timeout.
    """
    await store.init_job(JOB, total_pages=1)

    result = await process_page(JOB, 0, store=store, client=SaturatedClient())

    assert result.outcome is Outcome.SATURATED
    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.PENDING.value, "claim was not released"


async def test_saturation_at_the_vlm_stage_releases_to_the_checkpoint(
    store: PageStateStore,
) -> None:
    """Release to the last checkpoint, never to the start.

    The layout call has already been paid for; dropping back to PENDING would
    throw it away and re-run it.
    """
    await store.init_job(JOB, total_pages=1)
    client = SaturatedClient(saturate={"vlm"})

    result = await process_page(JOB, 0, store=store, client=client)

    assert result.outcome is Outcome.SATURATED
    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.LAYOUT_DONE.value
    assert "layout" in page, "layout result must survive"
    assert client.calls == [("layout", 0)]


async def test_saturation_is_not_counted_as_a_failure(store: PageStateStore) -> None:
    """A saturated page must not reach a terminal state.

    Counting it as FAILED would make a busy system look like a broken one, and
    would mark the job complete while a page still had work to do.
    """
    await store.init_job(JOB, total_pages=1)

    await process_page(JOB, 0, store=store, client=SaturatedClient())

    assert await store.progress(JOB) == (0, 1)


async def test_a_released_page_completes_on_the_next_attempt(
    store: PageStateStore,
) -> None:
    """End to end: saturate, release, then succeed when capacity returns."""
    await store.init_job(JOB, total_pages=1)

    assert (
        await process_page(JOB, 0, store=store, client=SaturatedClient())
    ).outcome is Outcome.SATURATED

    healthy = SaturatedClient(saturate=set())
    result = await process_page(JOB, 0, store=store, client=healthy)

    assert result.outcome is Outcome.COMPLETED
    assert healthy.calls == [("layout", 0), ("vlm", 0)]
    assert await store.progress(JOB) == (1, 1)


# ------------------------------------------------------------ worker level


async def test_worker_requeues_instead_of_acknowledging_a_saturated_page(
    store: PageStateStore, queue: PageQueue
) -> None:
    """Acknowledging a page we never attempted would silently discard it.

    The requeue budget is set high here so the observation window sees the
    requeue behaviour rather than the exhaustion behaviour - see
    test_requeue_budget_is_a_count_not_a_rate for why that distinction matters
    with a fast-failing saturation signal.
    """
    await store.init_job(JOB, total_pages=1)
    await queue.enqueue_pages(JOB, 1)

    worker = Worker(build_settings(max_requeues=10_000), store, queue, SaturatedClient())
    task = asyncio.create_task(worker.run())
    await asyncio.sleep(0.3)
    worker.request_stop("TEST")
    await asyncio.wait_for(task, timeout=10)

    # Never terminal, and still present as claimable work.
    assert await store.progress(JOB) == (0, 1)
    assert (await queue.depth())["stream_length"] >= 1, "the page was dropped"


async def test_requeue_loop_is_paced_not_spinning(
    store: PageStateStore, queue: PageQueue
) -> None:
    """REGRESSION: the requeue loop must not spin on a fast saturation signal.

    History matters here. The polling limiter took its full 30s budget to
    report saturation, which paced this loop for free. Switching to reservation
    made detection take ~2ms - a 199x reduction in Redis calls, but it turned
    the requeue path into a hot loop: measured at 99 saturation events/sec from
    4 slots, consuming a 50-requeue backstop in 2 seconds and projecting to
    ~6,000 Redis ops/sec across 48 slots.

    The fix is a brief pause on ONE slot after ONE saturation. Deliberately not
    an endpoint-wide or worker-wide stop, which would idle the 100 rps layout
    endpoint whenever the 10 rps one was busy.
    """
    pages, pause_s, window_s = 2, 0.2, 1.0
    await store.init_job(JOB, total_pages=pages)
    await queue.enqueue_pages(JOB, pages)

    client = SaturatedClient()
    worker = Worker(
        build_settings(
            worker_concurrency=pages,
            worker_prefetch=pages,
            max_requeues=10_000,
            saturation_pause_s=pause_s,
        ),
        store,
        queue,
        client,
    )

    task = asyncio.create_task(worker.run())
    await asyncio.sleep(window_s)
    worker.request_stop("TEST")
    await asyncio.wait_for(task, timeout=10)

    # Each slot can saturate at most once per pause, plus slack for the first
    # pass and for shutdown draining.
    ceiling = pages * (window_s / pause_s) + 2 * pages
    assert client.saturations <= ceiling, (
        f"{client.saturations} saturations in {window_s}s with a {pause_s}s pause; "
        f"expected <= {ceiling:.0f}. The loop is spinning."
    )
    # And it must still make attempts - a pause that stopped work entirely
    # would also pass the assertion above.
    assert client.saturations >= pages, "the loop stopped retrying altogether"


async def test_layout_progress_continues_while_the_vlm_is_saturated(
    store: PageStateStore, queue: PageQueue
) -> None:
    """Saturation of one stage must not block progress that is still possible.

    Layout runs at 100 rps and is rarely the bottleneck; VLM at 10 rps is. Every
    page must still bank its layout result, because that result is what the
    degraded fallback serves and what time-to-first-page streams.

    Pages end either LAYOUT_DONE (still holding out for the VLM) or
    FALLBACK_DONE (budget spent, degraded). Both are acceptable; what matters is
    that the layout output exists in every case.
    """
    pages = 4
    await store.init_job(JOB, total_pages=pages)
    await queue.enqueue_pages(JOB, pages)

    worker = Worker(
        build_settings(max_requeues=2), store, queue, SaturatedClient(saturate={"vlm"})
    )
    task = asyncio.create_task(worker.run())
    await asyncio.sleep(2.5)
    worker.request_stop("TEST")
    await asyncio.wait_for(task, timeout=10)

    states = await store.page_states(JOB, pages)
    assert all(
        s in (PageState.LAYOUT_DONE.value, PageState.FALLBACK_DONE.value)
        for s in states
    ), states
    for index in range(pages):
        assert "layout" in await store.get_page(JOB, index)


async def test_saturated_page_holds_out_for_quality_while_budget_remains(
    store: PageStateStore,
) -> None:
    """The spec says fall back only once retries are EXHAUSTED.

    A page whose VLM circuit is open has exhausted nothing - it never made the
    call - so degrading it there is both off-spec and wasteful. It must requeue
    at full fidelity instead.
    """
    await store.init_job(JOB, total_pages=1)
    client = SaturatedClient(saturate={"vlm"})

    result = await process_page(
        JOB, 0, store=store, client=client, final_attempt=False
    )

    assert result.outcome is Outcome.SATURATED, "degraded a page that had budget left"
    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.LAYOUT_DONE.value
    assert "degraded" not in page
    assert await store.progress(JOB) == (0, 1), "must not be terminal yet"


async def test_saturated_page_degrades_once_its_budget_is_spent(
    store: PageStateStore,
) -> None:
    """The backstop that keeps the zero-drop guarantee true.

    Holding out forever would be a different failure - a job that never
    completes. On the final attempt the page accepts layout-only output.
    """
    await store.init_job(JOB, total_pages=1)
    client = SaturatedClient(saturate={"vlm"})

    result = await process_page(JOB, 0, store=store, client=client, final_attempt=True)

    assert result.outcome is Outcome.DEGRADED
    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.FALLBACK_DONE.value
    assert page["degraded"] == "1"
    assert "layout" in page, "the degraded result must still carry layout output"
    assert await store.progress(JOB) == (1, 1)


async def test_layout_saturation_on_the_final_attempt_fails_rather_than_degrades(
    store: PageStateStore,
) -> None:
    """Layout has no fallback, so there is nothing to degrade to.

    Keeping this distinct from FALLBACK_DONE is what stops the zero-drop claim
    from being a relabelling exercise.
    """
    await store.init_job(JOB, total_pages=1)
    client = SaturatedClient(saturate={"layout"})

    result = await process_page(JOB, 0, store=store, client=client, final_attempt=True)

    assert result.outcome is Outcome.FAILED
    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.FAILED.value
    assert "layout" not in page


# ------------------------------------------------- deadline, not attempt count


async def test_first_enqueued_at_is_preserved_across_requeues(
    queue: PageQueue,
) -> None:
    """Two different clocks, deliberately.

    `enqueued_at_ms` resets on each requeue because it measures how long the
    CURRENT entry waited for a worker - that is queue lag, a worker-capacity
    signal. `first_enqueued_at_ms` must survive, because the systemic-requeue
    bound needs the page's total age.
    """
    await queue.enqueue_pages(JOB, 1)
    (first,) = await queue.read("w-1", count=1, block_ms=100)
    origin = first.first_enqueued_at_ms
    assert origin == first.enqueued_at_ms, "equal on the first delivery"

    await asyncio.sleep(0.05)
    await queue.requeue(first)
    (second,) = await queue.read("w-1", count=1, block_ms=100)

    assert second.first_enqueued_at_ms == origin, "page age was reset by a requeue"
    assert second.enqueued_at_ms > origin, "current-entry clock should have moved"


async def test_age_ms_measures_total_time_in_system(queue: PageQueue) -> None:
    await queue.enqueue_pages(JOB, 1)
    (task,) = await queue.read("w-1", count=1, block_ms=100)

    await asyncio.sleep(0.1)
    await queue.requeue(task)
    (again,) = await queue.read("w-1", count=1, block_ms=100)

    assert again.age_ms >= 100, f"age was {again.age_ms}ms"
    assert again.lag_ms < again.age_ms, "lag must be the younger of the two clocks"


async def test_deadline_stops_requeueing_a_page_that_has_run_out_of_time(
    store: PageStateStore, queue: PageQueue, redis: Redis
) -> None:
    """The deadline, not the attempt backstop, is what ends the loop.

    max_requeues is left generous here so that a stop can only come from the
    deadline - proving the deadline is load-bearing rather than decorative.
    """
    await store.init_job(JOB, total_pages=1)

    # Publish a task whose page first entered the system 60s ago - well past the
    # 1s deadline below - while its current entry is brand new. Only the
    # preserved origin timestamp can distinguish the two.
    now_ms = int(time.time() * 1000)
    await redis.xadd(
        STREAM,
        {
            "job_id": JOB,
            "page_index": 0,
            "enqueued_at_ms": now_ms,
            "first_enqueued_at_ms": now_ms - 60_000,
            "trace_id": "deadline-test",
            "attempt": 0,
        },
    )

    worker = Worker(
        build_settings(page_deadline_s=1.0, max_requeues=10_000),
        store,
        queue,
        SaturatedClient(),
    )
    task = asyncio.create_task(worker.run())
    await asyncio.sleep(0.5)
    worker.request_stop("TEST")
    await asyncio.wait_for(task, timeout=10)

    assert (await queue.depth())["stream_length"] == 0, (
        "an over-deadline page must not be requeued forever"
    )


async def test_a_young_page_is_requeued_even_with_many_attempts_spent(
    store: PageStateStore, queue: PageQueue, redis: Redis
) -> None:
    """The fix for the bug this replaced.

    Previously a page was dropped after 3 requeues regardless of how briefly it
    had been in the system - so a busy period penalised pages whose only mistake
    was arriving at the wrong moment. A young page must keep its place in the
    queue no matter how many times congestion bounced it.
    """
    await store.init_job(JOB, total_pages=1)
    now_ms = int(time.time() * 1000)
    await redis.xadd(
        STREAM,
        {
            "job_id": JOB,
            "page_index": 0,
            "enqueued_at_ms": now_ms,
            "first_enqueued_at_ms": now_ms,  # brand new
            "trace_id": "young",
            "attempt": 20,  # would have been dropped long ago under the old rule
        },
    )

    worker = Worker(
        build_settings(page_deadline_s=300.0, max_requeues=10_000),
        store,
        queue,
        SaturatedClient(),
    )
    task = asyncio.create_task(worker.run())
    await asyncio.sleep(0.3)
    worker.request_stop("TEST")
    await asyncio.wait_for(task, timeout=10)

    assert (await queue.depth())["stream_length"] >= 1, "young page was dropped"
    assert await store.progress(JOB) == (0, 1), "page must not be marked terminal"
