"""Worker dispatch-loop tests.

These verify the *bounded dispatch* invariant, which is the worker's primary
memory boundary: the worker must never hold more than `worker_concurrency`
pages, no matter how deep the queue is. Asserting it directly is the only way
the claim means anything - a semaphore-based version would pass a "how many run
at once" test while still having read the entire backlog into its PEL.

The model client is a probe that records peak simultaneous calls, so the bound
is measured rather than inferred.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.config import Settings
from app.queue.state import PageState, PageStateStore
from app.queue.streams import GROUP, STREAM, PageQueue
from app.worker.main import Worker

JOB = "worker-job"


class ConcurrencyProbe:
    """Model client that records how many calls are in flight simultaneously."""

    def __init__(self, delay_s: float = 0.02, fail_pages: set[int] | None = None) -> None:
        self.delay_s = delay_s
        self.fail_pages = fail_pages or set()
        self.current = 0
        self.peak = 0
        self.calls: list[tuple[str, int]] = []

    async def _call(self, stage: str, page_index: int) -> dict[str, Any]:
        self.calls.append((stage, page_index))
        self.current += 1
        self.peak = max(self.peak, self.current)
        try:
            await asyncio.sleep(self.delay_s)
            if page_index in self.fail_pages:
                raise RuntimeError(f"page {page_index} is poison")
            if stage == "layout":
                return {"model": "fast-layout", "boxes": []}
            return {"model": "heavy-vlm", "text": "t", "confidence": 0.9}
        finally:
            self.current -= 1

    async def layout(
        self, job_id: str, page_index: int, *, page_ref: str | None = None, **_: object
    ):
        return await self._call("layout", page_index)

    async def vlm(
        self, job_id: str, page_index: int, *, page_ref: str | None = None, **_: object
    ):
        return await self._call("vlm", page_index)


def build_settings(**overrides: Any) -> Settings:
    """Init kwargs take precedence over environment in pydantic-settings, so a
    developer's exported ORCH_* vars cannot skew these tests."""
    defaults: dict[str, Any] = {
        "worker_concurrency": 4,
        "worker_prefetch": 4,
        "worker_block_ms": 50,
        "result_ttl_s": 60,
    }
    return Settings(**{**defaults, **overrides})


async def run_until_complete(
    worker: Worker, store: PageStateStore, pages: int, timeout_s: float = 20.0
) -> None:
    """Run the dispatch loop until the job finishes, then stop it cleanly."""
    task = asyncio.create_task(worker.run())
    try:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            done, total = await store.progress(JOB)
            if total and done >= pages:
                break
            await asyncio.sleep(0.02)
        worker.request_stop("TEST")
        await asyncio.wait_for(task, timeout=timeout_s)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_worker_never_exceeds_its_concurrency_limit(
    store: PageStateStore, queue: PageQueue
) -> None:
    """The bound is the worker's memory ceiling; 60 queued pages must not breach it."""
    pages = 60
    await store.init_job(JOB, total_pages=pages)
    await queue.enqueue_pages(JOB, pages)

    probe = ConcurrencyProbe(delay_s=0.02)
    worker = Worker(build_settings(worker_concurrency=4, worker_prefetch=4), store, queue, probe)

    await run_until_complete(worker, store, pages)

    assert probe.peak <= 4, f"peak in-flight model calls was {probe.peak}, limit is 4"
    assert probe.peak > 1, "should actually be concurrent, not accidentally serial"
    assert await store.progress(JOB) == (pages, pages)


async def test_raising_concurrency_raises_observed_parallelism(
    store: PageStateStore, queue: PageQueue
) -> None:
    """Control for the test above: the bound tracks the setting, it is not a fluke."""
    pages = 60
    await store.init_job(JOB, total_pages=pages)
    await queue.enqueue_pages(JOB, pages)

    probe = ConcurrencyProbe(delay_s=0.02)
    worker = Worker(build_settings(worker_concurrency=12, worker_prefetch=8), store, queue, probe)

    await run_until_complete(worker, store, pages)

    assert 4 < probe.peak <= 12, f"peak was {probe.peak}, expected to scale toward 12"


async def test_worker_does_not_pull_the_whole_backlog_into_its_pending_list(
    store: PageStateStore, queue: PageQueue, redis: Any
) -> None:
    """The distinction between bounded dispatch and a semaphore, made testable.

    A semaphore-guarded version would read every task first and park the excess,
    so the PEL would hold the entire backlog - and a crash would leave all of it
    to reclaim. Bounded dispatch keeps the backlog in Redis, where it belongs.
    """
    pages = 80
    await store.init_job(JOB, total_pages=pages)
    await queue.enqueue_pages(JOB, pages)

    probe = ConcurrencyProbe(delay_s=0.05)
    worker = Worker(build_settings(worker_concurrency=4, worker_prefetch=4), store, queue, probe)

    peak_pending = 0
    task = asyncio.create_task(worker.run())
    try:
        for _ in range(60):
            summary = await redis.xpending(STREAM, GROUP)
            peak_pending = max(peak_pending, int(summary.get("pending", 0)))
            done, _total = await store.progress(JOB)
            if done >= pages:
                break
            await asyncio.sleep(0.02)
    finally:
        worker.request_stop("TEST")
        await asyncio.wait_for(task, timeout=20)

    assert peak_pending <= 4, (
        f"worker held {peak_pending} unacknowledged entries; bounded dispatch "
        "should keep this at or below the concurrency limit of 4"
    )


async def test_one_poisoned_page_does_not_stop_its_siblings(
    store: PageStateStore, queue: PageQueue
) -> None:
    """Failure isolation.

    Each page runs as its own task with its own error handling, so a page that
    raises cannot abandon the batch. `asyncio.gather` over a batch would let one
    unhandled exception take the rest down with it.
    """
    pages = 10
    await store.init_job(JOB, total_pages=pages)
    await queue.enqueue_pages(JOB, pages)

    probe = ConcurrencyProbe(delay_s=0.01, fail_pages={3, 7})
    worker = Worker(build_settings(worker_concurrency=4, worker_prefetch=4), store, queue, probe)

    await run_until_complete(worker, store, pages)

    states = await store.page_states(JOB, pages)
    assert states[3] == PageState.FAILED.value
    assert states[7] == PageState.FAILED.value
    healthy = [s for i, s in enumerate(states) if i not in (3, 7)]
    assert all(s == PageState.DONE.value for s in healthy), states
    # Every page terminal, so the job still completes rather than hanging.
    assert await store.progress(JOB) == (pages, pages)


async def test_every_task_is_acknowledged_so_the_stream_drains(
    store: PageStateStore, queue: PageQueue
) -> None:
    """Unacked entries accumulate forever; the stream must end up empty."""
    pages = 20
    await store.init_job(JOB, total_pages=pages)
    await queue.enqueue_pages(JOB, pages)

    worker = Worker(build_settings(), store, queue, ConcurrencyProbe(delay_s=0.01))
    await run_until_complete(worker, store, pages)

    assert await queue.depth() == {"stream_length": 0, "pending": 0, "backlog": 0}


async def test_shutdown_drains_in_flight_pages_instead_of_abandoning_them(
    store: PageStateStore, queue: PageQueue
) -> None:
    """SIGTERM must finish what is in hand.

    Dropping in-flight pages would leave them in a *_RUNNING state with no owner,
    recoverable only by the reaper's idle timeout - a needless delay when the
    process had time to finish cleanly.
    """
    pages = 8
    await store.init_job(JOB, total_pages=pages)
    await queue.enqueue_pages(JOB, pages)

    probe = ConcurrencyProbe(delay_s=0.15)
    worker = Worker(build_settings(worker_concurrency=8, worker_prefetch=8), store, queue, probe)

    task = asyncio.create_task(worker.run())
    # Let the loop pick up work and get pages in flight, then stop it mid-page.
    await asyncio.sleep(0.2)
    in_flight_at_stop = len(worker._in_flight)  # noqa: SLF001 - asserting internals is the point
    worker.request_stop("TEST")
    await asyncio.wait_for(task, timeout=20)

    assert in_flight_at_stop > 0, "test did not actually catch pages in flight"

    states = await store.page_states(JOB, pages)
    running = [s for s in states if s and s.endswith("_RUNNING")]
    assert running == [], f"pages abandoned mid-stage: {states}"

    # Nothing is UNACKNOWLEDGED: every page this worker held was either
    # finished or handed back, so none is waiting on a reaper timeout.
    depth = await queue.depth()
    assert depth["pending"] == 0, f"pages left in the PEL: {depth}"

    # But the queue is NOT empty, and that is correct. With the stage handoff a
    # page whose layout commits during the drain is requeued for its VLM stage
    # rather than holding the worker for up to 3s on a 10 rps endpoint. Its
    # layout result is durably committed and the entry is immediately claimable
    # by any surviving replica, so shutting down fast loses nothing.
    #
    # THE invariant: no page is both non-terminal and absent from the queue.
    # Every unfinished page has an entry waiting for a worker, so nothing has
    # to be rediscovered by a timeout and nothing is lost.
    #
    # Two non-terminal shapes are legitimate here and both are accounted for:
    # LAYOUT_DONE for a page handed off mid-drain, and PENDING for one the
    # worker never read because it had already stopped accepting work. What
    # would be a bug is a page in neither the queue nor a terminal state.
    unfinished = [s for s in states if s not in ("DONE", "FALLBACK_DONE", "FAILED")]
    assert depth["backlog"] == len(unfinished), (
        f"unfinished pages not all queued: {depth} {states}"
    )
    assert all(
        s in ("PENDING", "LAYOUT_DONE", "DONE") for s in states
    ), f"unexpected state after a clean shutdown: {states}"
    # And each queued page is at a checkpoint, never mid-stage.
    assert not any(s.endswith("_RUNNING") for s in states), states


@pytest.mark.parametrize("concurrency,prefetch", [(1, 1), (3, 1), (2, 8)])
async def test_dispatch_is_correct_across_concurrency_prefetch_combinations(
    store: PageStateStore, queue: PageQueue, concurrency: int, prefetch: int
) -> None:
    """prefetch > concurrency must still respect the concurrency ceiling.

    read() is called with min(prefetch, capacity), so an over-large prefetch is
    harmless. Getting that min() backwards is an easy mistake that this catches.
    """
    pages = 12
    await store.init_job(JOB, total_pages=pages)
    await queue.enqueue_pages(JOB, pages)

    probe = ConcurrencyProbe(delay_s=0.01)
    worker = Worker(
        build_settings(worker_concurrency=concurrency, worker_prefetch=prefetch),
        store,
        queue,
        probe,
    )

    await run_until_complete(worker, store, pages)

    assert probe.peak <= concurrency
    assert await store.progress(JOB) == (pages, pages)
