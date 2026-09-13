"""Recovering a worker's OWN interrupted pages.

A worker that is SIGKILLed leaves pages in a `*_RUNNING` state and its tasks in
its Pending Entries List. On restart it re-reads that PEL (`read_own_pending`)
so its work resumes immediately instead of waiting for a reaper's idle timeout.

The bug this file pins is the interaction between those two facts. The pipeline
treats a `*_RUNNING` page as "another worker owns this, stand down" - correct
when that worker is alive, and exactly wrong for a task that came out of THIS
worker's own PEL, because the owner was its own previous incarnation and is
gone. It stood down and then ACKED, so the page was left non-terminal with no
queue entry at all: unreachable by any worker, invisible to any reaper.

Measured live before the fix, with one `docker kill -9` during a 120-page run:

    108 DONE
     12 VLM_RUNNING      <- stranded, queue completely drained

`done_count` stalls at 108/120, so the job never reports complete and an SSE
subscriber waits out `sse_max_duration_s` for a `job.complete` that cannot
arrive. That is a zero-drop violation, not a delay.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.queue.state import PageState, PageStateStore
from app.queue.streams import LEAD_STREAM, PageQueue
from app.worker.pipeline import Outcome, process_page

JOB = "recover-job"


class FakeModelClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def layout(self, job_id: str, page_index: int, **_: object) -> dict[str, Any]:
        self.calls.append(("layout", page_index))
        return {"model": "fast-layout", "boxes": []}

    async def vlm(self, job_id: str, page_index: int, **_: object) -> dict[str, Any]:
        self.calls.append(("vlm", page_index))
        return {"model": "vlm", "text": "ok", "confidence": 0.9}

    def count(self, stage: str) -> int:
        return sum(1 for s, _ in self.calls if s == stage)


# ------------------------------------------------------- the queue side


async def test_own_pending_marks_tasks_as_recovered(queue: PageQueue) -> None:
    """The pipeline cannot infer this. A task carries no hint of whether it is
    a first delivery or a resumption, and the two need opposite handling of a
    `*_RUNNING` page, so the fact has to travel with the task."""
    await queue.enqueue_pages(JOB, 3)
    delivered = await queue.read("w-1", count=10, block_ms=100)

    assert all(not t.recovered for t in delivered)

    recovered = await queue.read_own_pending("w-1", count=10)

    assert len(recovered) == 3
    assert all(t.recovered for t in recovered)
    # And the lane is preserved, so the ack still targets the right stream.
    assert {t.stream for t in recovered} == {LEAD_STREAM, *{t.stream for t in delivered}}


# ------------------------------------------------- the pipeline side


async def test_a_page_left_mid_vlm_by_a_dead_worker_is_reclaimed(
    store: PageStateStore,
) -> None:
    """REGRESSION. The stranding bug, at its smallest.

    Page is in VLM_RUNNING with its layout already committed - exactly what a
    kill during the heavy stage leaves behind. Delivered as a recovered task,
    it must be rolled back to its checkpoint and finished, NOT stood down from.
    """
    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await store.transition(JOB, 0, PageState.LAYOUT_DONE, fields={"layout": "{}"})
    await store.transition(JOB, 0, PageState.VLM_RUNNING)

    client = FakeModelClient()
    result = await process_page(
        JOB, 0, store=store, client=client, reclaim=True
    )

    assert result.outcome in (Outcome.COMPLETED, Outcome.RESUMED)
    assert (await store.get_page(JOB, 0))["state"] == PageState.DONE.value
    assert client.count("vlm") == 1
    assert client.count("layout") == 0, "a committed stage must not re-run"


async def test_a_page_left_mid_layout_by_a_dead_worker_is_reclaimed(
    store: PageStateStore,
) -> None:
    """The other running state. Rolls back to PENDING, so layout DOES re-run -
    it was never committed, so there is nothing to preserve."""
    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)

    client = FakeModelClient()
    result = await process_page(JOB, 0, store=store, client=client, reclaim=True)

    assert result.outcome is Outcome.COMPLETED
    assert (await store.get_page(JOB, 0))["state"] == PageState.DONE.value
    assert client.count("layout") == 1


async def test_without_reclaim_a_running_page_is_still_left_alone(
    store: PageStateStore,
) -> None:
    """The fix must NOT widen to ordinary deliveries.

    A `*_RUNNING` page on a first delivery means a LIVE worker holds the claim,
    and rolling that back would have two workers running the same page and
    double-charging the model. Standing down is correct there; the distinction
    is entirely whether the previous owner was us.
    """
    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)

    client = FakeModelClient()
    result = await process_page(JOB, 0, store=store, client=client, reclaim=False)

    assert result.outcome is Outcome.OWNED_BY_OTHER
    assert (await store.get_page(JOB, 0))["state"] == PageState.LAYOUT_RUNNING.value
    assert client.calls == []


async def test_a_reclaimed_page_that_finished_meanwhile_is_a_no_op(
    store: PageStateStore,
) -> None:
    """Another replica's reaper may have already recovered and finished it.
    Reclaiming must notice, not redo the work."""
    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await store.transition(JOB, 0, PageState.LAYOUT_DONE, fields={"layout": "{}"})
    await store.transition(JOB, 0, PageState.VLM_RUNNING)
    await store.transition(JOB, 0, PageState.DONE, fields={"vlm": "{}"})

    client = FakeModelClient()
    result = await process_page(JOB, 0, store=store, client=client, reclaim=True)

    assert result.outcome is Outcome.ALREADY_TERMINAL
    assert client.calls == []


async def test_reclaim_does_not_double_count_the_job_counter(
    store: PageStateStore,
) -> None:
    """done_count is what `job.complete` and the status endpoint are built on,
    so a reclaimed page must land exactly once."""
    await store.init_job(JOB, 2)
    for page in range(2):
        await store.transition(JOB, page, PageState.LAYOUT_RUNNING)

    client = FakeModelClient()
    for page in range(2):
        await process_page(JOB, page, store=store, client=client, reclaim=True)
    # Redelivered again afterwards, as at-least-once allows.
    for page in range(2):
        await process_page(JOB, page, store=store, client=client, reclaim=True)

    done, total = await store.progress(JOB)

    assert (done, total) == (2, 2)


async def test_every_page_reaches_a_terminal_state_after_reclaim(
    store: PageStateStore,
) -> None:
    """The zero-drop property, stated directly: after recovery no page may be
    left non-terminal, because nothing else will ever look at it."""
    pages = 12
    await store.init_job(JOB, pages)
    # A realistic mix of what a kill leaves behind.
    for page in range(pages):
        await store.transition(JOB, page, PageState.LAYOUT_RUNNING)
        if page % 2 == 0:
            await store.transition(
                JOB, page, PageState.LAYOUT_DONE, fields={"layout": "{}"}
            )
            await store.transition(JOB, page, PageState.VLM_RUNNING)

    client = FakeModelClient()
    for page in range(pages):
        await process_page(JOB, page, store=store, client=client, reclaim=True)

    states = await store.page_states(JOB, pages)

    assert all(s == PageState.DONE.value for s in states), states
    assert await store.progress(JOB) == (pages, pages)


# ------------------------------------------------- the crash net


async def test_force_terminal_degrades_a_page_that_has_layout(
    store: PageStateStore,
) -> None:
    """The transition table picks the terminal state, not the call site.

    A page with a committed layout result has a usable answer, so giving up on
    it means FALLBACK_DONE - and FALLBACK_DONE is legal from LAYOUT_DONE and
    VLM_RUNNING precisely because of that.
    """
    from app.worker.pipeline import force_terminal

    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await store.transition(JOB, 0, PageState.LAYOUT_DONE, fields={"layout": "{}"})

    landed = await force_terminal(JOB, 0, store, "boom")

    assert landed == PageState.FALLBACK_DONE.value
    page = await store.get_page(JOB, 0)
    assert page["degraded"] == "1"
    assert await store.progress(JOB) == (1, 1), "must count towards completion"


async def test_force_terminal_fails_a_page_with_nothing_to_fall_back_to(
    store: PageStateStore,
) -> None:
    """No committed layout, so there is no reduced answer to serve. FAILED is
    legal from PENDING and LAYOUT_RUNNING and not from LAYOUT_DONE, which is
    the same distinction stated in the table."""
    from app.worker.pipeline import force_terminal

    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)

    landed = await force_terminal(JOB, 0, store, "boom")

    assert landed == PageState.FAILED.value
    assert await store.progress(JOB) == (1, 1)


async def test_force_terminal_is_a_no_op_on_a_finished_page(
    store: PageStateStore,
) -> None:
    """It must never double-count done_count or overwrite a real result."""
    from app.worker.pipeline import force_terminal

    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await store.transition(JOB, 0, PageState.LAYOUT_DONE, fields={"layout": "{}"})
    await store.transition(JOB, 0, PageState.VLM_RUNNING)
    await store.transition(JOB, 0, PageState.DONE, fields={"vlm": "{}"})

    landed = await force_terminal(JOB, 0, store, "boom")

    assert landed == PageState.DONE.value
    assert await store.progress(JOB) == (1, 1)


async def test_force_terminal_tolerates_a_missing_page(
    store: PageStateStore,
) -> None:
    """State may have expired by TTL while the page was in flight. Cleanup
    must never be the thing that raises."""
    from app.worker.pipeline import force_terminal

    assert await force_terminal("gone-job", 3, store, "boom") == "MISSING"
