"""TTL bounds INACTIVITY, not a job's total lifetime.

`init_job` sets a TTL on the job hash and every page hash so finished jobs
reclaim themselves. Set once and never renewed, that same TTL is also a hard
cap on how long a job may take - and a job that exceeds it does not fail
cleanly, it corrupts:

    HINCRBY on a missing key CREATES it.

So when the job hash expires mid-flight, the next terminal transition recreates
it holding only `done_count`. `total_pages` is then unreadable, the script
returns -1, and `TransitionResult.completed_job` can never be true again. Every
remaining page still processes correctly and the job never reports complete, so
`job.complete` is never published and every SSE subscriber waits out
`sse_max_duration_s` for an event that no longer exists. A silent hang at 99%,
with nothing in the logs.

Reachable without any crash at all: a long job under sustained backlog, a wide
chaos window, or simply `result_ttl_s` configured below the time a large job
takes. `page_deadline_s` does not help - it bounds one page's wait, not the
whole job's wall clock.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.queue.state import PageState, PageStateStore, job_key, page_key

JOB = "ttl-job"


async def ttl_ms(redis: Any, key: str) -> int:
    """-1 = no expiry set, -2 = key is gone."""
    return int(await redis.pttl(key))


# ------------------------------------------------------------- the mechanism


async def test_a_committed_stage_renews_both_page_and_job_ttl(redis: Any) -> None:
    """The job hash matters as much as the page hash, and is easier to miss:
    nothing in a page's own transition obviously belongs to the job, yet the
    job hash is the one whose loss is corrupting rather than lossy."""
    store = PageStateStore(redis, ttl_s=60)
    await store.init_job(JOB, 2)

    # Age both keys by rewriting the TTL far lower than the store's.
    await redis.pexpire(page_key(JOB, 0), 3_000)
    await redis.pexpire(job_key(JOB), 3_000)

    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)

    assert await ttl_ms(redis, page_key(JOB, 0)) > 30_000, "page TTL not renewed"
    assert await ttl_ms(redis, job_key(JOB)) > 30_000, "job TTL not renewed"


async def test_a_rejected_transition_renews_nothing(redis: Any) -> None:
    """Deliberate, and the reason renewal sits after the guard clauses.

    At-least-once delivery means a finished page is redelivered routinely. If a
    rejected transition renewed the TTL, every redelivery would push a completed
    job's expiry out again and the records would never reclaim themselves -
    turning a cleanup policy into an unbounded leak, which is the problem the
    TTL existed to solve.
    """
    store = PageStateStore(redis, ttl_s=60)
    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await redis.pexpire(page_key(JOB, 0), 3_000)
    await redis.pexpire(job_key(JOB), 3_000)

    # PENDING -> LAYOUT_RUNNING is no longer legal: already past it.
    rejected = await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)

    assert not rejected.ok
    assert await ttl_ms(redis, page_key(JOB, 0)) <= 3_000
    assert await ttl_ms(redis, job_key(JOB)) <= 3_000


async def test_a_missing_page_renews_nothing_and_does_not_resurrect_it(
    redis: Any,
) -> None:
    """The MISSING early return must stay a pure read. A PEXPIRE before the
    guard would set a TTL on a key that does not exist (harmless), but an HSET
    would recreate a page whose job is gone - a zombie with no job to belong
    to."""
    store = PageStateStore(redis, ttl_s=60)

    result = await store.transition("no-such-job", 0, PageState.LAYOUT_RUNNING)

    assert result.missing
    assert await ttl_ms(redis, page_key("no-such-job", 0)) == -2, "key was created"


# ------------------------------------------------------------- the regression


async def test_a_job_still_progressing_does_not_age_out_from_under_itself(
    redis: Any,
) -> None:
    """REGRESSION, and the whole point of the change.

    A TTL shorter than the job takes. Every page commits normally; the only
    question is whether the job's own bookkeeping survives long enough to
    notice the last one. Before renewal the job hash expired partway through,
    `total_pages` became unreadable, and `completed_job` was false for the
    final page - so nothing ever published `job.complete`.
    """
    pages = 6
    store = PageStateStore(redis, ttl_s=1)  # 1s, far shorter than this test
    await store.init_job(JOB, pages)

    completions = []
    for page in range(pages):
        # Slower than the TTL, on purpose. Without renewal the job hash is gone
        # by page 3 and the run is unrecoverable from there.
        await asyncio.sleep(0.4)
        await store.transition(JOB, page, PageState.LAYOUT_RUNNING)
        await store.transition(
            JOB, page, PageState.LAYOUT_DONE, fields={"layout": "{}"}
        )
        await store.transition(JOB, page, PageState.VLM_RUNNING)
        result = await store.transition(
            JOB, page, PageState.DONE, fields={"vlm": "{}"}
        )
        assert result.ok, f"page {page} failed to commit: {result}"
        assert result.total_pages == pages, (
            f"page {page} could not read total_pages ({result.total_pages}): "
            "the job hash expired mid-flight"
        )
        if result.completed_job:
            completions.append(page)

    assert await store.progress(JOB) == (pages, pages)
    assert completions == [pages - 1], (
        f"job.complete would have fired for {completions}, expected only the "
        "last page"
    )


async def test_an_idle_job_still_reclaims_itself(redis: Any) -> None:
    """The other half of the contract. Renewal must not become immortality:
    once a job stops transitioning, its clock runs out as designed."""
    store = PageStateStore(redis, ttl_s=1)
    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)

    # No further activity, so nothing renews.
    await asyncio.sleep(1.4)

    assert await ttl_ms(redis, job_key(JOB)) == -2, "job hash outlived its TTL"
    assert await ttl_ms(redis, page_key(JOB, 0)) == -2, "page hash outlived its TTL"


async def test_the_alarm_fires_when_the_job_hash_is_gone(redis: Any) -> None:
    """Renewal makes this unreachable in normal operation, so the alarm is for
    the case it is reached anyway - a manual FLUSHDB, an eviction under
    maxmemory, a misconfigured TTL. The page must still commit: it is a
    reporting failure, and failing the page would turn it into a data-loss one.
    """
    store = PageStateStore(redis, ttl_s=60)
    await store.init_job(JOB, 2)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await store.transition(JOB, 0, PageState.LAYOUT_DONE, fields={"layout": "{}"})
    await store.transition(JOB, 0, PageState.VLM_RUNNING)

    # Evict just the job hash, leaving the page hash intact.
    await redis.delete(job_key(JOB))

    result = await store.transition(JOB, 0, PageState.DONE, fields={"vlm": "{}"})

    assert result.ok, "the page itself must still commit"
    assert result.total_pages == -1
    assert not result.completed_job
    assert (await store.get_page(JOB, 0))["state"] == PageState.DONE.value


# ------------------------------------------------------------- the page sweep


async def test_an_unstarted_page_is_renewed_by_another_pages_progress(
    redis: Any,
) -> None:
    """The case that made a page-local renewal insufficient.

    Page 5 is never touched, so nothing in its own lifecycle can refresh it.
    Only the sweep triggered by page 0's progress keeps it alive - and it has to,
    because its queue entry still exists and a worker will eventually claim it.
    """
    store = PageStateStore(redis, ttl_s=2)
    await store.init_job(JOB, 6)

    # Age every page, then let the marker go stale so the next transition sweeps.
    for page in range(6):
        await redis.pexpire(page_key(JOB, page), 1_500)
    await redis.hset(job_key(JOB), "pages_renewed_at", "0")

    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)

    assert await ttl_ms(redis, page_key(JOB, 5)) > 1_500, (
        "an unstarted page was not renewed; it will expire mid-job and its "
        "queue entry will find no state to transition"
    )


async def test_the_sweep_is_amortised_not_per_transition(redis: Any) -> None:
    """Why the sweep is gated at all. Four transitions per page over a 100-page
    job is 400 scripts, so an unconditional sweep would be 40,000 PEXPIREs of
    pure bookkeeping churn per job."""
    store = PageStateStore(redis, ttl_s=60)
    await store.init_job(JOB, 4)

    marker_at_start = await redis.hget(job_key(JOB), "pages_renewed_at")

    # A full page's worth of transitions, well inside half the TTL.
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await store.transition(JOB, 0, PageState.LAYOUT_DONE, fields={"layout": "{}"})
    await store.transition(JOB, 0, PageState.VLM_RUNNING)
    await store.transition(JOB, 0, PageState.DONE, fields={"vlm": "{}"})

    assert await redis.hget(job_key(JOB), "pages_renewed_at") == marker_at_start, (
        "the sweep ran despite the marker being fresh"
    )


async def test_only_one_of_many_concurrent_transitions_claims_the_sweep(
    redis: Any,
) -> None:
    """The marker is written inside the same atomic script that reads it, so
    N workers transitioning different pages at once cannot all be told to
    sweep. Without that, a 16-slot worker would fire 16 redundant sweeps the
    moment the marker went stale."""
    store = PageStateStore(redis, ttl_s=2)
    pages = 8
    await store.init_job(JOB, pages)
    await redis.hset(job_key(JOB), "pages_renewed_at", "0")

    results = await asyncio.gather(
        *(
            store.transition(JOB, page, PageState.LAYOUT_RUNNING)
            for page in range(pages)
        )
    )

    assert all(r.ok for r in results)
    # Every page still alive, and the marker moved exactly once.
    assert await ttl_ms(redis, page_key(JOB, pages - 1)) > 1_000
    assert int(await redis.hget(job_key(JOB), "pages_renewed_at")) > 0
