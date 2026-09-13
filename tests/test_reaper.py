"""Reclaiming work from a worker that will never come back.

`read_own_pending` covers a worker RESTARTING: the consumer name is the
container hostname, so the new process finds its own pending list. It cannot
cover a worker being REPLACED - `--force-recreate`, a rescheduled pod, a
scale-down - because the new process has a new name and the dead consumer's
Pending Entries List is owned by nobody. Those entries are skipped by
`XREADGROUP >` (already delivered) and will never be acked (no live owner), so
the pages are non-terminal AND unreachable.

Three separate properties are pinned here, and the last two are things the
reaper could plausibly BREAK rather than fix:

  recovery      an orphan is reclaimed, rolled back to its last committed
                checkpoint, and requeued so it finishes.
  no stranding  the reclaimer never shares an entry id with the previous owner,
                because XACK is group-scoped: if it did, the old owner's ack
                would XDEL the only queue entry for a page the new owner is
                mid-flight on.
  no self-harm  a worker's own entries are never reclaimed - neither by the
                owner filter nor after a lease renewal.

Every test uses the SHIPPED idle threshold and backdates entries to make them
eligible. See `abandon` for why the obvious shortcut is worse than useless.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.config import Settings
from app.queue.state import TERMINAL_STATES, PageState, PageStateStore
from app.queue.streams import GROUP, LEAD_STREAM, STREAM, STREAMS, PageQueue
from app.worker.reaper import Reaper

JOB = "reap-job"
DEAD = "worker-that-died"
ALIVE = "worker-that-lives"

MIN_IDLE_MS = int(Settings().reaper_min_idle_s * 1000)
"""The threshold that actually ships, so the tests measure the real thing."""

IDLE_MS = MIN_IDLE_MS * 2
"""How far entries are backdated: unambiguously abandoned, not borderline."""


def make_reaper(
    settings: Settings, store: PageStateStore, queue: PageQueue, consumer: str = ALIVE
) -> Reaper:
    return Reaper(settings=settings, store=store, queue=queue, consumer=consumer)


async def deliver_to(queue: PageQueue, consumer: str, count: int) -> list[Any]:
    """Put entries into `consumer`'s pending list and leave them there."""
    return await queue.read(consumer, count=count, block_ms=50)


async def abandon(redis: Any, consumer: str, idle_ms: int = IDLE_MS) -> int:
    """Backdate `consumer`'s pending entries so they look truly abandoned.

    The obvious shortcut - collapsing `reaper_min_idle_s` to ~0 so freshly
    delivered entries qualify at once - was tried first and is worse than
    useless. It makes eligibility a race against test execution speed, so
    recovery cases found nothing to recover and passed only on their weaker
    assertions; and it destroys the exact property the concurrency test exists
    to check, because the interlock between two reapers IS the idle threshold,
    so a threshold near zero has no interlock. Both outcomes were properties of
    the fixture rather than of the code.

    XCLAIM's IDLE option sets an entry's last-delivery time directly, which is
    what makes this deterministic without sleeping for 30 seconds.
    min_idle_time=0 so the backdating is unconditional, and justid=True so it
    does not disturb delivery_count.
    """
    aged = 0
    for name in STREAMS:
        rows = await redis.xpending_range(name, GROUP, min="-", max="+", count=1000)
        ids = [row["message_id"] for row in rows if row["consumer"] == consumer]
        if not ids:
            continue
        await redis.xclaim(
            name,
            GROUP,
            consumer,
            min_idle_time=0,
            message_ids=ids,
            idle=idle_ms,
            justid=True,
        )
        aged += len(ids)
    return aged


async def orphan(
    queue: PageQueue, redis: Any, count: int, consumer: str = DEAD
) -> list[Any]:
    """Deliver `count` entries to a worker, then have that worker die."""
    delivered = await deliver_to(queue, consumer, count)
    await abandon(redis, consumer)
    return delivered


# ------------------------------------------------------- the queue primitive


async def test_an_abandoned_entry_of_another_consumer_is_claimable(
    queue: PageQueue, redis: Any
) -> None:
    await queue.enqueue_pages(JOB, 2)
    delivered = await orphan(queue, redis, 2)
    assert len(delivered) == 2

    claimed = await queue.claim_orphans(ALIVE, min_idle_ms=MIN_IDLE_MS, count=10)

    assert {t.page_index for t in claimed} == {0, 1}
    # The lane survives the claim. An entry id is only meaningful together with
    # its stream, so acking a lead entry against main is a silent no-op that
    # leaves it in the PEL forever.
    assert {t.stream for t in claimed} == {LEAD_STREAM, STREAM}


async def test_a_fresh_entry_is_not_claimable(queue: PageQueue) -> None:
    """The threshold is the whole safety mechanism, so it has to be real: a
    just-delivered entry belongs to a worker that is presumably working on it."""
    await queue.enqueue_pages(JOB, 3)
    await deliver_to(queue, DEAD, 3)

    claimed = await queue.claim_orphans(ALIVE, min_idle_ms=MIN_IDLE_MS, count=10)

    assert claimed == []


async def test_a_consumer_never_claims_its_own_entries(
    queue: PageQueue, redis: Any
) -> None:
    """REGRESSION for the bug a bare XAUTOCLAIM would have introduced.

    XAUTOCLAIM selects purely on idle time and cannot exclude a consumer. A
    worker's own task, legitimately parked for up to `rate_limit_max_wait_s`
    waiting on a VLM token, is indistinguishable by idle time from a dead
    worker's task - so the reaper would roll back a page its own process is
    actively working on. Note the entries here are backdated past the
    threshold, so idle time alone WOULD select them: the owner filter is what
    saves us.
    """
    await queue.enqueue_pages(JOB, 4)
    await deliver_to(queue, ALIVE, 4)
    await abandon(redis, ALIVE)

    claimed = await queue.claim_orphans(ALIVE, min_idle_ms=MIN_IDLE_MS, count=10)

    assert claimed == []


async def test_only_the_other_consumers_entries_are_claimed(
    queue: PageQueue, redis: Any
) -> None:
    """The filter must be per-entry, not per-scan: one lane holds entries from a
    dead worker and a live one at the same time."""
    await queue.enqueue_pages(JOB, 6)
    mine = await deliver_to(queue, ALIVE, 3)
    theirs = await deliver_to(queue, DEAD, 3)
    assert mine and theirs
    # Backdate BOTH, so only ownership can distinguish them.
    await abandon(redis, ALIVE)
    await abandon(redis, DEAD)

    claimed = await queue.claim_orphans(ALIVE, min_idle_ms=MIN_IDLE_MS, count=10)

    claimed_ids = {t.entry_id for t in claimed}
    assert claimed_ids == {t.entry_id for t in theirs}
    assert claimed_ids.isdisjoint({t.entry_id for t in mine})


async def test_a_claim_is_conditional_on_the_entry_still_being_idle(
    queue: PageQueue, redis: Any
) -> None:
    """Why XPENDING-then-XCLAIM is not the read-modify-write race it resembles.

    Two reapers both see an entry as idle, both claim it, the page is requeued
    twice - that is the shape of the bug, and XCLAIM's min-idle-time argument is
    what prevents it. It is mandatory and CONDITIONAL: an entry whose idle clock
    has been reset below the threshold is not claimed and does not appear in the
    reply. The first claimant resets idle to 0, so every other claimant's XCLAIM
    atomically returns nothing. That is a compare-and-set on idle time, which is
    why no lock or leader election is needed across replicas.
    """
    await queue.enqueue_pages(JOB, 2)
    await orphan(queue, redis, 2)

    first = await queue.claim_orphans("reaper-a", min_idle_ms=MIN_IDLE_MS, count=10)
    assert len(first) == 2

    second = await queue.claim_orphans("reaper-b", min_idle_ms=MIN_IDLE_MS, count=10)

    assert second == []


async def test_claiming_is_bounded_by_count(queue: PageQueue, redis: Any) -> None:
    """Per lane, as the config says. An operator flushing a consumer group or a
    mass eviction would otherwise make recovery its own outage."""
    await queue.enqueue_pages(JOB, 20)
    await orphan(queue, redis, 20)

    claimed = await queue.claim_orphans(ALIVE, min_idle_ms=MIN_IDLE_MS, count=3)

    # 3 from the main lane plus at most 1 from lead, which only ever holds
    # page 0. The point is that the bound is per lane and finite, not that it
    # is exactly 3.
    assert 3 <= len(claimed) <= 4


# ------------------------------------------------------------ lease renewal


async def test_renewing_a_lease_keeps_an_entry_out_of_the_reaper(
    queue: PageQueue, redis: Any
) -> None:
    """The property that lets `reaper_min_idle_s` be 30s instead of ~130s.

    Idle time in a PEL measures time since DELIVERY, not since progress, so
    without renewal a healthy worker's clock rises exactly as fast as a dead
    one's - and the only safe threshold would be above the longest legitimate
    hold (three attempts of a token wait plus a call, plus backoff). Renewal
    makes the clock mean "has this worker checked in", so the threshold can be
    a few missed check-ins.
    """
    await queue.enqueue_pages(JOB, 3)
    held = await deliver_to(queue, ALIVE, 3)
    await abandon(redis, ALIVE)

    # Established first: without renewal these ARE claimable by someone else.
    # Otherwise the assertion below could pass for any reason at all.
    assert await queue.claim_orphans("other", min_idle_ms=MIN_IDLE_MS, count=10) != []
    await abandon(redis, "other")
    await queue.claim_orphans(ALIVE, min_idle_ms=MIN_IDLE_MS, count=10)
    await abandon(redis, ALIVE)

    renewed = await queue.renew_leases(ALIVE, held)

    assert renewed == 3
    assert await queue.claim_orphans("other", min_idle_ms=MIN_IDLE_MS, count=10) == []


async def test_renewal_does_not_inflate_the_delivery_counter(
    queue: PageQueue, redis: Any
) -> None:
    """JUSTID is load-bearing, not an optimisation.

    Plain XCLAIM increments delivery_count, so a page legitimately held for 90s
    would look as though it had been delivered 18 times - poisoning the one
    counter that distinguishes a genuinely redelivered page from a slow one.
    """
    await queue.enqueue_pages(JOB, 2)
    held = await deliver_to(queue, ALIVE, 2)

    for _ in range(5):
        await queue.renew_leases(ALIVE, held)

    rows = await redis.xpending_range(STREAM, GROUP, min="-", max="+", count=10)
    assert rows, "entries should still be pending"
    assert all(int(row["times_delivered"]) == 1 for row in rows), rows


async def test_renewing_nothing_is_a_no_op(queue: PageQueue) -> None:
    """An idle worker holds no entries, and the maintenance loop still ticks."""
    assert await queue.renew_leases(ALIVE, []) == 0


# ------------------------------------------------------------- reaper policy


async def test_a_page_left_mid_vlm_is_rolled_back_and_requeued(
    settings: Settings, store: PageStateStore, queue: PageQueue, redis: Any
) -> None:
    """The headline path. The page keeps the layout result it already paid for:
    VLM_RUNNING rolls back to LAYOUT_DONE, never to PENDING."""
    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await store.transition(JOB, 0, PageState.LAYOUT_DONE, fields={"layout": "{}"})
    await store.transition(JOB, 0, PageState.VLM_RUNNING)
    await queue.enqueue_pages(JOB, 1)
    await orphan(queue, redis, 1)

    reaper = make_reaper(settings, store, queue)
    claimed = await reaper.scan()

    assert claimed == 1
    assert reaper.stats.requeued == 1
    assert (await store.get_page(JOB, 0))["state"] == PageState.LAYOUT_DONE.value


async def test_the_requeued_entry_is_a_new_id_in_the_main_lane(
    settings: Settings, store: PageStateStore, queue: PageQueue, redis: Any
) -> None:
    """REGRESSION for the stranding bug that reclaiming-in-place would introduce.

    XACK is GROUP-scoped, not consumer-scoped. If the reclaimer reused the
    original entry id, a previous owner that was merely slow rather than dead
    would later ack it - XDELing the only queue entry for a page the reclaimer
    is mid-flight on. Should the reclaimer then die, the page is stranded, and
    the recovery mechanism would have manufactured the failure it exists to
    prevent. A fresh entry makes the old owner's ack a harmless no-op on an id
    that no longer exists.
    """
    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await queue.enqueue_pages(JOB, 1)
    original = (await orphan(queue, redis, 1))[0]

    await make_reaper(settings, store, queue).scan()

    # The original is settled, so the dead worker's hypothetical late ack has
    # nothing left to destroy.
    assert await queue.claim_orphans("anyone", min_idle_ms=0, count=10) == []

    fresh = await deliver_to(queue, ALIVE, 5)
    assert len(fresh) == 1
    assert fresh[0].entry_id != original.entry_id
    assert fresh[0].page_index == 0
    # Main lane, even though page 0 arrived via lead: it has been delivered
    # once, so it is no longer first-page-critical, and admitting reclaims to
    # the priority lane would let a crash loop fill it with work that cannot run.
    assert fresh[0].stream == STREAM


async def test_a_reclaim_preserves_total_page_age(
    settings: Settings, store: PageStateStore, queue: PageQueue, redis: Any
) -> None:
    """`first_enqueued_at_ms` is the bound on systemic requeueing. Losing it
    across a reclaim would reset the page's deadline every time a worker died,
    so a crash loop could bounce one page forever."""
    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await queue.enqueue_pages(JOB, 1)
    original = (await orphan(queue, redis, 1))[0]

    await make_reaper(settings, store, queue).scan()
    fresh = (await deliver_to(queue, ALIVE, 5))[0]

    assert fresh.first_enqueued_at_ms == original.first_enqueued_at_ms
    assert fresh.attempt == original.attempt + 1, "a reclaim spends an attempt"


async def test_a_page_that_committed_before_the_crash_is_only_settled(
    settings: Settings, store: PageStateStore, queue: PageQueue, redis: Any
) -> None:
    """A worker that finished a page and died before XACK. Common, benign, and
    the correct response is to settle the entry and do nothing else - rerunning
    would duplicate work and requeueing would loop."""
    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await store.transition(JOB, 0, PageState.LAYOUT_DONE, fields={"layout": "{}"})
    await store.transition(JOB, 0, PageState.VLM_RUNNING)
    await store.transition(JOB, 0, PageState.DONE, fields={"vlm": "{}"})
    await queue.enqueue_pages(JOB, 1)
    await orphan(queue, redis, 1)

    reaper = make_reaper(settings, store, queue)
    await reaper.scan()

    assert reaper.stats.already_terminal == 1
    assert reaper.stats.requeued == 0
    assert await store.progress(JOB) == (1, 1), "done_count must not double"
    assert (await queue.depth())["stream_length"] == 0


async def test_an_orphan_whose_job_state_expired_is_settled_not_retried(
    settings: Settings, store: PageStateStore, queue: PageQueue, redis: Any
) -> None:
    """Page state carries a TTL; a queue entry in a dead consumer's PEL does
    not. There is nothing left to transition and nothing that could consume a
    result, so the entry must be settled or every scan re-examines it forever."""
    await queue.enqueue_pages("vanished-job", 1)
    await orphan(queue, redis, 1)

    reaper = make_reaper(settings, store, queue)
    await reaper.scan()

    assert reaper.stats.expired == 1
    assert (await queue.depth())["stream_length"] == 0


async def test_an_exhausted_orphan_is_forced_terminal_not_requeued(
    settings: Settings, store: PageStateStore, queue: PageQueue, redis: Any
) -> None:
    """The zero-drop rule for this path: giving up may never mean leaving a page
    non-terminal. A page with a committed layout has a usable answer, so it
    degrades to FALLBACK_DONE and still counts towards the job - which is what
    lets the job complete and its subscribers receive a `job.complete` rather
    than waiting out the SSE duration limit for an event that cannot arrive."""
    exhausted = settings.model_copy(update={"page_deadline_s": 0.0})

    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await store.transition(JOB, 0, PageState.LAYOUT_DONE, fields={"layout": "{}"})
    await store.transition(JOB, 0, PageState.VLM_RUNNING)
    await queue.enqueue_pages(JOB, 1)
    await orphan(queue, redis, 1)

    reaper = make_reaper(exhausted, store, queue)
    await reaper.scan()

    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.FALLBACK_DONE.value
    assert page["degraded"] == "1"
    assert await store.progress(JOB) == (1, 1)
    assert reaper.stats.forced_terminal == 1
    assert (await queue.depth())["stream_length"] == 0


async def test_an_exhausted_orphan_with_no_layout_fails(
    settings: Settings, store: PageStateStore, queue: PageQueue, redis: Any
) -> None:
    """Nothing committed, so there is no reduced answer to serve. Still
    terminal, still counted, never silent."""
    exhausted = settings.model_copy(update={"page_deadline_s": 0.0})

    await store.init_job(JOB, 1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await queue.enqueue_pages(JOB, 1)
    await orphan(queue, redis, 1)

    reaper = make_reaper(exhausted, store, queue)
    await reaper.scan()

    assert (await store.get_page(JOB, 0))["state"] == PageState.FAILED.value
    assert await store.progress(JOB) == (1, 1)


async def test_every_page_of_a_wholly_abandoned_job_becomes_reachable(
    settings: Settings, store: PageStateStore, queue: PageQueue, redis: Any
) -> None:
    """The zero-drop invariant for the whole mechanism, stated directly: after a
    scan, no page may be both non-terminal and absent from the queue.

    A realistic mix of what a SIGKILL leaves behind - some pages untouched,
    some mid-layout, some mid-VLM.
    """
    pages = 9
    await store.init_job(JOB, pages)
    for page in range(pages):
        if page % 3 == 1:
            await store.transition(JOB, page, PageState.LAYOUT_RUNNING)
        elif page % 3 == 2:
            await store.transition(JOB, page, PageState.LAYOUT_RUNNING)
            await store.transition(
                JOB, page, PageState.LAYOUT_DONE, fields={"layout": "{}"}
            )
            await store.transition(JOB, page, PageState.VLM_RUNNING)
    await queue.enqueue_pages(JOB, pages)
    await orphan(queue, redis, pages)

    await make_reaper(settings, store, queue).scan()

    states = await store.page_states(JOB, pages)
    reachable = (await queue.depth())["stream_length"]

    non_terminal = [
        index
        for index, state in enumerate(states)
        if PageState(state) not in TERMINAL_STATES
    ]
    assert len(non_terminal) == reachable, (
        f"{len(non_terminal)} non-terminal pages but {reachable} queue entries"
    )
    # And none is left claiming to be owned by the dead worker, or the next
    # delivery would stand down as OWNED_BY_OTHER and ack - stranding it after
    # all, which is the bug the own-pending fix closed on its own path.
    assert not any(
        state in (PageState.LAYOUT_RUNNING.value, PageState.VLM_RUNNING.value)
        for state in states
    ), states


async def test_a_scan_with_nothing_to_do_is_silent(
    settings: Settings, store: PageStateStore, queue: PageQueue
) -> None:
    """The normal case, run every `reaper_interval_s` on every replica forever.
    It must not requeue, force, or count anything."""
    reaper = make_reaper(settings, store, queue)

    assert await reaper.scan() == 0
    assert reaper.stats.snapshot() == {
        "scans": 1,
        "claimed": 0,
        "requeued": 0,
        "forced_terminal": 0,
        "already_terminal": 0,
        "expired": 0,
        "leases_renewed": 0,
    }


async def test_concurrent_reapers_requeue_a_page_exactly_once(
    settings: Settings, store: PageStateStore, queue: PageQueue, redis: Any
) -> None:
    """Every replica runs a reaper with no leader election, so this is the
    normal configuration rather than an edge case. The conditional claim is what
    makes it safe; this asserts the observable consequence."""
    pages = 6
    await store.init_job(JOB, pages)
    for page in range(pages):
        await store.transition(JOB, page, PageState.LAYOUT_RUNNING)
    await queue.enqueue_pages(JOB, pages)
    await orphan(queue, redis, pages)

    reapers = [make_reaper(settings, store, queue, f"reaper-{i}") for i in range(4)]
    await asyncio.gather(*(r.scan() for r in reapers))

    assert sum(r.stats.requeued for r in reapers) == pages
    assert (await queue.depth())["stream_length"] == pages
