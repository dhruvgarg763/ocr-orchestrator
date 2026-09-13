"""Redis Streams queue tests.

The properties under test are exactly the ones that make crash recovery
possible, so they are verified against a real Redis rather than asserted in a
comment:

  * delivery is EXCLUSIVE - two consumers in a group never get the same entry
  * delivery is TRACKED   - an unacknowledged entry stays in the PEL
  * ack SETTLES           - removed from the PEL and deleted from the stream
"""

from __future__ import annotations

import time

from redis.asyncio import Redis

from app.queue.streams import (
    GROUP,
    LEAD_STREAM,
    POISON_STREAM,
    STREAM,
    STREAMS,
    PageQueue,
    PageTask,
)


async def test_ensure_group_is_idempotent(redis: Redis) -> None:
    """Every replica calls this at startup; only one can win the race.

    Redis answers BUSYGROUP to the losers, which must be treated as success and
    not as a crash-looping worker.
    """
    q = PageQueue(redis)
    await q.ensure_group()
    await q.ensure_group()  # must not raise

    for name in STREAMS:
        groups = await redis.xinfo_groups(name)
        assert [g["name"] for g in groups] == [GROUP], f"{name} has no group"


async def test_page_zero_goes_to_the_priority_lane(
    queue: PageQueue, redis: Redis
) -> None:
    """Time-to-first-page is a FAIRNESS problem, not a latency one.

    Under one FIFO stream the 50th job's first page sits behind 979 pages
    belonging to clients already being served, so it is laid out last:
    measured p95 time-to-first-page of 14,511 ms against a 200 ms target. The
    layout endpoint's own 100 rps makes 1,000 pages a 10 second floor, so no
    amount of tuning reaches the target - only serving each job's FIRST page
    ahead of other jobs' LATER pages does.

    One page per job is all the metric needs, which is what keeps the lane
    small enough to drain instantly.
    """
    await queue.enqueue_pages("job-1", 5)

    assert await redis.xlen(LEAD_STREAM) == 1
    assert await redis.xlen(STREAM) == 4


async def test_a_single_page_job_is_all_lead(queue: PageQueue, redis: Redis) -> None:
    await queue.enqueue_pages("job-1", 1)

    assert await redis.xlen(LEAD_STREAM) == 1
    assert await redis.xlen(STREAM) == 0


async def test_the_lead_lane_is_served_before_a_deep_backlog(
    queue: PageQueue, redis: Redis
) -> None:
    """The property the whole two-lane design exists for.

    A job arriving behind a large backlog must still get a page promptly. With
    one stream its first page would be read only after every queued page ahead
    of it.
    """
    await queue.enqueue_pages("early", 60)
    await queue.enqueue_pages("late", 5)

    tasks = await queue.read("w-1", count=8, block_ms=100)

    assert any(t.job_id == "late" for t in tasks), "the newcomer got nothing"
    late = [t for t in tasks if t.job_id == "late"]
    assert [t.page_index for t in late] == [0]
    assert late[0].stream == LEAD_STREAM


async def test_a_task_knows_which_lane_delivered_it(queue: PageQueue) -> None:
    """An entry id is only meaningful together with its stream, and acking
    against the wrong one fails silently."""
    await queue.enqueue_pages("job-1", 3)

    tasks = await queue.read("w-1", count=10, block_ms=100)
    by_page = {t.page_index: t.stream for t in tasks}

    assert by_page[0] == LEAD_STREAM
    assert by_page[1] == by_page[2] == STREAM


async def test_acking_against_the_wrong_lane_is_a_silent_no_op(
    queue: PageQueue, redis: Redis
) -> None:
    """REGRESSION, and why `ack(stream=...)` has no default.

    XACK against a stream that never held the entry returns 0 rather than
    raising, so the entry stays in the PEL and the reaper later redelivers a
    page that was already finished. This is exactly what the tests themselves
    did when the argument was defaultable - caught only because a depth
    assertion disagreed by one.
    """
    await queue.enqueue_pages("job-1", 1)
    (lead,) = await queue.read("w-1", count=5, block_ms=100)
    assert lead.stream == LEAD_STREAM

    await queue.ack(lead.entry_id, stream=STREAM)  # the wrong lane

    assert (await redis.xpending(LEAD_STREAM, GROUP))["pending"] == 1, (
        "the entry is still pending, with no error raised anywhere"
    )

    await queue.ack(lead.entry_id, stream=lead.stream)

    assert (await redis.xpending(LEAD_STREAM, GROUP))["pending"] == 0


async def test_a_requeue_never_enters_the_priority_lane(
    queue: PageQueue, redis: Redis
) -> None:
    """A page delivered once is no longer first-page-critical. Admitting
    retries would let a saturated endpoint fill the lane with work that cannot
    run, destroying the emptiness that makes it fast."""
    await queue.enqueue_pages("job-1", 1)
    (lead,) = await queue.read("w-1", count=5, block_ms=100)

    await queue.requeue(lead)

    assert await redis.xlen(LEAD_STREAM) == 0
    assert await redis.xlen(STREAM) == 1


async def test_a_stage_handoff_does_not_spend_an_attempt(queue: PageQueue) -> None:
    """The attempt counter bounds how long CONGESTION may bounce a page. A
    handoff is progress, so charging it would make every page spend an attempt
    on its own happy path."""
    await queue.enqueue_pages("job-1", 1)
    (task,) = await queue.read("w-1", count=5, block_ms=100)

    await queue.requeue(task, count_attempt=False)
    (again,) = await queue.read("w-1", count=5, block_ms=100)

    assert again.attempt == task.attempt == 0

    await queue.requeue(again)
    (third,) = await queue.read("w-1", count=5, block_ms=100)

    assert third.attempt == 1, "a real retry still counts"


async def test_the_read_bound_holds_across_both_lanes(
    queue: PageQueue, redis: Redis
) -> None:
    """REGRESSION. COUNT is applied PER STREAM, measured.

    A combined XREADGROUP over both lanes at the full count returns up to
    2 x count entries, and every one lands in this consumer's PEL - so the
    over-read cannot be discarded. At count=1 an evenly split budget also gave
    1 + 1 = 2. The dispatch loop's "resident tasks <= worker_concurrency" is a
    hard claim, so this bound has to be exact at every size.
    """
    await queue.enqueue_pages("a", 40)
    await queue.enqueue_pages("b", 40)
    await queue.enqueue_pages("c", 40)

    for count in (1, 2, 3, 4, 8, 16):
        tasks = await queue.read(f"w-{count}", count=count, block_ms=50)
        assert len(tasks) <= count, f"count={count} over-read {len(tasks)}"


async def test_enqueue_pages_writes_one_entry_per_page(queue: PageQueue, redis: Redis) -> None:
    count = await queue.enqueue_pages("job-1", 7, trace_id="trace-abc")

    assert count == 7
    # Six in the main lane plus page 0 in the priority lane. The count is what
    # the caller cares about; the split is an internal scheduling decision.
    assert await redis.xlen(STREAM) == 6
    assert await redis.xlen(LEAD_STREAM) == 1


async def test_enqueued_task_carries_the_trace_id_across_the_process_boundary(
    queue: PageQueue,
) -> None:
    """ContextVars do not survive a process hop; a stream field does."""
    await queue.enqueue_pages("job-1", 1, trace_id="trace-abc")

    (task,) = await queue.read("w-1", count=10, block_ms=100)

    assert task.job_id == "job-1"
    assert task.page_index == 0
    assert task.trace_id == "trace-abc"


async def test_read_delivers_and_records_in_the_pending_list(
    queue: PageQueue, redis: Redis
) -> None:
    """The PEL entry is the crash-recovery safety net.

    With BRPOP there would be no record at all, so a worker dying here would
    lose the task silently.
    """
    await queue.enqueue_pages("job-1", 3)

    tasks = await queue.read("w-1", count=3, block_ms=100)

    assert len(tasks) == 3
    # A list comprehension, not a generator: `await` inside a generator
    # expression makes it an async generator, which sum() cannot consume.
    pending = sum(
        [int((await redis.xpending(name, GROUP))["pending"]) for name in STREAMS]
    )
    assert pending == 3
    # Still in the streams: delivery does not consume.
    assert sum([await redis.xlen(name) for name in STREAMS]) == 3


async def test_ack_removes_from_pel_and_deletes_the_entry(
    queue: PageQueue, redis: Redis
) -> None:
    """XACK alone leaves the entry in the stream forever.

    The task stream cannot use MAXLEN to bound itself - trimming by length
    discards the OLDEST entries, which in a task queue are unprocessed work. So
    entries are deleted once acknowledged instead, which is safe because group
    delivery is exclusive.
    """
    await queue.enqueue_pages("job-1", 3)
    tasks = await queue.read("w-1", count=3, block_ms=100)

    for task in tasks:
        await queue.ack(task.entry_id, stream=task.stream)

    assert (await redis.xpending(STREAM, GROUP))["pending"] == 0
    assert await redis.xlen(STREAM) == 0, "stream must not grow without bound"


async def test_delivery_is_exclusive_between_consumers(queue: PageQueue) -> None:
    """The core scaling property: N workers share work, they do not duplicate it.

    Without it, scaling to 3 replicas would triple the model bill rather than
    tripling throughput.
    """
    await queue.enqueue_pages("job-1", 4)

    first = await queue.read("w-1", count=4, block_ms=100)
    second = await queue.read("w-2", count=4, block_ms=100)

    assert len(first) == 4
    assert second == [], "w-2 must not see entries already delivered to w-1"


async def test_read_own_pending_recovers_this_consumers_unacked_work(
    queue: PageQueue,
) -> None:
    """A restarted worker resumes its own interrupted pages immediately.

    Consumer identity is the container hostname, which is stable across
    restarts, so "own pending" is meaningful. Without this the work would wait
    for the reaper's idle timeout instead.
    """
    await queue.enqueue_pages("job-1", 2)
    delivered = await queue.read("w-1", count=2, block_ms=100)
    assert len(delivered) == 2

    # --- w-1 dies here without acking, then restarts under the same name ---
    recovered = await queue.read_own_pending("w-1", count=10)

    assert {t.page_index for t in recovered} == {0, 1}
    assert {t.entry_id for t in recovered} == {t.entry_id for t in delivered}


async def test_own_pending_is_scoped_to_the_consumer(queue: PageQueue) -> None:
    """w-2 must not be able to steal w-1's in-flight work by asking politely.

    Claiming another consumer's entries requires XAUTOCLAIM with an idle
    threshold, so a live worker's tasks cannot be yanked out from under it.
    """
    await queue.enqueue_pages("job-1", 2)
    await queue.read("w-1", count=2, block_ms=100)

    assert await queue.read_own_pending("w-2", count=10) == []


async def test_depth_separates_backlog_from_pending(queue: PageQueue) -> None:
    """Two different failure signals, so they must be two different numbers.

    Rising backlog  => workers cannot keep up with ingestion (add workers).
    Rising pending  => workers are stalled or dying (investigate workers).
    """
    await queue.enqueue_pages("job-1", 10)
    assert await queue.depth() == {"stream_length": 10, "pending": 0, "backlog": 10}

    tasks = await queue.read("w-1", count=4, block_ms=100)
    assert await queue.depth() == {"stream_length": 10, "pending": 4, "backlog": 6}

    for task in tasks:
        await queue.ack(task.entry_id, stream=task.stream)
    assert await queue.depth() == {"stream_length": 6, "pending": 0, "backlog": 6}


async def test_read_blocks_then_returns_empty_when_the_queue_is_idle(
    queue: PageQueue,
) -> None:
    """A bounded block is what lets the worker loop notice a stop request.

    With block=0 (forever), SIGTERM would hang until the next task arrived.
    """
    started = time.monotonic()
    tasks = await queue.read("w-1", count=1, block_ms=200)
    elapsed = time.monotonic() - started

    assert tasks == []
    assert elapsed >= 0.15, "should have actually blocked rather than busy-polled"


async def test_prefetch_count_bounds_how_much_a_worker_holds(queue: PageQueue) -> None:
    """The worker's primary memory boundary.

    Resident tasks are at most count x replicas, regardless of queue depth. This
    is why a 100,000-page backlog cannot blow up worker memory.
    """
    await queue.enqueue_pages("job-1", 100)

    tasks = await queue.read("w-1", count=8, block_ms=100)

    assert len(tasks) == 8, "must not hand over the whole backlog"


def test_lag_ms_measures_queue_wait_not_service_time() -> None:
    task = PageTask(
        entry_id="1-0",
        job_id="j",
        page_index=0,
        enqueued_at_ms=int(time.time() * 1000) - 5_000,
        trace_id="t",
    )

    assert 4_500 < task.lag_ms < 5_500


def test_lag_ms_is_never_negative() -> None:
    """Clock skew between the API and worker containers must not produce
    negative lag, which would corrupt any percentile computed from it."""
    task = PageTask(
        entry_id="1-0",
        job_id="j",
        page_index=0,
        enqueued_at_ms=int(time.time() * 1000) + 10_000,  # "enqueued in the future"
        trace_id="t",
    )

    assert task.lag_ms == 0.0


async def test_read_recreates_a_vanished_consumer_group(
    queue: PageQueue, redis: Redis
) -> None:
    """REGRESSION: the worker must not crash-loop on a missing group.

    The group is created once at startup, so a Redis restart without
    persistence, a failover to an empty replica, or an operator flush used to
    take down every worker and keep them down until each was manually
    restarted - observed as an endless
    `NOGROUP No such key 'stream:pages' or consumer group 'workers'`.

    Recovery is trivially correct: recreate the group and return empty. Only
    NOGROUP is handled; any other ResponseError is a real bug and still raises.
    """
    await queue.enqueue_pages("job", 2)
    await redis.delete(STREAM)  # the group goes with the stream

    tasks = await queue.read("w-1", count=5, block_ms=50)

    assert tasks == [], "should return empty rather than raise"
    groups = await redis.xinfo_groups(STREAM)
    assert any(g["name"] == GROUP for g in groups), "group was not recreated"

    # And the queue is usable again afterwards.
    await queue.enqueue_pages("job2", 1)
    assert len(await queue.read("w-1", count=5, block_ms=200)) == 1


async def test_a_lead_only_job_does_not_wait_out_the_block(
    queue: PageQueue,
) -> None:
    """REGRESSION. A 20x latency cliff at exactly pages == 1.

    The dispatch loop's WAIT used to sit on the main stream alone, so a
    lead-lane XADD could not wake an idle worker. Multi-page jobs hid it
    completely: their pages 1..n-1 land in the main stream and do the waking.
    A one-page job is entirely lead-lane, so nothing woke the worker and its
    only page waited out worker_block_ms. Measured on an IDLE system - the
    best case - before the fix:

        1-page job    p50 1499 ms   max 2840 ms
        2-page job    p50   74 ms

    Here the read starts FIRST, against empty lanes, so it is genuinely
    blocked; the page is then published while it waits. A read issued after
    the XADD would pass on the non-blocking probe and prove nothing.
    """
    import asyncio

    async def publish_soon() -> None:
        await asyncio.sleep(0.05)
        await queue.enqueue_pages("late-single", 1)

    asyncio.create_task(publish_soon())

    started = asyncio.get_running_loop().time()
    tasks = await queue.read("w-1", count=8, block_ms=5_000)
    waited_ms = (asyncio.get_running_loop().time() - started) * 1000

    assert [t.page_index for t in tasks] == [0]
    assert tasks[0].stream == LEAD_STREAM
    assert waited_ms < 1_000, (
        f"woke after {waited_ms:.0f}ms - the wait is not watching the lead lane"
    )


async def test_the_read_bound_holds_on_the_blocking_path_too(
    queue: PageQueue,
) -> None:
    """The combined blocking read halves its per-stream COUNT, because COUNT is
    applied per stream and both lanes are listed."""
    import asyncio

    async def publish_soon() -> None:
        await asyncio.sleep(0.05)
        await queue.enqueue_pages("burst", 40)

    asyncio.create_task(publish_soon())

    tasks = await queue.read("w-1", count=4, block_ms=5_000)

    assert 0 < len(tasks) <= 4, f"blocking path over-read: {len(tasks)}"


async def test_a_capped_main_budget_does_not_block_on_an_empty_lane(
    queue: PageQueue,
) -> None:
    """REGRESSION. Waiting on the wrong thing looked like a throughput tax.

    When the reserved lane is enabled and main is at its cap, lead is the only
    lane the worker may read. Blocking on it for worker_block_ms starved the
    main lane: with 760 pages queued, in-flight pages fell from 48 to 2 and
    throughput dropped ~40%. The caller must get control back immediately so
    it can wait on its in-flight tasks, which is the event that actually
    matters (a slot freeing).
    """
    import asyncio

    await queue.enqueue_pages("plenty", 40)
    # Drain the lead entry so the lane is genuinely empty.
    await queue.read("w-0", count=40, block_ms=100)

    started = asyncio.get_running_loop().time()
    tasks = await queue.read("w-1", lead_count=4, main_count=0, block_ms=5_000)
    waited_ms = (asyncio.get_running_loop().time() - started) * 1000

    assert tasks == []
    assert waited_ms < 500, (
        f"blocked {waited_ms:.0f}ms on an empty reserved lane while main work waited"
    )


async def test_own_pending_respects_its_count_across_both_lanes(
    queue: PageQueue,
) -> None:
    """REGRESSION. The `read()` bound bug, repeated in the recovery path.

    COUNT is applied PER STREAM, so one call over both lanes at `count`
    returns up to 2 x count - measured, count=4 returned 8 (4 lead + 4 main).
    This runs at startup with count=worker_concurrency, so a restarting worker
    spawned twice its concurrency limit in coroutines at exactly the moment it
    is most loaded.
    """
    for i in range(5):
        await queue.enqueue_pages(f"j{i}", 6)

    # Deliver everything to one consumer without acking, so it all sits in
    # that consumer's PEL and is eligible for own-pending recovery.
    while await queue.read("w-1", count=50, block_ms=50):
        pass

    assert (await queue.depth())["pending"] == 30

    for count in (1, 2, 4, 8):
        recovered = await queue.read_own_pending("w-1", count=count)
        assert len(recovered) <= count, (
            f"count={count} recovered {len(recovered)} tasks"
        )

    # And it must still reach BOTH lanes, or a dead worker's lead task would
    # wait for the reaper instead of resuming immediately.
    everything = await queue.read_own_pending("w-1", count=100)
    assert {t.stream for t in everything} == {LEAD_STREAM, STREAM}


async def test_a_capped_main_budget_waits_on_the_lead_lane_when_asked(
    queue: PageQueue,
) -> None:
    """REGRESSION, the other direction.

    When main is at its reservation cap the reserved slots are the only ones
    that can act, so the LEAD LANE is what the worker must wait on. Returning
    immediately instead meant a first page started only when some unrelated
    page happened to finish - measured as p95 time-to-first-page of 333-406ms
    into a busy system, at every reserve value tried.

    The read must therefore be genuinely blocked on the lane and wake on
    arrival, not poll and miss.
    """
    import asyncio

    async def publish_soon() -> None:
        await asyncio.sleep(0.05)
        await queue.enqueue_pages("newcomer", 1)

    asyncio.create_task(publish_soon())

    started = asyncio.get_running_loop().time()
    tasks = await queue.read(
        "w-1", lead_count=4, main_count=0, block_ms=5_000, lead_block_ms=2_000
    )
    waited_ms = (asyncio.get_running_loop().time() - started) * 1000

    assert [t.page_index for t in tasks] == [0]
    assert tasks[0].stream == LEAD_STREAM
    assert waited_ms < 1_000, f"did not wake on arrival ({waited_ms:.0f}ms)"


async def test_the_lead_wait_is_bounded_not_indefinite(queue: PageQueue) -> None:
    """The opposite failure. Blocking for the full worker_block_ms left
    capacity freed DURING the wait unused - in-flight pages fell from 48 to 2
    with 760 queued and throughput dropped ~40%. The wait must expire quickly
    enough that the caller can refill the main lane."""
    import asyncio

    started = asyncio.get_running_loop().time()
    tasks = await queue.read(
        "w-1", lead_count=4, main_count=0, block_ms=10_000, lead_block_ms=150
    )
    waited_ms = (asyncio.get_running_loop().time() - started) * 1000

    assert tasks == []
    assert waited_ms < 1_000, f"waited {waited_ms:.0f}ms; should honour the bound"


# --------------------------------------------------------------------------
# Poison entries: a malformed task must not take the worker down
#
# REGRESSION. `_parse` runs inside `read()`, upstream of every per-task
# try/except in the worker, so a KeyError on one entry's fields propagated out
# of Worker.run() and exited the process. All three replicas read the same
# entry and died the same way, and a crashed worker never XACKs - so the entry
# was still there on restart. Measured against the live stack:
#
#     worker-1 restarts=0 state=exited
#     worker-2 restarts=0 state=exited
#     worker-3 restarts=0 state=exited
#     queue: stream_length=24, pending=0, backlog=24
#
# A single bad message caused a permanent, self-sustaining outage with 24
# pages stranded. These tests pin the three properties that fix it: reading
# does not raise, the entry is settled so it cannot be redelivered forever,
# and it is counted rather than silently dropped.
# --------------------------------------------------------------------------


def _malformed_fields() -> dict[str, str]:
    """Exactly the entry that caused the outage: no `enqueued_at_ms`."""
    return {
        "job_id": "poison-job",
        "page_index": "0",
        "first_enqueued_at_ms": str(int(time.time() * 1000)),
        "trace_id": "poison",
    }


async def test_malformed_entry_does_not_raise_out_of_read(queue, redis):
    """The whole bug in one assertion: read() used to raise KeyError here."""
    await redis.xadd(STREAM, _malformed_fields())

    tasks = await queue.read("c1", count=10, block_ms=50)

    assert tasks == [], "a malformed entry must not be returned as a task"


async def test_malformed_entry_is_settled_not_redelivered(queue, redis):
    """Ack + XDEL, so the crash cannot resurrect itself on restart.

    This is the property that made the original bug permanent rather than
    transient: an unacked entry comes back to the next worker to read.
    """
    await redis.xadd(STREAM, _malformed_fields())

    await queue.read("c1", count=10, block_ms=50)

    depth = await queue.depth()
    assert depth["pending"] == 0, "quarantined entry left in the PEL"
    assert depth["backlog"] == 0
    assert await redis.xlen(STREAM) == 0, "quarantined entry left in the stream"

    # And a second reader finds nothing, which is the redelivery check stated
    # from the other side.
    assert await queue.read("c2", count=10, block_ms=50) == []


async def test_malformed_entry_is_counted_and_inspectable(queue, redis):
    """Counted, never silently swallowed - and diagnosable after the fact."""
    await redis.xadd(STREAM, _malformed_fields())

    await queue.read("c1", count=10, block_ms=50)

    assert await queue.poison_count() == 1

    quarantined = await redis.xrange(POISON_STREAM)
    assert len(quarantined) == 1
    _entry_id, fields = quarantined[0]
    assert fields["origin_stream"] == STREAM
    assert "enqueued_at_ms" in fields["reason"], fields["reason"]
    # The producer bug must be diagnosable from the quarantine alone.
    assert fields["field:job_id"] == "poison-job"


async def test_good_entries_survive_a_poison_entry_in_the_same_batch(queue, redis):
    """The blast radius is one entry, not the batch.

    XREADGROUP returns a batch; handling the fault per entry rather than per
    read is what keeps the other pages moving.
    """
    await redis.xadd(STREAM, _malformed_fields())
    await queue.enqueue_pages("good-job", 3, trace_id="t")

    tasks = await queue.read("c1", count=10, block_ms=50)

    assert sorted(t.page_index for t in tasks) == [0, 1, 2]
    assert all(t.job_id == "good-job" for t in tasks)
    assert await queue.poison_count() == 1


async def test_malformed_orphan_does_not_take_down_the_reaper(queue, redis):
    """The reaper path needs the same guard, and needs it more.

    claim_orphans runs on a timer, so a malformed orphan would crash every
    replica repeatedly without any job being submitted at all.
    """
    entry_id = await redis.xadd(STREAM, _malformed_fields())
    # Make it look abandoned by a consumer that will never come back.
    await redis.xclaim(
        STREAM,
        GROUP,
        "ghost",
        min_idle_time=0,
        message_ids=[entry_id],
        idle=120_000,
        justid=True,
        force=True,
    )

    claimed = await queue.claim_orphans("c1", min_idle_ms=1000, count=10)

    assert claimed == []
    assert await queue.poison_count() == 1
    assert await redis.xlen(STREAM) == 0


async def test_a_non_integer_field_is_quarantined_too(queue, redis):
    """ValueError, not just KeyError - int() on a present-but-garbage field."""
    fields = _malformed_fields()
    fields["enqueued_at_ms"] = "not-a-number"
    await redis.xadd(STREAM, fields)

    assert await queue.read("c1", count=10, block_ms=50) == []
    assert await queue.poison_count() == 1
