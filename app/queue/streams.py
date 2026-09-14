"""Page task queue, on a Redis Stream with a consumer group.

`XREADGROUP` (not `BRPOP`) records delivery in the consumer's Pending
Entries List, so a worker SIGKILLed mid-page leaves recoverable evidence
rather than losing the task silently - at the cost of at-least-once
delivery, which the state CAS (app/queue/state.py) and Idempotency-Key
(app/worker/client.py) make safe to redeliver. Page 0 of every job goes to
a separate `stream:pages:lead`, drained preferentially, because under one
FIFO stream the 50th job's first page sat at queue position ~980 (p95 TTFP
14.5s against a 200ms target) - requeues and stage handoffs stay on the
main stream since a page already delivered once is no longer
first-page-critical. The task stream has no `MAXLEN` (trimming would
discard unprocessed work); result streams do (app/queue/results.py),
because their entries are disposable notifications about state already
committed elsewhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from common.logging import get_logger

log = get_logger("queue")

STREAM = "stream:pages"
LEAD_STREAM = "stream:pages:lead"
"""Page 0 of every job. Priority lane for time-to-first-page."""

STREAMS = (LEAD_STREAM, STREAM)
"""Read order. Listed lead-first because XREADGROUP returns results in the
order the streams are given."""

GROUP = "workers"

POISON_STREAM = "stream:pages:poison"
"""Entries that could not be parsed into a PageTask.

Capped with MAXLEN, unlike the task streams, because these entries are not
work: nothing will ever process them, so discarding the oldest loses no job.
The cap is what stops a malformed producer from filling Redis.
"""

POISON_MAXLEN = 1000

POISON_COUNTER_KEY = "metrics:poison_entries_total"
"""Counted, never silently swallowed - the zero-drop claim is about pages
reaching a *recorded* terminus, and a quarantined entry is a terminus."""

# Sentinel ids used by XREADGROUP:
#   ">" = entries never delivered to this group
#   "0" = this consumer's own pending entries, oldest first
NEW_MESSAGES = ">"
OWN_PENDING = "0"


@dataclass(frozen=True)
class PageTask:
    entry_id: str
    """The stream entry id. Needed to XACK, so it must survive the whole job."""

    job_id: str
    page_index: int
    enqueued_at_ms: int
    trace_id: str
    """Carried through the queue so one trace id spans api -> worker -> mock.
    Context does not cross a process boundary by itself."""

    recovered: bool = False
    """True if this task came out of THIS consumer's own pending list.

    The pipeline cannot infer it. A task carries no hint of whether it is a
    first delivery or a resumption, and the two need OPPOSITE handling of a
    page found in a `*_RUNNING` state: on a first delivery that state means a
    live worker holds the claim and we must stand down, but on a resumption the
    claim was left by our own previous incarnation, which is gone - standing
    down there stranded the page (see tests/test_own_pending_recovery.py).
    """

    stream: str = STREAM
    """Which stream delivered this task.

    Carried rather than inferred: an entry id is only meaningful together with
    its stream, so acknowledging a lead task against the main stream would
    silently fail to remove it from the PEL - and the reaper would later
    redeliver a page that had already been processed.
    """

    attempt: int = 0
    """How many times this task has been requeued. Carried on the entry so the
    count survives being handled by a different worker each time. Kept for
    observability and as a backstop; the requeue *bound* is the deadline below,
    not this counter."""

    first_enqueued_at_ms: int = 0
    """When this page FIRST entered the system, preserved across requeues.

    `enqueued_at_ms` is reset by each requeue - correct, because it measures how
    long the current entry waited for a worker. But a bound on systemic
    requeueing needs total age, so that survives separately.

    Why a deadline rather than an attempt count: saturation is a property of the
    endpoint, not of the page. Charging a page's attempt budget for a system-wide
    condition drops pages that did nothing wrong except arrive during a busy
    period - which is a zero-drop violation. Total age is the honest bound: a
    page gets a fair share of wall-clock time regardless of how many times
    congestion bounced it.
    """

    @property
    def age_ms(self) -> float:
        """Total time since this page first entered the system."""
        origin = self.first_enqueued_at_ms or self.enqueued_at_ms
        return max(0.0, time.time() * 1000 - origin)

    @property
    def lag_ms(self) -> float:
        """How long this task waited before a worker picked it up.

        Queue lag, not service latency: the headline number for whether workers
        are keeping up with ingestion.
        """
        return max(0.0, time.time() * 1000 - self.enqueued_at_ms)


def _parse(stream: str, entry_id: str, fields: dict[str, str]) -> PageTask:
    return PageTask(
        entry_id=entry_id,
        stream=stream,
        job_id=fields["job_id"],
        page_index=int(fields["page_index"]),
        enqueued_at_ms=int(fields["enqueued_at_ms"]),
        trace_id=fields.get("trace_id", ""),
        attempt=int(fields.get("attempt", 0)),
        # Falls back to enqueued_at_ms for entries written before this field
        # existed, and for the first delivery where the two are equal anyway.
        first_enqueued_at_ms=int(
            fields.get("first_enqueued_at_ms") or fields["enqueued_at_ms"]
        ),
    )


class PageQueue:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def ensure_group(self) -> None:
        """Create the consumer group, idempotently.

        mkstream=True creates the stream too, so a worker can start before any
        job has been submitted. BUSYGROUP just means another replica won the
        race to create it - expected, not an error.
        """
        for name in STREAMS:
            try:
                await self._redis.xgroup_create(name, GROUP, id="0", mkstream=True)
                log.info("consumer_group_created", stream=name, group=GROUP)
            except ResponseError as exc:
                if "BUSYGROUP" not in str(exc):
                    raise

    # --------------------------------------------------------------- producer

    async def enqueue_pages(
        self, job_id: str, total_pages: int, *, trace_id: str = ""
    ) -> int:
        """Append one task per page in a single round trip.

        Pipelined because time-to-first-page is graded at under 200ms: 100
        sequential XADDs would spend most of that budget on network round trips.

        Only a page *reference* is enqueued - job id and index. The page bytes
        stay on disk and are read on demand by whichever worker claims the task,
        which is what keeps memory O(1) per page rather than O(document).
        """
        now_ms = int(time.time() * 1000)
        async with self._redis.pipeline(transaction=False) as pipe:
            for index in range(total_pages):
                # Page 0 to the priority lane. One page per job is all
                # time-to-first-page needs, and holding the lane to one entry
                # per job is exactly what lets it drain inside the layout
                # endpoint's burst instead of behind 979 other pages.
                pipe.xadd(
                    LEAD_STREAM if index == 0 else STREAM,
                    {
                        "job_id": job_id,
                        "page_index": index,
                        "enqueued_at_ms": now_ms,
                        "first_enqueued_at_ms": now_ms,
                        "trace_id": trace_id,
                    },
                )
            ids = await pipe.execute()

        log.info("pages_enqueued", job_id=job_id, count=len(ids))
        return len(ids)

    # --------------------------------------------------------------- consumer

    async def read(
        self,
        consumer: str,
        *,
        block_ms: int,
        count: int | None = None,
        lead_count: int = 0,
        main_count: int = 0,
        lead_block_ms: int = 0,
    ) -> list[PageTask]:
        """Claim undelivered tasks from both lanes, blocking up to `block_ms`.

        `count` is the prefetch bound. It is the difference between holding a
        handful of tasks in memory and holding the whole backlog: with N
        replicas, resident tasks are at most count x N regardless of how deep
        the queue is. This is the primary memory boundary for the worker.

        Blocking rather than polling means no idle CPU burn and no added latency
        - Redis wakes the worker the moment a task arrives.

        Two ways to ask, because there are two genuinely different needs:

          count=N                 up to N tasks, whichever lane they are in.
                                  For callers with no per-lane policy.
          lead_count/main_count   separate per-lane budgets. The dispatch loop
                                  uses this, because the caller is the only one
                                  that knows how many of its slots are free PER
                                  LANE - and a single count split internally
                                  cannot express a reservation, whose whole
                                  point is that its budget is unavailable to
                                  the other lane.

        Each lane needs its own COUNT for a measured reason. XREADGROUP applies
        COUNT PER STREAM, so one combined call at the full count returns up to
        2 x count entries - and every one lands in this consumer's PEL, so the
        over-read cannot simply be discarded.
        """
        try:
            response = await self._read_lanes(
                consumer,
                lead_count=count if count is not None else lead_count,
                main_count=count if count is not None else main_count,
                block_ms=block_ms,
                total_cap=count,
                lead_block_ms=lead_block_ms,
            )
        except ResponseError as exc:
            # NOGROUP means the stream or the group has vanished underneath us.
            # The group is created once at startup, so without this the worker
            # crash-loops forever on a condition that is trivially recoverable:
            # a Redis restart without persistence, a failover to an empty
            # replica, or an operator flush would take down every replica and
            # keep them down until each was manually restarted.
            #
            # Narrowed to NOGROUP deliberately - any other ResponseError is a
            # real bug and must still surface.
            if "NOGROUP" not in str(exc):
                raise
            log.warning("consumer_group_missing_recreating", consumer=consumer)
            await self.ensure_group()
            return []

        return await self._flatten(response)

    async def read_own_pending(self, consumer: str, *, count: int) -> list[PageTask]:
        """Re-read tasks this consumer was given but never acknowledged.

        A worker restarting under a stable name (the container hostname) uses
        this to resume its own interrupted work immediately, without waiting for
        the reaper's idle timeout. Recovering another consumer's pending entries
        needs XAUTOCLAIM (Step 14).

        Both lanes, because a worker that died holding a lead task must recover
        it - otherwise that job's first page waits for the reaper's idle
        timeout, which is precisely the latency the priority lane exists to
        avoid.

        REGRESSION: read one lane at a time, for the same reason `read()` does.
        COUNT is applied PER STREAM, so a single call over both lanes at
        `count` returns up to 2 x count - measured, `count=4` returned 8. This
        runs at startup with count=worker_concurrency, so the over-read spawned
        twice the concurrency limit in coroutines and broke the bounded-dispatch
        invariant at exactly the moment a restarting worker is most loaded.
        """
        lead = (
            await self._redis.xreadgroup(
                groupname=GROUP,
                consumername=consumer,
                streams={LEAD_STREAM: OWN_PENDING},
                count=count,
            )
            or []
        )
        taken = sum(len(entries) for _stream, entries in lead)
        main: Any = []
        if taken < count:
            main = (
                await self._redis.xreadgroup(
                    groupname=GROUP,
                    consumername=consumer,
                    streams={STREAM: OWN_PENDING},
                    count=count - taken,
                )
                or []
            )
        tasks = [
            replace(task, recovered=True)
            for task in await self._flatten(lead + main)
        ]
        if tasks:
            log.warning(
                "resuming_own_pending", consumer=consumer, count=len(tasks)
            )
        return tasks

    async def claim_orphans(
        self, consumer: str, *, min_idle_ms: int, count: int
    ) -> list[PageTask]:
        """Take ownership of entries pending on a consumer that stopped renewing.

        The gap `read_own_pending` cannot close. A consumer name is the container
        hostname, so a worker that RESTARTS finds its own pending list and
        resumes immediately. A worker that is REPLACED does not: the new process
        has a new name, and the dead consumer's PEL entries are delivered (so
        `XREADGROUP >` skips them) and owned by nobody (so no live consumer will
        ever ack them). Non-terminal and unreachable.

        XPENDING-then-XCLAIM, not XAUTOCLAIM
        ------------------------------------
        XAUTOCLAIM is the one-call version and is the obvious choice, but it
        selects purely on idle time and cannot exclude a consumer - including
        THIS one. A worker's own task, legitimately parked for up to
        `rate_limit_max_wait_s` on a VLM token, is indistinguishable by idle
        time from a dead worker's task, so the reaper would roll back a page its
        own process is actively working. Owner filtering is not expressible in
        XAUTOCLAIM, so the two-call form is the correct primitive and the extra
        round trip is paid on an idle path once per interval.

        Why the read-then-claim is not a race
        -------------------------------------
        It looks like a classic read-modify-write: two reapers both see an entry
        as idle, both claim it, and the page is requeued twice. It is not, and
        the reason is that XCLAIM's min-idle-time argument is MANDATORY and
        CONDITIONAL - an entry whose idle clock has already been reset below it
        is not claimed and does not appear in the reply. The first claimant's
        XCLAIM resets idle to 0, so every other claimant's XCLAIM atomically
        returns nothing for that id. Passing the same threshold used for the scan
        turns the claim into a compare-and-set on idle time, which is why no
        leader election or lock is needed across replicas.

        Even so, a duplicate would be harmless rather than corrupting: the state
        CAS admits exactly one claimant, and the deterministic `Idempotency-Key`
        makes a second model call for the same (job, page, stage) coalesce onto
        the first. The conditional claim is what makes it not happen; those two
        are what make it survivable if it ever did.
        """
        claimed: list[PageTask] = []
        for name in STREAMS:
            pending = await self._redis.xpending_range(
                name,
                GROUP,
                min="-",
                max="+",
                count=count,
                idle=min_idle_ms,
            )
            # Never our own. Defence in depth: lease renewal should mean our
            # entries are never idle enough to appear here at all, so anything
            # that does appear under our name is a bug in renewal - and
            # reclaiming it would be the reaper sabotaging its own worker.
            ids = [
                row["message_id"]
                for row in pending
                if row.get("consumer") != consumer
            ]
            if not ids:
                continue

            entries = (
                await self._redis.xclaim(
                    name,
                    GROUP,
                    consumer,
                    # Conditional: the entry is claimed only if it is STILL this
                    # idle, which is what makes concurrent reapers safe.
                    min_idle_time=min_idle_ms,
                    message_ids=ids,
                )
                or []
            )
            for entry_id, fields in entries:
                # XCLAIM returns nothing for an id whose stream entry has been
                # XDELed, and (Redis >= 7) removes it from the PEL as a side
                # effect - which is exactly the tombstone cleanup wanted here.
                # An entry acked-then-crashed-before-XDEL would otherwise sit in
                # the PEL forever, and every scan would re-examine it.
                if not fields:
                    continue
                # Quarantined for the same reason as in `_flatten`, and this
                # path matters more: the reaper runs on a timer, so a
                # malformed orphan would take out every replica repeatedly
                # without any job needing to be submitted at all.
                try:
                    claimed.append(_parse(name, entry_id, fields))
                except (KeyError, ValueError, TypeError) as exc:
                    await self._quarantine(name, entry_id, fields, exc)

        if claimed:
            log.warning(
                "orphans_claimed",
                consumer=consumer,
                count=len(claimed),
                min_idle_ms=min_idle_ms,
            )
        return claimed

    async def renew_leases(self, consumer: str, tasks: list[PageTask]) -> int:
        """Reset the idle clock on entries this worker is still working on.

        What makes `reaper_min_idle_s` a liveness signal rather than a guess.
        Idle time in a PEL measures time since DELIVERY, so without renewal it
        rises on a healthy worker exactly as fast as on a dead one, and the only
        safe threshold is one above the longest legitimate hold (~130s here:
        three attempts of a token wait plus a call, plus backoff). Renewal makes
        the clock measure "has this worker checked in", so the threshold can be
        a few missed check-ins instead.

        JUSTID is load-bearing, not an optimisation. Plain XCLAIM increments
        delivery_count, so a page held for 90s would look as though it had been
        delivered 18 times - poisoning the one counter that distinguishes a
        genuinely redelivered page from a slow one. JUSTID resets idle time and
        leaves the counter alone.

        One call per lane carrying every id, so the cost is O(lanes) round trips
        regardless of how many pages are in flight.
        """
        by_stream: dict[str, list[str]] = {}
        for task in tasks:
            by_stream.setdefault(task.stream, []).append(task.entry_id)

        renewed = 0
        for name, ids in by_stream.items():
            # min_idle_time=0 so renewal is unconditional: we hold these
            # entries and want the clock reset however recently it last was.
            result = await self._redis.xclaim(
                name,
                GROUP,
                consumer,
                min_idle_time=0,
                message_ids=ids,
                justid=True,
            )
            renewed += len(result or [])
        return renewed

    async def requeue(self, task: PageTask, *, count_attempt: bool = True) -> str:
        """Return a task to the queue for a later attempt.

        Used when a failure is systemic rather than page-specific - no
        rate-limit token available, or a circuit breaker open. Retrying in place
        would pin the page while the worker could be serving others.

        ORDER MATTERS: publish the replacement FIRST, then acknowledge the
        original. Crashing between the two leaves the original still pending, so
        it is redelivered and the page runs twice - harmless, because processing
        is idempotent. Acknowledging first and then crashing would lose the task
        entirely. When one ordering risks a duplicate and the other risks a
        loss, always choose the duplicate.

        There is no delayed delivery in Redis Streams, so the replacement is
        immediately eligible. That is acceptable: the token bucket throttles
        whoever picks it up, which is the same backpressure by a different
        route. The persisted attempt counter is what stops it spinning forever.

        `count_attempt=False` is for a STAGE HANDOFF rather than a retry - a
        page whose layout succeeded, being put back so its VLM stage can be
        claimed by a fresh slot. The distinction is not cosmetic: the attempt
        counter and the page deadline exist to bound how long a page can be
        bounced by CONGESTION, and charging a successful stage transition
        against that budget would degrade pages for making progress. Every
        page would spend one of its attempts on its own happy path.
        """
        attempt = task.attempt + 1 if count_attempt else task.attempt
        # Always the MAIN lane, even for a task that arrived via lead. It has
        # been delivered once, so it is no longer first-page-critical - and
        # admitting retries to the priority lane would let a saturated endpoint
        # fill it with work that cannot run, destroying the emptiness that
        # makes it fast.
        new_id = await self._redis.xadd(
            STREAM,
            {
                "job_id": task.job_id,
                "page_index": task.page_index,
                # Reset: this measures how long the NEW entry waits for a worker.
                "enqueued_at_ms": int(time.time() * 1000),
                # Preserved: total age is what bounds systemic requeueing.
                "first_enqueued_at_ms": task.first_enqueued_at_ms or task.enqueued_at_ms,
                "trace_id": task.trace_id,
                "attempt": attempt,
            },
        )
        await self.ack(task.entry_id, stream=task.stream)

        log.info(
            "task_requeued",
            job_id=task.job_id,
            page_index=task.page_index,
            attempt=attempt,
            # Distinguished in the logs, or a handoff looks exactly like a
            # congestion retry and the requeue rate becomes uninterpretable.
            reason="handoff" if not count_attempt else "retry",
            new_entry_id=new_id,
        )
        return new_id

    async def ack(self, entry_id: str, *, stream: str) -> None:
        """Settle an entry: remove it from the PEL, then delete it.

        XACK alone would leave the entry in the stream forever, so the stream
        would grow without bound. XDEL is safe here because a consumer group
        delivers an entry to exactly one consumer, so nobody else can still
        need it.

        `stream` is REQUIRED, with no default. Acknowledging against the
        wrong stream fails SILENTLY - XACK on a stream that never held the
        entry is simply a no-op returning 0 - so the entry stays in the PEL
        and the reaper later redelivers a page that was already finished. A
        default turned that into a one-word mistake; making it explicit turns
        it into a type error. This was not hypothetical: it is exactly what
        the queue tests did, and the depth assertions are what caught it. Acknowledging a lead
        entry against the main stream would silently leave it in the PEL, and
        the reaper would later redeliver a page that was already finished.
        """
        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.xack(stream, GROUP, entry_id)
            pipe.xdel(stream, entry_id)
            await pipe.execute()

    async def _read_lanes(
        self,
        consumer: str,
        *,
        lead_count: int,
        main_count: int,
        block_ms: int,
        total_cap: int | None = None,
        lead_block_ms: int = 0,
    ) -> Any:
        """Probe the lead lane, then read the main one.

        Two calls rather than one so each lane gets its own COUNT without the
        combined read's doubling. The lead probe is non-blocking, so when the
        lane is empty - the common case, since it holds one entry per job - it
        costs one extra round trip and adds no latency.

        The block moves to whichever call can still afford to wait: if lead
        returned work we do not block at all, so those first pages start now
        instead of after a 2 second idle wait.
        """
        # Both probes below are NON-blocking; the wait is a single combined
        # read at the end that watches both lanes. Getting this wrong in either
        # direction has bitten: waiting on only one lane made a lead-only job
        # sit for worker_block_ms, and skipping the wait entirely (when a
        # budget worked out to zero) turned the dispatch loop into a hot spin.
        lead: Any = []
        if lead_count > 0:
            lead = (
                await self._redis.xreadgroup(
                    groupname=GROUP,
                    consumername=consumer,
                    streams={LEAD_STREAM: NEW_MESSAGES},
                    count=lead_count,
                    block=None,
                )
                or []
            )

        if total_cap is not None:
            # Shared-budget mode. The main budget is what LEAD ACTUALLY LEFT,
            # computed after the fact rather than pre-split: at count=1 an even
            # split gives 1 + 1 = 2, so the smallest possible prefetch would
            # read twice its bound. The dispatch loop's "resident tasks <=
            # worker_concurrency" is a hard claim, not an approximate one.
            taken = sum(len(entries) for _stream, entries in lead)
            main_count = min(main_count, total_cap - taken)

        if main_count <= 0:
            # Main is at its reservation cap, so the reserved slots are the
            # only ones that can act - which makes the LEAD LANE the thing to
            # wait on, not our in-flight tasks.
            #
            # This line has been wrong in both directions, and the two failures
            # are worth keeping side by side:
            #
            #   blocking for worker_block_ms (2s) starved the main lane.
            #   Capacity freed DURING the wait went unused because the worker
            #   was parked in Redis: in-flight pages fell from 48 to 2 with 760
            #   queued and throughput dropped ~40%.
            #
            #   returning immediately starved the LEAD lane. The caller fell
            #   back to waiting on in-flight tasks, so a first page started
            #   only when an unrelated VLM call finished - a ~94ms mean with a
            #   long tail, measured as p95 time-to-first-page of 333-406ms for
            #   a job arriving into a busy system, at EVERY reserve value.
            #
            # A bounded wait is the resolution: long enough that a lead arrival
            # wakes us immediately, short enough that freed capacity is not
            # left idle. See worker_lead_poll_ms for the arithmetic.
            #
            # lead_block_ms=0 keeps the old immediate-return behaviour, which
            # is what the unit tests assert and what a caller with no reserved
            # lane wants.
            if lead or not lead_block_ms:
                return lead
            return (
                await self._redis.xreadgroup(
                    groupname=GROUP,
                    consumername=consumer,
                    streams={LEAD_STREAM: NEW_MESSAGES},
                    count=lead_count,
                    block=lead_block_ms,
                )
                or []
            )

        main = (
            await self._redis.xreadgroup(
                groupname=GROUP,
                consumername=consumer,
                streams={STREAM: NEW_MESSAGES},
                count=main_count,
                # Non-blocking. The wait happens below, across BOTH lanes.
                block=None,
            )
            or []
        )
        if lead or main or not block_ms:
            return lead + main

        # Nothing in either lane, so now we wait - and the wait MUST watch both.
        #
        # REGRESSION this fixes: the wait used to sit on the main stream alone,
        # so a lead-lane XADD could not wake an idle worker. A multi-page job
        # hid it, because its pages 1..n-1 land in the main stream and do the
        # waking. A ONE-page job is entirely lead-lane, so nothing woke the
        # worker and its only page waited out worker_block_ms. Measured on an
        # idle system:
        #
        #     1-page job    p50 1499 ms   max 2840 ms
        #     2-page job    p50   74 ms
        #
        # A 20x latency cliff at exactly pages == 1, on a completely ordinary
        # input, in the BEST case for the system.
        #
        # COUNT is per stream, so the budget is halved to keep the total within
        # `main_count`. Under-reading is free here by construction: we only
        # reach this line because both lanes were empty a moment ago, so
        # whatever arrives is a trickle - and the dispatch loop immediately
        # loops round to claim the rest.
        share = max(1, main_count // 2)
        return (
            await self._redis.xreadgroup(
                groupname=GROUP,
                consumername=consumer,
                # Lead first: XREADGROUP returns streams in the order given.
                streams={LEAD_STREAM: NEW_MESSAGES, STREAM: NEW_MESSAGES},
                count=share,
                block=block_ms,
            )
            or []
        )

    async def _flatten(self, response: Any) -> list[PageTask]:
        """XREADGROUP returns [(stream_name, [(id, {fields}), ...]), ...].

        A malformed entry is QUARANTINED here, not raised. This is not
        defensive padding - the raising version took the whole pipeline down,
        and it is worth being precise about how, because the shape of the
        failure is the reason the handling has to live at this exact spot.

        `_parse` runs inside `read()`, which is upstream of every per-task
        `try/except` in the worker. So a `KeyError` on one entry's fields does
        not fail that page - it propagates out of `Worker.run()` and exits the
        process. Every replica then reads the same entry and dies the same way,
        and because a crashed worker never `XACK`s, the entry is still there on
        restart. Measured, with three replicas and one entry missing
        `enqueued_at_ms`:

            worker-1 restarts=0 state=exited
            worker-2 restarts=0 state=exited
            worker-3 restarts=0 state=exited
            queue: stream_length=24, pending=0, backlog=24

        A permanent outage with 24 pages stranded and nothing alive to consume
        them - from a single bad message, and self-sustaining across restarts.
        That is the worst available outcome for a zero-drop guarantee, and it
        is strictly worse than losing the one entry that caused it.

        This is the same failure mode the NOGROUP branch in `read()` already
        guards, which is the useful lesson: that branch was written because a
        vanished group crash-looped every replica, and the general rule - a
        fault in the *read* path is unrecoverable in a way a fault in the
        *processing* path is not - was never extended to a fault in the data.

        So: ack the entry (stop the redelivery that makes it self-sustaining),
        copy it to a capped quarantine stream (so it is inspectable rather than
        gone), and increment a counter exported on /metrics (so it is counted
        rather than silent). The loop survives, every other page keeps moving,
        and the one unusable entry is accounted for.
        """
        if not response:
            return []
        tasks: list[PageTask] = []
        for stream, entries in response:
            for entry_id, fields in entries:
                try:
                    tasks.append(_parse(stream, entry_id, fields))
                except (KeyError, ValueError, TypeError) as exc:
                    await self._quarantine(stream, entry_id, fields, exc)
        return tasks

    async def _quarantine(
        self,
        stream: str,
        entry_id: str,
        fields: dict[str, str],
        exc: Exception,
    ) -> None:
        """Settle an unparseable entry and record it.

        Order matters: the quarantine copy and the counter are written BEFORE
        the ack. If this process dies between the two, the entry is still
        pending and gets redelivered - it is quarantined twice, which
        overcounts. Acking first would instead delete the entry before it was
        recorded, losing it silently. Given a choice between double-counting a
        malformed entry and losing it, overcounting is the honest error.
        """
        reason = f"{type(exc).__name__}: {exc}"
        log.error(
            "entry_quarantined",
            stream=stream,
            entry_id=entry_id,
            reason=reason,
            fields=sorted(fields),
        )
        try:
            async with self._redis.pipeline(transaction=False) as pipe:
                pipe.xadd(
                    POISON_STREAM,
                    {
                        "origin_stream": stream,
                        "origin_entry_id": entry_id,
                        "reason": reason,
                        # The original fields, so the producer bug is
                        # diagnosable from the quarantine alone.
                        **{f"field:{k}": v for k, v in fields.items()},
                    },
                    maxlen=POISON_MAXLEN,
                    approximate=True,
                )
                pipe.incr(POISON_COUNTER_KEY)
                await pipe.execute()
        except Exception:
            # A quarantine that fails must not resurrect the crash it exists to
            # prevent. Losing the audit copy is survivable; losing the worker
            # is what this whole method is for.
            log.exception("quarantine_write_failed", entry_id=entry_id)
        await self.ack(entry_id, stream=stream)

    async def poison_count(self) -> int:
        """How many entries have been quarantined. Exported on /metrics."""
        raw = await self._redis.get(POISON_COUNTER_KEY)
        return int(raw or 0)

    # ------------------------------------------------------------ observability

    async def consumers(self, *, min_idle_ms: int) -> dict[str, Any]:
        """Who holds what, and how much of it nobody is renewing any more.

        The evidence surface for the crash-recovery claim. "The job finished
        after a SIGKILL" is only convincing if you can also show that the dead
        worker's entries existed, were attributed to a name that never came
        back, and then went to zero - and that no page was quietly abandoned
        along the way. Without this, recovery is indistinguishable from the work
        never having been queued.

        `orphaned` uses the same idle threshold as the reaper, so this endpoint
        answers exactly the question the reaper asks. A number that stays above
        zero across several scans means recovery is not keeping up - a genuinely
        different signal from a spike, which is just a worker having died.
        """
        lanes: dict[str, Any] = {}
        total_orphaned = 0
        for name in STREAMS:
            try:
                rows = await self._redis.xinfo_consumers(name, GROUP)
            except ResponseError as exc:
                if "NOGROUP" not in str(exc) and "no such key" not in str(exc):
                    raise
                rows = []
            stale = await self._redis.xpending_range(
                name, GROUP, min="-", max="+", count=1000, idle=min_idle_ms
            )
            total_orphaned += len(stale)
            lanes[name] = {
                "consumers": [
                    {
                        "name": row["name"],
                        "pending": int(row["pending"]),
                        "idle_ms": int(row["idle"]),
                    }
                    for row in rows
                ],
                "orphaned": len(stale),
            }
        return {
            "min_idle_ms": min_idle_ms,
            "orphaned": total_orphaned,
            "lanes": lanes,
        }

    async def depth(self) -> dict[str, int]:
        """Queue depth and in-flight count.

        `backlog` is undelivered work; `pending` is delivered-but-unacknowledged.
        They mean different things: rising backlog means workers are too slow,
        rising pending means workers are stuck or dying.
        """
        async with self._redis.pipeline(transaction=False) as pipe:
            for name in STREAMS:
                pipe.xlen(name)
                pipe.xpending(name, GROUP)
            rows = await pipe.execute()

        length = pending_count = 0
        for index, name in enumerate(STREAMS):
            length += int(rows[index * 2])
            raw_pending = rows[index * 2 + 1]
            if isinstance(raw_pending, dict):
                pending_count += int(raw_pending.get("pending", 0))

        # Summed across both lanes: a page is admitted-but-unsettled whichever
        # lane carries it. Counting only the main stream would under-report by
        # one entry per in-flight job, and the admission watermark is built
        # directly on this number.
        return {
            "stream_length": length,
            "pending": pending_count,
            "backlog": max(0, length - pending_count),
        }
