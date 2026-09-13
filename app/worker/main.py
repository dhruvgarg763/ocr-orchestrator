"""Worker process: consume page tasks, run them through the pipeline.

Run with `python -m app.worker.main`.

Concurrency model: BOUNDED DISPATCH
-----------------------------------
The worker never reads more tasks than it has free capacity to run:

    capacity = concurrency - len(in_flight)
    tasks    = await queue.read(lead_count=..., main_count=...)

The tempting alternative is to read everything and guard execution with a
semaphore:

    sem = asyncio.Semaphore(16)
    async with sem: await process_page(task)      # the rest park here

That bounds concurrent *execution* but not what the process *holds*. By the time
the semaphore is consulted you have already created N coroutine objects (~3.2 KB
each, measured) and - worse - already read N tasks off the stream, moving every
one of them into this consumer's Pending Entries List. Die at that point and all
N need reclaiming rather than the handful actually being worked on.

Bounded dispatch leaves the backlog in Redis, which is where a backlog belongs:
durable, observable via XLEN, and someone else's memory. Resident tasks are
<= worker_concurrency by construction, whether the queue holds 10 pages or
100,000.

Failure isolation: each page runs as its own task and _handle catches
everything, so one poisoned page cannot take down the dispatch loop or its
siblings. With `asyncio.gather` over a batch, one unhandled exception would
abandon the whole batch.

Rate limiting (Step 7) is a shared Redis token bucket per endpoint, so the
limit holds across replicas: N in-process buckets would have permitted N x the
intended rate. Calls wait for a token instead of being rejected downstream,
which is the difference between backpressure and load shedding.

A circuit breaker (Step 9) sits in front of the limiter, also shared through
Redis so one replica's discovery of an outage stops all of them. When the VLM is
unavailable a page degrades to its committed layout output rather than failing,
which is what makes the zero-drop guarantee achievable.

Two lanes, and a budget partitioned between them
------------------------------------------------
Page 0 of every job goes to a priority lane (`stream:pages:lead`) so that a
job's FIRST page is not read behind other jobs' later pages - under one FIFO
stream the 50th job's first page sat at queue position ~980 and p95
time-to-first-page was 14.5s. `worker_lead_reserve` slots are withheld from the
main lane so a first page always has somewhere to run, which matters because
bounded dispatch only reads when a slot is free.

Acknowledgement policy - four distinct outcomes, four responses:

  terminal   ack. DONE, FALLBACK_DONE or FAILED: the page is finished, for
             better or worse, and redelivery would gain nothing.
  HANDOFF    REQUEUE without spending an attempt, and do not pause. Layout is
             committed and its page.partial is already on every subscriber's
             stream; the page goes back so a fresh slot claims its VLM stage
             rather than this one blocking for 1.5-3s on a 10 rps endpoint.
             Progress, not congestion - so it must not be charged against the
             attempt budget or every page would spend one on its happy path.
  SATURATED  REQUEUE, do not ack, then pause this slot briefly. Nothing was
             attempted and nothing degraded, so holding the page would pin a
             worker that could be serving others. Bounded by the page deadline.
  crash      neither. The task stays in the Pending Entries List, which is the
             safety net Step 14 harvests.

Crash recovery, in four layers (Step 14)
----------------------------------------
  per-stage commits  a crash costs one stage, not the page.
  read_own_pending   a worker RESTARTING under the same name (the container
                     hostname) resumes its own interrupted work at once.
  the reaper         a worker REPLACED rather than restarted leaves PEL entries
                     owned by a name that will never return. `_maintenance`
                     renews leases on what this worker holds and reclaims what
                     nobody is renewing - see app/worker/reaper.py.
  Idempotency-Key    a duplicate delivery never becomes a duplicate model call,
                     which is what makes all of the above safe to be
                     aggressive about.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import time
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.core.redis_client import close_redis, get_redis, init_redis
from app.pdf.splitter import (
    InvalidPdf,
    PageContent,
    delete_document,
    extract_page_async,
    orphan_candidates,
    page_ref as make_page_ref,
    pdf_path,
)
from app.queue.results import ResultPublisher
from app.queue.state import PageStateStore
from app.queue.streams import LEAD_STREAM, PageQueue, PageTask
from app.ratelimit.adaptive import AdaptiveRate
from app.ratelimit.breaker import CircuitBreaker
from app.ratelimit.token_bucket import TokenBucketLimiter
from app.worker.client import LAYOUT, VLM, ModelClient
from app.worker.metrics import WorkerMetrics
from app.worker.reaper import Reaper
from app.worker.pipeline import (
    Outcome,
    force_terminal,
    process_page,
    release_claim,
)
from common.logging import configure_logging, get_logger
from common.tracing import new_span_id, new_trace_id, set_trace_context

log = get_logger("worker")


def consumer_name() -> str:
    """Stable per container, so own-pending recovery works across restarts.

    A fresh UUID per process would orphan any work the previous process had
    pending, leaving it for the reaper's idle timeout instead of being picked up
    immediately.
    """
    return os.getenv("ORCH_CONSUMER_NAME") or socket.gethostname()


@dataclass
class WorkerStats:
    outcomes: dict[str, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)

    def record(self, outcome: Outcome) -> None:
        self.outcomes[outcome.value] = self.outcomes.get(outcome.value, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.outcomes.values())

    @property
    def pages_per_sec(self) -> float:
        elapsed = max(1e-6, time.monotonic() - self.started_at)
        return self.total / elapsed


class Worker:
    def __init__(
        self,
        settings: Settings,
        store: PageStateStore,
        queue: PageQueue,
        client: ModelClient,
        results: ResultPublisher | None = None,
        metrics: WorkerMetrics | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._queue = queue
        self._client = client
        self._results = results
        self._metrics = metrics
        self._consumer = consumer_name()
        self._stopping = asyncio.Event()
        self._in_flight: set[asyncio.Task[None]] = set()
        self._lead_in_flight = 0
        """Tasks currently running that arrived via the priority lane.

        Counted separately so the reservation means something: without it the
        lane's budget would be spent by whatever happened to be in flight."""
        self._held: dict[tuple[str, str], PageTask] = {}
        """Entries this worker currently owns in the consumer group's PEL.

        Tracked separately from `_in_flight`, which holds asyncio.Tasks and is
        wholesale REASSIGNED by `asyncio.wait`. Lease renewal needs the queue
        entries, not the coroutines, and needs them to survive that
        reassignment - deriving one from the other would couple the liveness
        signal to an unrelated bookkeeping detail.
        """
        self._stats = WorkerStats()
        self._last_sweep = 0.0
        self._reaper = Reaper(
            settings=settings,
            store=store,
            queue=queue,
            consumer=self._consumer,
            results=results,
        )

    def request_stop(self, signame: str) -> None:
        log.info("shutdown_requested", signal=signame, consumer=self._consumer)
        self._stopping.set()

    async def run(self) -> None:
        settings = self._settings
        await self._queue.ensure_group()
        log.info(
            "worker_started",
            consumer=self._consumer,
            concurrency=settings.worker_concurrency,
            prefetch=settings.worker_prefetch,
        )

        # Resume our own interrupted work before taking anything new, so a
        # restart finishes in-flight pages rather than starting fresh ones while
        # the old ones sit unacknowledged.
        for task in await self._queue.read_own_pending(
            self._consumer, count=settings.worker_concurrency
        ):
            self._spawn(task)

        maintenance = asyncio.create_task(self._maintenance())

        # Keep looping while there is anything to do OR anything still running:
        # on shutdown we stop accepting new work but must drain what we hold.
        while not self._stopping.is_set() or self._in_flight:
            capacity = settings.worker_concurrency - len(self._in_flight)

            if capacity > 0 and not self._stopping.is_set():
                # The budget is PARTITIONED, not shared. `worker_lead_reserve`
                # slots are only ever spent on the priority lane, so a new
                # job's first page never waits behind pages that are parked on
                # a VLM token - which is what made the lane ineffective before
                # the reservation existed.
                # Never the LAST slot: a reservation that consumed the whole
                # budget would leave the main lane unable to read at all, so a
                # single-slot worker would serve first pages and nothing else.
                lead_reserve = min(
                    settings.worker_lead_reserve,
                    max(0, settings.worker_concurrency - 1),
                )
                if lead_reserve == 0:
                    # Too little capacity to partition at all (concurrency=1).
                    # Fall back to a SHARED budget, because a reservation of
                    # zero is not "no priority" - it starved the lead lane
                    # completely, so page 0 of every job was never read and the
                    # job could never finish.
                    tasks = await self._queue.read(
                        self._consumer,
                        count=min(settings.worker_prefetch, capacity),
                        block_ms=settings.worker_block_ms,
                    )
                else:
                    lead_budget = max(0, lead_reserve - self._lead_in_flight)
                    main_budget = max(
                        0,
                        min(
                            settings.worker_prefetch,
                            # Main may use everything except the reservation.
                            capacity - lead_budget,
                        ),
                    )
                    tasks = await self._queue.read(
                        self._consumer,
                        lead_count=min(lead_budget, capacity),
                        main_count=main_budget,
                        block_ms=settings.worker_block_ms,
                        # Only matters when main_budget is 0. Then the reserved
                        # slots are the only ones that can act, so the lane
                        # itself is what we wait on - briefly, so that capacity
                        # freed during the wait is not left idle.
                        lead_block_ms=settings.worker_lead_poll_ms,
                    )
                for task in tasks:
                    self._spawn(task)
                if tasks:
                    # Fill remaining capacity immediately instead of waiting for
                    # one of these to finish first.
                    continue

            # Swept from the idle path on an interval: it is a few stats, and
            # doing it while the queue is empty keeps it off the hot path
            # entirely. One worker doing it is enough, and the Redis liveness
            # check makes a concurrent sweep harmless.
            now = time.monotonic()
            if now - self._last_sweep >= self._settings.orphan_sweep_interval_s:
                self._last_sweep = now
                try:
                    await self._sweep_orphan_documents()
                except Exception as exc:  # noqa: BLE001
                    # Housekeeping must never take down the dispatch loop.
                    log.warning("orphan_sweep_failed", error=type(exc).__name__)

            if self._in_flight:
                # Either we are at capacity, or the queue is idle and we are
                # just waiting for running pages. Both mean: block until at
                # least one finishes, then re-evaluate capacity.
                done, self._in_flight = await asyncio.wait(
                    self._in_flight, return_when=asyncio.FIRST_COMPLETED
                )
                self._reap(done)
            elif self._stopping.is_set():
                break

        # Cancelled only after the drain, so pages finishing during shutdown
        # keep their leases renewed to the last moment.
        maintenance.cancel()
        try:
            await maintenance
        except asyncio.CancelledError:
            pass

        log.info(
            "worker_stopped",
            consumer=self._consumer,
            processed=self._stats.total,
            outcomes=self._stats.outcomes,
            pages_per_sec=round(self._stats.pages_per_sec, 2),
            reaper=self._reaper.stats.snapshot(),
        )

    async def _maintenance(self) -> None:
        """Renew leases on held entries, and reclaim orphans left by the dead.

        A SEPARATE task, not a step in the dispatch loop, and that is the whole
        point. The dispatch loop parks in `asyncio.wait` whenever the worker is
        at capacity - which is precisely when it holds the most entries and has
        the most to lose - so a renewal folded into that loop would stop firing
        under exactly the load that makes it matter. It would also inherit
        `worker_block_ms` as its floor and the page duration as its ceiling: a
        worker holding sixteen 3s VLM calls would not renew for 3s, and one
        waiting out a 30s token wait would not renew for 30s, so the liveness
        signal would report "dead" for the healthiest worker in the fleet.

        Excluded from `_in_flight` deliberately. That set is the concurrency
        bound and is drained on shutdown; a housekeeping task inside it would
        consume a page slot and make the drain loop wait on something that never
        finishes on its own.
        """
        settings = self._settings
        last_scan = time.monotonic()

        while not self._stopping.is_set() or self._in_flight:
            await asyncio.sleep(settings.lease_renew_interval_s)

            # Renewal continues during shutdown drain. A worker gracefully
            # finishing its last pages is alive, and letting its leases lapse
            # would have another replica reclaim pages that are about to
            # complete - turning an orderly shutdown into duplicated work.
            try:
                await self._reaper.renew(list(self._held.values()))
            except Exception as exc:  # noqa: BLE001
                # A lapsed lease costs a redundant reclaim; a crashed
                # maintenance task costs every future renewal AND every future
                # reap. Never let one iteration end the loop.
                log.warning("lease_renew_failed", error=type(exc).__name__)

            # BEFORE the reaper gate below, not after. The `continue`s that
            # implement that gate fire on most iterations, so a flush placed at
            # the end of the loop body would only run on the rare iteration
            # that also scanned - making every metric as coarse as
            # `reaper_interval_s` and silently dropping the gauges in between.
            if self._metrics is not None:
                try:
                    await self._metrics.flush(
                        outcomes=self._stats.outcomes,
                        reaper=self._reaper.stats.snapshot(),
                        in_flight=len(self._held),
                    )
                except Exception as exc:  # noqa: BLE001
                    # Same rule as the renewal above: losing one flush costs an
                    # interval of resolution, losing the task costs every future
                    # renewal and reap. Metrics are never worth that.
                    log.warning("metrics_flush_failed", error=type(exc).__name__)

            now = time.monotonic()
            if not settings.reaper_enabled:
                continue
            if now - last_scan < settings.reaper_interval_s:
                continue
            last_scan = now
            try:
                await self._reaper.scan()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "reaper_scan_failed",
                    error=type(exc).__name__,
                    detail=str(exc)[:200],
                )

    async def _resolve_page(
        self, task: PageTask
    ) -> tuple[str | None, PageContent | None]:
        """Read exactly one page out of the job's document.

        Returns (None, None) for synthetic jobs - those created by POST /jobs
        with a page count rather than an upload - which is how the benchmark
        drives 1,000 pages without shipping 50 real PDFs.

        A stat per page rather than a flag on the task: the check costs about a
        microsecond and keeps the queue entry's size independent of the
        document's, which is what Step 11's watermark arithmetic assumes.

        A page that cannot be read is NOT failed here. The document is shared
        state that may be mid-write or on a volume that briefly misbehaved, and
        the layout endpoint can still be called with only a reference. Losing
        the descriptor degrades the request; failing the page would lose it.
        """
        try:
            document = pdf_path(self._settings.data_dir, task.job_id)
        except ValueError:
            # An id that cannot name a file has no document behind it. This is
            # a resolution question, not a processing one - raising here made
            # every page of such a job crash, which is exactly backwards.
            return None, None

        if not document.exists():
            return None, None

        reference = make_page_ref(task.job_id, task.page_index)
        try:
            content = await extract_page_async(
                document,
                task.page_index,
                text_limit=self._settings.pdf_text_sample_chars,
            )
        except InvalidPdf as exc:
            log.warning(
                "page_extract_failed",
                job_id=task.job_id,
                page_index=task.page_index,
                error=str(exc)[:200],
            )
            return reference, None

        return reference, content

    async def _sweep_orphan_documents(self) -> int:
        """Delete documents whose job no longer exists. Returns the count.

        The backstop for a leak the happy path cannot cover. Deleting on
        completion only fires when a worker acknowledges the LAST page of a
        job, so anything that prevents a job completing strands its upload:
        a page sitting in a dead worker's pending list, a process killed
        between the final ack and the delete, or a job whose Redis state
        expired while pages were still in flight.

        Redis liveness is the test, not age alone. Job state carries a TTL, so
        "the job hash is gone" means nothing can reference the file any more -
        a stronger signal than any timeout guess. The age gate exists only to
        avoid racing a job that is still being ingested.

        Deliberately not a separate process: it is a few stats per interval,
        and a sweeper that runs only when a worker is alive is a sweeper that
        cannot itself become the thing that needs monitoring.
        """
        candidates = orphan_candidates(
            self._settings.data_dir, min_age_s=self._settings.orphan_sweep_age_s
        )
        removed = 0
        for job_id in candidates:
            if await self._store.get_job(job_id):
                continue  # still live
            if delete_document(self._settings.data_dir, job_id):
                removed += 1
                log.warning("orphan_document_swept", job_id=job_id)
        return removed

    async def _maybe_delete_document(self, job_id: str) -> None:
        """Remove the source file once every page of the job is terminal.

        Uploads are the only resource here with no TTL: Redis keys expire on
        their own, a file does not. Without this, disk grows monotonically with
        every job ever ingested - a leak that only becomes visible in
        production, long after the run that caused it.

        Checked on terminal pages only, and any worker may be the one that
        happens to finish the job, so the delete has to tolerate having already
        been done - hence the idempotent `delete_document`.
        """
        done, total = await self._store.progress(job_id)
        if total and done >= total:
            delete_document(self._settings.data_dir, job_id)

    def _spawn(self, task: PageTask) -> None:
        if task.stream == LEAD_STREAM:
            self._lead_in_flight += 1
        self._held[(task.stream, task.entry_id)] = task
        self._in_flight.add(asyncio.create_task(self._handle(task)))

    def _reap(self, done: set[asyncio.Task[None]]) -> None:
        """Retrieve results so an exception is never silently swallowed.

        An un-retrieved task exception is only reported at garbage-collection
        time as "Task exception was never retrieved" - easy to miss, and it
        means a page died with no record. _handle should make this unreachable;
        this is the net under it.
        """
        for finished in done:
            if finished.cancelled():
                continue
            if (exc := finished.exception()) is not None:
                log.error(
                    "task_crashed", error=type(exc).__name__, detail=str(exc)[:200]
                )

    async def _handle(self, task: PageTask) -> None:
        """Wrapper that owns the lane accounting.

        The reserved-slot counter has to be released on EVERY exit path, and
        _handle_page has several - handoff, saturation, crash, terminal. A
        decrement per return is a leak waiting for the next branch someone
        adds, and a leaked reservation is permanent: the lane would quietly
        stop being served with nothing in the logs to say so.
        """
        started = time.monotonic()
        try:
            await self._handle_page(task)
        finally:
            self._held.pop((task.stream, task.entry_id), None)
            if task.stream == LEAD_STREAM:
                self._lead_in_flight = max(0, self._lead_in_flight - 1)
            if self._metrics is not None:
                # Timed in the wrapper, not in _handle_page, so every exit path
                # is covered - including the crash and saturation branches,
                # which are the ones whose duration is worth knowing.
                self._metrics.record_page(time.monotonic() - started)

    async def _handle_page(self, task: PageTask) -> None:
        # Adopt the trace id assigned at ingest, with a fresh span for this hop.
        # ContextVars are copied into each task at creation, so concurrent pages
        # cannot overwrite one another's context.
        set_trace_context(task.trace_id or new_trace_id(), new_span_id())

        started = time.monotonic()

        # Has this page run out of budget to keep holding out for full VLM
        # fidelity? Only the worker knows, because only it can see the page's
        # age and requeue count. The pipeline uses this to decide between
        # requeueing (preserve quality) and degrading (guarantee termination).
        age_s = task.age_ms / 1000
        final_attempt = (
            age_s >= self._settings.degrade_after_s
            or age_s >= self._settings.page_deadline_s
            or task.attempt + 1 >= self._settings.effective_max_requeues
        )

        # Resolve the page reference to real page data, if this job has a
        # document. Derived from the filesystem rather than carried on the task:
        # a `has_pdf` flag on every entry would grow the queue payload, and
        # admission control's memory arithmetic is stated per queued page.
        page_ref, page_content = await self._resolve_page(task)

        try:
            result = await process_page(
                task.job_id,
                task.page_index,
                store=self._store,
                client=self._client,
                page_ref=page_ref,
                page_content=page_content,
                final_attempt=final_attempt,
                results=self._results,
                handoff=self._settings.stage_handoff,
                # Only for a task out of our OWN pending list. It licenses
                # rolling back a *_RUNNING page, which is safe precisely
                # because the claim was left by this worker's previous
                # incarnation - and unsafe for anything else.
                reclaim=task.recovered,
            )
        except Exception as exc:  # noqa: BLE001
            # Isolation boundary: one bad page must not stop the dispatch loop
            # or its siblings.
            log.error(
                "page_crashed",
                job_id=task.job_id,
                page_index=task.page_index,
                error=type(exc).__name__,
                detail=str(exc)[:200],
            )
            # This used to ack unconditionally, which could STRAND the page:
            # process_page may have crashed with the page left in a *_RUNNING
            # state, so acking removed its only queue entry while the page was
            # still non-terminal - reachable by nobody, counted in no total,
            # and freezing done_count short of completion forever. Exactly the
            # failure that stranded 12 pages after a SIGKILL, arrived at by a
            # different route.
            #
            # process_page is supposed to make this unreachable (it handles
            # everything internally), so this is the net under it - and a net
            # that loses pages is not a net.
            reason = f"{type(exc).__name__}: {str(exc)[:120]}"
            if final_attempt:
                # No budget left to retry, so the page must become terminal
                # HERE. Which terminal state is decided by what it has already
                # committed, not by this call site.
                await force_terminal(
                    task.job_id,
                    task.page_index,
                    self._store,
                    reason,
                    results=self._results,
                )
                await self._queue.ack(task.entry_id, stream=task.stream)
                await self._maybe_delete_document(task.job_id)
                return

            # Budget remains: hand the page back at its last checkpoint so the
            # next delivery can claim it. Releasing first is essential - a page
            # left *_RUNNING would make the next delivery stand down as
            # OWNED_BY_OTHER and ack, stranding it after all.
            await release_claim(task.job_id, task.page_index, self._store)
            await self._queue.requeue(task)
            return

        if result.outcome is Outcome.HANDOFF:
            # Layout is committed and its partial event is already on every
            # subscriber's stream. Put the page back so a fresh slot claims its
            # VLM stage, and free this one NOW rather than blocking it for the
            # next 1.5-3s on a 10 rps endpoint.
            #
            # `count_attempt=False` because this is progress, not congestion.
            # The attempt counter and the page deadline bound how long a page
            # may be bounced by a busy system; charging a successful stage
            # transition against that budget would mean every page spent one of
            # its attempts on its own happy path, and long jobs would degrade
            # for no reason but their own success.
            #
            # No pause either, for the same reason: pausing exists to stop a
            # slot spinning on saturation, and nothing here was refused.
            await self._queue.requeue(task, count_attempt=False)
            self._stats.record(result.outcome)
            log.info(
                "stage_handoff",
                job_id=task.job_id,
                page_index=task.page_index,
                stage=result.detail,
                duration_ms=round((time.monotonic() - started) * 1000, 1),
            )
            return

        if result.outcome is Outcome.SATURATED:
            # Do NOT acknowledge a page we never attempted. The system is at
            # capacity, not broken: requeue so this worker can serve other work
            # and the page is tried again when tokens exist.
            #
            # Note what is NOT done here: the worker does not pause before
            # taking its next task. Backpressure is per-ENDPOINT, never
            # per-worker. Layout runs at 100 rps and VLM at 10, so a page
            # blocked on VLM says nothing about a page that needs layout -
            # pausing the worker would idle a 100 rps endpoint because a 10 rps
            # one is busy, and those pages could have been banking the layout
            # results that make the degraded fallback possible. The page itself
            # already backed off, jittered, inside limiter.acquire().
            #
            # The pipeline only returns SATURATED when budget remains - it
            # degrades or fails on the final attempt instead - so this branch
            # is simply "requeue". The guard below is defensive: a SATURATED
            # result on a final attempt would mean the two sides disagree, and
            # requeueing it would loop forever.
            if not final_attempt:
                await self._queue.requeue(task)
                self._stats.record(result.outcome)
                log.info(
                    "page_requeued",
                    job_id=task.job_id,
                    page_index=task.page_index,
                    attempt=task.attempt + 1,
                    age_s=round(age_s, 1),
                    endpoint=result.detail,
                )

                # Pace the loop before freeing this slot.
                #
                # Necessary because the limiter now REPORTS saturation in ~2ms
                # instead of after a 30s poll. Without a pause the loop spins:
                # measured at 99 saturation events/sec from 4 slots, which
                # consumed a 50-requeue backstop in 2 seconds and would put
                # ~6,000 Redis ops/sec of pure churn through 48 slots.
                #
                # Scoped as narrowly as possible: this pauses ONE slot, briefly,
                # after ONE saturation. It is not an endpoint-wide or
                # worker-wide stop - those would idle the 100 rps layout
                # endpoint because the 10 rps one is busy, and starve the layout
                # results that the degraded fallback and time-to-first-page both
                # depend on.
                #
                # Capped at the projected time to the next free slot, since
                # sleeping past that just wastes capacity.
                pause_s = min(
                    result.retry_after_s or self._settings.saturation_pause_s,
                    self._settings.saturation_pause_s,
                )
                if pause_s > 0:
                    await asyncio.sleep(pause_s)
                return

            # Out of budget. Step 9 degrades the page here instead of giving up;
            # for now it is acknowledged and left at its checkpoint, which makes
            # it visible as an incomplete page rather than an invisible loop.
            # Should be unreachable: the pipeline degrades or fails on a final
            # attempt rather than returning SATURATED. Logged loudly because it
            # means the worker and pipeline disagree about the budget.
            log.error(
                "saturated_on_final_attempt",
                job_id=task.job_id,
                page_index=task.page_index,
                attempts=task.attempt + 1,
                age_s=round(age_s, 1),
            )

        # Acknowledge: every other branch of process_page leaves the page
        # terminal or explicitly declines it, so redelivery would gain nothing.
        # A worker that dies before reaching this line leaves the task pending,
        # which is exactly the safety net Step 14 harvests.
        # Acknowledged against the stream that DELIVERED it. A lead-lane task
        # acked on the main stream would stay in the PEL, and the reaper would
        # later redeliver a page that was already finished.
        await self._queue.ack(task.entry_id, stream=task.stream)
        self._stats.record(result.outcome)
        await self._maybe_delete_document(task.job_id)

        log.info(
            "page_processed",
            job_id=task.job_id,
            page_index=task.page_index,
            outcome=result.outcome.value,
            detail=result.detail or None,
            queue_lag_ms=round(task.lag_ms, 1),
            duration_ms=round((time.monotonic() - started) * 1000, 1),
            # Emitted per page so the concurrency level is visible in the logs
            # rather than inferred.
            in_flight=len(self._in_flight),
        )


async def main() -> None:
    settings = get_settings()
    configure_logging("worker", settings.log_level)

    await init_redis(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        pool_timeout=settings.redis_pool_timeout,
    )

    store = PageStateStore(get_redis(), ttl_s=settings.result_ttl_s)
    queue = PageQueue(get_redis())

    # Publishing lives in the worker, not the API, because an event is a claim
    # about work that has been committed - and only the process that committed
    # it can make that claim without a second read to check.
    results = ResultPublisher(
        get_redis(),
        maxlen=settings.result_stream_maxlen,
        ttl_s=settings.result_ttl_s,
    )

    # One bucket per endpoint, shared by every replica through Redis. Keyed by
    # endpoint name only - deliberately NOT by worker - because the limit
    # belongs to the downstream service, not to any one client of it.
    limiters = {
        LAYOUT: TokenBucketLimiter(
            get_redis(),
            LAYOUT,
            rate=settings.layout_rps,
            burst=settings.layout_burst,
            ttl_s=settings.rate_limit_bucket_ttl_s,
            jitter_ms=settings.rate_limit_jitter_ms,
        ),
        VLM: TokenBucketLimiter(
            get_redis(),
            VLM,
            rate=settings.vlm_rps,
            burst=settings.vlm_burst,
            ttl_s=settings.rate_limit_bucket_ttl_s,
            jitter_ms=settings.rate_limit_jitter_ms,
        ),
    }

    # One breaker per endpoint, shared through Redis. Per-process breakers
    # would make every replica rediscover the same outage independently, each
    # hammering until it did - the blast radius would scale with replica count,
    # exactly as the in-process rate limiter did.
    def make_breaker(endpoint: str) -> CircuitBreaker:
        return CircuitBreaker(
            get_redis(),
            endpoint,
            window_s=settings.breaker_window_s,
            min_volume=settings.breaker_min_volume,
            failure_ratio=settings.breaker_failure_ratio,
            cooldown_s=settings.breaker_cooldown_s,
            max_probes=settings.breaker_max_probes,
            probe_successes=settings.breaker_probe_successes,
            ttl_s=settings.rate_limit_bucket_ttl_s,
        )

    breakers = {LAYOUT: make_breaker(LAYOUT), VLM: make_breaker(VLM)}

    # One AIMD controller per endpoint, shared through Redis like the bucket and
    # the breaker. Note max_rate is the ADVERTISED rate: this controller only
    # ratchets down from the published limit and recovers back up to it, because
    # unlike TCP we are not discovering unknown bandwidth - we are detecting
    # when real capacity has fallen below a figure we were given.
    def make_controller(endpoint: str, max_rate: float, slo_ms: float) -> AdaptiveRate:
        return AdaptiveRate(
            get_redis(),
            endpoint,
            max_rate=max_rate,
            min_rate=settings.aimd_min_rate,
            decrease_factor=settings.aimd_decrease_factor,
            increase_step=settings.aimd_increase_step,
            increase_after=settings.aimd_increase_after,
            refractory_ms=settings.aimd_refractory_ms,
            latency_slo_ms=slo_ms,
            latency_min_rate=max_rate * settings.aimd_latency_floor_fraction,
            latency_samples=settings.aimd_latency_samples,
            latency_min_samples=settings.aimd_latency_min_samples,
            ttl_s=settings.rate_limit_bucket_ttl_s,
        )

    controllers = (
        {
            LAYOUT: make_controller(
                LAYOUT, settings.layout_rps, settings.layout_latency_slo_ms
            ),
            VLM: make_controller(VLM, settings.vlm_rps, settings.vlm_latency_slo_ms),
        }
        if settings.adaptive_enabled
        else {}
    )

    metrics = WorkerMetrics(
        get_redis(),
        worker=consumer_name(),
        # Three missed flushes before this worker's gauges stop being counted -
        # the same tolerance, for the same reason, as the lease renewal in
        # app/worker/reaper.py. Shorter and a GC pause deletes a live worker's
        # in-flight count; longer and a dead one inflates the fleet total.
        gauge_ttl_s=settings.metrics_flush_interval_s * 3,
        prefetch_limit=settings.worker_concurrency,
    )

    async with ModelClient(
        settings.mock_base_url,
        timeout_s=settings.http_timeout_s,
        connect_timeout_s=settings.http_connect_timeout_s,
        max_connections=settings.http_max_connections,
        limiters=limiters,
        breakers=breakers,
        controllers=controllers,
        rate_limit_max_wait_s=settings.rate_limit_max_wait_s,
        max_attempts=settings.max_attempts,
        backoff_base_ms=settings.backoff_base_ms,
        backoff_max_ms=settings.backoff_max_ms,
        metrics=metrics,
    ) as client:
        worker = Worker(settings, store, queue, client, results, metrics=metrics)

        loop = asyncio.get_running_loop()
        for signame in ("SIGTERM", "SIGINT"):
            sig = getattr(signal, signame, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, worker.request_stop, signame)
            except NotImplementedError:
                # add_signal_handler is unavailable on Windows; containers are
                # Linux, so this only affects local runs outside Docker.
                signal.signal(sig, lambda *_s, _n=signame: worker.request_stop(_n))

        try:
            await worker.run()
        finally:
            await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
