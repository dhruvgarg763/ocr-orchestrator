"""Recover page tasks orphaned by a worker that will never come back.

The one recovery layer that covers a worker being REPLACED rather than
restarted: a consumer that stopped renewing its leases leaves PEL entries
delivered (so `XREADGROUP >` skips them) and owned by nobody, which is the
one state a zero-drop guarantee cannot survive. Never dispatches a claimed
orphan directly - `XACK` is group-scoped, so if the "dead" owner is
actually alive-and-slow, its eventual ack would `XDEL` the only queue
entry for a page this reaper is mid-flight on. Instead it rolls the page
back to its last committed checkpoint, publishes a fresh entry, and acks
the original, so nobody ever shares an entry id. A reclaim spends one
attempt (unlike a stage handoff or a saturation requeue) because a worker
dying while holding a specific page is weak evidence against that page,
and an uncounted reclaim would let a poison page cycle through every
replica forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings
from app.queue.results import ResultPublisher
from app.queue.state import TERMINAL_STATES, PageState, PageStateStore
from app.queue.streams import PageQueue, PageTask
from app.worker.pipeline import force_terminal, release_claim
from common.logging import get_logger
from common.tracing import new_span_id, set_trace_context

log = get_logger("reaper")


@dataclass
class ReaperStats:
    """Counters for the four things a claimed orphan can turn out to be.

    Split apart rather than totalled because they mean opposite things
    operationally. A rising `requeued` is recovery working. A rising
    `already_terminal` means workers are dying between commit and ack, which is
    harmless but points at a shutdown-path bug. A rising `expired` means job
    state is aging out while pages are still queued, i.e. `result_ttl_s` is too
    low for the current backlog. A rising `forced_terminal` means pages are
    being reclaimed repeatedly and giving up - a poison page, or a crash loop.
    """

    scans: int = 0
    claimed: int = 0
    requeued: int = 0
    forced_terminal: int = 0
    already_terminal: int = 0
    expired: int = 0
    leases_renewed: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "scans": self.scans,
            "claimed": self.claimed,
            "requeued": self.requeued,
            "forced_terminal": self.forced_terminal,
            "already_terminal": self.already_terminal,
            "expired": self.expired,
            "leases_renewed": self.leases_renewed,
        }


@dataclass
class Reaper:
    settings: Settings
    store: PageStateStore
    queue: PageQueue
    consumer: str
    results: ResultPublisher | None = None
    stats: ReaperStats = field(default_factory=ReaperStats)

    async def scan(self) -> int:
        """One pass. Returns how many orphans were claimed.

        Safe to run on every replica simultaneously - see
        `PageQueue.claim_orphans` for why the claim is conditional rather than
        racy - so there is no leader election, no lock, and no single reaper
        whose own death would disable recovery.
        """
        self.stats.scans += 1
        orphans = await self.queue.claim_orphans(
            self.consumer,
            min_idle_ms=int(self.settings.reaper_min_idle_s * 1000),
            count=self.settings.reaper_batch,
        )
        self.stats.claimed += len(orphans)
        for task in orphans:
            await self._recover(task)
        return len(orphans)

    async def _recover(self, task: PageTask) -> None:
        # Adopt the original trace id, with a fresh span. Without this the
        # recovery of a page appears in the logs under no trace at all, so the
        # one event you most want to correlate with an incident is the one event
        # that cannot be correlated.
        set_trace_context(task.trace_id, new_span_id())

        page = await self.store.get_page(task.job_id, task.page_index)

        if not page:
            # Job state aged out by TTL while this entry sat in a dead
            # consumer's PEL. There is nothing left to transition, and nothing
            # that could consume a result. Settle the entry so the scan does not
            # re-examine it forever.
            self.stats.expired += 1
            await self.queue.ack(task.entry_id, stream=task.stream)
            log.warning(
                "orphan_state_expired",
                job_id=task.job_id,
                page_index=task.page_index,
                age_s=round(task.age_ms / 1000, 1),
            )
            return

        if PageState(page["state"]) in TERMINAL_STATES:
            # The worker committed the page and died before acking. Common and
            # entirely benign - this is at-least-once behaving as designed - and
            # the correct response is to settle the entry and nothing else.
            # Re-running would be a duplicate; requeueing would loop.
            self.stats.already_terminal += 1
            await self.queue.ack(task.entry_id, stream=task.stream)
            return

        # Out of budget? Decided exactly as the worker's own crash handler
        # decides it, from the same fields, so a page cannot get a different
        # answer depending on which path noticed it was in trouble.
        age_s = task.age_ms / 1000
        exhausted = (
            age_s >= self.settings.page_deadline_s
            or task.attempt + 1 >= self.settings.effective_max_requeues
        )

        if exhausted:
            # Never leave it non-terminal. WHICH terminal state is the
            # transition table's decision, not this call site's: a page with a
            # committed layout degrades to FALLBACK_DONE and still counts
            # towards the job's completion, so the job finishes and its
            # subscribers get a job.complete instead of waiting out the SSE
            # duration limit for an event that can never arrive.
            landed = await force_terminal(
                task.job_id,
                task.page_index,
                self.store,
                f"reclaimed after {age_s:.0f}s, attempt {task.attempt + 1}",
                results=self.results,
            )
            self.stats.forced_terminal += 1
            await self.queue.ack(task.entry_id, stream=task.stream)
            log.error(
                "orphan_forced_terminal",
                job_id=task.job_id,
                page_index=task.page_index,
                state=landed,
                attempt=task.attempt + 1,
                age_s=round(age_s, 1),
            )
            return

        # ORDER MATTERS, and it is the opposite of intuitive. Release the claim
        # FIRST, then requeue.
        #
        # A page left in a *_RUNNING state is read by the next delivery as
        # "a live worker owns this" - so the pipeline stands down with
        # OWNED_BY_OTHER and the worker acks, destroying the fresh entry and
        # stranding the page. Requeueing without releasing would therefore
        # produce a queue entry that is guaranteed to be thrown away.
        #
        # Crashing between the two is safe: the page sits at its checkpoint with
        # its original entry still pending, so the next scan claims it again.
        # Crashing the other way round would leave a *_RUNNING page whose only
        # entry had been acked.
        await release_claim(task.job_id, task.page_index, self.store)
        await self.queue.requeue(task)
        self.stats.requeued += 1
        log.warning(
            "orphan_requeued",
            job_id=task.job_id,
            page_index=task.page_index,
            was=page["state"],
            attempt=task.attempt + 1,
            age_s=round(age_s, 1),
        )

    async def renew(self, tasks: list[PageTask]) -> int:
        """Tell Redis this worker is still alive on the entries it holds.

        The other half of the mechanism, and the reason `reaper_min_idle_s` can
        be 30s rather than the ~130s a legitimate hold can reach. See
        `PageQueue.renew_leases`.
        """
        if not tasks:
            return 0
        renewed = await self.queue.renew_leases(self.consumer, tasks)
        self.stats.leases_renewed += renewed
        return renewed
