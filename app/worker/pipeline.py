"""Process one page: layout, then VLM.

Resume awareness is the payoff of committing each stage separately in Step 4: a
page is dispatched to the stage its state says it needs, so a redelivered page
never repeats a stage that has already been committed.

Four ways a page can end short of a full result, handled differently on purpose:

  DEGRADED   the VLM could not produce a result AND the page is out of budget
             to keep trying. Completes as FALLBACK_DONE carrying its
             already-committed layout output plus a low confidence flag. A
             SUCCESS for zero-drop purposes: output was produced, fidelity was
             lost.
  SATURATED  no rate-limit token, or an open circuit, with budget remaining.
             Nothing was consumed; the claim is released and the task requeued
             at FULL fidelity.
  HANDOFF    layout committed and its `page.partial` event is out; the page is
             put back so a fresh slot claims its VLM stage. Not a failure and
             not a retry - see below.
  FAILED     layout itself could not be completed, or the layout endpoint was
             unavailable with no budget left. The genuinely unrecoverable case.
  (crash)    the worker dies. Nothing is acknowledged, so the queue's pending
             list holds the task and the reaper recovers it (Step 14).

Degrading is a LAST RESORT, not a first response
------------------------------------------------
The spec is specific: "If retries are exhausted, fall back to Fast-Layout-Model
with a low confidence flag." A page that arrives while the VLM's circuit is
open has exhausted nothing - it never made the call - so degrading it there
would be both off-spec and wasteful.

Measured on an 80% rejection storm lasting 40s with a 5s breaker cooldown:
degrading on the first open circuit produced 32 layout-only pages out of 40.
Waiting instead gives each page ~8 deliveries of 3 attempts, putting
P(never succeeding) near 0.5% - and the outage ends long before the page
deadline. So a page holds out for full fidelity until `final_attempt`, and only
then accepts a degraded result.

Holding out costs no time-to-first-page: layout output streams as soon as it
lands, and only the VLM upgrade waits.

The asymmetry between the two stages is deliberate. The VLM degrades because a
committed layout result is a usable answer - and needs no extra call, since
that result is already in hand. Layout cannot degrade, because layout IS the
fallback: a page with no layout has nothing to fall back to.

Two events per page, and why the slot is released between them
--------------------------------------------------------------
Each committed stage publishes to the job's result stream: `page.partial` when
layout lands (~50ms), `page.final` when the page becomes terminal. Both are
gated on the state transition having been made by THIS caller, so a redelivered
page is a no-op on the notification path exactly as it is on the state path.

The page is then handed back to the queue rather than held through the VLM
stage. Holding it makes the fast stage inherit the slow stage's queueing: with
48 worker slots and 1,000 pages every slot parks on a VLM token, later pages
are never read off the queue at all, and their layout - which a 100 rps
endpoint could do immediately - waits on a 10 rps one. Measured p95
time-to-first-page went from 14.5s to under 1s.

This adds no new state and no new recovery path. LAYOUT_DONE is already a
durable checkpoint and already a legal resume point, so the redelivered page
skips the stage it has committed - the same mechanism that makes a crash cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

import httpx

from app.queue.state import (
    RECLAIM_TARGET,
    RUNNING_STATES,
    TERMINAL_STATES,
    TRANSITIONS,
    PageState,
    PageStateStore,
    TransitionResult,
)
from app.pdf.splitter import PageContent
from app.queue.results import (
    JOB_COMPLETE,
    PAGE_FINAL,
    PAGE_PARTIAL,
    ResultPublisher,
)
from app.ratelimit.breaker import CircuitOpen
from app.ratelimit.token_bucket import RateLimitTimeout
from app.worker.client import ModelClient
from common.logging import get_logger

log = get_logger("pipeline")


class Outcome(str, Enum):
    COMPLETED = "completed"
    """Both stages succeeded on this attempt."""

    RESUMED = "resumed"
    """Layout was already committed; only the VLM stage ran. This is what a
    crash-recovered page looks like."""

    ALREADY_TERMINAL = "already_terminal"
    """Redelivered after completion. At-least-once makes this normal, and it
    must be a no-op rather than reprocessing."""

    OWNED_BY_OTHER = "owned_by_other"
    """Another worker holds the claim. Ours is not the attempt that proceeds."""

    DEGRADED = "degraded"
    """The VLM was unavailable, so the page carries layout-only output with a
    low confidence flag. Counted as a SUCCESS for the zero-drop guarantee: the
    page produced usable output. Only the fidelity was lost."""

    HANDOFF = "handoff"
    """Layout committed and its partial event is out; the page is being put
    back so a fresh slot can claim its VLM stage.

    Not a failure and not a retry - a deliberate release of the worker slot
    between two stages whose capacities differ by 10x. See the module docstring.
    """

    SATURATED = "saturated"
    """No rate-limit token within the wait budget. The page is untouched and its
    claim released, so the caller must REQUEUE rather than acknowledge. Not a
    failure: the system is at capacity, and the page has lost nothing."""

    FAILED = "failed"
    """Layout could not be completed. The only genuinely unrecoverable case -
    VLM failures degrade instead, so this no longer covers "retries exhausted"
    for the heavy stage."""


# Layout output alone is a real result - spatial blocks, types, reading order -
# so a page can be served at reduced fidelity rather than not at all.
FALLBACK_CONFIDENCE = 0.4


async def _emit(
    results: ResultPublisher | None,
    event: str,
    transition: TransitionResult,
    job_id: str,
    page_index: int,
    **payload: object,
) -> None:
    """Publish a page event, but ONLY if this caller made the transition.

    The `transition.ok` gate is the whole reason this is a helper rather than a
    publish call at each site. At-least-once delivery means a page can be
    handed to a worker twice; the state CAS makes the second attempt a no-op,
    and gating the event on the same CAS makes the notification a no-op too.
    Publishing unconditionally would emit two `page.final` events for one page,
    so a client counting events would think a 20-page job had 21 or 22 pages
    and never reconcile with `total_pages`.

    Ordering is deliberate: the state commit happens first, this second. The
    reverse would let a crash between them announce a result that was never
    committed - a client told page 7 is DONE while Redis still says
    VLM_RUNNING, which no amount of client-side retrying can repair. This way
    the only crash window loses a NOTIFICATION of state that is durably
    stored, which `GET /jobs/{id}` still reports and which `job.complete`
    still counts. Lose the cheap thing, never the authoritative one.

    Failures here are swallowed. The stream is a convenience built on top of
    committed state, so a Redis hiccup while publishing must not turn a
    successfully processed page into a failed one - that would let the
    observability path break the thing it observes.
    """
    if results is None or not transition.ok:
        return
    try:
        await results.publish(
            job_id,
            event,
            {"page_index": page_index, "state": transition.requested.value, **payload},
        )
        if transition.completed_job:
            # Exactly one worker can reach this, because HINCRBY handed exactly
            # one of them done_count == total_pages.
            await results.publish(
                job_id,
                JOB_COMPLETE,
                {
                    "total_pages": transition.total_pages,
                    "done": transition.done_count,
                },
            )
    except Exception as exc:  # noqa: BLE001
        # NOTE the kwarg name. `event` is structlog's own first positional -
        # the message name - so `log.warning("msg", event=...)` raises
        # TypeError: got multiple values for argument 'event'. Which happened
        # here, inside the handler whose entire purpose is to never propagate:
        # the publish failure was swallowed correctly and then the LOG CALL
        # raised, escaped _emit, and the pipeline's broad handler turned a page
        # whose result was already durably committed into FAILED.
        #
        # The general rule this earned: a handler that exists to guarantee
        # nothing escapes must contain nothing that can throw. Reserved kwarg
        # names in the logging library count.
        log.warning(
            "result_publish_failed",
            job_id=job_id,
            page_index=page_index,
            result_event=event,
            error=type(exc).__name__,
        )


async def _degrade(
    job_id: str,
    page_index: int,
    store: PageStateStore,
    cause: BaseException,
    results: ResultPublisher | None = None,
) -> PageResult:
    """Complete a page with layout-only output and a low confidence flag.

    Terminal and counted, so the job can finish. The distinction from FAILED is
    the whole point: FAILED means we produced nothing, FALLBACK_DONE means we
    produced less. Only the second one satisfies a zero-drop requirement.
    """
    reason = type(cause).__name__
    if isinstance(cause, httpx.HTTPStatusError):
        reason = f"HTTP {cause.response.status_code}"

    result = await store.transition(
        job_id,
        page_index,
        PageState.FALLBACK_DONE,
        fields={
            "degraded": 1,
            "confidence": FALLBACK_CONFIDENCE,
            "fallback_reason": reason,
        },
    )
    # A degraded page is still a FINAL page as far as a subscriber is
    # concerned: it is terminal, it carries output, and the client needs to stop
    # waiting for an upgrade that will never come. The `degraded` flag and the
    # low confidence are what tell it the difference, not a separate event type.
    await _emit(
        results,
        PAGE_FINAL,
        result,
        job_id,
        page_index,
        complete=True,
        degraded=True,
        confidence=FALLBACK_CONFIDENCE,
        fallback_reason=reason,
    )
    log.warning(
        "page_degraded",
        job_id=job_id,
        page_index=page_index,
        reason=reason,
        applied=result.ok,
    )
    return PageResult(Outcome.DEGRADED, job_id, page_index, reason)


async def release_claim(
    job_id: str, page_index: int, store: PageStateStore
) -> None:
    """Hand a claimed page back, without losing committed stage progress.

    Moves a *_RUNNING page to its last checkpoint: LAYOUT_RUNNING -> PENDING,
    VLM_RUNNING -> LAYOUT_DONE. Never back to the start, so a released page
    keeps the layout result it already paid for.
    """
    page = await store.get_page(job_id, page_index)
    if not page:
        return
    current = PageState(page["state"])
    if current not in RUNNING_STATES:
        return
    await store.transition(
        job_id,
        page_index,
        RECLAIM_TARGET[current],
        # Narrowed to the exact state observed, so a concurrent transition
        # cannot be clobbered by this rollback.
        allowed_from=[current],
    )


async def force_terminal(
    job_id: str,
    page_index: int,
    store: PageStateStore,
    reason: str,
    *,
    results: ResultPublisher | None = None,
) -> str:
    """Drive a page to the terminal state its progress permits. Returns it.

    The last-resort path for a page that cannot be processed and has no budget
    left to try again. It exists so that "give up" can never mean "leave it
    non-terminal", which is the one outcome a zero-drop guarantee cannot
    survive: a non-terminal page with no queue entry is reachable by nobody,
    counts towards no total, and freezes `done_count` short of completion so
    the job never reports complete and a subscriber waits for an event that
    cannot arrive.

    WHICH terminal state is not a choice made here - the transition table
    already encodes it. FALLBACK_DONE is legal from LAYOUT_DONE and
    VLM_RUNNING, FAILED from PENDING and LAYOUT_RUNNING, and that split is
    exactly right: a page whose layout is committed has a usable answer to
    degrade to, and a page without one has nothing. So the observed state
    selects the target, and the table stays the single source of truth.
    """
    page = await store.get_page(job_id, page_index)
    if not page:
        return "MISSING"

    state = PageState(page["state"])
    if state in TERMINAL_STATES:
        return state.value

    degradable = PageState.FALLBACK_DONE in TRANSITIONS[state]
    target = PageState.FALLBACK_DONE if degradable else PageState.FAILED
    fields: dict[str, str | int | float] = (
        {
            "degraded": 1,
            "confidence": FALLBACK_CONFIDENCE,
            "fallback_reason": reason,
        }
        if degradable
        else {"error": reason}
    )

    applied = await store.transition(job_id, page_index, target, fields=fields)
    if not applied.ok:
        # Another claimant moved it first. Whatever it did, the page is theirs.
        return applied.observed

    await _emit(
        results,
        PAGE_FINAL,
        applied,
        job_id,
        page_index,
        complete=True,
        degraded=degradable,
        **({"confidence": FALLBACK_CONFIDENCE} if degradable else {"error": reason}),
    )
    log.error(
        "page_forced_terminal",
        job_id=job_id,
        page_index=page_index,
        observed=state.value,
        terminal=target.value,
        reason=reason,
    )
    return target.value


@dataclass
class PageResult:
    outcome: Outcome
    job_id: str
    page_index: int
    detail: str = ""

    retry_after_s: float = 0.0
    """For SATURATED only: projected seconds until the endpoint has a free
    slot. The worker uses it to pace its requeue loop, since saturation is now
    detected in ~2ms and would otherwise spin."""


async def process_page(
    job_id: str,
    page_index: int,
    *,
    store: PageStateStore,
    client: ModelClient,
    page_ref: str | None = None,
    page_content: PageContent | None = None,
    final_attempt: bool = False,
    results: ResultPublisher | None = None,
    handoff: bool = False,
    reclaim: bool = False,
) -> PageResult:
    """Run one page through layout and then the VLM.

    `final_attempt` says this page has run out of budget to keep holding out
    for full fidelity - it has waited longer than `degrade_after_s`, or used up
    the requeue backstop. Only then may an unreachable VLM produce a degraded
    result. The caller owns that decision because only it knows the page's age.
    """
    page = await store.get_page(job_id, page_index)
    if not page:
        return PageResult(Outcome.FAILED, job_id, page_index, "page state missing")

    state = PageState(page["state"])

    # Redelivery after completion. Cheap to detect, and detecting it is what
    # keeps at-least-once delivery from double-charging for a page.
    if state in TERMINAL_STATES:
        return PageResult(Outcome.ALREADY_TERMINAL, job_id, page_index, state.value)

    # A *_RUNNING state means some worker claimed this page and has not
    # finished. What to do about it depends entirely on WHO that worker was.
    if state in RUNNING_STATES:
        if not reclaim:
            # A first delivery: the claim belongs to a worker that is presumed
            # alive, so interfering would run the same page twice and
            # double-charge the model. Stand down and let it finish.
            return PageResult(Outcome.OWNED_BY_OTHER, job_id, page_index, state.value)

        # A RESUMPTION out of our own pending list, so the worker that left
        # this claim was our own previous incarnation and is definitively gone.
        #
        # REGRESSION: standing down here STRANDED the page. The caller
        # acknowledges OWNED_BY_OTHER - correct when a live worker will finish
        # the page - so the entry left the queue while the page stayed
        # non-terminal, reachable by nobody and visible to no reaper. Measured
        # live with one `docker kill -9` during a 120-page run: 108 DONE and 12
        # pages stuck in VLM_RUNNING with the queue completely drained, so
        # done_count froze at 108/120 and the job could never report complete.
        #
        # Roll back to the last committed checkpoint instead. LAYOUT_RUNNING
        # goes to PENDING, VLM_RUNNING to LAYOUT_DONE - never to the start, so
        # a committed layout result is not thrown away and not paid for twice.
        await release_claim(job_id, page_index, store)

        page = await store.get_page(job_id, page_index)
        if not page:
            return PageResult(Outcome.FAILED, job_id, page_index, "page state missing")
        state = PageState(page["state"])

        if state in TERMINAL_STATES:
            # Another replica's reaper recovered and finished it while we were
            # restarting. Nothing to do, and nothing to undo.
            return PageResult(Outcome.ALREADY_TERMINAL, job_id, page_index, state.value)
        if state in RUNNING_STATES:
            # The rollback lost a race with another claimant. It owns the page
            # now, so this attempt stands down after all.
            return PageResult(Outcome.OWNED_BY_OTHER, job_id, page_index, state.value)

    # Serialised once, used by both stages. The descriptor is bounded by
    # construction (geometry plus an optionally-truncated text sample), which is
    # what keeps the request body O(1) in the page's real size.
    descriptor = page_content.as_payload() if page_content else None

    resumed = state is PageState.LAYOUT_DONE

    try:
        # -- stage 1: layout ------------------------------------------------
        if state is PageState.PENDING:
            claim = await store.transition(job_id, page_index, PageState.LAYOUT_RUNNING)
            if not claim.ok:
                # Lost the race between reading the state and claiming it. The
                # atomic CAS is what makes this a harmless miss rather than a
                # double execution.
                return PageResult(
                    Outcome.OWNED_BY_OTHER, job_id, page_index, claim.observed
                )

            layout = await client.layout(
                job_id, page_index, page_ref=page_ref, page_content=descriptor
            )
            committed = await store.transition(
                job_id,
                page_index,
                PageState.LAYOUT_DONE,
                # Result and state are written by the same Lua script, so there
                # is no window where the state claims a result that is absent.
                fields={"layout": json.dumps(layout, separators=(",", ":"))},
            )

            # THE time-to-first-page move. Layout returns in ~50ms and the VLM
            # takes 1.5-3s, so a stream that emitted one event per page could
            # not put its first byte on the wire inside the 200ms target no
            # matter how the pipeline was tuned - the number is smaller than a
            # single VLM call. Emitting here makes first-byte latency a
            # property of the FAST stage, and the heavy stage an upgrade
            # delivered later against the same page_index.
            #
            # `complete: false` is the contract: the client may render this and
            # must expect exactly one page.final for the same page.
            await _emit(
                results,
                PAGE_PARTIAL,
                committed,
                job_id,
                page_index,
                complete=False,
                layout=layout,
            )

            if handoff:
                # STAGE HANDOFF. Stop here and let the caller requeue, so this
                # slot is free immediately instead of spending the next 1.5-3s
                # blocked on a VLM token.
                #
                # Without it the two stages share a slot, so the FAST stage
                # inherits the SLOW stage's queueing: with 48 slots and 100
                # pages every slot parks on a VLM token, pages 49-100 are never
                # read off the queue at all, and their layout - which the
                # 100 rps endpoint could have done immediately - waits for a
                # 10 rps endpoint to drain. Measured time-to-first-page went
                # from 4.9s to 0.2s at p95 by releasing here.
                #
                # Safe because LAYOUT_DONE is already a durable checkpoint
                # (Step 4) and already a legal resume point (Step 9's crash
                # recovery): the redelivered page skips the stage it has
                # committed and runs only the VLM. This adds no new state and
                # no new recovery path - it reuses the one that makes a crash
                # cheap.
                return PageResult(Outcome.HANDOFF, job_id, page_index, "layout")

        # -- stage 2: VLM ---------------------------------------------------
        claim = await store.transition(job_id, page_index, PageState.VLM_RUNNING)
        if not claim.ok:
            return PageResult(Outcome.OWNED_BY_OTHER, job_id, page_index, claim.observed)

        try:
            vlm = await client.vlm(
                job_id, page_index, page_ref=page_ref, page_content=descriptor
            )

        except (RateLimitTimeout, CircuitOpen) as exc:
            # We never reached the VLM: no token, or its circuit is open. This
            # page has exhausted NOTHING, so degrading it now would be wrong on
            # two counts.
            #
            # The spec is explicit - "If retries are exhausted, fall back to
            # Fast-Layout-Model with a low confidence flag." A page that never
            # attempted the call has not exhausted its retries.
            #
            # And it is wasteful. Measured on an 80% rejection storm lasting
            # 40s with a 5s breaker cooldown: a page gets ~8 deliveries, each
            # with 3 attempts, so P(never succeeding) is about 0.5% - and the
            # outage ends long before the page deadline. Degrading on the first
            # open circuit turned 32 of 40 pages into layout-only results that
            # would almost all have succeeded at full fidelity.
            #
            # So: hold out for quality, bounded by `final_attempt`. Requeue and
            # let the breaker's cooldown do its job.
            #
            # This does not cost time-to-first-page: layout output is streamed
            # as soon as it lands, and only the VLM upgrade is delayed.
            if not final_attempt:
                raise

            # Out of budget. Degrade rather than wait forever - this is the
            # backstop that keeps the zero-drop guarantee true.
            return await _degrade(job_id, page_index, store, exc, results)

        except BaseException as exc:
            # Retries genuinely exhausted, or a permanent error. THIS is the
            # case the spec describes, and the page really does have nothing
            # better available.
            #
            # By now the layout result is already committed (Step 4's per-stage
            # checkpoint) and is useful on its own - bounding boxes, block
            # types, reading order - so an unavailable VLM costs fidelity, not
            # the page. Note that "fall back to Fast-Layout-Model" needs no
            # extra call: that result is already in hand.
            return await _degrade(job_id, page_index, store, exc, results)

        committed = await store.transition(
            job_id,
            page_index,
            PageState.DONE,
            fields={
                "vlm": json.dumps(vlm, separators=(",", ":")),
                "confidence": vlm.get("confidence", 1.0),
                "degraded": 0,
            },
        )
        await _emit(
            results,
            PAGE_FINAL,
            committed,
            job_id,
            page_index,
            complete=True,
            degraded=False,
            confidence=vlm.get("confidence", 1.0),
            vlm=vlm,
        )

    except (RateLimitTimeout, CircuitOpen) as exc:
        # Systemic, not page-specific: we never reached the model, so nothing
        # has been consumed and nothing is degraded. Release the claim so the
        # requeued task is claimable again - leaving it in a *_RUNNING state
        # would make the next delivery see OWNED_BY_OTHER and stand down, and
        # the page would stall until the reaper's idle timeout.
        #
        # In practice this is now the LAYOUT stage only: the VLM stage handles
        # its own failures above, degrading instead of propagating. And that
        # asymmetry is the point - layout has no fallback, because layout IS the
        # fallback. A page with no layout result has nothing to degrade to, so
        # the only correct move is to put it back and try again later, bounded
        # by the page deadline.
        await release_claim(job_id, page_index, store)

        if final_attempt:
            # Only reachable from the LAYOUT stage: the VLM stage degrades on
            # its final attempt rather than propagating. A page with no layout
            # result has nothing to fall back to, so this is genuinely
            # unrecoverable - the one remaining path to FAILED.
            failed = await store.transition(
                job_id,
                page_index,
                PageState.FAILED,
                fields={"error": f"{type(exc).__name__}: {exc.endpoint} unavailable"},
            )
            # A failed page gets a page.final too. Silence here would be the
            # worst outcome for a subscriber: the job never reaches
            # done == total from its point of view, so it waits for an event
            # that cannot arrive and the connection hangs until its own
            # timeout. "0% unhandled" has to mean visible, not merely counted.
            await _emit(
                results,
                PAGE_FINAL,
                failed,
                job_id,
                page_index,
                complete=True,
                degraded=False,
                error=f"{type(exc).__name__}: {exc.endpoint} unavailable",
            )
            log.error(
                "page_failed_no_fallback",
                job_id=job_id,
                page_index=page_index,
                endpoint=exc.endpoint,
                cause=type(exc).__name__,
            )
            return PageResult(Outcome.FAILED, job_id, page_index, exc.endpoint)

        log.info(
            "page_saturated",
            job_id=job_id,
            page_index=page_index,
            endpoint=exc.endpoint,
            cause=type(exc).__name__,
            # Both, and clearly distinguished: retry_after is a projection,
            # waited is elapsed. Conflating them made logs claim a 5s wait for
            # a call that returned in 1.7ms.
            retry_after_s=round(exc.retry_after_s, 2),
            waited_s=round(getattr(exc, "waited_s", 0.0), 2),
        )
        return PageResult(
            Outcome.SATURATED,
            job_id,
            page_index,
            exc.endpoint,
            retry_after_s=exc.retry_after_s,
        )

    except Exception as exc:  # noqa: BLE001
        # In practice the LAYOUT stage only. The VLM stage catches its own
        # failures above and degrades, so what reaches here is a page with no
        # committed layout result - nothing to fall back TO, which is why this
        # is the one remaining path to FAILED rather than FALLBACK_DONE.
        #
        # Reached only after the client has classified the error and spent its
        # retry budget, or judged the failure permanent.
        log.warning(
            "page_failed",
            job_id=job_id,
            page_index=page_index,
            error=type(exc).__name__,
            detail=str(exc)[:200],
        )
        # Default predecessors, which after the table fix include VLM_RUNNING.
        # Narrowing this would strand a page that failed mid-VLM: no terminal
        # state, no done_count increment, and a job that never completes.
        failed = await store.transition(
            job_id,
            page_index,
            PageState.FAILED,
            fields={"error": f"{type(exc).__name__}: {str(exc)[:200]}"},
        )
        await _emit(
            results,
            PAGE_FINAL,
            failed,
            job_id,
            page_index,
            complete=True,
            degraded=False,
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
        return PageResult(Outcome.FAILED, job_id, page_index, type(exc).__name__)

    return PageResult(
        Outcome.RESUMED if resumed else Outcome.COMPLETED, job_id, page_index
    )
