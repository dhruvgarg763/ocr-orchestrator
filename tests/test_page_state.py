"""Page state machine tests.

Two groups:

  * Pure-logic tests on the transition table - fast, no Redis, and they catch
    the class of bug where the table and its derived inverse disagree.
  * Behavioural tests against a real Redis, including the concurrency test that
    is the entire reason transitions are a Lua script.
"""

from __future__ import annotations

import asyncio
from collections import deque

import pytest
from redis.asyncio import Redis

from app.queue.state import (
    PREDECESSORS,
    RECLAIM_TARGET,
    RUNNING_STATES,
    TERMINAL_STATES,
    TRANSITIONS,
    PageState,
    PageStateStore,
    can_transition,
    page_key,
)

JOB = "test-job"


# ======================================================================
# Transition table invariants (no Redis)
# ======================================================================


def test_terminal_states_are_exactly_the_sinks() -> None:
    """TERMINAL_STATES is derived, so this pins the intent.

    If someone later gives DONE an outgoing edge, the derived set changes
    silently and terminal pages would stop incrementing done_count. This test
    turns that into a failure.
    """
    assert TERMINAL_STATES == {
        PageState.DONE,
        PageState.FALLBACK_DONE,
        PageState.FAILED,
    }


def test_predecessors_is_a_faithful_inverse_of_transitions() -> None:
    """The inverted table must agree with the forward table in both directions.

    PREDECESSORS is what `transition()` uses to decide legality by default, so a
    mismatch here would let an illegal transition through or block a legal one.
    """
    for src, targets in TRANSITIONS.items():
        for dst in targets:
            assert src in PREDECESSORS[dst], f"{src} -> {dst} missing from inverse"

    for dst, sources in PREDECESSORS.items():
        for src in sources:
            assert dst in TRANSITIONS[src], f"{src} -> {dst} in inverse only"


def test_every_state_is_reachable_from_pending() -> None:
    """An unreachable state is dead code that silently never runs."""
    seen = {PageState.PENDING}
    queue = deque([PageState.PENDING])
    while queue:
        for nxt in TRANSITIONS[queue.popleft()]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    assert seen == set(PageState), f"unreachable: {set(PageState) - seen}"


def test_no_state_transitions_to_itself() -> None:
    """A self-edge would let a terminal page re-increment done_count."""
    for src, targets in TRANSITIONS.items():
        assert src not in targets


def test_every_reclaim_target_is_a_legal_transition() -> None:
    """The crash reaper uses RECLAIM_TARGET; it must not produce illegal moves."""
    assert set(RECLAIM_TARGET) == RUNNING_STATES
    for running, target in RECLAIM_TARGET.items():
        assert can_transition(running, target)


def test_reclaim_goes_to_a_checkpoint_not_back_to_the_start() -> None:
    """A VLM crash must not discard the already-committed layout result.

    This is the whole payoff of committing each stage separately: re-running a
    3s VLM call is acceptable, re-running the layout call is waste.
    """
    assert RECLAIM_TARGET[PageState.VLM_RUNNING] == PageState.LAYOUT_DONE
    assert RECLAIM_TARGET[PageState.LAYOUT_RUNNING] == PageState.PENDING


# ======================================================================
# Behaviour against real Redis
# ======================================================================


async def test_init_job_creates_every_page_in_pending(store: PageStateStore) -> None:
    await store.init_job(JOB, total_pages=5, pdf_path="/data/x.pdf")

    job = await store.get_job(JOB)
    assert job["total_pages"] == "5"
    assert job["done_count"] == "0"
    assert job["pdf_path"] == "/data/x.pdf"

    assert await store.page_states(JOB, 5) == [PageState.PENDING.value] * 5


async def test_init_job_sets_ttl_on_all_keys(store: PageStateStore, redis: Redis) -> None:
    """Without a TTL, Redis memory grows with every job ever submitted."""
    await store.init_job(JOB, total_pages=3)

    assert 0 < await redis.ttl(f"job:{JOB}") <= 60
    for index in range(3):
        assert 0 < await redis.ttl(page_key(JOB, index)) <= 60


async def test_legal_transition_succeeds(store: PageStateStore) -> None:
    await store.init_job(JOB, total_pages=1)

    result = await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)

    assert result.ok
    assert result.done_count == -1, "non-terminal transitions must not touch the counter"
    assert (await store.get_page(JOB, 0))["state"] == PageState.LAYOUT_RUNNING.value


async def test_illegal_transition_is_rejected_and_leaves_state_intact(
    store: PageStateStore,
) -> None:
    """PENDING -> DONE must be impossible: it would skip both model calls."""
    await store.init_job(JOB, total_pages=1)

    result = await store.transition(JOB, 0, PageState.DONE)

    assert not result.ok
    assert result.observed == PageState.PENDING.value
    assert (await store.get_page(JOB, 0))["state"] == PageState.PENDING.value
    assert (await store.get_job(JOB))["done_count"] == "0"


async def test_transition_on_missing_page_reports_missing(store: PageStateStore) -> None:
    """Distinguishable from a rejection: MISSING means a bug, not a lost race."""
    result = await store.transition("no-such-job", 0, PageState.LAYOUT_RUNNING)

    assert not result.ok
    assert result.missing


async def test_terminal_transition_increments_done_count(store: PageStateStore) -> None:
    await store.init_job(JOB, total_pages=3)
    for index in range(3):
        await store.transition(JOB, index, PageState.LAYOUT_RUNNING)
        await store.transition(JOB, index, PageState.LAYOUT_DONE)
        await store.transition(JOB, index, PageState.VLM_RUNNING)

    assert (await store.transition(JOB, 0, PageState.DONE)).done_count == 1
    assert (await store.transition(JOB, 1, PageState.FALLBACK_DONE)).done_count == 2
    assert await store.progress(JOB) == (2, 3)


async def test_replaying_a_terminal_transition_does_not_double_count(
    store: PageStateStore,
) -> None:
    """At-least-once delivery means this WILL happen; it must be harmless.

    The second attempt is rejected because DONE is not a legal predecessor of
    DONE, so done_count stays correct. `already_applied` lets the caller treat
    it as benign instead of logging a false alarm.
    """
    await store.init_job(JOB, total_pages=1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await store.transition(JOB, 0, PageState.LAYOUT_DONE)
    await store.transition(JOB, 0, PageState.VLM_RUNNING)

    first = await store.transition(JOB, 0, PageState.DONE)
    second = await store.transition(JOB, 0, PageState.DONE)

    assert first.ok and first.done_count == 1
    assert not second.ok
    assert second.already_applied
    assert await store.progress(JOB) == (1, 1)


async def test_extra_fields_are_written_atomically_with_the_state(
    store: PageStateStore,
) -> None:
    """State and payload land in the same script, so they cannot disagree.

    If these were two commands there would be a window where state says
    LAYOUT_DONE but the result field is empty - and a reader in that window
    would stream an empty page result.
    """
    await store.init_job(JOB, total_pages=1)

    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING, fields={"worker": "w-1"})
    await store.transition(
        JOB, 0, PageState.LAYOUT_DONE, fields={"layout": '{"boxes":[]}', "confidence": 0.9}
    )

    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.LAYOUT_DONE.value
    assert page["layout"] == '{"boxes":[]}'
    assert page["confidence"] == "0.9"
    assert page["worker"] == "w-1"


async def test_narrowing_allowed_from_blocks_an_otherwise_legal_transition(
    store: PageStateStore,
) -> None:
    """The reaper needs this: reclaim VLM_RUNNING pages ONLY.

    LAYOUT_DONE is reachable from both LAYOUT_RUNNING and VLM_RUNNING. A reaper
    that used the default predecessor set would also yank pages that are
    legitimately mid-layout.
    """
    await store.init_job(JOB, total_pages=1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)

    result = await store.transition(
        JOB, 0, PageState.LAYOUT_DONE, allowed_from=[PageState.VLM_RUNNING]
    )

    assert not result.ok
    assert result.observed == PageState.LAYOUT_RUNNING.value


async def test_unreachable_target_raises_rather_than_silently_never_matching(
    store: PageStateStore,
) -> None:
    await store.init_job(JOB, total_pages=1)

    with pytest.raises(ValueError, match="unreachable"):
        await store.transition(JOB, 0, PageState.PENDING, allowed_from=[])


# ----------------------------------------------------------------------
# The concurrency tests: the reason transitions are a Lua script
# ----------------------------------------------------------------------


async def test_concurrent_claims_produce_exactly_one_winner(
    store: PageStateStore,
) -> None:
    """50 workers race to claim one page. Exactly one may win.

    This is the core safety property. A second winner means the page is
    processed twice: two VLM calls (double cost, ~3s each) and duplicate
    results pushed into the client's stream.
    """
    await store.init_job(JOB, total_pages=1)

    results = await asyncio.gather(
        *(store.transition(JOB, 0, PageState.LAYOUT_RUNNING) for _ in range(50))
    )

    assert sum(1 for r in results if r.ok) == 1
    losers = [r for r in results if not r.ok]
    assert len(losers) == 49
    # Losers observe the winner's state, which tells them why they lost.
    assert all(r.observed == PageState.LAYOUT_RUNNING.value for r in losers)


async def test_naive_check_then_act_loses_the_race(redis: Redis) -> None:
    """Demonstrates the bug the Lua script exists to prevent.

    This is the implementation you get by default if you ask for "claim a page
    in Redis": read, check, write. Concurrent callers all read PENDING before
    any of them writes, so all of them believe they own the page.

    The `await asyncio.sleep(0)` between READ and WRITE is not a trick to force
    a failure - it is the realistic case, and leaving it out produces a
    dangerously misleading result. Measured directly:

        no gap between read and write : 1 winner  (looks correct!)
        gap of a single event-loop tick: 20 winners
        gap of 10ms                    : 3 winners

    With no gap the naive code passes a 50-way concurrency test *by accident*,
    because BlockingConnectionPool serialises tasks while they contend for
    connections - task 1 finishes its whole read-write cycle before task 2 even
    gets a connection. The real worker awaits a 50ms-3000ms HTTP model call in
    that gap, making the window millions of times wider.

    That is the worst kind of race: it passes locally, passes in CI, and
    corrupts data in production once real latency widens the window.
    """
    key = page_key("naive", 0)
    await redis.hset(key, mapping={"state": PageState.PENDING.value})

    winners = 0

    async def naive_claim() -> None:
        nonlocal winners
        current = await redis.hget(key, "state")  # READ
        await asyncio.sleep(0)  # stands in for the model call
        if current == PageState.PENDING.value:  # CHECK
            await redis.hset(key, "state", PageState.LAYOUT_RUNNING.value)  # WRITE
            winners += 1

    await asyncio.gather(*(naive_claim() for _ in range(50)))

    assert winners > 1, "expected the naive version to over-claim"


async def test_atomic_claim_survives_the_same_gap_that_breaks_the_naive_one(
    store: PageStateStore,
) -> None:
    """The control for the test above: same widened window, correct result.

    Proves the fix is the atomicity of the script and not an artifact of
    timing - because the compare and the set happen inside Redis, there is no
    client-side window to widen at all.
    """
    await store.init_job(JOB, total_pages=1)

    async def atomic_claim() -> bool:
        await asyncio.sleep(0)  # same gap as the naive version
        return (await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)).ok

    results = await asyncio.gather(*(atomic_claim() for _ in range(50)))

    assert sum(results) == 1


async def test_reclaim_after_crash_returns_page_to_its_checkpoint(
    store: PageStateStore,
) -> None:
    """Simulates the Module D scenario at the state-machine level.

    A worker claimed the page, finished layout, started the VLM, then died. The
    reaper must return it to LAYOUT_DONE so a replacement resumes at the VLM
    stage - not at the beginning.
    """
    await store.init_job(JOB, total_pages=1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING, fields={"worker": "dead-1"})
    await store.transition(JOB, 0, PageState.LAYOUT_DONE, fields={"layout": '{"boxes":[1]}'})
    await store.transition(JOB, 0, PageState.VLM_RUNNING, fields={"worker": "dead-1"})

    # --- worker is SIGKILLed here; the reaper acts ---
    reclaimed = await store.transition(
        JOB,
        0,
        RECLAIM_TARGET[PageState.VLM_RUNNING],
        allowed_from=[PageState.VLM_RUNNING],
    )

    assert reclaimed.ok
    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.LAYOUT_DONE.value
    assert page["layout"] == '{"boxes":[1]}', "layout result must survive the crash"

    # A replacement worker can now claim the VLM stage and finish.
    assert (await store.transition(JOB, 0, PageState.VLM_RUNNING)).ok
    assert (await store.transition(JOB, 0, PageState.DONE)).ok
    assert await store.progress(JOB) == (1, 1)


async def test_concurrent_reclaim_produces_exactly_one_winner(
    store: PageStateStore,
) -> None:
    """Several reaper instances may fire at once; only one may reclaim."""
    await store.init_job(JOB, total_pages=1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await store.transition(JOB, 0, PageState.LAYOUT_DONE)
    await store.transition(JOB, 0, PageState.VLM_RUNNING)

    results = await asyncio.gather(
        *(
            store.transition(
                JOB, 0, PageState.LAYOUT_DONE, allowed_from=[PageState.VLM_RUNNING]
            )
            for _ in range(20)
        )
    )

    assert sum(1 for r in results if r.ok) == 1


async def test_bump_attempts_is_atomic(store: PageStateStore) -> None:
    """HINCRBY, not read-add-write: concurrent retries must not lose a count.

    An undercounted attempts field means the retry budget is never exhausted and
    a permanently failing page retries forever.
    """
    await store.init_job(JOB, total_pages=1)

    await asyncio.gather(*(store.bump_attempts(JOB, 0) for _ in range(100)))

    assert (await store.get_page(JOB, 0))["attempts"] == "100"


async def test_full_happy_path_for_many_pages(store: PageStateStore) -> None:
    """End-to-end shape: every page reaches a terminal state exactly once."""
    pages = 20
    await store.init_job(JOB, total_pages=pages)

    async def run_page(index: int) -> None:
        assert (await store.transition(JOB, index, PageState.LAYOUT_RUNNING)).ok
        assert (await store.transition(JOB, index, PageState.LAYOUT_DONE)).ok
        assert (await store.transition(JOB, index, PageState.VLM_RUNNING)).ok
        # Odd pages degrade, as if the VLM had exhausted its retries.
        target = PageState.DONE if index % 2 == 0 else PageState.FALLBACK_DONE
        assert (await store.transition(JOB, index, target)).ok

    await asyncio.gather(*(run_page(i) for i in range(pages)))

    done, total = await store.progress(JOB)
    assert (done, total) == (pages, pages)

    states = await store.page_states(JOB, pages)
    assert all(PageState(s) in TERMINAL_STATES for s in states)
