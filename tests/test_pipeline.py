"""Per-page pipeline tests.

A fake client is used instead of the real mock server. That is deliberate: these
tests are about the pipeline's *control flow* - which stage runs, which is
skipped, what happens on failure - and a real 2-second VLM call would make the
suite slow without testing anything extra. The mock server's own behaviour is
covered in test_mock_primitives.py, and the two are wired together in the
integration run.

The fake records every call, so "did the layout stage re-run?" is a direct
assertion rather than an inference.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from app.queue.state import PageState, PageStateStore
from app.worker.client import idempotency_key
from app.worker.pipeline import FALLBACK_CONFIDENCE, Outcome, process_page

JOB = "pipe-job"


class FakeModelClient:
    """Records calls; optionally fails a chosen stage.

    `**_` on both stage methods absorbs kwargs the real client gains over time.
    Without it a double fails on the NEXT parameter added upstream - which has
    now happened three times in this build (the AIMD `rate`, and the Step 12
    `page_content`) - and it fails as a TypeError swallowed into Outcome.FAILED,
    which looks like a pipeline bug rather than a stale fixture.
    """

    def __init__(self, fail_on: set[str] | None = None, delay_s: float = 0.0) -> None:
        self.calls: list[tuple[str, int]] = []
        self.fail_on = fail_on or set()
        self.delay_s = delay_s

    def _record(self, stage: str, page_index: int) -> None:
        self.calls.append((stage, page_index))

    def count(self, stage: str) -> int:
        return sum(1 for s, _ in self.calls if s == stage)

    async def layout(
        self,
        job_id: str,
        page_index: int,
        *,
        page_ref: str | None = None,
        **_: object,
    ) -> dict[str, Any]:
        self._record("layout", page_index)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if "layout" in self.fail_on:
            raise RuntimeError("layout exploded")
        return {"model": "fast-layout", "boxes": [{"x": 1, "y": 2, "w": 3, "h": 4}]}

    async def vlm(
        self,
        job_id: str,
        page_index: int,
        *,
        page_ref: str | None = None,
        **_: object,
    ) -> dict[str, Any]:
        self._record("vlm", page_index)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if "vlm" in self.fail_on:
            raise RuntimeError("vlm exploded")
        return {"model": "heavy-vlm", "text": "hello", "confidence": 0.91}


# ---------------------------------------------------------------- happy path


async def test_happy_path_runs_both_stages_once_and_stores_both_results(
    store: PageStateStore,
) -> None:
    await store.init_job(JOB, total_pages=1)
    client = FakeModelClient()

    result = await process_page(JOB, 0, store=store, client=client)

    assert result.outcome is Outcome.COMPLETED
    assert client.calls == [("layout", 0), ("vlm", 0)]

    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.DONE.value
    assert json.loads(page["layout"])["boxes"][0]["x"] == 1
    assert json.loads(page["vlm"])["text"] == "hello"
    assert page["confidence"] == "0.91"
    assert page["degraded"] == "0"
    assert await store.progress(JOB) == (1, 1)


# ------------------------------------------------------------------- resume


async def test_resume_from_layout_done_skips_the_layout_call(
    store: PageStateStore,
) -> None:
    """The payoff of per-stage commits, asserted directly.

    A page recovered after a crash mid-VLM must NOT re-run layout. Committing
    only at the end of the page would make this impossible - the whole page
    would have to be redone.
    """
    await store.init_job(JOB, total_pages=1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING)
    await store.transition(
        JOB, 0, PageState.LAYOUT_DONE, fields={"layout": '{"boxes":["preexisting"]}'}
    )

    client = FakeModelClient()
    result = await process_page(JOB, 0, store=store, client=client)

    assert result.outcome is Outcome.RESUMED
    assert client.count("layout") == 0, "layout must not be repeated"
    assert client.count("vlm") == 1

    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.DONE.value
    # The original layout result is intact, not overwritten.
    assert json.loads(page["layout"])["boxes"] == ["preexisting"]


# ------------------------------------------------------------- redelivery


async def test_redelivered_completed_page_is_a_no_op(store: PageStateStore) -> None:
    """At-least-once delivery makes this routine, so it must cost nothing.

    The state check short-circuits before any HTTP call, which is the difference
    between a redelivered page costing 0ms and costing a 3s inference.
    """
    await store.init_job(JOB, total_pages=1)
    await process_page(JOB, 0, store=store, client=FakeModelClient())

    client = FakeModelClient()
    result = await process_page(JOB, 0, store=store, client=client)

    assert result.outcome is Outcome.ALREADY_TERMINAL
    assert client.calls == []
    assert await store.progress(JOB) == (1, 1), "done_count must not double"


async def test_page_claimed_by_another_worker_is_left_alone(
    store: PageStateStore,
) -> None:
    """A *_RUNNING state means someone owns it. Interfering would duplicate work.

    Only the reaper may move such a page, and only after an idle threshold
    proves the owner is gone.
    """
    await store.init_job(JOB, total_pages=1)
    await store.transition(JOB, 0, PageState.LAYOUT_RUNNING, fields={"worker": "other"})

    client = FakeModelClient()
    result = await process_page(JOB, 0, store=store, client=client)

    assert result.outcome is Outcome.OWNED_BY_OTHER
    assert client.calls == []


async def test_missing_page_state_fails_loudly(store: PageStateStore) -> None:
    """A task for a job that was never initialised is a bug, not a lost race."""
    client = FakeModelClient()

    result = await process_page("never-created", 0, store=store, client=client)

    assert result.outcome is Outcome.FAILED
    assert client.calls == []


# -------------------------------------------------------------- failures


async def test_layout_failure_marks_the_page_failed_with_the_reason(
    store: PageStateStore,
) -> None:
    await store.init_job(JOB, total_pages=1)

    result = await process_page(
        JOB, 0, store=store, client=FakeModelClient(fail_on={"layout"})
    )

    assert result.outcome is Outcome.FAILED
    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.FAILED.value
    assert "layout exploded" in page["error"]
    # FAILED is terminal, so the job can still reach completion rather than
    # hanging forever on an unfinished page.
    assert await store.progress(JOB) == (1, 1)


async def test_vlm_failure_degrades_rather_than_failing(store: PageStateStore) -> None:
    """The zero-drop mechanism: a VLM failure costs fidelity, not the page.

    The layout result is already committed by the time the VLM runs, and layout
    output is usable on its own - boxes, block types, reading order. So the page
    completes as FALLBACK_DONE with a low confidence flag instead of being lost.

    Also still the regression guard for the original bug this test was written
    for: a VLM error once left the page stranded in VLM_RUNNING with no terminal
    state and no done_count increment, so the job could never report complete.
    The final assertion is what pins that.
    """
    await store.init_job(JOB, total_pages=1)

    result = await process_page(
        JOB, 0, store=store, client=FakeModelClient(fail_on={"vlm"})
    )

    assert result.outcome is Outcome.DEGRADED
    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.FALLBACK_DONE.value
    assert page["degraded"] == "1"
    assert float(page["confidence"]) == FALLBACK_CONFIDENCE
    assert page["fallback_reason"] == "RuntimeError"
    # The layout result is what makes the degraded page worth serving.
    assert json.loads(page["layout"])["boxes"][0]["x"] == 1
    assert await store.progress(JOB) == (1, 1), "page must not be stranded"


async def test_layout_failure_still_fails_because_there_is_no_fallback(
    store: PageStateStore,
) -> None:
    """The asymmetry, asserted.

    Layout cannot degrade because layout IS the fallback - a page with no layout
    result has nothing to fall back to. This is the only remaining path to
    FAILED, and keeping it distinct from FALLBACK_DONE is what makes the
    zero-drop claim honest rather than a relabelling exercise.
    """
    await store.init_job(JOB, total_pages=1)

    result = await process_page(
        JOB, 0, store=store, client=FakeModelClient(fail_on={"layout"})
    )

    assert result.outcome is Outcome.FAILED
    page = await store.get_page(JOB, 0)
    assert page["state"] == PageState.FAILED.value
    assert "layout" not in page, "there is no partial result to serve"


async def test_layout_result_survives_a_later_vlm_failure(
    store: PageStateStore,
) -> None:
    """Work already paid for must not be discarded by a downstream failure.

    This is the precondition for the degraded fallback: a page whose VLM failed
    still holds the layout result it degrades to.
    """
    await store.init_job(JOB, total_pages=1)

    await process_page(JOB, 0, store=store, client=FakeModelClient(fail_on={"vlm"}))

    page = await store.get_page(JOB, 0)
    assert json.loads(page["layout"])["boxes"][0]["x"] == 1


# ------------------------------------------------------------ concurrency


async def test_concurrent_processing_of_one_page_does_exactly_one_unit_of_work(
    store: PageStateStore,
) -> None:
    """Ten workers handed the same page must produce one set of model calls.

    A delay is used so every task reads PENDING before any of them claims it -
    the realistic case, since the real gap between reading state and claiming it
    spans a network round trip. The atomic CAS is what collapses ten attempts
    into one.

    The assertion is on WORK DONE, not on which task gets which label, because
    two interleavings are both correct and the test was flaky for asserting
    only the first:

      * the usual one - a task claims PENDING, runs both stages, returns
        COMPLETED, and the other nine are turned away at the claim.
      * the rarer one - a task is first scheduled AFTER the winner has already
        committed LAYOUT_DONE, so it reads LAYOUT_DONE rather than PENDING,
        claims the VLM stage, and finishes the page as RESUMED. The original
        winner then loses its own VLM claim and returns OWNED_BY_OTHER.

    In both cases exactly one layout call and one VLM call happen and the page
    lands DONE once, which is the property that matters. Asserting
    `COMPLETED == 1` asserted a scheduling order instead, and failed under load
    when the suite ran the whole file against a busy Redis.
    """
    await store.init_job(JOB, total_pages=1)
    client = FakeModelClient(delay_s=0.01)

    results = await asyncio.gather(
        *(process_page(JOB, 0, store=store, client=client) for _ in range(10))
    )

    outcomes = [r.outcome for r in results]
    finishers = [o for o in outcomes if o in (Outcome.COMPLETED, Outcome.RESUMED)]
    assert len(finishers) == 1, f"more than one task did the work: {outcomes}"
    assert all(
        o in (Outcome.OWNED_BY_OTHER, Outcome.ALREADY_TERMINAL) for o in outcomes
        if o not in (Outcome.COMPLETED, Outcome.RESUMED)
    ), outcomes

    # The invariants that actually encode "exactly one unit of work".
    assert client.count("layout") == 1
    assert client.count("vlm") == 1
    assert await store.progress(JOB) == (1, 1)
    assert (await store.get_page(JOB, 0))["state"] == PageState.DONE.value


async def test_many_pages_all_reach_done(store: PageStateStore) -> None:
    pages = 25
    await store.init_job(JOB, total_pages=pages)
    client = FakeModelClient()

    results = await asyncio.gather(
        *(process_page(JOB, i, store=store, client=client) for i in range(pages))
    )

    assert all(r.outcome is Outcome.COMPLETED for r in results)
    assert client.count("layout") == pages
    assert client.count("vlm") == pages
    assert await store.progress(JOB) == (pages, pages)


# ----------------------------------------------------------- idempotency key


def test_idempotency_key_is_stable_and_stage_scoped() -> None:
    """Stable across processes, distinct per stage.

    If layout and VLM shared a key, the VLM call would be served the cached
    layout response. If the key were unstable, a restarted worker would compute
    a different key for the same work and pay twice.
    """
    a = idempotency_key("job", 3, "layout")
    assert a == idempotency_key("job", 3, "layout")
    assert a != idempotency_key("job", 3, "vlm")
    assert a != idempotency_key("job", 4, "layout")
    assert a != idempotency_key("other", 3, "layout")


def test_idempotency_key_digest_is_hashlib_not_builtin_hash() -> None:
    """Pins the value, so swapping in builtin hash() fails the suite.

    hash() on str is salted per process, so it would silently break replay
    safety across a restart - the exact scenario the key exists for.
    """
    assert idempotency_key("job", 3, "layout").startswith("fc9f9c3a7e4f")
