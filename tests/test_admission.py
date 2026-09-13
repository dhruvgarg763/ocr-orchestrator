"""Admission control tests, against real Redis.

The depth signal is XLEN on the real stream, so these tests enqueue real tasks
rather than stubbing a number - the coupling between "what ack() deletes" and
"what admission counts" is exactly what would break silently.

The two tests that matter most are
`test_hysteresis_prevents_flapping_at_the_watermark` (a single threshold
oscillates on every page completion) and
`test_a_shed_job_is_counted_not_silently_dropped` (the "0% unhandled" claim is
only meaningful if refusals are recorded).
"""

from __future__ import annotations

import asyncio

import pytest
from redis.asyncio import Redis

from app.api.admission import AdmissionController
from app.queue.streams import PageQueue
from app.ratelimit.adaptive import AdaptiveRate, Congestion


def make_controller(redis: Redis, **kwargs) -> AdmissionController:
    defaults = {
        "high_watermark": 100,
        "low_watermark_fraction": 0.8,
        "drain_rate": 10.0,
        "retry_after_cap_s": 300,
        "retry_after_jitter": 0.0,  # deterministic; the jitter test opts in
        "ttl_s": 60,
    }
    return AdmissionController(redis, **{**defaults, **kwargs})


async def fill(queue: PageQueue, pages: int) -> None:
    """Put real work in the stream, since XLEN is the signal under test."""
    await queue.enqueue_pages("filler", pages)


# ------------------------------------------------------------------- basics


async def test_admits_when_the_queue_is_empty(redis: Redis, queue: PageQueue) -> None:
    """An idle system must never refuse. Admission control is a capacity bound,
    not a rate limit."""
    verdict = await make_controller(redis).evaluate(50)

    assert verdict.admitted
    assert verdict.depth == 0
    assert not verdict.shedding


async def test_refuses_once_the_watermark_would_be_exceeded(
    redis: Redis, queue: PageQueue
) -> None:
    controller = make_controller(redis, high_watermark=100)
    await fill(queue, 95)

    verdict = await controller.evaluate(50)

    assert not verdict.admitted
    assert verdict.depth == 95
    assert verdict.retry_after_s >= 1


async def test_the_job_counts_toward_its_own_admission(
    redis: Redis, queue: PageQueue
) -> None:
    """A watermark that only checks the CURRENT depth can be overshot by an
    entire job, and a bound that does not bind is not a bound.

    The cost is that a large job can be refused where a small one fits. With
    max_pages at 100 against a watermark in the thousands that unfairness is
    small and bounded, so the exactness is worth more.
    """
    controller = make_controller(redis, high_watermark=100)
    await fill(queue, 60)

    assert (await controller.evaluate(40)).admitted, "60 + 40 == 100 fits exactly"
    await controller.reset()
    assert not (await controller.evaluate(41)).admitted, "60 + 41 exceeds 100"


async def test_depth_tracks_settled_work_not_total_work(
    redis: Redis, queue: PageQueue
) -> None:
    """XLEN is the signal precisely because ack() deletes the entry.

    If ack only acknowledged and left the entry in the stream, depth would grow
    monotonically forever and admission would seize permanently after the first
    N pages - regardless of how much work had actually completed.
    """
    controller = make_controller(redis, high_watermark=100)
    await fill(queue, 90)
    assert not (await controller.evaluate(20)).admitted

    tasks = await queue.read("w-1", count=90, block_ms=100)
    for task in tasks:
        await queue.ack(task.entry_id, stream=task.stream)

    verdict = await controller.evaluate(20)
    assert verdict.admitted, "settled work must stop counting"
    assert verdict.depth == 0


async def test_in_flight_work_still_counts(redis: Redis, queue: PageQueue) -> None:
    """Delivered-but-unacknowledged pages are still resident.

    They occupy the PEL and their state hashes exist, so excluding them would
    under-count exactly the work most likely to be re-delivered.
    """
    controller = make_controller(redis, high_watermark=100)
    await fill(queue, 95)
    await queue.read("w-1", count=95, block_ms=100)  # delivered, NOT acked

    verdict = await controller.evaluate(20)

    assert not verdict.admitted
    assert verdict.depth == 95


# --------------------------------------------------------------- hysteresis


async def test_hysteresis_prevents_flapping_at_the_watermark(
    redis: Redis, queue: PageQueue
) -> None:
    """THE test for this step.

    With a single threshold the system oscillates on every page completion: at
    the mark it refuses, one page drains so it admits, the next job pushes it
    back over. Clients would see an unpredictable mix of 202s and 503s.

    Two marks make it a Schmitt trigger. Having tripped at 100, the controller
    must keep refusing at 99, 95, 90 - all the way down to the recovery mark at
    80 - so the state is stable rather than knife-edge.
    """
    controller = make_controller(redis, high_watermark=100, low_watermark_fraction=0.8)
    await fill(queue, 100)
    assert not (await controller.evaluate(1)).admitted, "should trip"

    tasks = await queue.read("w-1", count=100, block_ms=200)

    # Drain to 85: above the recovery mark of 80, so still shedding.
    for task in tasks[:15]:
        await queue.ack(task.entry_id, stream=task.stream)
    verdict = await controller.evaluate(1)
    assert not verdict.admitted, f"recovered too early at depth {verdict.depth}"
    assert verdict.shedding

    # Drain past 80 and it must recover.
    for task in tasks[15:25]:
        await queue.ack(task.entry_id, stream=task.stream)
    verdict = await controller.evaluate(1)
    assert verdict.admitted, f"failed to recover at depth {verdict.depth}"
    assert not verdict.shedding


async def test_trips_and_recoveries_are_counted(redis: Redis, queue: PageQueue) -> None:
    """A system that flaps is visible in these counters even when the
    instantaneous state looks fine, which is the only way to catch a badly
    tuned watermark in production."""
    controller = make_controller(redis, high_watermark=10, low_watermark_fraction=0.8)
    await fill(queue, 10)
    await controller.evaluate(1)

    tasks = await queue.read("w-1", count=10, block_ms=200)
    for task in tasks[:5]:
        await queue.ack(task.entry_id, stream=task.stream)
    await controller.evaluate(1)

    stats = await controller.stats()
    assert stats["trips"] == 1
    assert stats["recoveries"] == 1


async def test_low_watermark_fraction_of_one_is_rejected(redis: Redis) -> None:
    """At 1 the two marks coincide, the hysteresis disappears, and the
    controller is back to flapping - so the config must not permit it."""
    with pytest.raises(ValueError):
        make_controller(redis, low_watermark_fraction=1.0)


# -------------------------------------------------------------- accounting


async def test_a_shed_job_is_counted_not_silently_dropped(
    redis: Redis, queue: PageQueue
) -> None:
    """The evidence behind "0% unhandled".

    A refusal that appears nowhere is indistinguishable from a drop. Every shed
    job has to land in both the counters and the logs, so the benchmark can
    report an honest admitted-versus-shed tally rather than a success rate over
    only the requests that happened to be accepted.
    """
    controller = make_controller(redis, high_watermark=100)
    await fill(queue, 100)

    for _ in range(3):
        await controller.evaluate(20)

    stats = await controller.stats()
    assert stats["shed_jobs"] == 3
    assert stats["shed_pages"] == 60
    assert stats["shedding"] is True


async def test_admitted_work_is_counted_too(redis: Redis, queue: PageQueue) -> None:
    """Both halves, or the ratio is unknowable."""
    controller = make_controller(redis, high_watermark=1_000)

    await controller.evaluate(10)
    await controller.evaluate(30)

    stats = await controller.stats()
    assert stats["admitted_jobs"] == 2
    assert stats["admitted_pages"] == 40
    assert stats["shed_jobs"] == 0


# ------------------------------------------------------------ retry-after


async def test_retry_after_is_derived_from_the_drain_rate(
    redis: Redis, queue: PageQueue
) -> None:
    """Not a constant. The client is told how long the backlog actually needs.

    Depth 100, recovery mark 80, so 20 pages must clear; at 10 pages/sec that
    is 2 seconds.
    """
    controller = make_controller(
        redis, high_watermark=100, low_watermark_fraction=0.8, drain_rate=10.0
    )
    await fill(queue, 100)

    verdict = await controller.evaluate(50)

    assert verdict.retry_after_s == 2


async def test_retry_after_uses_the_discovered_rate_not_the_advertised_one(
    redis: Redis, queue: PageQueue
) -> None:
    """Composes with Step 10, and this is the point of doing so.

    If the VLM has degraded to 2.5 rps, the same backlog takes 4x as long to
    clear. Quoting a figure derived from the advertised 10 rps guarantees the
    client returns too early and is refused again - so the controller reads the
    rate AIMD actually discovered.
    """
    degraded = AdaptiveRate(
        redis, "vlm", max_rate=10.0, min_rate=1.0, refractory_ms=0, ttl_s=60
    )
    for _ in range(4):  # 10 -> 2.401 rps
        await degraded.on_congestion(Congestion.REJECTED)

    controller = make_controller(
        redis,
        high_watermark=100,
        low_watermark_fraction=0.8,
        drain_rate=10.0,
        controller=degraded,
    )
    await fill(queue, 100)

    verdict = await controller.evaluate(50)

    # 20 pages at ~2.4 rps is ~8.3s, versus 2s at the advertised rate.
    assert verdict.retry_after_s >= 8, verdict.retry_after_s


async def test_retry_after_is_capped(redis: Redis, queue: PageQueue) -> None:
    """"Come back in two hours" is not actionable - a client that waits that
    long has been dropped, not deferred."""
    controller = make_controller(
        redis, high_watermark=100, drain_rate=0.1, retry_after_cap_s=30
    )
    await fill(queue, 100)

    assert (await controller.evaluate(50)).retry_after_s == 30


async def test_retry_after_is_never_zero(redis: Redis, queue: PageQueue) -> None:
    """RFC 9110 Retry-After is integer seconds, and 0 invites an immediate
    retry - which is indistinguishable from no backoff at all."""
    controller = make_controller(
        redis, high_watermark=100, low_watermark_fraction=0.99, drain_rate=1_000.0
    )
    await fill(queue, 100)

    assert (await controller.evaluate(50)).retry_after_s >= 1


async def test_retry_after_is_jittered(redis: Redis, queue: PageQueue) -> None:
    """Step 8's lesson, applied at the edge.

    Fifty clients handed an identical `Retry-After: 60` all return in the same
    instant and recreate the overload that caused the refusal. Jitter converts
    that convoy back into a queue.
    """
    controller = make_controller(
        redis,
        high_watermark=100,
        low_watermark_fraction=0.5,
        drain_rate=1.0,
        retry_after_jitter=0.5,
    )
    await fill(queue, 100)

    values = {(await controller.evaluate(50)).retry_after_s for _ in range(30)}

    assert len(values) > 1, f"all refusals quoted the same delay: {values}"
    assert min(values) >= 50, values


# ------------------------------------------------------------ distributed


async def test_shed_state_is_shared_between_api_replicas(
    redis: Redis, queue: PageQueue
) -> None:
    """Hysteresis is stateful, so the flag has to be shared.

    Per-process flags would let one replica sit in the shedding state while
    another admitted freely, and a client's experience would depend on which
    instance the load balancer picked.
    """
    first = make_controller(redis, high_watermark=100)
    second = make_controller(redis, high_watermark=100)
    await fill(queue, 100)

    await first.evaluate(1)  # trips
    tasks = await queue.read("w-1", count=100, block_ms=200)
    for task in tasks[:15]:  # down to 85, still above the 80 recovery mark
        await queue.ack(task.entry_id, stream=task.stream)

    verdict = await second.evaluate(1)
    assert not verdict.admitted, "second replica did not see the shed state"
    assert verdict.shedding


async def test_unserialised_admission_overshoots_the_watermark(
    redis: Redis, queue: PageQueue
) -> None:
    """Why the API holds a lock across check and enqueue.

    The controller is a pure function of XLEN, and XLEN does not move until the
    caller enqueues - a separate round trip. So concurrent callers that check
    before any of them enqueues all admit against the same depth.

    Measured against the live API at the assignment's own benchmark
    concurrency (50 concurrent POSTs of 20 pages, watermark 400) before the
    fix: 130-150% overshoot, and one trial admitted 1,000 pages against a 400
    limit while shedding NOTHING. A bound that can be exceeded 2.5x is not a
    bound.
    """
    controller = make_controller(redis, high_watermark=100)
    await fill(queue, 50)

    # No serialisation: every check runs before any enqueue.
    verdicts = await asyncio.gather(*(controller.evaluate(10) for _ in range(20)))
    admitted = sum(1 for v in verdicts if v.admitted)

    assert admitted == 20, "all 20 saw depth=50 and admitted against it"
    assert admitted * 10 + 50 > 100, "which overshoots the watermark"


async def test_serialising_check_and_enqueue_makes_the_bound_exact(
    redis: Redis, queue: PageQueue
) -> None:
    """The fix, as the API applies it.

    Holding a lock across check-then-enqueue makes every check observe all
    prior enqueues, so admitted work lands exactly on the watermark rather than
    2.5x past it. Verified against the live API: 0% overshoot across three
    trials, admitting exactly 400 pages against a 400 watermark.

    It is also FASTER under overload - 351-487ms for 50 POSTs versus 934ms -
    because a refusal skips the state-init and enqueue that an admission
    performs. Overload protection should make overload cheap.

    With N API replicas the overshoot returns, but bounded by (N-1) x
    max_pages: a provable constant, unlike the previous bound of "however many
    requests a client chooses to send at once". Making it exact across replicas
    would mean folding the depth check, the state init and the enqueue into one
    Lua script - correct, but it couples three subsystems, and is only worth it
    if the API needs to scale horizontally.
    """
    controller = make_controller(redis, high_watermark=100)
    lock = asyncio.Lock()

    async def admit_and_enqueue(index: int) -> bool:
        async with lock:
            verdict = await controller.evaluate(10)
            if verdict.admitted:
                await queue.enqueue_pages(f"job-{index}", 10)
            return verdict.admitted

    results = await asyncio.gather(*(admit_and_enqueue(i) for i in range(20)))

    admitted_pages = sum(10 for ok in results if ok)
    assert admitted_pages == 100, f"admitted {admitted_pages}, watermark is 100"
    assert (await queue.depth())["stream_length"] == 100, "exactly at the bound"


# ---------------------------------------------------------------- hygiene


async def test_state_key_expires(redis: Redis, queue: PageQueue) -> None:
    controller = make_controller(redis, ttl_s=60)
    await controller.evaluate(1)

    assert 0 < await redis.ttl(controller.key) <= 60


@pytest.mark.parametrize(
    "kwargs",
    [
        {"high_watermark": 0},
        {"low_watermark_fraction": 0},
        {"low_watermark_fraction": 1.0},
        {"drain_rate": 0},
    ],
)
async def test_invalid_config_is_rejected(redis: Redis, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        make_controller(redis, **kwargs)
