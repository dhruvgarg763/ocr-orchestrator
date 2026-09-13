"""AIMD controller tests, against real Redis.

The control law is evaluated in Lua so that N replicas share one verdict, which
is the part a fake would have to reimplement. The two tests that matter most
are `test_refractory_period_prevents_one_event_collapsing_the_rate` (without it
a single congestion event floors the rate) and
`test_latency_breach_decreases_the_rate_without_any_rejection` (the signal
neither the token bucket nor the breaker can see).
"""

from __future__ import annotations

import asyncio

import pytest
from redis.asyncio import Redis

from app.ratelimit.adaptive import AdaptiveRate, Congestion

ENDPOINT = "vlm"


def make_controller(redis: Redis, **kwargs) -> AdaptiveRate:
    defaults = {
        "max_rate": 10.0,
        "min_rate": 1.0,
        "decrease_factor": 0.7,
        "increase_step": 1.0,
        "increase_after": 5,
        "refractory_ms": 0.0,  # off by default; the refractory tests opt in
        "ttl_s": 60,
    }
    return AdaptiveRate(redis, ENDPOINT, **{**defaults, **kwargs})


# ------------------------------------------------------------------ start


async def test_starts_at_the_advertised_rate_not_the_floor(redis: Redis) -> None:
    """A cold controller must not throttle.

    Starting at the floor would make every fresh deployment crawl through a
    ramp-up that looks exactly like an outage. We assume the endpoint honours
    its own published limit until it tells us otherwise.
    """
    assert await make_controller(redis, max_rate=10.0).current() == 10.0


async def test_read_does_not_mutate_the_rate(redis: Redis) -> None:
    """The rate is read once per request, so a read with side effects would
    make the controller a function of traffic volume rather than of health."""
    controller = make_controller(redis)
    for _ in range(20):
        await controller.current()

    assert await controller.current() == 10.0


# --------------------------------------------------------- decrease side


async def test_congestion_decreases_multiplicatively(redis: Redis) -> None:
    """x0.7 per event, not -1.

    Multiplicative decrease reaches safety in O(log) steps. The cost of running
    too fast is congestion collapse - pages destroyed, endpoint degraded - so
    the exit has to be quick.
    """
    controller = make_controller(redis, max_rate=10.0, decrease_factor=0.7)

    rates = []
    for _ in range(4):
        rates.append((await controller.on_congestion(Congestion.REJECTED)).rate)

    assert rates == pytest.approx([7.0, 4.9, 3.43, 2.401], rel=1e-3)


async def test_rate_never_falls_below_the_floor(redis: Redis) -> None:
    """The floor is not cosmetic - a rate of 0 is an unrecoverable deadlock.

    At rate 0 the bucket never grants a token, so the controller can never
    observe a success, so it can never climb back out. Multiplicative decrease
    approaches zero asymptotically, which is exactly the direction that needs a
    hard stop.
    """
    controller = make_controller(redis, min_rate=1.0)

    for _ in range(50):
        await controller.on_congestion(Congestion.REJECTED)

    rate = await controller.current()
    assert rate == 1.0
    assert rate > 0, "a zero rate can never recover"


async def test_at_floor_is_reported_distinctly_from_a_decrease(
    redis: Redis,
) -> None:
    """Telemetry has to distinguish "still backing off" from "bottomed out",
    because the second means the endpoint is worse than our policy can absorb.
    """
    controller = make_controller(redis, min_rate=1.0)
    for _ in range(20):
        await controller.on_congestion(Congestion.REJECTED)

    verdict = await controller.on_congestion(Congestion.REJECTED)
    assert verdict.reason == "at_floor"
    assert not verdict.changed


# --------------------------------------------------------- increase side


async def test_successes_increase_additively(redis: Redis) -> None:
    """+1 per streak, not x2.

    A multiplicative increase overshoots capacity on every probe - 1,2,4,8,16
    blows straight past a true limit of 10 - giving a large-amplitude sawtooth
    that spends half its life in overload. Additive increase keeps the
    oscillation small and centred just below capacity.
    """
    controller = make_controller(redis, increase_after=5, increase_step=1.0)
    for _ in range(20):
        await controller.on_congestion(Congestion.REJECTED)
    assert await controller.current() == 1.0

    rates = []
    for i in range(1, 21):
        verdict = await controller.on_success(2_000)
        if verdict.changed:
            rates.append(verdict.rate)

    assert rates == pytest.approx([2.0, 3.0, 4.0, 5.0])


async def test_increase_requires_a_full_success_streak(redis: Redis) -> None:
    """Stepping up on every single success would race back to the advertised
    rate immediately after one failure, defeating the point of backing off."""
    controller = make_controller(redis, increase_after=5)
    await controller.on_congestion(Congestion.REJECTED)
    assert await controller.current() == 7.0

    for _ in range(4):
        assert not (await controller.on_success(2_000)).changed

    assert (await controller.on_success(2_000)).changed, "5th success should step up"


async def test_congestion_resets_progress_toward_an_increase(
    redis: Redis,
) -> None:
    """Otherwise a failure arriving one success short of the threshold is still
    followed by an increase - raising the rate in direct response to evidence
    that it is already too high."""
    controller = make_controller(redis, increase_after=5)
    await controller.on_congestion(Congestion.REJECTED)

    for _ in range(4):
        await controller.on_success(2_000)
    verdict = await controller.on_congestion(Congestion.REJECTED)
    assert verdict.successes == 0

    # The banked streak is gone, so one more success must not step up.
    assert not (await controller.on_success(2_000)).changed


async def test_rate_never_exceeds_the_advertised_ceiling(redis: Redis) -> None:
    """One-sided, unlike TCP.

    Textbook AIMD has no ceiling because available bandwidth is unknown. We
    know the published limit, and probing above it would only earn 429s. So
    this controller detects capacity BELOW spec and recovers to spec - it does
    not hunt for capacity above it.
    """
    controller = make_controller(redis, max_rate=10.0, increase_after=1)

    for _ in range(200):
        await controller.on_success(2_000)

    assert await controller.current() == 10.0


async def test_full_cycle_decays_then_recovers_to_the_ceiling(
    redis: Redis,
) -> None:
    """The shape the whole step exists to produce."""
    controller = make_controller(redis, increase_after=5)

    for _ in range(4):
        await controller.on_congestion(Congestion.REJECTED)
    degraded = await controller.current()
    assert degraded < 3.0, degraded

    for _ in range(200):
        await controller.on_success(2_000)

    assert await controller.current() == 10.0


# ------------------------------------------------------------- refractory


async def test_refractory_period_prevents_one_event_collapsing_the_rate(
    redis: Redis,
) -> None:
    """THE test for this step.

    When the rate is cut there are already ~rate x latency requests in flight,
    admitted at the OLD rate and about to fail against the same overloaded
    endpoint. Compounding a decrease per failure gives 0.7^22 = 0.0004 of the
    original rate from a SINGLE congestion event - measured at 1.0 rps (the
    floor, 10% of advertised) without the guard, versus 7.0 rps with it.

    TCP solves the identical problem with "one reduction per RTT": the window
    halves once per loss EVENT, not once per lost packet.
    """
    controller = make_controller(redis, refractory_ms=5_000)

    for _ in range(22):  # one in-flight batch, all rejected at once
        await controller.on_congestion(Congestion.REJECTED)

    assert await controller.current() == 7.0, "compounded within one event"


async def test_suppressed_decreases_are_reported_as_refractory(
    redis: Redis,
) -> None:
    """Distinguishable in telemetry from a decrease that actually happened, or
    the logs would over-report how hard the controller is backing off."""
    controller = make_controller(redis, refractory_ms=5_000)

    first = await controller.on_congestion(Congestion.REJECTED)
    second = await controller.on_congestion(Congestion.REJECTED)

    assert first.reason == "decrease" and first.changed
    assert second.reason == "refractory" and not second.changed


async def test_refractory_window_expires(redis: Redis) -> None:
    """It delays repeat decreases, it must not prevent them - a sustained
    outage has to keep pushing the rate down."""
    controller = make_controller(redis, refractory_ms=200)

    await controller.on_congestion(Congestion.REJECTED)
    assert await controller.current() == 7.0

    await asyncio.sleep(0.25)
    await controller.on_congestion(Congestion.REJECTED)

    assert await controller.current() == pytest.approx(4.9)


async def test_refractory_still_resets_the_success_streak(redis: Redis) -> None:
    """A suppressed decrease is still evidence. Ignoring it entirely would let
    an in-flight batch of failures be followed by an increase."""
    controller = make_controller(redis, refractory_ms=5_000, increase_after=5)
    await controller.on_congestion(Congestion.REJECTED)

    for _ in range(4):
        await controller.on_success(2_000)
    await controller.on_congestion(Congestion.REJECTED)  # suppressed

    assert not (await controller.on_success(2_000)).changed, "streak survived"


# ---------------------------------------------------------------- latency


async def test_latency_breach_decreases_the_rate_without_any_rejection(
    redis: Redis,
) -> None:
    """The signal nothing else in the stack can see.

    The assignment names it: "when downstream endpoints return HTTP 429 OR
    LATENCY SPIKES". An endpoint that accepts everything but answers in 14s
    returns no 429s, so the token bucket sees nothing wrong and the breaker's
    failure ratio stays at zero - while in-flight work piles up and pages blow
    their deadlines.
    """
    controller = make_controller(
        redis,
        latency_slo_ms=4_500,
        latency_min_samples=20,
        increase_after=10_000,  # isolate the decrease
    )

    for _ in range(25):
        healthy = await controller.on_success(2_200)
    assert healthy.rate == 10.0, "healthy latency must not throttle"
    assert healthy.p95_ms == pytest.approx(2_200)

    for _ in range(25):
        degraded = await controller.on_success(14_000)

    assert degraded.rate < 10.0
    assert degraded.p95_ms > 4_500


async def test_latency_decrease_is_attributed_separately(redis: Redis) -> None:
    """"Slow" and "rejected" need different labels: they point at different
    causes and, in production, at different fixes."""
    controller = make_controller(
        redis, latency_slo_ms=1_000, latency_min_samples=5, increase_after=10_000
    )

    for _ in range(6):
        verdict = await controller.on_success(9_000)

    assert verdict.reason == "decrease_latency"


async def test_no_latency_verdict_below_the_minimum_sample_count(
    redis: Redis,
) -> None:
    """A p95 over 3 samples is not a p95.

    Without this floor, two slow warm-up calls would throttle a healthy system
    before it had served enough traffic to know anything.
    """
    controller = make_controller(
        redis, latency_slo_ms=1_000, latency_min_samples=20, increase_after=10_000
    )

    for _ in range(19):
        verdict = await controller.on_success(50_000)  # far past the SLO

    assert verdict.p95_ms == -1, "should report no verdict yet"
    assert await controller.current() == 10.0, "throttled on insufficient evidence"


async def test_latency_tracking_can_be_disabled(redis: Redis) -> None:
    """slo=0 turns it off, so the layout endpoint (50ms, where scheduling noise
    alone moves the percentile) can opt out."""
    controller = make_controller(
        redis, latency_slo_ms=0, latency_min_samples=5, increase_after=10_000
    )

    for _ in range(30):
        verdict = await controller.on_success(60_000)

    assert verdict.p95_ms == -1
    assert await controller.current() == 10.0


async def test_latency_ring_buffer_is_bounded(redis: Redis) -> None:
    """O(1) memory per endpoint. An unbounded list of every latency ever
    recorded is a slow leak that only shows up under sustained load."""
    controller = make_controller(
        redis, latency_slo_ms=100_000, latency_samples=50, increase_after=10_000
    )

    for _ in range(500):
        await controller.on_success(2_000)

    assert await redis.llen(controller.latency_key) == 50


async def test_p95_is_a_percentile_not_a_mean(redis: Redis) -> None:
    """A mean would let a degrading tail hide behind fast requests.

    90 fast and 10 slow gives a mean of 1800ms - comfortably under any
    threshold 9000ms would breach - while the p95 sits in the slow tail. That
    gap is the reason for a percentile rather than an average.

    Note the index arithmetic: nearest-rank p95 over 100 samples is the 95th
    smallest, so exactly 5% slow samples land ON the boundary and p95 reads as
    fast. That is correct - "95% of values are <= this" - but it means the test
    needs more than 5% slow to demonstrate anything.
    """
    controller = make_controller(
        redis, latency_slo_ms=100_000, latency_min_samples=20, increase_after=10_000
    )

    for _ in range(90):
        await controller.on_success(1_000)
    for _ in range(10):
        verdict = await controller.on_success(9_000)

    assert verdict.p95_ms == pytest.approx(9_000)


# ------------------------------------------------------------ distributed


async def test_state_is_shared_between_controller_instances(
    redis: Redis,
) -> None:
    """The reason this lives in Redis.

    With per-process controllers each replica would rediscover the same
    degradation independently, and the aggregate rate would be N x whatever
    each one decided was safe - which defeats the point of backing off at all.
    """
    first = make_controller(redis)
    second = make_controller(redis)

    await first.on_congestion(Congestion.REJECTED)

    assert await second.current() == 7.0


async def test_concurrent_congestion_reports_do_not_overdraw(
    redis: Redis,
) -> None:
    """50 replicas reporting at once must not each apply their own x0.7.

    Two mechanisms combine here: Lua makes the read-modify-write atomic, and
    the refractory window collapses one event into one decrease. Without the
    first, concurrent reports would interleave and lose decrements; without the
    second, they would compound.
    """
    controller = make_controller(redis, refractory_ms=5_000)

    await asyncio.gather(
        *(controller.on_congestion(Congestion.REJECTED) for _ in range(50))
    )

    assert await controller.current() == 7.0


async def test_endpoints_are_controlled_independently(redis: Redis) -> None:
    """A degraded VLM must not throttle layout - layout is where the degraded
    fallback's input comes from, and where time-to-first-page is won."""
    vlm = AdaptiveRate(redis, "vlm", max_rate=10.0, refractory_ms=0, ttl_s=60)
    layout = AdaptiveRate(redis, "layout", max_rate=100.0, refractory_ms=0, ttl_s=60)

    for _ in range(5):
        await vlm.on_congestion(Congestion.REJECTED)

    assert await vlm.current() < 10.0
    assert await layout.current() == 100.0
    assert vlm.key != layout.key


# ---------------------------------------------------------------- hygiene


async def test_controller_key_expires(redis: Redis) -> None:
    """Idle endpoints reclaim themselves instead of accumulating one key per
    endpoint per deployment forever. Expiry is safe: a missing controller
    restarts at the advertised rate."""
    controller = make_controller(redis, ttl_s=60)
    await controller.on_success(2_000)

    assert 0 < await redis.ttl(controller.key) <= 60


async def test_reset_returns_to_the_advertised_rate(redis: Redis) -> None:
    controller = make_controller(redis)
    for _ in range(5):
        await controller.on_congestion(Congestion.REJECTED)

    await controller.reset()

    assert await controller.current() == 10.0


@pytest.mark.parametrize(
    "kwargs,why",
    [
        ({"max_rate": 0}, "a zero ceiling permits nothing"),
        ({"min_rate": 0}, "a zero floor is an unrecoverable deadlock"),
        ({"min_rate": 20, "max_rate": 10}, "floor above ceiling is incoherent"),
        ({"decrease_factor": 1.0}, "factor of 1 is not a decrease"),
        ({"decrease_factor": 1.5}, "factor above 1 is an increase"),
        ({"increase_step": 0}, "a zero step never recovers"),
        ({"increase_after": 0}, "needs at least one success"),
    ],
)
async def test_invalid_config_is_rejected(
    redis: Redis, kwargs: dict, why: str
) -> None:
    with pytest.raises(ValueError):
        make_controller(redis, **kwargs)


# ------------------------------------------------- two floors, two signals


async def test_latency_decrease_stops_at_a_higher_floor_than_a_429(
    redis: Redis,
) -> None:
    """The two signals carry different weights of evidence.

    A 429 is the endpoint saying directly that we are too fast, so backing off
    to the hard floor is justified. A latency breach cannot distinguish
    slowness WE caused (relieved by backing off) from slowness in their own GC
    pause or dependency (not relieved, and throttling just discards
    throughput).

    Measured consequence of conflating them: the mock caps rate but not
    concurrency, so at 13s latency it still serves its full 10 rps. Flooring
    the rate on that signal cut a 60-page job to 34 pages in the window where
    adaptive=off completed all 60.
    """
    controller = make_controller(
        redis,
        max_rate=10.0,
        min_rate=1.0,
        latency_min_rate=5.0,
        latency_slo_ms=4_500,
        latency_min_samples=20,
        increase_after=10_000,
    )

    for _ in range(60):  # sustained, far past the SLO
        await controller.on_success(14_000)

    assert await controller.current() == 5.0, "latency alone must not floor the rate"


async def test_an_explicit_429_still_reaches_the_hard_floor(
    redis: Redis,
) -> None:
    """The higher latency floor must not weaken the definitive signal."""
    controller = make_controller(
        redis, max_rate=10.0, min_rate=1.0, latency_min_rate=5.0
    )

    for _ in range(30):
        await controller.on_congestion(Congestion.REJECTED)

    assert await controller.current() == 1.0


async def test_latency_floor_must_lie_between_the_other_two(
    redis: Redis,
) -> None:
    """A floor outside [min_rate, max_rate] is incoherent and would silently
    either disable the hard floor or throttle a healthy endpoint."""
    with pytest.raises(ValueError):
        make_controller(redis, max_rate=10.0, min_rate=1.0, latency_min_rate=20.0)
    with pytest.raises(ValueError):
        make_controller(redis, max_rate=10.0, min_rate=2.0, latency_min_rate=1.0)
