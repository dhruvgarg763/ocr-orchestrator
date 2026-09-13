"""Circuit breaker tests, against real Redis.

The state machine is evaluated inside a Lua script so that N replicas share one
verdict, which is exactly what a fake would have to reimplement. The two tests
that matter most are the HALF_OPEN ones: closing on a timer instead of probing
is the classic mistake, and it re-kills a recovering service with full load.
"""

from __future__ import annotations

import asyncio

import pytest
from redis.asyncio import Redis

from app.ratelimit.breaker import BreakerState, CircuitBreaker

ENDPOINT = "vlm"


def make_breaker(redis: Redis, **kwargs) -> CircuitBreaker:
    defaults = {
        "window_s": 10.0,
        "min_volume": 4,
        "failure_ratio": 0.5,
        "cooldown_s": 0.3,
        "max_probes": 2,
        "probe_successes": 2,
        "ttl_s": 60,
    }
    return CircuitBreaker(redis, ENDPOINT, **{**defaults, **kwargs})


# ------------------------------------------------------------------- closed


async def test_starts_closed_and_allows(redis: Redis) -> None:
    """A cold breaker must not block. Starting open would look like an outage."""
    verdict = await make_breaker(redis).allow()

    assert verdict.allowed
    assert verdict.state is BreakerState.CLOSED


async def test_successes_alone_never_open_the_circuit(redis: Redis) -> None:
    breaker = make_breaker(redis)
    for _ in range(50):
        await breaker.record_success()

    assert (await breaker.allow()).state is BreakerState.CLOSED


async def test_min_volume_prevents_opening_on_noise(redis: Redis) -> None:
    """1 failure out of 2 is 50%, but it is not evidence.

    Without this guard a quiet endpoint opens its circuit on a single unlucky
    call, and every page routed through it degrades needlessly.
    """
    breaker = make_breaker(redis, min_volume=10, failure_ratio=0.5)

    for _ in range(4):  # 4 failures, below min_volume
        await breaker.record_failure()

    verdict = await breaker.allow()
    assert verdict.allowed
    assert verdict.state is BreakerState.CLOSED


async def test_opens_once_volume_and_ratio_are_both_met(redis: Redis) -> None:
    breaker = make_breaker(redis, min_volume=4, failure_ratio=0.5)

    await breaker.record_success()
    await breaker.record_success()
    await breaker.record_failure()
    still_closed = await breaker.allow()
    assert still_closed.state is BreakerState.CLOSED, "1/3 is below the ratio"

    await breaker.record_failure()  # now 2 failures / 4 total = 50%
    opened = await breaker.allow()

    assert not opened.allowed
    assert opened.state is BreakerState.OPEN
    assert opened.retry_after_s > 0


async def test_open_circuit_reports_how_long_to_wait(redis: Redis) -> None:
    """The caller needs the number to pace itself; guessing produces a herd."""
    breaker = make_breaker(redis, min_volume=2, cooldown_s=5.0)
    await breaker.record_failure()
    await breaker.record_failure()

    verdict = await breaker.allow()

    assert not verdict.allowed
    assert 4.0 < verdict.retry_after_s <= 5.0


async def test_rolling_window_forgets_old_failures(redis: Redis) -> None:
    """Counters that never reset keep an hour-old outage alive forever.

    Only recent history predicts the next request, so the window must expire.
    """
    breaker = make_breaker(redis, window_s=0.3, min_volume=4, failure_ratio=0.75)

    await breaker.record_failure()
    await breaker.record_failure()
    await breaker.record_failure()
    assert (await breaker.allow()).state is BreakerState.CLOSED  # 3 < min_volume

    await asyncio.sleep(0.35)  # window rolls, counters reset
    await breaker.record_failure()
    await breaker.record_failure()

    assert (await breaker.allow()).state is BreakerState.CLOSED, (
        "pre-window failures should have been forgotten"
    )


# ---------------------------------------------------------------- half open


async def test_open_becomes_half_open_after_the_cooldown(redis: Redis) -> None:
    breaker = make_breaker(redis, min_volume=2, cooldown_s=0.2)
    await breaker.record_failure()
    await breaker.record_failure()
    assert (await breaker.allow()).state is BreakerState.OPEN

    await asyncio.sleep(0.25)
    verdict = await breaker.allow()

    assert verdict.allowed, "a probe should be admitted"
    assert verdict.state is BreakerState.HALF_OPEN


async def test_half_open_admits_only_a_bounded_number_of_probes(
    redis: Redis,
) -> None:
    """The whole reason HALF_OPEN exists.

    Closing on a timer would send full production load at a service that may
    still be dead, re-killing it and restarting the cycle. Probing costs 2
    requests to answer the same question.

    The probe count lives in Redis so that N replicas cannot each send "just
    one" and collectively flood a recovering endpoint.
    """
    breaker = make_breaker(redis, min_volume=2, cooldown_s=0.2, max_probes=2)
    await breaker.record_failure()
    await breaker.record_failure()
    await asyncio.sleep(0.25)

    verdicts = [await breaker.allow() for _ in range(6)]

    assert sum(1 for v in verdicts if v.allowed) == 2, (
        f"admitted {sum(1 for v in verdicts if v.allowed)} probes, max is 2"
    )
    assert all(v.state is BreakerState.HALF_OPEN for v in verdicts)


async def test_one_probe_failure_reopens_immediately(redis: Redis) -> None:
    """The probe existed to answer a question, and the answer was no.

    Waiting for a ratio to build again would mean sending more doomed traffic.
    """
    breaker = make_breaker(redis, min_volume=2, cooldown_s=0.2)
    await breaker.record_failure()
    await breaker.record_failure()
    await asyncio.sleep(0.25)
    assert (await breaker.allow()).state is BreakerState.HALF_OPEN

    verdict = await breaker.record_failure()

    assert verdict.state is BreakerState.OPEN
    assert not (await breaker.allow()).allowed, "cooldown should have restarted"


async def test_enough_probe_successes_close_the_circuit(redis: Redis) -> None:
    breaker = make_breaker(
        redis, min_volume=2, cooldown_s=0.2, max_probes=3, probe_successes=2
    )
    await breaker.record_failure()
    await breaker.record_failure()
    await asyncio.sleep(0.25)
    await breaker.allow()

    first = await breaker.record_success()
    assert first.state is BreakerState.HALF_OPEN, "one success is not enough"

    second = await breaker.record_success()
    assert second.state is BreakerState.CLOSED

    verdict = await breaker.allow()
    assert verdict.allowed and verdict.state is BreakerState.CLOSED


async def test_closing_resets_the_counters(redis: Redis) -> None:
    """Stale failures must not immediately reopen a recovered circuit."""
    breaker = make_breaker(
        redis, min_volume=2, cooldown_s=0.2, failure_ratio=0.5, probe_successes=1
    )
    await breaker.record_failure()
    await breaker.record_failure()
    await asyncio.sleep(0.25)
    await breaker.allow()
    closed = await breaker.record_success()

    assert closed.state is BreakerState.CLOSED
    assert closed.failures == 0
    assert closed.successes == 0


# -------------------------------------------------------------- distributed


async def test_state_is_shared_between_breaker_instances(redis: Redis) -> None:
    """The reason this lives in Redis.

    With per-process breakers, every replica must rediscover the same outage
    and each keeps hammering until it does - so the blast radius scales with
    replica count. Shared state means one discovery stops all of them.
    """
    first = make_breaker(redis, min_volume=2)
    second = make_breaker(redis, min_volume=2)

    await first.record_failure()
    await first.record_failure()

    verdict = await second.allow()
    assert not verdict.allowed, "second replica did not see the first's verdict"
    assert verdict.state is BreakerState.OPEN


async def test_concurrent_probes_respect_the_global_limit(redis: Redis) -> None:
    """Three replicas racing to probe must not collectively exceed max_probes."""
    breakers = [make_breaker(redis, min_volume=2, cooldown_s=0.2, max_probes=2) for _ in range(3)]
    await breakers[0].record_failure()
    await breakers[0].record_failure()
    await asyncio.sleep(0.25)

    verdicts = await asyncio.gather(*(b.allow() for b in breakers for _ in range(4)))

    assert sum(1 for v in verdicts if v.allowed) == 2


async def test_independent_endpoints_have_independent_breakers(
    redis: Redis,
) -> None:
    """A dead VLM must not stop layout work - that is where the fallback's
    input comes from."""
    vlm = CircuitBreaker(redis, "vlm", min_volume=2, cooldown_s=5)
    layout = CircuitBreaker(redis, "layout", min_volume=2, cooldown_s=5)

    await vlm.record_failure()
    await vlm.record_failure()

    assert not (await vlm.allow()).allowed
    assert (await layout.allow()).allowed
    assert vlm.key != layout.key


# -------------------------------------------------------------- hygiene


async def test_breaker_key_expires(redis: Redis) -> None:
    breaker = make_breaker(redis, ttl_s=60)
    await breaker.allow()

    assert 0 < await redis.ttl(breaker.key) <= 60


async def test_peek_does_not_mutate_state(redis: Redis) -> None:
    """/metrics must be able to read the breaker without consuming a probe."""
    breaker = make_breaker(redis, min_volume=2, cooldown_s=0.2, max_probes=1)
    await breaker.record_failure()
    await breaker.record_failure()
    await asyncio.sleep(0.25)

    for _ in range(5):
        await breaker.peek()

    # The single probe slot must still be available.
    assert (await breaker.allow()).allowed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"failure_ratio": 0},
        {"failure_ratio": 1.5},
        {"min_volume": 0},
        {"max_probes": 0},
        {"probe_successes": 0},
    ],
)
async def test_invalid_config_is_rejected(redis: Redis, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        make_breaker(redis, **kwargs)


async def test_peek_reports_what_allow_would_say(redis: Redis) -> None:
    """peek must be usable for /metrics, not permanently report 'blocked'."""
    breaker = make_breaker(redis, min_volume=2, cooldown_s=5.0)

    assert (await breaker.peek()).allowed, "closed breaker should read as allowing"

    await breaker.record_failure()
    await breaker.record_failure()

    verdict = await breaker.peek()
    assert not verdict.allowed
    assert verdict.state is BreakerState.OPEN
    assert verdict.retry_after_s > 0
