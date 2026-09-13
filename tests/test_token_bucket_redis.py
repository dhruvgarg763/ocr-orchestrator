"""Distributed token bucket tests.

Run against a real Redis, because the property under test is that the Lua script
executes atomically inside the server. A fake would reimplement that and prove
nothing.

The decisive test is `aggregate_rate_holds_across_independent_clients`: several
limiter instances, as separate worker replicas would have, sharing one bucket.
That is exactly the case the in-process limiter gets wrong.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from redis.asyncio import Redis

from app.ratelimit.token_bucket import RateLimitTimeout, TokenBucketLimiter

ENDPOINT = "test-endpoint"


def make_limiter(redis: Redis, **kwargs) -> TokenBucketLimiter:
    defaults = {"rate": 10.0, "burst": 10, "ttl_s": 60, "jitter_ms": 0.0}
    return TokenBucketLimiter(redis, ENDPOINT, **{**defaults, **kwargs})


# --------------------------------------------------------------- basic shape


async def test_bucket_starts_full(redis: Redis) -> None:
    """A cold or expired bucket must allow its configured burst.

    Starting empty would throttle every deployment from zero for a full burst
    window, which looks like an outage.
    """
    limiter = make_limiter(redis, rate=10, burst=10)

    granted = [await limiter.try_acquire() for _ in range(10)]

    assert all(a.granted for a in granted)
    denied = await limiter.try_acquire()
    assert not denied.granted

    # A full token at 10/sec would be 100ms, but the ten acquires above each
    # cost a network round trip (~3ms), and the bucket refills during them. So
    # the reported wait is for the FRACTION still missing - measured at 68ms.
    # This is a genuine difference from the in-process limiter, where ten
    # acquires are instantaneous; asserting exactly 100ms here would be
    # asserting that Redis calls are free.
    assert 0 < denied.wait_ms <= 100, denied


async def test_denied_acquire_does_not_consume_budget(redis: Redis) -> None:
    """A rejected caller must not spend tokens it did not receive.

    If denial consumed budget, a burst of contenders would starve the bucket and
    nobody would make progress.
    """
    limiter = make_limiter(redis, rate=10, burst=10)
    for _ in range(10):
        await limiter.try_acquire()

    before = await limiter.peek()
    for _ in range(20):
        await limiter.try_acquire()
    after = await limiter.peek()

    # Only refill should have changed the count, never the failed attempts.
    assert after >= before


async def test_refill_is_proportional_to_elapsed_time(redis: Redis) -> None:
    limiter = make_limiter(redis, rate=20, burst=20)
    for _ in range(20):
        await limiter.try_acquire()

    await asyncio.sleep(0.5)  # 0.5s * 20/sec = 10 tokens

    assert 8.5 <= await limiter.peek() <= 11.5


async def test_refill_is_capped_at_burst(redis: Redis) -> None:
    """The cap is what makes a limiter a limiter.

    Uncapped, an idle bucket accrues unbounded credit and can then discharge it
    all at once - the exact spike the limit exists to prevent.
    """
    limiter = make_limiter(redis, rate=1000, burst=5)
    for _ in range(5):
        await limiter.try_acquire()

    await asyncio.sleep(0.2)  # 200 tokens if uncapped

    assert await limiter.peek() <= 5.0


@pytest.mark.parametrize("rate,burst", [(0, 10), (10, 0), (-1, 10)])
async def test_invalid_config_is_rejected(redis: Redis, rate: float, burst: int) -> None:
    with pytest.raises(ValueError):
        TokenBucketLimiter(redis, ENDPOINT, rate=rate, burst=burst)


# ------------------------------------------------------- the distributed case


async def test_bucket_state_is_shared_between_limiter_instances(redis: Redis) -> None:
    """Two instances, one bucket. This is the whole point of the exercise.

    Separate in-process limiters would each have their own 10 tokens, for 20
    total. Sharing through Redis means the second instance sees what the first
    consumed.
    """
    first = make_limiter(redis, rate=10, burst=10)
    second = make_limiter(redis, rate=10, burst=10)

    for _ in range(10):
        assert (await first.try_acquire()).granted

    assert not (await second.try_acquire()).granted, (
        "second limiter got a token the shared bucket had already spent"
    )


async def test_aggregate_rate_holds_across_independent_clients(redis: Redis) -> None:
    """THE test. Three replicas at 10 rps must total 10 rps, not 30.

    Measured with the in-process limiter for comparison:
        1 replica  ->  9 req/s   (correct)
        3 replicas -> 27 req/s   (3x violation)
        5 replicas -> 45 req/s   (4x violation)

    The violation scales with replica count, so it appears precisely when the
    system is scaled out - and never in single-process development.
    """
    limiters = [make_limiter(redis, rate=10, burst=10) for _ in range(3)]
    for limiter in limiters:  # drain the shared burst to measure steady state
        while (await limiter.try_acquire()).granted:
            pass

    granted = 0
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        for limiter in limiters:
            if (await limiter.try_acquire()).granted:
                granted += 1
        await asyncio.sleep(0.005)

    assert granted <= 13, f"3 replicas sent {granted} req/s against a 10 rps limit"
    assert granted >= 6, f"only {granted} req/s; the limiter is over-throttling"


async def test_concurrent_acquires_never_overdraw_the_bucket(redis: Redis) -> None:
    """100 simultaneous claimants against 10 tokens must yield exactly 10.

    Without Lua this is a read-modify-write race: every claimant reads the same
    token count before any of them writes, and all 100 believe they were
    granted.
    """
    limiter = make_limiter(redis, rate=0.001, burst=10)  # refill ~never

    results = await asyncio.gather(*(limiter.try_acquire() for _ in range(100)))

    assert sum(1 for r in results if r.granted) == 10


async def test_naive_read_modify_write_overdraws(redis: Redis) -> None:
    """Demonstrates the bug the Lua script prevents, executably.

    The gap between reading the token count and writing it back is a network
    round trip, and the real caller then spends 50-3000ms in an HTTP request -
    so in production that window is enormous.
    """
    key = "naive-bucket"
    await redis.hset(key, mapping={"tokens": "10"})
    granted = 0

    async def naive_acquire() -> None:
        nonlocal granted
        tokens = float(await redis.hget(key, "tokens"))  # READ
        await asyncio.sleep(0)  # the window: a round trip, or a model call
        if tokens >= 1:  # CHECK
            await redis.hset(key, "tokens", tokens - 1)  # WRITE
            granted += 1

    await asyncio.gather(*(naive_acquire() for _ in range(100)))

    assert granted > 10, "expected the naive version to overdraw"


# --------------------------------------------------------------- acquire()


async def test_acquire_waits_instead_of_failing(redis: Redis) -> None:
    """Waiting is what makes this backpressure rather than load shedding.

    The page is delayed, never dropped - which is the difference between the
    zero-drop requirement passing and failing.
    """
    limiter = make_limiter(redis, rate=20, burst=2)
    for _ in range(2):
        await limiter.try_acquire()

    started = time.monotonic()
    waited = await limiter.acquire(max_wait_s=5)
    elapsed = time.monotonic() - started

    assert waited > 0, "should have reported a wait"
    # One token at 20/sec = 50ms.
    assert 0.02 <= elapsed <= 0.5


async def test_acquire_raises_a_distinct_error_when_its_budget_runs_out(
    redis: Redis,
) -> None:
    """Bounded waiting, with a typed error.

    Waiting forever would let tasks pile up invisibly. RateLimitTimeout is its
    own type so Step 8 can treat "saturated" as retryable, distinct from a
    model error.
    """
    limiter = make_limiter(redis, rate=0.001, burst=1)
    await limiter.try_acquire()

    with pytest.raises(RateLimitTimeout) as excinfo:
        await limiter.acquire(max_wait_s=0.3)

    assert excinfo.value.endpoint == ENDPOINT
    assert excinfo.value.waited_s >= 0


async def test_many_waiters_all_eventually_proceed_and_respect_the_rate(
    redis: Redis,
) -> None:
    """No starvation, and the rate still holds while everyone is queued."""
    limiter = make_limiter(redis, rate=50, burst=5, jitter_ms=25.0)

    started = time.monotonic()
    await asyncio.gather(*(limiter.acquire(max_wait_s=10) for _ in range(30)))
    elapsed = time.monotonic() - started

    # 30 requests, 5 free from the burst, 25 at 50/sec = 0.5s minimum.
    assert elapsed >= 0.4, f"finished in {elapsed:.2f}s - faster than the rate allows"
    assert elapsed < 5.0, f"took {elapsed:.2f}s - starvation or lost wakeups"


async def test_jitter_spreads_wakeups_rather_than_synchronising_them(
    redis: Redis,
) -> None:
    """Every waiter computes the SAME wait_ms.

    Without jitter they wake in the same instant, race for one token, and fire a
    synchronised burst of Redis calls - a convoy instead of a queue. With
    jitter, completion times spread out.
    """
    limiter = make_limiter(redis, rate=40, burst=1, jitter_ms=25.0)
    await limiter.try_acquire()

    async def timed() -> float:
        await limiter.acquire(max_wait_s=10)
        return time.monotonic()

    finishes = sorted(await asyncio.gather(*(timed() for _ in range(12))))
    gaps = [b - a for a, b in zip(finishes, finishes[1:])]

    assert max(gaps) > 0, "all waiters completed at the same instant"


# ------------------------------------------------------------------ hygiene


async def test_bucket_key_expires_so_idle_endpoints_do_not_leak(redis: Redis) -> None:
    """One key per endpoint is small, but unbounded key growth is still a leak.

    Expiry is safe because a missing bucket starts full.
    """
    limiter = make_limiter(redis, ttl_s=60)
    await limiter.try_acquire()

    ttl = await redis.ttl(limiter.key)
    assert 0 < ttl <= 60


async def test_independent_endpoints_do_not_share_a_bucket(redis: Redis) -> None:
    """The fast layout model must not be throttled by VLM saturation."""
    layout = TokenBucketLimiter(redis, "layout", rate=100, burst=100, jitter_ms=0.0)
    vlm = TokenBucketLimiter(redis, "vlm", rate=10, burst=10, jitter_ms=0.0)

    while (await vlm.try_acquire()).granted:  # saturate the VLM bucket
        pass

    assert (await layout.try_acquire()).granted
    assert layout.key != vlm.key


async def test_saturation_reports_projection_and_elapsed_separately(
    redis: Redis,
) -> None:
    """REGRESSION: the two durations must not be conflated.

    `retry_after_s` is a PROJECTION - how far away the next free slot is.
    `waited_s` is ELAPSED time actually spent asleep. Since the limiter
    reserves rather than polls, saturation is detected up front and nothing is
    slept, so waited_s is 0.

    The original bug reported the projection as though it were elapsed: logs
    claimed a ~5s wait for a call that returned in 1.7ms, which makes the
    telemetry actively misleading about where time is going. `waited_s >= 0`
    alone cannot catch that.
    """
    limiter = make_limiter(redis, rate=10, burst=10)
    # Drive the balance far negative so the next reservation cannot fit.
    for _ in range(60):
        await limiter.try_acquire(max_wait_s=30)

    started = time.monotonic()
    with pytest.raises(RateLimitTimeout) as excinfo:
        await limiter.acquire(max_wait_s=1.0)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, f"should fail fast, took {elapsed * 1000:.0f}ms"
    assert excinfo.value.waited_s == 0.0, "claimed to have waited when it did not"
    assert excinfo.value.retry_after_s > 1.0, "projection should exceed the budget"
