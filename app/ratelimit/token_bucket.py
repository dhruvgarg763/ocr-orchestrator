"""Distributed token bucket, enforced inside Redis.

Why not the in-process bucket from mock_model/ratelimit.py
----------------------------------------------------------
That one is correct for the mock, because the mock is a single server enforcing
its own published limit - it is the sole authority. A *client-side* limiter
cannot work that way. Each worker replica would hold its own bucket and each
would faithfully enforce 10 rps, for an aggregate of N x 10. Measured:

    1 replica  x 10 rps  ->   9 requests/sec   (correct)
    3 replicas x 10 rps  ->  27 requests/sec   (3x violation)
    5 replicas x 10 rps  ->  45 requests/sec   (4x violation)

The error is proportional to how well you have scaled out, so the bug appears
exactly when the system starts succeeding.

Three things therefore have to be shared, not just one:

  1. the token count  -> lives in a Redis hash
  2. the read-modify-write -> a Lua script, which Redis runs atomically, so 24
     concurrent claimants cannot all observe the same token and overdraw
  3. the CLOCK -> taken inside the script via redis.call('TIME')

Point 3 is the easy one to miss. If `now` were passed in as an argument, every
container would supply its own clock. Measured with a client-supplied clock:

  * A *constant* offset turns out to be survivable. The `elapsed > 0` guard
    below means a lagging replica simply never refills, so the fastest clock
    becomes the de-facto authority and the aggregate rate stays roughly right -
    unfair, but not a violation.
  * A clock *lead* is not survivable. A replica 3s ahead, arriving at a bucket
    another replica had just drained, computed elapsed = +3000ms and was granted
    10 requests immediately - the entire burst, minted from nothing.

Real clocks do exactly that: NTP steps them forward, VMs resume from suspend,
containers start on hosts with drifted clocks. Taking the time inside the script
means there is exactly one clock and the whole class of bug disappears. (It
makes the script non-deterministic, which mattered under Redis <= 4 command
replication; Redis 5+ uses effects replication, so it is safe.)

Reservation, not polling
------------------------
The script does not merely report how long to wait - it hands out a slot. When
no token is free it deducts the cost anyway, driving the balance NEGATIVE, and
returns the delay until that debt is paid off. The caller sleeps once and
proceeds. The negative balance IS the queue, recorded in the bucket.

The obvious alternative - return a wait and have the caller re-poll - collapses
at scale, because every waiter computes the same delay, wakes in the same
instant, and all but `rate` of them are denied again. Measured on a 10 rps
bucket:

    waiters   grants   Redis calls   wasted
         16       16            35      54%
         48       48           751      94%
        150      150         9,063      98%
        400      310        61,692      99%

199 round trips per granted token at 400 waiters. With reservation it is 1, and
grants follow arrival order rather than being re-raffled every tick. This is
what Guava's RateLimiter.acquire() does, and it is still a token bucket - the
balance simply carries a debt.

The debt is bounded: a reservation is only issued if it fits inside the caller's
max_wait, so the balance cannot fall below -(max_wait * rate).
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis

from app.core.redis_client import register_script
from common.logging import get_logger

log = get_logger("ratelimit")


class RateLimitTimeout(Exception):
    """Raised when no slot could be reserved inside the caller's wait budget.

    Deliberately a distinct exception type: it means "the system is saturated",
    which is a retryable, expected condition - not a model error. The worker
    relies on that distinction to requeue rather than fail the page.

    The two durations are different and must not be conflated:

      retry_after_s  how far in the future the next free slot is. A PROJECTION,
                     and the useful number for pacing a retry.
      waited_s       time this caller actually spent asleep. Zero when the
                     reservation was refused outright, which is now the normal
                     case: the reserving limiter detects saturation in ~2ms
                     rather than after a 30s poll.

    Reporting the projection as though it were elapsed time was an actual bug -
    logs claimed a 5s wait for a call that returned in 1.7ms.
    """

    def __init__(self, endpoint: str, retry_after_s: float, waited_s: float = 0.0) -> None:
        super().__init__(
            f"{endpoint}: no slot available; next free in {retry_after_s:.2f}s "
            f"(waited {waited_s:.2f}s)"
        )
        self.endpoint = endpoint
        self.retry_after_s = retry_after_s
        self.waited_s = waited_s


# KEYS[1] bucket hash
# ARGV[1] rate (tokens/sec)  ARGV[2] burst  ARGV[3] cost  ARGV[4] ttl (ms)
# ARGV[5] max_wait_ms - the longest reservation the caller will accept
# returns {granted, wait_ms, tokens_x1000}
#
# `granted == 1` with `wait_ms > 0` means RESERVED: the caller sleeps wait_ms
# and then proceeds, with no second call. See the class docstring for why that
# matters (measured: 199 Redis round trips per grant became 1).
_BUCKET_LUA = """
local rate       = tonumber(ARGV[1])
local burst      = tonumber(ARGV[2])
local cost       = tonumber(ARGV[3])
local ttl_ms     = tonumber(ARGV[4])
local max_wait_ms = tonumber(ARGV[5])

-- One clock for every replica. Passing `now` in as an argument would let a
-- container with a fast clock mint free tokens.
local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + tonumber(t[2]) / 1000

local data   = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts     = tonumber(data[2])

if tokens == nil or ts == nil then
  -- First use (or the key expired while idle): start full, so a cold system is
  -- allowed its configured burst rather than being throttled from zero.
  tokens = burst
  ts = now_ms
end

-- Lazy refill: derive tokens from elapsed time instead of running a timer.
-- O(1), no background task, exact to the microsecond. Guarded against a
-- non-positive interval so a clock adjustment cannot subtract tokens.
local elapsed = now_ms - ts
if elapsed > 0 then
  tokens = math.min(burst, tokens + elapsed * rate / 1000)
  ts = now_ms
end

local granted = 0
local wait_ms = 0

if tokens >= cost then
  tokens = tokens - cost
  granted = 1
else
  -- Round up: waking early would only burn another round trip.
  wait_ms = math.ceil((cost - tokens) * 1000 / rate)

  if wait_ms <= max_wait_ms then
    -- RESERVE. Deducting below zero puts this caller in a queue that is
    -- recorded in the bucket itself, so it sleeps once and then proceeds
    -- without polling again. The negative balance is exactly the backlog of
    -- promised tokens, and refill pays it off at `rate`.
    --
    -- The alternative - returning a wait and having the caller re-poll -
    -- generates a herd: every waiter computes the same wait, wakes together,
    -- and all but `rate` of them are denied again. Measured at 400 waiters:
    -- 61,692 Redis calls to grant 310 tokens, 99% of them wasted.
    tokens = tokens - cost
    granted = 1
  end
  -- Over max_wait: NOT reserved. Leaving the balance untouched is what bounds
  -- how negative it can go - to -(max_wait_ms/1000 * rate) - and what lets the
  -- caller raise RateLimitTimeout instead of promising a slot nobody waits for.
end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', ts)
-- Idle buckets reclaim themselves rather than accumulating one key per
-- endpoint per deployment forever.
redis.call('PEXPIRE', KEYS[1], ttl_ms)

return {granted, wait_ms, math.floor(tokens * 1000)}
"""


@dataclass(frozen=True)
class Acquisition:
    granted: bool
    wait_ms: int
    tokens: float
    """Tokens left after this call, for X-RateLimit-style reporting."""


class TokenBucketLimiter:
    def __init__(
        self,
        redis: Redis,
        endpoint: str,
        *,
        rate: float,
        burst: int,
        ttl_s: int = 300,
        jitter_ms: float = 25.0,
        key_prefix: str = "rl",
    ) -> None:
        if rate <= 0 or burst <= 0:
            raise ValueError("rate and burst must be positive")
        self._redis = redis
        self.endpoint = endpoint
        self.rate = rate
        self.burst = burst
        self._ttl_ms = ttl_s * 1000
        self._jitter_ms = jitter_ms
        self.key = f"{key_prefix}:{endpoint}"

    @property
    def _script(self) -> Any:
        # Lazy: register_script needs a live client, which exists only after
        # lifespan startup.
        return register_script("token_bucket", _BUCKET_LUA)

    async def try_acquire(
        self, cost: float = 1.0, *, max_wait_s: float = 0.0, rate: float | None = None
    ) -> Acquisition:
        """One call. Returns whether a slot was obtained, and when it starts.

        `max_wait_s=0` is a pure non-blocking check: grant only if a token is
        available right now. A positive value permits a RESERVATION - the script
        may return `granted=True` together with a non-zero `wait_ms`, meaning
        "this slot is yours, start using it in wait_ms".

        `rate` overrides the configured rate for this call, which is how the
        AIMD controller steers the bucket (app/ratelimit/adaptive.py). The rate
        was always a script argument rather than stored state, so a moving
        setpoint costs nothing: refill is derived from elapsed time at whatever
        rate is supplied, so lowering it slows the next refill immediately
        without invalidating tokens already granted.
        """
        granted, wait_ms, tokens_milli = await self._script(
            keys=[self.key],
            args=[
                self.rate if rate is None else rate,
                self.burst,
                cost,
                self._ttl_ms,
                max_wait_s * 1000,
            ],
        )
        return Acquisition(
            granted=bool(granted), wait_ms=int(wait_ms), tokens=int(tokens_milli) / 1000
        )

    async def acquire(
        self, cost: float = 1.0, *, max_wait_s: float = 30.0, rate: float | None = None
    ) -> float:
        """Reserve a slot, wait for it, return the seconds spent waiting.

        Waiting rather than failing is what makes this backpressure instead of
        load shedding: the page is delayed, never dropped. The wait is bounded
        so a genuinely wedged downstream surfaces as a typed error with a metric
        attached, instead of tasks accumulating in memory forever.

        ONE Redis call and ONE sleep - not a polling loop. The distinction is
        not stylistic. A polling implementation makes every waiter compute the
        same delay, wake together, and contend again; at 400 waiters on a 10 rps
        bucket that measured 61,692 Redis calls to grant 310 tokens - 199 round
        trips per grant, 99% wasted. Reserving makes it exactly 1, and grants
        follow arrival order instead of being re-raffled on every tick.
        """
        result = await self.try_acquire(cost, max_wait_s=max_wait_s, rate=rate)

        if not result.granted:
            # The reservation would have exceeded the budget, so none was taken:
            # no slot is held and nothing needs releasing.
            # waited_s=0: nothing was slept. Saturation is detected up front
            # now, and claiming otherwise would corrupt the telemetry.
            raise RateLimitTimeout(
                self.endpoint, retry_after_s=result.wait_ms / 1000, waited_s=0.0
            )

        if result.wait_ms <= 0:
            return 0.0

        # Jitter is a small ABSOLUTE spread, not a fraction of the wait.
        #
        # Proportional jitter was right for the old polling loop, where waiters
        # woke in a herd and needed scattering. With reservations they are
        # already spaced exactly 1/rate apart, so scaling a long wait by up to
        # 1.25x does two bad things: it delays the whole queue (measured 8.4
        # grants/sec against a 10 rps limit - under-driving the endpoint by
        # 16%), and it REORDERS callers, because a caller holding an earlier
        # slot can draw more jitter than one behind it. That showed up as 136
        # FIFO inversions across 313 grants.
        #
        # A few milliseconds of absolute spread keeps adjacent slots off the
        # same timer tick without either effect.
        sleep_s = result.wait_ms / 1000 + random.random() * self._jitter_ms / 1000
        await asyncio.sleep(sleep_s)
        return sleep_s

    async def peek(self) -> float:
        """Token balance, without consuming or reserving anything.

        May be NEGATIVE, which is meaningful rather than an error: it is the
        backlog of slots already promised to waiting callers. -25 at 10 rps
        means the next arrival would be told to come back in 2.5 seconds.
        """
        return (await self.try_acquire(cost=0.0, max_wait_s=0.0)).tokens

    async def reset(self) -> None:
        """Drop the bucket so the next caller starts full. Tests only."""
        await self._redis.delete(self.key)
