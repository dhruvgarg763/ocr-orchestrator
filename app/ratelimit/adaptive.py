"""AIMD rate controller: discover the rate actually available.

The token bucket enforces the rate we were told; this discovers the rate
actually available when an endpoint degrades - including a latency spike
with no 429s at all, which the bucket cannot see. Asymmetric by design
(TCP congestion control applied to a dispatcher): multiplicative decrease
(x0.7) reaches safety in O(log) steps because overload is costly, additive
increase (+1) probes gently because being merely slow is cheap. Only LOAD
signals (429, timeout, p95 breach) count as congestion - 5xx is excluded
because the mock's 5% baseline failure rate has nothing to do with load
and would otherwise drag the rate down by noise; sustained 5xx is the
circuit breaker's job. A refractory period after each decrease stops the
~rate*latency requests already in flight at the old rate from each
triggering their own cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from redis.asyncio import Redis

from app.core.redis_client import register_script
from common.logging import get_logger

log = get_logger("aimd")


class Congestion(str, Enum):
    """Why we are being asked to slow down. Recorded for telemetry so a rate
    decay can be attributed rather than guessed at."""

    REJECTED = "rejected"
    """An explicit 429. The endpoint told us directly."""

    TIMEOUT = "timeout"
    """No answer at all - the endpoint is past the point of rejecting cleanly."""

    LATENCY = "latency"
    """p95 breached its SLO with no rejections. The signal a token bucket and a
    failure-ratio breaker are both blind to."""


# KEYS[1] controller hash   KEYS[2] latency ring buffer
# ARGV[1]  op: "success" | "congestion" | "read"
# ARGV[2]  min_rate          ARGV[3]  max_rate
# ARGV[4]  decrease_factor   ARGV[5]  increase_step
# ARGV[6]  increase_after    ARGV[7]  refractory_ms
# ARGV[8]  ttl_ms            ARGV[9]  latency_ms (-1 = none)
# ARGV[10] latency_slo_ms (0 = disabled)
# ARGV[11] ring_size         ARGV[12] min_samples
# returns {rate_x1000, changed, reason, p95_ms, successes}
_AIMD_LUA = """
local op              = ARGV[1]
local min_rate        = tonumber(ARGV[2])
local max_rate        = tonumber(ARGV[3])
local decrease_factor = tonumber(ARGV[4])
local increase_step   = tonumber(ARGV[5])
local increase_after  = tonumber(ARGV[6])
local refractory_ms   = tonumber(ARGV[7])
local ttl_ms          = tonumber(ARGV[8])
local latency_ms      = tonumber(ARGV[9])
local latency_slo_ms  = tonumber(ARGV[10])
local ring_size       = tonumber(ARGV[11])
local min_samples     = tonumber(ARGV[12])
local latency_floor   = tonumber(ARGV[13])

-- One clock for every replica, as with the bucket and the breaker: a container
-- running fast must not be able to expire another replica's refractory window.
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + tonumber(t[2]) / 1000

local h = redis.call('HMGET', KEYS[1], 'rate', 'successes', 'last_decrease_at')
local rate      = tonumber(h[1])
local successes = tonumber(h[2]) or 0
local last_dec  = tonumber(h[3]) or 0

if rate == nil then
  -- Start at the ADVERTISED rate, not at the floor. Starting low would
  -- throttle every cold deployment for the whole ramp-up period, which looks
  -- exactly like an outage. We assume the endpoint honours its own spec until
  -- it tells us otherwise.
  rate = max_rate
end

local changed = 0
local reason = 'none'
local p95 = -1

-- ---------------------------------------------------------------- latency
-- Recorded on success only. A failed request's latency measures how fast the
-- endpoint says no, which is not the quantity of interest.
if op == 'success' and latency_ms >= 0 and latency_slo_ms > 0 then
  redis.call('LPUSH', KEYS[2], latency_ms)
  redis.call('LTRIM', KEYS[2], 0, ring_size - 1)
  redis.call('PEXPIRE', KEYS[2], ttl_ms)

  local n = redis.call('LLEN', KEYS[2])
  -- A p95 over 3 samples is not a p95. Waiting for a minimum sample count
  -- stops a couple of slow warm-up requests from throttling a healthy system.
  if n >= min_samples then
    local raw = redis.call('LRANGE', KEYS[2], 0, -1)
    local nums = {}
    for i = 1, #raw do nums[i] = tonumber(raw[i]) end
    table.sort(nums)
    local idx = math.ceil(0.95 * #nums)
    if idx < 1 then idx = 1 end
    p95 = nums[idx]
  end
end

-- -------------------------------------------------------------- congestion
local congested = false
if op == 'congestion' then
  congested = true
  reason = 'signal'
elseif p95 >= 0 and latency_slo_ms > 0 and p95 > latency_slo_ms then
  -- Accepted-but-slow is still congestion, and it is the case a rate limiter
  -- cannot see: no request was rejected, so nothing else in the stack reacts.
  congested = true
  reason = 'latency'
end

if congested then
  -- Progress toward an increase is discarded. Without this a failure arriving
  -- one success short of the threshold would still be followed by an increase.
  successes = 0

  if (now - last_dec) < refractory_ms then
    -- Inside the refractory window. The ~rate x latency requests already in
    -- flight were admitted at the old rate and are about to fail against the
    -- same overloaded endpoint; compounding a decrease per failure gives
    -- 0.7^22 of the original rate from ONE event. TCP's equivalent rule is one
    -- reduction per RTT.
    reason = 'refractory'
  else
    -- TWO floors, because the two signals carry different weights of evidence.
    --
    -- A 429 or a timeout is the endpoint telling us directly that we are too
    -- fast; that justifies backing off all the way to min_rate.
    --
    -- A latency breach is AMBIGUOUS. Slowness caused by our load is relieved
    -- by backing off; slowness from a GC pause, a cold cache or a slow
    -- dependency of theirs is not, and throttling then just discards
    -- throughput. Latency alone cannot separate the two - measured here: the
    -- mock has no concurrency cap, so at 13s latency it still serves its full
    -- 10 rps, and flooring the rate cut a 60-page job to 34 pages in the same
    -- window while adaptive=off completed all 60.
    --
    -- So we will give up half our throughput to protect an endpoint that might
    -- be struggling because of us, but not 90% on evidence we cannot attribute.
    local floor = min_rate
    if reason == 'latency' then floor = latency_floor end

    local new_rate = rate * decrease_factor
    if new_rate < floor then new_rate = floor end
    -- A floor above zero is mandatory either way: at rate 0 no token is ever
    -- granted, the controller can never observe a success, and it could never
    -- climb out.
    if new_rate ~= rate then
      rate = new_rate
      changed = 1
      if reason == 'signal' then reason = 'decrease' end
      if reason == 'latency' then reason = 'decrease_latency' end
    else
      reason = 'at_floor'
    end
    last_dec = now
  end

-- ---------------------------------------------------------------- success
elseif op == 'success' then
  successes = successes + 1
  if successes >= increase_after then
    successes = 0
    if rate < max_rate then
      rate = rate + increase_step
      if rate > max_rate then rate = max_rate end
      changed = 1
      reason = 'increase'
      -- Note the ceiling: max_rate is the ADVERTISED limit. Probing past it
      -- would only earn 429s, so this controller recovers to the published
      -- rate and stops. It detects capacity below spec, it does not hunt for
      -- capacity above it.
    end
  end
end

redis.call('HSET', KEYS[1],
  'rate', rate, 'successes', successes, 'last_decrease_at', last_dec,
  'updated_at', now)
redis.call('PEXPIRE', KEYS[1], ttl_ms)

-- Round, do not truncate. 10 * 0.7 * 0.7 is 4.8999999999999995 in floating
-- point, and flooring the encoded form reports 4.899 rps for a rate that is
-- 4.9. The stored value keeps full precision so nothing accumulates, but a
-- truncated read is needlessly hard to assert against and to eyeball in logs.
return {math.floor(rate * 1000 + 0.5), changed, reason, math.floor(p95 + 0.5), successes}
"""


@dataclass(frozen=True)
class RateVerdict:
    rate: float
    """Requests per second currently permitted."""

    changed: bool
    """True if this call moved the rate, for log-on-transition only."""

    reason: str
    """increase | decrease | decrease_latency | refractory | at_floor | none"""

    p95_ms: float
    """-1 when latency tracking is off or there are too few samples."""

    successes: int
    """Consecutive successes banked toward the next increase."""


class AdaptiveRate:
    """Per-endpoint AIMD controller, evaluated inside Redis.

    Shared state, for the same reason the bucket and breaker share theirs: with
    per-process controllers each replica would have to rediscover the same
    degradation independently, and the aggregate rate would be N x whatever
    each replica individually decided was safe - which defeats the point of
    backing off at all.
    """

    def __init__(
        self,
        redis: Redis,
        endpoint: str,
        *,
        max_rate: float,
        min_rate: float = 1.0,
        decrease_factor: float = 0.7,
        increase_step: float = 1.0,
        increase_after: int = 20,
        refractory_ms: float = 1_000.0,
        latency_slo_ms: float = 0.0,
        latency_min_rate: float | None = None,
        latency_samples: int = 100,
        latency_min_samples: int = 20,
        ttl_s: int = 300,
        key_prefix: str = "aimd",
    ) -> None:
        if max_rate <= 0 or min_rate <= 0:
            raise ValueError("rates must be positive")
        if min_rate > max_rate:
            raise ValueError("min_rate cannot exceed max_rate")
        if not 0 < decrease_factor < 1:
            raise ValueError("decrease_factor must be in (0, 1) to be a decrease")
        if increase_step <= 0 or increase_after < 1:
            raise ValueError("increase_step and increase_after must be positive")

        self._redis = redis
        self.endpoint = endpoint
        self.key = f"{key_prefix}:{endpoint}"
        self.latency_key = f"{key_prefix}:{endpoint}:lat"
        self.max_rate = max_rate
        self.min_rate = min_rate
        # Defaults to half the advertised rate: enough to relieve an endpoint we
        # might be overloading, bounded so an unattributable slowdown cannot
        # cost us everything.
        self.latency_min_rate = (
            max_rate * 0.5 if latency_min_rate is None else latency_min_rate
        )
        if not min_rate <= self.latency_min_rate <= max_rate:
            raise ValueError("latency_min_rate must sit between min_rate and max_rate")
        self._args = [
            min_rate,
            max_rate,
            decrease_factor,
            increase_step,
            increase_after,
            refractory_ms,
            ttl_s * 1000,
        ]
        self._latency_args = [
            latency_slo_ms,
            latency_samples,
            latency_min_samples,
            self.latency_min_rate,
        ]

    @property
    def _script(self) -> Any:
        return register_script("aimd", _AIMD_LUA)

    async def _run(self, op: str, latency_ms: float = -1.0) -> RateVerdict:
        rate_milli, changed, reason, p95, successes = await self._script(
            keys=[self.key, self.latency_key],
            args=[op, *self._args, latency_ms, *self._latency_args],
        )
        verdict = RateVerdict(
            rate=int(rate_milli) / 1000,
            changed=bool(changed),
            reason=reason if isinstance(reason, str) else reason.decode(),
            p95_ms=float(p95),
            successes=int(successes),
        )
        if verdict.changed:
            log.info(
                "rate_adapted",
                endpoint=self.endpoint,
                rate=round(verdict.rate, 2),
                max_rate=self.max_rate,
                reason=verdict.reason,
                p95_ms=verdict.p95_ms if verdict.p95_ms >= 0 else None,
            )
        return verdict

    async def current(self) -> float:
        """The rate to hand the token bucket for the next request."""
        return (await self._run("read")).rate

    async def on_success(self, latency_ms: float) -> RateVerdict:
        """Bank a success, and check p95 for an accepted-but-slow endpoint."""
        return await self._run("success", latency_ms=latency_ms)

    async def on_congestion(self, kind: Congestion) -> RateVerdict:
        """Report a load signal - 429 or timeout. NOT a 5xx; see the module
        docstring for why baseline failures must not drive this controller."""
        verdict = await self._run("congestion")
        log.debug(
            "congestion_reported",
            endpoint=self.endpoint,
            kind=kind.value,
            rate=round(verdict.rate, 2),
            reason=verdict.reason,
        )
        return verdict

    async def reset(self) -> None:
        """Back to the advertised rate. Tests and operator intervention only."""
        await self._redis.delete(self.key, self.latency_key)
