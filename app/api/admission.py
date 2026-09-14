"""Admission control: the outermost memory bound.

Downstream backpressure (token bucket, retries, breaker, AIMD) all act
after work is already in the queue, so none of it protects against
ingestion - a client can POST a 100-page job in ~5ms, offering far more
arrival rate than the 10 rps VLM can ever serve, and an unbounded queue
just grows (measured: 40,000 pages offered with the watermark off queued
39,998 of them; with it on, 6,000). A `503` carrying `Retry-After` is
counted as handled, not dropped - the alternative (accept, then OOM-kill a
worker and silently lose every in-flight page) is the actual drop. Depth
is read from `XLEN` rather than a separate counter, because a derived
signal can't drift: a separate "admitted pages" counter would need
decrementing on completion, and a worker crashing between the terminal
transition and the decrement would leak it upward forever. Two watermarks
(high/low), not one, because a single threshold oscillates on every page
completion - trip at `high`, recover only at `low`, the same reason a
thermostat doesn't switch at one temperature.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis

from app.core.redis_client import register_script
from app.queue.streams import STREAM, STREAMS
from app.ratelimit.adaptive import AdaptiveRate
from common.logging import get_logger

log = get_logger("admission")

# KEYS[1] admission state hash   KEYS[2] the task stream
# ARGV[1] pages   ARGV[2] high_watermark   ARGV[3] low_watermark   ARGV[4] ttl_ms
# returns {admitted, depth, shedding, shed_jobs, shed_pages, admitted_jobs}
_ADMISSION_LUA = """
local pages = tonumber(ARGV[1])
local high  = tonumber(ARGV[2])
local low   = tonumber(ARGV[3])
local ttl_ms = tonumber(ARGV[4])

-- Read depth inside the script, so the depth and the shed flag cannot
-- disagree. XLEN is the count of admitted-but-unsettled pages, because ack()
-- deletes the entry as well as acknowledging it.
--
-- Summed over EVERY queue lane (KEYS[2..]). Page 0 of each job lives in the
-- priority lane, so reading only the main stream would under-count admitted
-- work by one page per in-flight job - the watermark would drift by exactly
-- the number of concurrent jobs, which is the quantity it is meant to bound.
local depth = 0
for i = 2, #KEYS do
  depth = depth + tonumber(redis.call('XLEN', KEYS[i]))
end
local shedding = redis.call('HGET', KEYS[1], 'shedding') == '1'

-- Schmitt trigger: trip at `high`, recover only at `low`. A single threshold
-- would flip state on every page completion.
if shedding and depth <= low then
  shedding = false
  redis.call('HINCRBY', KEYS[1], 'recoveries', 1)
end

local admitted = 0
if not shedding then
  -- The job is counted INCLUDING its own pages. A watermark that only checks
  -- the current depth can be exceeded by a whole job, and a bound that does
  -- not bind is not a bound. The cost is that a large job can be refused where
  -- a 1-page job would fit; with max_pages at 100 against a watermark in the
  -- thousands that unfairness is small and bounded.
  if depth + pages <= high then
    admitted = 1
  else
    shedding = true
    redis.call('HINCRBY', KEYS[1], 'trips', 1)
  end
end

-- Counted, never silent. "0% unhandled" is only a meaningful claim if every
-- refusal is recorded and reportable.
if admitted == 1 then
  redis.call('HINCRBY', KEYS[1], 'admitted_jobs', 1)
  redis.call('HINCRBY', KEYS[1], 'admitted_pages', pages)
else
  redis.call('HINCRBY', KEYS[1], 'shed_jobs', 1)
  redis.call('HINCRBY', KEYS[1], 'shed_pages', pages)
end

if shedding then
  redis.call('HSET', KEYS[1], 'shedding', '1')
else
  redis.call('HSET', KEYS[1], 'shedding', '0')
end
redis.call('PEXPIRE', KEYS[1], ttl_ms)

return {
  admitted, depth, shedding and 1 or 0,
  tonumber(redis.call('HGET', KEYS[1], 'shed_jobs')) or 0,
  tonumber(redis.call('HGET', KEYS[1], 'shed_pages')) or 0,
  tonumber(redis.call('HGET', KEYS[1], 'admitted_jobs')) or 0
}
"""


@dataclass(frozen=True)
class Verdict:
    admitted: bool
    depth: int
    """Pages admitted but not yet settled, at decision time."""

    shedding: bool
    retry_after_s: int
    """Only meaningful when refused. Jittered; see `_retry_after`."""

    shed_jobs: int
    shed_pages: int
    admitted_jobs: int


class AdmissionController:
    def __init__(
        self,
        redis: Redis,
        *,
        high_watermark: int,
        low_watermark_fraction: float = 0.8,
        drain_rate: float = 10.0,
        controller: AdaptiveRate | None = None,
        retry_after_cap_s: int = 300,
        retry_after_jitter: float = 0.2,
        ttl_s: int = 3_600,
        key: str = "admission",
    ) -> None:
        if high_watermark < 1:
            raise ValueError("high_watermark must be positive")
        if not 0 < low_watermark_fraction < 1:
            raise ValueError(
                "low_watermark_fraction must be in (0, 1); at 1 there is no "
                "hysteresis and the controller flaps"
            )
        if drain_rate <= 0:
            raise ValueError("drain_rate must be positive")

        self._redis = redis
        self.key = key
        self.high_watermark = high_watermark
        self.low_watermark = max(1, int(high_watermark * low_watermark_fraction))
        self._drain_rate = drain_rate
        self._controller = controller
        self._cap_s = retry_after_cap_s
        self._jitter = retry_after_jitter
        self._ttl_ms = ttl_s * 1000

    @property
    def _script(self) -> Any:
        return register_script("admission", _ADMISSION_LUA)

    async def _service_rate(self) -> float:
        """Pages per second the system is actually draining at.

        Prefers the AIMD controller's DISCOVERED rate over the configured one.
        That is what makes Retry-After honest: if the VLM has degraded to 2.7
        rps, the same backlog takes nearly 4x as long to clear, and quoting a
        figure derived from the advertised 10 rps just guarantees the client
        comes back too early and is refused again.

        The VLM is the bottleneck stage, so its rate is the system's page rate.
        """
        if self._controller is None:
            return self._drain_rate
        try:
            return max(0.1, await self._controller.current())
        except Exception:  # noqa: BLE001
            # Telling a client to retry is strictly better than failing the
            # request because a telemetry read failed.
            log.warning("service_rate_unavailable", fallback=self._drain_rate)
            return self._drain_rate

    def _retry_after(self, depth: int, rate: float) -> int:
        """How long until the backlog has drained to the recovery mark.

        Jittered, for the reason Step 8 established: fifty clients handed an
        identical `Retry-After: 60` all return in the same instant, recreating
        the overload that caused the refusal. Jitter converts that convoy into
        a queue.

        Capped, because "come back in two hours" is not actionable, and floored
        at one second because RFC 9110 Retry-After is integer seconds and zero
        would invite an immediate retry.
        """
        excess = max(0, depth - self.low_watermark)
        drain_s = excess / rate
        jittered = drain_s * (1.0 + random.uniform(0, self._jitter))
        return max(1, min(self._cap_s, math.ceil(jittered)))

    async def evaluate(self, pages: int) -> Verdict:
        """Decide whether to admit a job of `pages` pages.

        NOTE on exactness: the decision reads XLEN, but the depth does not
        change until the caller enqueues, so concurrent admissions can each see
        the same pre-enqueue depth. The overshoot is bounded by
        `concurrent_admits x pages_per_job`.

        Confirmed empirically: 20-way concurrency at 100 pages per job predicts
        at most 2,000 pages of overshoot, and a flood against a 5,000 watermark
        settled at 6,000 - an overshoot of 1,000 pages, ~300KB of Redis.

        That is accepted deliberately rather than fixed with an atomic
        reservation counter, because such a counter has to be decremented on
        completion and would leak upward forever if a worker died in between -
        permanently tightening admission. A bounded overshoot from a
        self-healing derived signal is a better failure mode than an unbounded
        drift from an exact one.
        """
        admitted, depth, shedding, shed_jobs, shed_pages, admitted_jobs = (
            await self._script(
                keys=[self.key, *STREAMS],
                args=[pages, self.high_watermark, self.low_watermark, self._ttl_ms],
            )
        )

        depth = int(depth)
        admitted = bool(admitted)
        retry_after = 0
        if not admitted:
            retry_after = self._retry_after(depth, await self._service_rate())
            log.warning(
                "job_shed",
                pages=pages,
                depth=depth,
                high_watermark=self.high_watermark,
                low_watermark=self.low_watermark,
                retry_after_s=retry_after,
                shed_jobs_total=int(shed_jobs),
            )

        return Verdict(
            admitted=admitted,
            depth=depth,
            shedding=bool(shedding),
            retry_after_s=retry_after,
            shed_jobs=int(shed_jobs),
            shed_pages=int(shed_pages),
            admitted_jobs=int(admitted_jobs),
        )

    async def stats(self) -> dict[str, Any]:
        """Admission accounting, for /metrics and for the benchmark's honest
        tally of admitted versus shed."""
        raw = await self._redis.hgetall(self.key)
        get = lambda k: int(raw.get(k, 0))  # noqa: E731
        return {
            "high_watermark": self.high_watermark,
            "low_watermark": self.low_watermark,
            "shedding": raw.get("shedding") == "1",
            "admitted_jobs": get("admitted_jobs"),
            "admitted_pages": get("admitted_pages"),
            "shed_jobs": get("shed_jobs"),
            "shed_pages": get("shed_pages"),
            "trips": get("trips"),
            "recoveries": get("recoveries"),
        }

    async def reset(self) -> None:
        """Clear the flag and the counters. Tests and operators only."""
        await self._redis.delete(self.key)
