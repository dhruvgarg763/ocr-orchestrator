"""Distributed circuit breaker, evaluated inside Redis.

Retries handle sparse faults, not a sustained outage: under an 80%
rejection rate, buying 99% success needs 21 attempts against an endpoint
already rejecting everything. `CLOSED -> OPEN` on a failure-ratio
threshold (gated by a minimum call volume, so 1 failure in 2 calls can't
trip it), `OPEN -> HALF_OPEN` after a cooldown, `HALF_OPEN -> CLOSED`
after N consecutive probe successes or back to `OPEN` on any failure.
`HALF_OPEN` rather than closing on a timer, because a timer sends full
production load at a service that may still be dead. State lives in
Redis so one replica's discovery of an outage stops all of them, rather
than each independently re-discovering it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from redis.asyncio import Redis

from app.core.redis_client import register_script
from common.logging import get_logger

log = get_logger("breaker")


class BreakerState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpen(Exception):
    """Raised instead of calling an endpoint whose breaker is open.

    A distinct type from RateLimitTimeout, because the correct response is
    different. Saturation means "busy, come back shortly" - so the page is
    requeued and keeps its full fidelity. An open circuit means "this endpoint
    is not answering", and waiting does not help: the page should degrade to
    whatever output it already has.
    """

    def __init__(self, endpoint: str, retry_after_s: float) -> None:
        super().__init__(f"{endpoint}: circuit open, retry in {retry_after_s:.1f}s")
        self.endpoint = endpoint
        self.retry_after_s = retry_after_s


# KEYS[1] breaker hash
# ARGV[1]  op: "allow" | "success" | "failure" | "peek"
# ARGV[2]  window_ms       ARGV[3] min_volume     ARGV[4] failure_ratio
# ARGV[5]  cooldown_ms     ARGV[6] max_probes     ARGV[7] probe_successes
# ARGV[8]  ttl_ms
# returns {allowed, state, retry_after_ms, failures, successes}
_BREAKER_LUA = """
local op              = ARGV[1]
local window_ms       = tonumber(ARGV[2])
local min_volume      = tonumber(ARGV[3])
local failure_ratio   = tonumber(ARGV[4])
local cooldown_ms     = tonumber(ARGV[5])
local max_probes      = tonumber(ARGV[6])
local probe_successes = tonumber(ARGV[7])
local ttl_ms          = tonumber(ARGV[8])

-- One clock for every replica, same reasoning as the token bucket: a
-- container running fast must not be able to expire another's cooldown early.
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + tonumber(t[2]) / 1000

local h = redis.call('HMGET', KEYS[1],
  'state', 'window_start', 'failures', 'successes',
  'opened_at', 'probes', 'probe_ok')

local state        = h[1] or 'CLOSED'
local window_start = tonumber(h[2]) or now
local failures     = tonumber(h[3]) or 0
local successes    = tonumber(h[4]) or 0
local opened_at    = tonumber(h[5]) or 0
local probes       = tonumber(h[6]) or 0
local probe_ok     = tonumber(h[7]) or 0

-- Roll the window while closed. Only meaningful in CLOSED: while OPEN the
-- counters are frozen evidence, and while HALF_OPEN the probe counters govern.
if state == 'CLOSED' and (now - window_start) >= window_ms then
  window_start = now
  failures = 0
  successes = 0
end

-- An OPEN breaker whose cooldown has elapsed becomes HALF_OPEN on the next
-- touch. Done here rather than on a timer so no background task is needed.
if state == 'OPEN' and (now - opened_at) >= cooldown_ms then
  state = 'HALF_OPEN'
  probes = 0
  probe_ok = 0
end

local allowed = 0
local retry_after_ms = 0

if op == 'allow' then
  if state == 'OPEN' then
    retry_after_ms = math.ceil(cooldown_ms - (now - opened_at))
    if retry_after_ms < 0 then retry_after_ms = 0 end
  elseif state == 'HALF_OPEN' then
    if probes < max_probes then
      -- Admit a probe. Counted so that concurrent callers across all replicas
      -- cannot collectively send a flood of "one" probe each.
      probes = probes + 1
      allowed = 1
    else
      -- Probe slots taken; the outcome of those decides. Come back soon rather
      -- than after a full cooldown.
      retry_after_ms = math.ceil(cooldown_ms / 4)
    end
  else
    allowed = 1
  end

elseif op == 'peek' then
  -- Report what `allow` WOULD say, without consuming a probe slot. Explicit
  -- rather than falling through the dispatch, so /metrics does not read a
  -- permanently false `allowed`.
  --
  -- Note it can still perform the lazy OPEN -> HALF_OPEN and window-roll
  -- transitions above. Those are expiry, not consumption: they would have
  -- happened on the next real call anyway, and deferring them would make peek
  -- report a state that no longer exists.
  if state == 'OPEN' then
    retry_after_ms = math.ceil(cooldown_ms - (now - opened_at))
    if retry_after_ms < 0 then retry_after_ms = 0 end
  elseif state == 'HALF_OPEN' then
    if probes < max_probes then allowed = 1 end
  else
    allowed = 1
  end

elseif op == 'success' then
  if state == 'HALF_OPEN' then
    probe_ok = probe_ok + 1
    if probes > 0 then probes = probes - 1 end
    if probe_ok >= probe_successes then
      -- Recovered. Counters reset so stale failures cannot immediately reopen.
      state = 'CLOSED'
      window_start = now
      failures = 0
      successes = 0
      probes = 0
      probe_ok = 0
    end
  else
    successes = successes + 1
  end
  allowed = 1

elseif op == 'failure' then
  if state == 'HALF_OPEN' then
    -- A single probe failure reopens immediately. The probe existed precisely
    -- to answer this question, and the answer is no.
    state = 'OPEN'
    opened_at = now
    probes = 0
    probe_ok = 0
    retry_after_ms = cooldown_ms
  else
    failures = failures + 1
    local total = failures + successes
    -- min_volume guard: 1 failure out of 2 requests is noise, not a signal.
    if total >= min_volume and (failures / total) >= failure_ratio then
      state = 'OPEN'
      opened_at = now
      probes = 0
      probe_ok = 0
      retry_after_ms = cooldown_ms
    end
  end
end

redis.call('HSET', KEYS[1],
  'state', state, 'window_start', window_start,
  'failures', failures, 'successes', successes,
  'opened_at', opened_at, 'probes', probes, 'probe_ok', probe_ok)
redis.call('PEXPIRE', KEYS[1], ttl_ms)

return {allowed, state, math.floor(retry_after_ms), failures, successes}
"""


@dataclass(frozen=True)
class BreakerVerdict:
    allowed: bool
    state: BreakerState
    retry_after_s: float
    failures: int
    successes: int


class CircuitBreaker:
    def __init__(
        self,
        redis: Redis,
        endpoint: str,
        *,
        window_s: float = 10.0,
        min_volume: int = 10,
        failure_ratio: float = 0.5,
        cooldown_s: float = 5.0,
        max_probes: int = 3,
        probe_successes: int = 2,
        ttl_s: int = 300,
        key_prefix: str = "cb",
    ) -> None:
        if not 0 < failure_ratio <= 1:
            raise ValueError("failure_ratio must be in (0, 1]")
        if min_volume < 1 or max_probes < 1 or probe_successes < 1:
            raise ValueError("min_volume, max_probes, probe_successes must be >= 1")

        self._redis = redis
        self.endpoint = endpoint
        self.key = f"{key_prefix}:{endpoint}"
        self._args = [
            window_s * 1000,
            min_volume,
            failure_ratio,
            cooldown_s * 1000,
            max_probes,
            probe_successes,
            ttl_s * 1000,
        ]

    @property
    def _script(self) -> Any:
        return register_script("circuit_breaker", _BREAKER_LUA)

    async def _run(self, op: str) -> BreakerVerdict:
        allowed, state, retry_ms, failures, successes = await self._script(
            keys=[self.key], args=[op, *self._args]
        )
        return BreakerVerdict(
            allowed=bool(allowed),
            state=BreakerState(state),
            retry_after_s=int(retry_ms) / 1000,
            failures=int(failures),
            successes=int(successes),
        )

    async def allow(self) -> BreakerVerdict:
        """Ask permission. Records a probe slot if the state is HALF_OPEN."""
        return await self._run("allow")

    async def record_success(self) -> BreakerVerdict:
        return await self._run("success")

    async def record_failure(self) -> BreakerVerdict:
        return await self._run("failure")

    async def peek(self) -> BreakerVerdict:
        """Read state without recording anything, for /metrics and tests."""
        return await self._run("peek")

    async def reset(self) -> None:
        """Force closed. Tests and operator intervention only."""
        await self._redis.delete(self.key)
