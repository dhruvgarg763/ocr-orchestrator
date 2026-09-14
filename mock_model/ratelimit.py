"""In-process token bucket for the mock inference server.

Refill is computed lazily from elapsed time, not a background timer:
O(1) per call, exact rather than tick-quantised. Deliberately in-process,
which is correct here because the mock is a single server enforcing its
own published limit - the orchestrator's client-side limiter can't work
this way (N replicas would give N x the intended rate; see
app/ratelimit/token_bucket.py). Correctness assumes one process per
bucket, which is why the mock runs single-worker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    rate: float
    """Sustained refill, tokens per second."""

    burst: int
    """Bucket capacity: the largest instantaneous spike tolerated."""

    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rate <= 0 or self.burst <= 0:
            raise ValueError("rate and burst must be positive")
        self._tokens = float(self.burst)  # start full: an idle client may burst
        # monotonic(), never time(): the wall clock can step backwards (NTP
        # correction, DST, manual change). A negative `elapsed` would either
        # mint free tokens or stall the bucket until the clock caught up.
        self._last_refill = time.monotonic()

    def _refill(self, now: float) -> None:
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(float(self.burst), self._tokens + elapsed * self.rate)
            self._last_refill = now

    def acquire(self, cost: float = 1.0) -> float:
        """Try to take `cost` tokens.

        Returns 0.0 when granted, otherwise the seconds until enough tokens
        exist. Returning a wait time rather than a bool lets the caller sleep
        exactly long enough instead of spin-polling - the same reason the Redis
        version in Step 7 returns wait_ms.

        Thread/task safety: the read-modify-write below contains no `await`, so
        no other coroutine can interleave with it - asyncio only switches tasks
        at suspension points. An asyncio.Lock would therefore be pure overhead.
        If an `await` is ever introduced into this method, that ceases to be
        true and a lock becomes mandatory.
        """
        now = time.monotonic()
        self._refill(now)

        if self._tokens >= cost:
            self._tokens -= cost
            return 0.0

        return (cost - self._tokens) / self.rate

    @property
    def remaining(self) -> int:
        """Whole tokens available now, for X-RateLimit-Remaining."""
        self._refill(time.monotonic())
        return int(self._tokens)
