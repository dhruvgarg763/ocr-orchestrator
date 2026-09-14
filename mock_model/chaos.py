"""On-demand fault injection.

Duress that arrives 2% of the time at random is untestable, so a chaos
rule lets a test force e.g. "80% of VLM calls return 429 for 30 seconds"
and assert zero pages were dropped. Rules are time-boxed so a test that
crashes mid-run can't leave the mock permanently broken for every test
after it; expiry is checked lazily on read rather than by a background
reaper task.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ChaosRule:
    status: int
    """HTTP status to force, e.g. 429 to exercise backpressure, 500 for retries."""

    ratio: float
    """Fraction of requests affected, 0.0-1.0. Partial failure is the realistic
    case and the harder one to handle correctly."""

    extra_latency_ms: float
    """Added delay. Models a degraded-but-alive endpoint, which is what drives
    the adaptive limiter down in Step 10."""

    expires_at: float
    """Deadline on time.monotonic()."""

    def remaining_s(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())


class ChaosController:
    def __init__(self) -> None:
        self._rules: dict[str, ChaosRule] = {}

    def set(
        self,
        endpoint: str,
        *,
        status: int = 429,
        ratio: float = 1.0,
        seconds: float = 30.0,
        extra_latency_ms: float = 0.0,
    ) -> ChaosRule:
        rule = ChaosRule(
            status=status,
            ratio=ratio,
            extra_latency_ms=extra_latency_ms,
            expires_at=time.monotonic() + seconds,
        )
        self._rules[endpoint] = rule
        return rule

    def clear(self, endpoint: str | None = None) -> None:
        if endpoint is None:
            self._rules.clear()
        else:
            self._rules.pop(endpoint, None)

    def active(self, endpoint: str) -> ChaosRule | None:
        """Return the rule if one is live, expiring it lazily otherwise."""
        rule = self._rules.get(endpoint)
        if rule is None:
            return None
        if rule.remaining_s() <= 0:
            del self._rules[endpoint]
            return None
        return rule

    def roll(self, endpoint: str) -> ChaosRule | None:
        """Should *this* request be faulted? Applies the rule's ratio."""
        rule = self.active(endpoint)
        if rule is None:
            return None
        return rule if random.random() < rule.ratio else None

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            endpoint: {
                "status": rule.status,
                "ratio": rule.ratio,
                "extra_latency_ms": rule.extra_latency_ms,
                "expires_in_s": round(rule.remaining_s(), 2),
            }
            for endpoint in list(self._rules)
            if (rule := self.active(endpoint)) is not None
        }
