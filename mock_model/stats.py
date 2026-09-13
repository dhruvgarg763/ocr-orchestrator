"""Per-endpoint counters, exposed via GET /admin/call-counts.

These are the mock's side of the story. The orchestrator will report its own
metrics in Step 18, and the interesting assertions are cross-checks between the
two: e.g. orchestrator "pages completed" == mock "executions", proving no page
was double-processed or silently skipped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EndpointStats:
    requests: int = 0
    """Every inbound request, including ones rejected before doing work."""

    executions: int = 0
    """Requests where the model actually ran (idempotency cache miss)."""

    replays: int = 0
    """Requests served from the idempotency cache."""

    rate_limited: int = 0
    """429s returned by the token bucket."""

    injected_failures: int = 0
    """5xx returned by the random failure rate or a chaos rule."""

    chaos_rejections: int = 0
    """Requests faulted specifically by an active chaos rule."""


@dataclass
class Stats:
    endpoints: dict[str, EndpointStats] = field(default_factory=dict)

    def of(self, endpoint: str) -> EndpointStats:
        return self.endpoints.setdefault(endpoint, EndpointStats())

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {name: asdict(s) for name, s in self.endpoints.items()}

    def reset(self) -> None:
        self.endpoints.clear()
