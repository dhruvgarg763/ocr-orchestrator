"""Worker-side metrics: instrument the events, derive the rest.

Model call outcomes/duration, rate-limit waits, retries, and page duration
are instrumented directly at the call site since nothing else counts
them. Page outcomes, reaper actions, and in-flight count are DERIVED at
flush time from state that already has an owner (`WorkerStats`,
`ReaperStats`, `Worker._held`), rather than duplicated into a parallel
counter that could drift from it. `sync()` pushes the DIFFERENCE since
the last flush, because those objects are absolute running totals while
`common.metrics.MetricsSink` flushes counters as deltas into a shared
Redis hash - feeding it absolutes would report the full lifetime total on
every flush. A worker restart loses at most one flush interval either
way, and the fleet-wide total never goes backwards, which is what
Prometheus needs from a counter.
"""

from __future__ import annotations

from typing import Any, Mapping

from common.metrics import MetricsSink, Registry

__all__ = ["WorkerMetrics"]

# Which per-attempt outcomes end a page's life, and how the zero-drop metric
# should read them. Outcomes absent from this table are deliberately
# non-terminal: HANDOFF releases the slot mid-page, SATURATED leaves the page
# untouched for requeue, OWNED_BY_OTHER means this attempt was not the one that
# proceeded. Counting any of them as terminal would inflate the denominator of
# the zero-drop calculation with work that had not finished.
_TERMINAL_RESULT: dict[str, str] = {
    "completed": "success",
    "resumed": "success",
    "degraded": "success",
    "failed": "failed",
    "already_terminal": "duplicate",
}


class WorkerMetrics:
    """Everything a worker exports, plus the flush that ships it."""

    def __init__(
        self,
        redis: Any,
        *,
        worker: str,
        gauge_ttl_s: float,
        prefetch_limit: int,
    ) -> None:
        self.worker = worker
        self.registry = Registry()

        # --- instrumented at the call site
        self.model_requests = self.registry.counter(
            "orch_model_requests_total",
            "Model calls attempted, by endpoint and outcome.",
            ["endpoint", "outcome"],
        )
        self.model_duration = self.registry.histogram(
            "orch_model_request_duration_seconds",
            "Latency of one model call, excluding the rate-limit wait.",
            ["endpoint"],
        )
        self.retries = self.registry.counter(
            "orch_model_retries_total",
            "Retry attempts issued, by endpoint.",
            ["endpoint"],
        )
        self.rate_limit_waits = self.registry.counter(
            "orch_rate_limit_waits_total",
            "Times a worker blocked waiting for a rate-limit token.",
            ["endpoint"],
        )
        self.rate_limit_wait_seconds = self.registry.counter(
            "orch_rate_limit_wait_seconds_total",
            "Cumulative time spent waiting for tokens - the cost of backpressure.",
            ["endpoint"],
        )
        self.page_duration = self.registry.histogram(
            "orch_page_duration_seconds",
            "Wall time from claiming a page to reaching a terminal state.",
        )

        # --- derived at flush from objects that already own the number
        self.page_attempts = self.registry.counter(
            "orch_page_attempts_total",
            "Page processing attempts, by per-attempt outcome.",
            ["outcome"],
        )
        """Attempts, NOT pages, and the distinction is not pedantic.

        `Outcome` describes one pass through the pipeline, and a two-stage page
        deliberately produces two: HANDOFF when layout commits and the slot is
        released, then RESUMED when the VLM stage runs. A 15-page job therefore
        records 30 attempts and zero `completed` - measured, and the reason this
        counter was renamed. Read as "pages" it would report double the traffic
        and no successes, which is worse than having no counter at all."""

        self.pages_terminal = self.registry.counter(
            "orch_pages_terminal_total",
            "Pages reaching a terminal state, classified for the zero-drop metric.",
            ["result"],
        )
        """The graded view, derived from the same outcomes so the two cannot drift.

        `success` covers RESUMED, COMPLETED and DEGRADED - a degraded page
        produced usable output at reduced fidelity, which the zero-drop
        guarantee counts as served, not dropped. `failed` is FAILED alone.
        `duplicate` is ALREADY_TERMINAL, which at-least-once delivery makes
        routine and which must not be counted as new work."""
        self.reaper_actions = self.registry.counter(
            "orch_reaper_actions_total",
            "Orphaned entries handled by the reaper, by action.",
            ["action"],
        )
        self.in_flight = self.registry.gauge(
            "orch_worker_in_flight",
            "Pages this worker is currently processing.",
            ["worker"],
        )
        self.prefetch_limit_gauge = self.registry.gauge(
            "orch_worker_prefetch_limit",
            "Configured concurrency ceiling for this worker.",
            ["worker"],
        )
        self.prefetch_limit_gauge.set(float(prefetch_limit), worker=worker)

        self._sink = MetricsSink(
            redis, self.registry, worker=worker, gauge_ttl_s=gauge_ttl_s
        )
        self._last_outcomes: dict[str, int] = {}
        self._last_reaper: dict[str, int] = {}

    # ------------------------------------------------------------ call sites

    def record_model_call(
        self, endpoint: str, outcome: str, duration_s: float
    ) -> None:
        self.model_requests.inc(endpoint=endpoint, outcome=outcome)
        # Duration is recorded for failures too. A 429 returns in a millisecond
        # and a timeout takes the full budget; averaging only successes would
        # hide the second and make the endpoint look healthier under load than
        # it is.
        self.model_duration.observe(duration_s, endpoint=endpoint)

    def record_retry(self, endpoint: str) -> None:
        self.retries.inc(endpoint=endpoint)

    def record_rate_limit_wait(self, endpoint: str, waited_s: float) -> None:
        self.rate_limit_waits.inc(endpoint=endpoint)
        self.rate_limit_wait_seconds.inc(waited_s, endpoint=endpoint)

    def record_page(self, duration_s: float) -> None:
        self.page_duration.observe(duration_s)

    # --------------------------------------------------------------- derived

    def sync(
        self,
        *,
        outcomes: Mapping[str, int],
        reaper: Mapping[str, int],
        in_flight: int,
    ) -> None:
        """Fold absolute running totals into counter deltas.

        Guarded with `max(0, ...)`: a caller passing a total that went
        BACKWARDS (a reset stats object, a test reusing this instance) would
        otherwise push a negative delta into Redis and make the fleet counter
        non-monotonic - the one property counters must not lose. Clamping turns
        that into a dropped sample instead of corrupted history.
        """
        for outcome, total in outcomes.items():
            delta = max(0, int(total) - self._last_outcomes.get(outcome, 0))
            if delta:
                self.page_attempts.inc(delta, outcome=outcome)
                result = _TERMINAL_RESULT.get(outcome)
                if result is not None:
                    self.pages_terminal.inc(delta, result=result)
            self._last_outcomes[outcome] = int(total)

        for action, total in reaper.items():
            delta = max(0, int(total) - self._last_reaper.get(action, 0))
            if delta:
                self.reaper_actions.inc(delta, action=action)
            self._last_reaper[action] = int(total)

        self.in_flight.set(float(in_flight), worker=self.worker)

    async def flush(
        self,
        *,
        outcomes: Mapping[str, int],
        reaper: Mapping[str, int],
        in_flight: int,
    ) -> None:
        self.sync(outcomes=outcomes, reaper=reaper, in_flight=in_flight)
        await self._sink.flush()
