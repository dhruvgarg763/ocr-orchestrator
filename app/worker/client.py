"""HTTP client for the mock inference endpoints.

Rate limiting is enforced HERE rather than in the pipeline, deliberately: one
choke point that every call must pass through cannot be forgotten at a new call
site. A limiter applied by the caller is a limiter someone eventually skips.

Four gates guard every call, and their ORDER is the important thing:

  1. circuit breaker   is this endpoint answering at all? Cheapest check, so it
                       runs first - an open circuit must not consume a
                       rate-limit slot a healthy endpoint could have used.
  2. AIMD setpoint     how fast may we go, given what we have observed? Supplies
                       the rate; it does not enforce it.
  3. token bucket      enforces that rate across every replica, reserving a slot
                       and waiting rather than rejecting.
  4. retries           bounded, classified, full-jitter backoff, honouring
                       Retry-After.

On the way back, the outcome is reported to both the breaker (is it alive?) and
the controller (how fast may I go?) - deliberately different questions, which is
why a 5xx counts for the first and not the second.

Two things have been here from the start, because retrofitting them is invasive:

  * `Idempotency-Key`, derived deterministically from (job, page, stage). At
    least-once delivery guarantees a page will sometimes be sent twice; this is
    what makes the second send free instead of a duplicate 3s inference.
  * `traceparent`, so one trace id spans api -> worker -> mock. ContextVars do
    not cross a process boundary; a header does.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from types import TracebackType
from typing import Any, Mapping

import httpx

from app.ratelimit.adaptive import AdaptiveRate, Congestion
from app.ratelimit.breaker import BreakerState, CircuitBreaker, CircuitOpen
from app.ratelimit.retry import Disposition, backoff_delay_s, classify, retry_after_ms
from app.ratelimit.token_bucket import TokenBucketLimiter
from common.logging import get_logger
from common.tracing import format_traceparent

log = get_logger("model-client")

LAYOUT_PATH = "/v1/predict/layout"
VLM_PATH = "/v1/predict/vlm"

LAYOUT = "layout"
VLM = "vlm"


def _describe(exc: BaseException) -> str:
    """Compact, greppable error label for structured logs.

    An HTTP status is the single most useful fact about a failure, so it is
    hoisted into the label rather than buried in a stack trace.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def idempotency_key(job_id: str, page_index: int, stage: str) -> str:
    """Stable key for one (page, stage) unit of work.

    hashlib, never builtin hash(): hash() on str is salted per process, so a
    worker restart would produce a different key for the same work and defeat
    the entire mechanism in exactly the crash scenario it exists for.
    """
    return hashlib.sha256(f"{job_id}:{page_index}:{stage}".encode()).hexdigest()[:32]


class ModelClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float,
        connect_timeout_s: float,
        max_connections: int,
        limiters: Mapping[str, TokenBucketLimiter] | None = None,
        breakers: Mapping[str, CircuitBreaker] | None = None,
        controllers: Mapping[str, AdaptiveRate] | None = None,
        rate_limit_max_wait_s: float = 30.0,
        max_attempts: int = 3,
        backoff_base_ms: float = 200.0,
        backoff_max_ms: float = 5_000.0,
        transport: httpx.AsyncBaseTransport | None = None,
        metrics: Any = None,
    ) -> None:
        """`transport` is a seam for tests.

        The retry loop's behaviour - how many requests it issues, which headers
        it repeats, how it reads Retry-After - is only observable from the wire.
        Injecting a transport lets that be asserted directly instead of inferred
        from a running server's logs.
        """
        # Optional so unit tests can exercise the HTTP path without Redis. In
        # the worker they are always supplied.
        self._metrics = metrics
        """Optional so the unit tests can drive the HTTP path with no Redis.
        Every call site below is None-guarded for that reason, not because
        metrics are optional in the worker - there they are always wired."""

        self._limiters = dict(limiters or {})
        self._breakers = dict(breakers or {})
        self._controllers = dict(controllers or {})
        self._max_wait_s = rate_limit_max_wait_s
        self.rate_limit_wait_s = 0.0
        """Cumulative time spent waiting for tokens. Surfaced as a metric in
        Step 18: it is the honest cost of backpressure, and distinguishes
        "throttled" from "slow model"."""

        self._max_attempts = max_attempts
        self._backoff_base_ms = backoff_base_ms
        self._backoff_max_ms = backoff_max_ms

        self.retries = 0
        """Retry attempts issued. Rising retries with stable goodput means the
        system is absorbing faults; rising retries with falling goodput means it
        is thrashing."""
        self._last_state: dict[str, BreakerState] = {}
        """Last observed breaker state per endpoint, so recovery is logged once
        rather than on every subsequent success."""

        self.retry_successes = 0
        """Calls that succeeded only because of a retry - i.e. pages that would
        have been dropped before Step 8."""

        self._client = httpx.AsyncClient(
            base_url=base_url,
            # Separate connect timeout: failing to *reach* the service should be
            # detected in ~2s, while a legitimately slow VLM gets the full budget.
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            # A bounded pool is a backpressure lever in its own right - once
            # every connection is busy, the next call waits instead of opening
            # unbounded sockets.
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ModelClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _post(
        self,
        path: str,
        stage: str,
        job_id: str,
        page_index: int,
        page_ref: str | None,
        page_content: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Breaker FIRST, before the rate limiter. Order matters: an open circuit
        # means we are not going to send the request at all, so reserving a
        # rate-limit slot for it would hold capacity that a healthy endpoint
        # could have used. Checking the cheap gate before the expensive one.
        breaker = self._breakers.get(stage)
        if breaker is not None:
            verdict = await breaker.allow()
            if not verdict.allowed:
                raise CircuitOpen(stage, verdict.retry_after_s)

        # Ask permission BEFORE sending. The whole point is to stop issuing
        # requests we already know will be rejected: a 429 costs the endpoint
        # real work and destroys a page, while waiting for a token costs only
        # time. This is the difference between load shedding and backpressure.
        # The AIMD controller supplies the bucket's SETPOINT. The bucket still
        # does the enforcing - shaping traffic, reserving slots, holding the
        # limit across replicas - it just no longer assumes the advertised rate
        # is the achievable one.
        #
        # One extra Redis round trip per request, which at 10 rps is ~10 ops/sec
        # against a server that does 100k. Caching the rate locally for a few
        # hundred ms would remove it, at the cost of a decrease taking that long
        # to bite; not worth the staleness for a saving this small.
        controller = self._controllers.get(stage)
        rate = await controller.current() if controller is not None else None

        limiter = self._limiters.get(stage)
        if limiter is not None:
            waited = await limiter.acquire(max_wait_s=self._max_wait_s, rate=rate)
            if waited > 0:
                self.rate_limit_wait_s += waited
                log.debug(
                    "rate_limit_wait",
                    endpoint=stage,
                    waited_ms=round(waited * 1000, 1),
                    job_id=job_id,
                    page_index=page_index,
                )
                if self._metrics is not None:
                    self._metrics.record_rate_limit_wait(stage, waited)

        started = time.monotonic()
        try:
            response = await self._client.post(
                path,
                json={
                    "job_id": job_id,
                    "page_index": page_index,
                    # A REFERENCE plus a small descriptor, never the page bytes.
                    # This is what keeps the request body O(1) in page size and
                    # the queue entry O(1) in document size.
                    "page_ref": page_ref,
                    "page": page_content,
                },
                headers={
                    # The SAME key on every attempt. That is what makes retrying
                    # a timeout safe: if the original request did land and only
                    # the response was lost, the retry is served from the
                    # server's cache instead of re-running the inference.
                    "Idempotency-Key": idempotency_key(job_id, page_index, stage),
                    "traceparent": format_traceparent(),
                },
            )
            response.raise_for_status()
        except BaseException as exc:
            if self._metrics is not None:
                # Outcome is bucketed, not free-form: an unbounded `outcome`
                # label (an exception message, say) would create a new
                # Prometheus series per distinct error string, which is the
                # classic way to take down a monitoring system with cardinality.
                if isinstance(exc, httpx.HTTPStatusError):
                    code = exc.response.status_code
                    outcome = "429" if code == 429 else f"http_{code // 100}xx"
                elif isinstance(exc, httpx.TimeoutException):
                    outcome = "timeout"
                else:
                    outcome = "error"
                self._metrics.record_model_call(
                    stage, outcome, time.monotonic() - started
                )

            if controller is not None:
                # ONLY load signals steer the rate. 429 means "too fast" in so
                # many words; a timeout means the endpoint is past the point of
                # even rejecting cleanly.
                #
                # 5xx is deliberately absent, and that is the important line in
                # this method. The mock emits a 5% baseline failure rate that
                # has nothing to do with load. Counting those as congestion
                # would trigger a decrease roughly every 20 requests - about as
                # often as a success streak earns an increase - so the rate
                # would be dragged permanently below the achievable one by noise
                # the endpoint was always going to produce. Sustained 5xx is the
                # breaker's problem; this controller answers "how fast may I
                # go", not "is it alive".
                if isinstance(exc, httpx.HTTPStatusError):
                    if exc.response.status_code == 429:
                        await controller.on_congestion(Congestion.REJECTED)
                elif isinstance(exc, httpx.TimeoutException):
                    await controller.on_congestion(Congestion.TIMEOUT)

            if breaker is not None:
                # Only PERMANENT and RETRY count against the breaker. A 400 is a
                # bug in our request, not evidence the endpoint is unhealthy -
                # but it is still a failed call, and excluding it would let a
                # systematically malformed request go unnoticed. Saturation is
                # excluded because no request was made.
                if classify(exc) is not Disposition.SATURATED:
                    verdict = await breaker.record_failure()
                    if verdict.state is BreakerState.OPEN:
                        log.warning(
                            "circuit_opened",
                            endpoint=stage,
                            failures=verdict.failures,
                            successes=verdict.successes,
                            retry_after_s=verdict.retry_after_s,
                        )
            raise

        if self._metrics is not None:
            self._metrics.record_model_call(
                stage, "success", time.monotonic() - started
            )

        if controller is not None:
            # Latency is fed in on success only: a failed request's duration
            # measures how fast the endpoint says no, which is not the quantity
            # we are trying to control. The controller checks p95 against its
            # SLO here, which is how an accepted-but-slow endpoint - invisible
            # to both the bucket and the breaker - still produces a decrease.
            await controller.on_success((time.monotonic() - started) * 1000)

        if breaker is not None:
            verdict = await breaker.record_success()
            if verdict.state is BreakerState.CLOSED and self._last_state.get(
                stage
            ) is not BreakerState.CLOSED:
                log.info("circuit_closed", endpoint=stage)
            self._last_state[stage] = verdict.state

        return response.json()

    async def _post_with_retries(
        self,
        path: str,
        stage: str,
        job_id: str,
        page_index: int,
        page_ref: str | None,
        page_content: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Bounded retries with classification and full-jitter backoff.

        Retries happen here rather than at the queue level because these faults
        are short-lived - a transient 5xx or a straggler 429 resolves in
        milliseconds. Returning the page to the queue would cost a redelivery
        round trip and lose the stage progress already committed.

        SATURATED is the exception: no rate-limit token after the full wait
        budget means the constraint is systemic, and retrying in place would
        pin this page while the worker could be serving others. It propagates
        so the caller can requeue.
        """
        last_exc: BaseException | None = None

        for attempt in range(self._max_attempts):
            try:
                result = await self._post(
                    path, stage, job_id, page_index, page_ref, page_content
                )
                if attempt > 0:
                    self.retry_successes += 1
                    log.info(
                        "retry_succeeded",
                        endpoint=stage,
                        job_id=job_id,
                        page_index=page_index,
                        attempt=attempt + 1,
                    )
                return result

            except BaseException as exc:  # noqa: BLE001 - classified below
                last_exc = exc
                disposition = classify(exc)

                if disposition in (Disposition.SATURATED, Disposition.CIRCUIT_OPEN):
                    # Neither is solved by waiting longer here. Saturation needs
                    # a requeue; an open circuit needs a degraded fallback. Both
                    # decisions belong to the pipeline, which knows whether the
                    # page has a usable partial result. Retrying an open circuit
                    # would defeat the breaker entirely.
                    raise

                if disposition is Disposition.PERMANENT:
                    log.warning(
                        "permanent_failure",
                        endpoint=stage,
                        job_id=job_id,
                        page_index=page_index,
                        error=_describe(exc),
                    )
                    raise

                if attempt == self._max_attempts - 1:
                    # Budget spent. Propagated so the pipeline can decide: at
                    # the VLM stage this becomes a degraded FALLBACK_DONE built
                    # from the already-committed layout result, so the page is
                    # not lost. This is the exhausted-retries case the spec
                    # names as the trigger for falling back.
                    log.warning(
                        "retries_exhausted",
                        endpoint=stage,
                        job_id=job_id,
                        page_index=page_index,
                        attempts=self._max_attempts,
                        error=_describe(exc),
                    )
                    raise

                delay_s = backoff_delay_s(
                    attempt,
                    base_ms=self._backoff_base_ms,
                    max_ms=self._backoff_max_ms,
                    retry_after_ms_hint=retry_after_ms(exc),
                )
                self.retries += 1
                if self._metrics is not None:
                    self._metrics.record_retry(stage)
                log.info(
                    "retrying",
                    endpoint=stage,
                    job_id=job_id,
                    page_index=page_index,
                    attempt=attempt + 1,
                    of=self._max_attempts,
                    delay_ms=round(delay_s * 1000, 1),
                    error=_describe(exc),
                )
                await asyncio.sleep(delay_s)

        # Unreachable: the final attempt either returns or raises above.
        raise last_exc if last_exc else RuntimeError("retry loop fell through")

    async def layout(
        self,
        job_id: str,
        page_index: int,
        *,
        page_ref: str | None = None,
        page_content: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._post_with_retries(
            LAYOUT_PATH, LAYOUT, job_id, page_index, page_ref, page_content
        )

    async def vlm(
        self,
        job_id: str,
        page_index: int,
        *,
        page_ref: str | None = None,
        page_content: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._post_with_retries(
            VLM_PATH, VLM, job_id, page_index, page_ref, page_content
        )
