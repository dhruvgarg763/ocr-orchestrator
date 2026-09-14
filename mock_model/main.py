"""Mock inference engines - a tunable adversary for the orchestrator.

Per spec: layout 50ms/100rps/2% failures, VLM 1500-3000ms/10rps/5%
failures. Both endpoints share one code path (`_handle`) parameterised by
bucket, latency and payload generator, so the ordering guarantees are
provably identical for both rather than duplicated and drifting. Fixed
request order: Idempotency-Key check (replay from cache, or coalesce onto
an in-flight leader - consuming no token either way, since the gap
between checking and filling the cache is as wide as the model latency) ->
chaos rule (no token, no latency) -> token bucket (429 immediately, no
latency) -> sleep -> random failure (token already spent, models the
model genuinely running and then failing) -> generate output. Rate
limiting precedes latency because a real gateway rejects at the edge
without doing work, and keeps 429s fast so the backpressure signal isn't
itself delayed.
"""

from __future__ import annotations

import asyncio
import math
import random
from typing import Any, Callable

from fastapi import FastAPI, Header, HTTPException, Response

from common.logging import configure_logging, get_logger
from common.middleware import TraceMiddleware
from mock_model.chaos import ChaosController
from mock_model.config import MockSettings, get_mock_settings
from mock_model.idempotency import IdempotencyStore
from mock_model.payloads import layout_payload, vlm_payload
from mock_model.ratelimit import TokenBucket
from mock_model.schemas import ChaosRequest, PredictRequest
from mock_model.stats import Stats

log = get_logger("mock")

LAYOUT = "layout"
VLM = "vlm"


def _rate_limit_headers(wait_s: float, bucket: TokenBucket) -> dict[str, str]:
    """Tell the client exactly how long to wait.

    RFC 9110 only allows integer delay-seconds in Retry-After, which is far too
    coarse for a 10 RPS limiter: rounding a 100ms wait up to 1s would waste 90%
    of the capacity. So we send the spec-compliant integer for dumb clients AND
    an exact millisecond header, which our own client prefers.
    """
    return {
        "Retry-After": str(max(1, math.ceil(wait_s))),
        "X-Retry-After-Ms": str(int(wait_s * 1000)),
        "X-RateLimit-Limit": str(bucket.rate),
        "X-RateLimit-Burst": str(bucket.burst),
        "X-RateLimit-Remaining": str(bucket.remaining),
    }


def create_app() -> FastAPI:
    settings: MockSettings = get_mock_settings()
    configure_logging("mock-model", settings.log_level)

    buckets = {
        LAYOUT: TokenBucket(rate=settings.layout_rps, burst=settings.layout_burst),
        VLM: TokenBucket(rate=settings.vlm_rps, burst=settings.vlm_burst),
    }
    chaos = ChaosController()
    idem = IdempotencyStore(
        ttl_s=settings.idempotency_ttl_s,
        max_entries=settings.idempotency_max_entries,
    )
    stats = Stats()

    app = FastAPI(title="Mock Vision Models")
    app.add_middleware(TraceMiddleware)

    # Exposed for tests and for the admin endpoints below.
    app.state.buckets = buckets
    app.state.chaos = chaos
    app.state.idem = idem
    app.state.stats = stats

    async def _handle(
        endpoint: str,
        req: PredictRequest,
        idem_key: str | None,
        latency_ms: Callable[[], float],
        failure_rate: float,
        payload: Callable[[str, int], dict[str, Any]],
    ) -> dict[str, Any]:
        st = stats.of(endpoint)
        st.requests += 1

        # -- 2. idempotent replay: free, no token, no latency ---------------
        if idem_key:
            cached = idem.get(idem_key)
            if cached is not None:
                st.replays += 1
                log.info(
                    "idempotent_replay",
                    endpoint=endpoint,
                    job_id=req.job_id,
                    page_index=req.page_index,
                    idempotency_key=idem_key,
                )
                return cached

            # Cache miss, but an identical request may be mid-flight. Claim
            # leadership or wait for whoever already has it - a completed-result
            # cache alone cannot close a window as wide as the model latency.
            is_leader, handle = idem.begin(idem_key)
            if not is_leader:
                st.replays += 1
                log.info(
                    "idempotent_coalesced",
                    endpoint=endpoint,
                    job_id=req.job_id,
                    page_index=req.page_index,
                    idempotency_key=idem_key,
                )
                return await idem.join(handle)

            try:
                body = await _execute(endpoint, req, latency_ms, failure_rate, payload)
            except BaseException as exc:
                # Includes CancelledError: a client disconnect must still
                # release followers, or they wait on an Event nobody will set.
                idem.abandon(idem_key, exc)
                raise
            count = idem.record_execution(idem_key)
            if count > 1:
                # Never expected. Loud, because it invalidates the Module D claim.
                log.error(
                    "duplicate_execution", idempotency_key=idem_key, executions=count
                )
            idem.put(idem_key, body)
            idem.finish(idem_key, body)
            return body

        # No idempotency key: nothing to coalesce or replay.
        return await _execute(endpoint, req, latency_ms, failure_rate, payload)

    async def _execute(
        endpoint: str,
        req: PredictRequest,
        latency_ms: Callable[[], float],
        failure_rate: float,
        payload: Callable[[str, int], dict[str, Any]],
    ) -> dict[str, Any]:
        """Steps 3-7: the part that actually costs rate-limit budget and time."""
        st = stats.of(endpoint)
        bucket = buckets[endpoint]

        # -- 3. chaos: fault before spending any budget ---------------------
        rule = chaos.active(endpoint)
        faulted = chaos.roll(endpoint) if rule is not None else None
        if faulted is not None and faulted.status >= 400:
            st.chaos_rejections += 1
            if faulted.status == 429:
                st.rate_limited += 1
                raise HTTPException(
                    status_code=429,
                    detail=f"{endpoint} rate limited (chaos)",
                    headers=_rate_limit_headers(1.0, bucket),
                )
            st.injected_failures += 1
            raise HTTPException(
                status_code=faulted.status, detail=f"{endpoint} chaos failure"
            )

        # -- 4. token bucket: reject at the edge, fast ----------------------
        wait_s = bucket.acquire()
        if wait_s > 0:
            st.rate_limited += 1
            log.info("rate_limited", endpoint=endpoint, wait_ms=round(wait_s * 1000, 1))
            raise HTTPException(
                status_code=429,
                detail=f"{endpoint} rate limit exceeded",
                headers=_rate_limit_headers(wait_s, bucket),
            )

        # -- 5. simulated inference latency ---------------------------------
        total_ms = latency_ms() + (rule.extra_latency_ms if rule is not None else 0.0)
        await asyncio.sleep(total_ms / 1000)

        # -- 6. random failure: the model ran, then failed -------------------
        if random.random() < failure_rate:
            st.injected_failures += 1
            log.warning(
                "injected_failure",
                endpoint=endpoint,
                job_id=req.job_id,
                page_index=req.page_index,
            )
            raise HTTPException(status_code=500, detail=f"{endpoint} model failure")

        # -- 7. produce output ------------------------------------------------
        # Recording and caching happen in _handle, which owns the key.
        st.executions += 1
        return payload(req.job_id, req.page_index)

    # ------------------------------------------------------------------ models

    @app.post("/v1/predict/layout")
    async def predict_layout(
        req: PredictRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return await _handle(
            LAYOUT,
            req,
            idempotency_key,
            latency_ms=lambda: settings.layout_latency_ms,
            failure_rate=settings.layout_failure_rate,
            payload=layout_payload,
        )

    @app.post("/v1/predict/vlm")
    async def predict_vlm(
        req: PredictRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return await _handle(
            VLM,
            req,
            idempotency_key,
            latency_ms=lambda: random.uniform(
                settings.vlm_latency_min_ms, settings.vlm_latency_max_ms
            ),
            failure_rate=settings.vlm_failure_rate,
            payload=vlm_payload,
        )

    # ------------------------------------------------------------------- admin

    @app.post("/admin/chaos")
    async def set_chaos(req: ChaosRequest) -> dict[str, Any]:
        chaos.set(
            req.endpoint,
            status=req.status,
            ratio=req.ratio,
            seconds=req.seconds,
            extra_latency_ms=req.extra_latency_ms,
        )
        log.warning("chaos_enabled", **req.model_dump())
        return {"chaos": chaos.snapshot()}

    @app.delete("/admin/chaos")
    async def clear_chaos(endpoint: str | None = None) -> dict[str, Any]:
        chaos.clear(endpoint)
        log.info("chaos_cleared", endpoint=endpoint or "all")
        return {"chaos": chaos.snapshot()}

    @app.get("/admin/chaos")
    async def get_chaos() -> dict[str, Any]:
        return {"chaos": chaos.snapshot()}

    @app.get("/admin/call-counts")
    async def call_counts() -> dict[str, Any]:
        """The Module D evidence endpoint.

        After a SIGKILL mid-job, `idempotency.duplicate_executions` must be
        empty and each endpoint's `executions` must equal the page count.
        """
        return {
            "endpoints": stats.snapshot(),
            "idempotency": idem.stats(),
            "buckets": {
                name: {"rate": b.rate, "burst": b.burst, "remaining": b.remaining}
                for name, b in buckets.items()
            },
        }

    @app.post("/admin/reset")
    async def reset() -> dict[str, str]:
        stats.reset()
        idem.reset()
        chaos.clear()
        return {"status": "reset"}

    @app.get("/health")
    async def health(response: Response) -> dict[str, str]:
        # Never rate limited or faulted: an unhealthy-looking mock during a
        # chaos test would send docker-compose into a restart loop.
        response.headers["Cache-Control"] = "no-store"
        return {"status": "ok"}

    return app


app = create_app()
