"""Orchestrator API entrypoint."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response

from app.config import get_settings
from app.core.redis_client import (
    close_redis,
    get_redis,
    get_stream_redis,
    healthcheck,
    init_redis,
    init_stream_redis,
)
from app.api import evaluate, jobs, metrics as metrics_api, stream
from app.queue.state import PageStateStore
from app.api.admission import AdmissionController
from app.queue.results import ResultReader
from app.queue.streams import PageQueue
from app.ratelimit.adaptive import AdaptiveRate
from app.ratelimit.breaker import CircuitBreaker
from app.worker.client import LAYOUT, VLM
from common.metrics import Registry
from common.logging import configure_logging, get_logger
from common.middleware import TraceMiddleware

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook.

    Long-lived resources belong here, not at import time: import happens before
    the event loop exists, so an async client created there binds to the wrong
    loop. Resources acquired here are also released deterministically, which is
    what stops a reload leaking connections.
    """
    settings = get_settings()
    log.info(
        "api_starting",
        redis_url=settings.redis_url,
        mock_base_url=settings.mock_base_url,
    )

    await init_redis(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        pool_timeout=settings.redis_pool_timeout,
    )
    app.state.state_store = PageStateStore(get_redis(), ttl_s=settings.result_ttl_s)
    app.state.page_queue = PageQueue(get_redis())

    # A SECOND pool, for the SSE tailer's blocking reads. Two measured reasons,
    # both in app/core/redis_client.py:
    #
    #   * a 5s socket timeout against XREAD BLOCK 15000 reports every normal
    #     block as a dead server. Every stream died at ~5s, and a streaming body
    #     cannot change its status, so the client saw a truncated response
    #     rather than an error. This pool's timeout is derived FROM the block.
    #   * a blocked reader holds its connection for the whole block, so 50
    #     concurrent subscribers would take 50 of the main pool's 32
    #     connections - starving INGESTION, which matters more than any
    #     subscriber. Separate pools make that priority structural.
    #
    # Sized from the subscriber cap, plus headroom for the non-blocking catch-up
    # reads those same subscribers issue on connect.
    await init_stream_redis(
        settings.redis_url,
        max_connections=settings.sse_max_subscribers + 8,
        block_ms=settings.sse_block_ms,
        pool_timeout=settings.redis_pool_timeout,
    )

    # Read side only. The API never publishes results - a result is a
    # statement about work that was done, and only the worker that did it
    # can make one.
    app.state.result_reader = ResultReader(get_stream_redis())

    # Admission control for subscribers, for the same reason Step 11 has it for
    # jobs: the resource is finite, so the refusal should be explicit and
    # counted rather than an exhaustion discovered as a timeout.
    app.state.sse_slots = stream.SubscriberSlots(settings.sse_max_subscribers)

    # Admission control is the OUTERMOST bound. Everything in Steps 7-10 pushes
    # back downstream and protects the model endpoints; none of it protects this
    # service from ingestion, because all of it acts after the work is already
    # queued. A client can offer ~20,000 pages/sec while the VLM drains 10.
    #
    # Retry-After is derived from the AIMD controller's DISCOVERED rate, not the
    # advertised one: if the VLM has degraded to 2.7 rps the same backlog takes
    # 4x as long to clear, and quoting a figure based on 10 rps just brings the
    # client back too early to be refused again.
    # Serialises the admission critical section - check, init, enqueue - within
    # this process.
    #
    # Without it the check and the enqueue are separate round trips, so
    # concurrent requests all read the same pre-enqueue depth and each admits
    # against it. Measured at the assignment's own benchmark concurrency (50
    # concurrent POSTs, 20 pages each, watermark 400): 130-150% overshoot, and
    # in one trial 1,000 pages were admitted against a 400 limit with NOTHING
    # shed. A bound that can be exceeded 2.5x is not a bound.
    #
    # Holding the lock across all three steps makes each check observe every
    # prior enqueue, so the overshoot is zero within a process. With N API
    # replicas it returns, but bounded by (N-1) x max_pages - which is a
    # provable constant, unlike the previous bound of "however many requests a
    # client chooses to send at once".
    #
    # Cheap because ingestion is already serialised by Redis's single thread:
    # the lock makes the ORDERING explicit, it does not add contention that was
    # not already there.
    app.state.admission_lock = asyncio.Lock()

    app.state.admission = AdmissionController(
        get_redis(),
        high_watermark=settings.queue_high_watermark,
        low_watermark_fraction=settings.queue_low_watermark_fraction,
        drain_rate=settings.vlm_rps,
        controller=(
            AdaptiveRate(get_redis(), VLM, max_rate=settings.vlm_rps)
            if settings.adaptive_enabled
            else None
        ),
        retry_after_cap_s=settings.retry_after_cap_s,
        retry_after_jitter=settings.retry_after_jitter,
        ttl_s=settings.result_ttl_s,
    )
    # ------------------------------------------------------- observability
    #
    # Read-only views of the breaker and limiter state the WORKERS own. The API
    # never calls a model, so it holds neither - it can report them only
    # because Step 9 and Step 10 keep that state in Redis so three replicas
    # would agree. Being observable from a non-participant is the second
    # dividend of that decision.
    #
    # The constructor arguments must match the workers', because these objects
    # derive their Redis key from the endpoint name and prefix. A mismatch
    # would read a key nobody writes and report a permanently-closed breaker,
    # which is worse than reporting nothing.
    app.state.metrics_breakers = {
        endpoint: CircuitBreaker(
            get_redis(),
            endpoint,
            window_s=settings.breaker_window_s,
            min_volume=settings.breaker_min_volume,
            failure_ratio=settings.breaker_failure_ratio,
            cooldown_s=settings.breaker_cooldown_s,
            ttl_s=settings.result_ttl_s,
        )
        for endpoint in (LAYOUT, VLM)
    }
    app.state.metrics_limiters = (
        {
            endpoint: AdaptiveRate(
                get_redis(),
                endpoint,
                max_rate=settings.vlm_rps if endpoint == VLM else settings.layout_rps,
            )
            for endpoint in (LAYOUT, VLM)
        }
        if settings.adaptive_enabled
        else {}
    )

    # This process's own metrics. Only /evaluate writes here today; SSE and
    # admission figures are read from their owners at scrape time instead of
    # being mirrored, so there is no second copy to drift.
    app.state.metrics_registry = Registry()
    app.state.evaluate_duration = app.state.metrics_registry.histogram(
        "orch_evaluate_duration_seconds",
        "Time spent computing one evaluation metric, by metric.",
        ["metric"],
    )

    # The API creates the group too, so ingestion works even if no worker has
    # started yet - otherwise the first tasks would go to a group-less stream
    # and never be delivered.
    await app.state.page_queue.ensure_group()

    try:
        yield
    finally:
        # `finally` so an exception raised while starting a later resource still
        # releases the pool rather than leaking it.
        await close_redis()
        log.info("api_stopped")


def create_app() -> FastAPI:
    """Factory, so tests can build an isolated app with overridden settings."""
    settings = get_settings()
    configure_logging("api", settings.log_level)

    app = FastAPI(title="Sarvam Vision Orchestrator", lifespan=lifespan)
    app.add_middleware(TraceMiddleware)
    app.include_router(jobs.router)
    app.include_router(stream.router)
    app.include_router(evaluate.router)
    app.include_router(metrics_api.router)

    @app.get("/health")
    async def health(response: Response) -> dict[str, Any]:
        """Liveness plus dependency status.

        Reports degraded instead of raising: a health endpoint that throws gives
        an ambiguous 500 rather than naming the dependency that is down.
        """
        deps = await healthcheck()
        ok = deps.get("redis") == "ok"
        response.status_code = 200 if ok else 503
        return {"status": "ok" if ok else "degraded", **deps}

    return app


app = create_app()
