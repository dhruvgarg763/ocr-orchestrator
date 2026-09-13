"""GET /metrics - Prometheus exposition, assembled from three sources.

A scrape here is a fan-in, not a read of one counter table. The values come
from wherever they already live, which is the point: nothing in this file is a
second source of truth for a number some other module already owns.

    Redis, written by the workers      retries, 429s, page outcomes, in-flight,
                                       reaper actions, model latency histogram
                                       (see common/metrics.py for why the
                                       workers cannot be scraped directly)
    Redis, native application state    queue backlog/pending per lane, breaker
                                       state, adaptive limit, admission tallies
    This process                       SSE subscribers, /evaluate timings

Constraints a scrape endpoint has that an ordinary route does not
---------------------------------------------------------------
It is hit every 15 seconds forever, by a client that will not stop, and a
timeout on it looks like the service being down. So:

  cheap        one pipelined round trip for the Redis-backed families, and no
               key scans proportional to job or page count. `KEYS`/`SCAN` over
               `job:*` would make scrape cost grow with retention - the
               terminal-state counters are therefore incremented by the workers
               on transition, never derived by counting hashes.

  non-blocking everything here is awaited I/O or dictionary arithmetic. The
               same argument as app/api/evaluate.py, for the same reason: this
               process serves the SSE streams the TTFP metric is measured on.

  best-effort  a scrape must not fail because one dependency is briefly
               unavailable. A degraded scrape that reports what it can is worth
               more than a 500, because the metrics are how you find out what
               is wrong - and losing them exactly when something breaks is the
               worst possible time.

No transactional consistency across families, deliberately
----------------------------------------------------------
Queue depth is read microseconds apart from the page counters, so the two can
disagree by a page or two. Making them consistent would mean holding a lock
across the whole scrape, blocking the pipeline every 15 seconds to make a
monitoring snapshot tidier than the thing it monitors. Prometheus is built for
this - every scrape is a sample, and rates are computed across scrapes.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Request, Response

from app.config import get_settings
from app.ratelimit.breaker import BreakerState
from app.queue.streams import GROUP, LEAD_STREAM, POISON_COUNTER_KEY, STREAM
from common.logging import get_logger
from common.metrics import (
    COUNTER_KEY,
    DEFAULT_BUCKETS,
    GAUGE_KEY_PREFIX,
    HISTOGRAM_COUNT_KEY,
    HISTOGRAM_SUM_KEY,
    MetricFamily,
    Registry,
    Sample,
    decode_field,
    histogram_samples,
    render,
)

log = get_logger("metrics")

router = APIRouter()

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

LANES = {LEAD_STREAM: "lead", STREAM: "main"}

# Which flushed metric names are histograms, and their help text. Counters and
# gauges carry no such table because their names are self-describing; a
# histogram needs its bucket layout known at render time to rebuild the
# cumulative form.
_WORKER_HISTOGRAMS = {
    "orch_model_request_duration_seconds": (
        "Latency of one model call, including the wait for a rate-limit token.",
    ),
    "orch_page_duration_seconds": (
        "Wall time from a page being claimed to reaching a terminal state.",
    ),
}

_WORKER_COUNTER_HELP = {
    "orch_page_attempts_total":
        "Page processing attempts by outcome; a two-stage page yields handoff "
        "then resumed, so this is attempts and not pages.",
    "orch_pages_terminal_total":
        "Pages reaching a terminal state: success (incl. degraded), failed, "
        "or duplicate redelivery.",
    "orch_rate_limit_wait_seconds_total":
        "Cumulative seconds spent waiting for rate-limit tokens.",
    "orch_model_requests_total": "Model calls attempted, by endpoint and outcome.",
    "orch_model_retries_total": "Retry attempts issued, by endpoint.",
    "orch_rate_limit_waits_total": "Times a worker waited on a rate-limit token.",
    "orch_fallbacks_total": "Pages degraded to the layout model after VLM exhaustion.",
    "orch_reaper_actions_total": "Orphaned entries handled by the reaper, by action.",
    "orch_requeues_total": "Pages returned to the queue rather than failed.",
}

_WORKER_GAUGE_HELP = {
    "orch_worker_in_flight": "Pages a worker is currently processing.",
    "orch_worker_prefetch_limit": "Configured concurrency ceiling per worker.",
}


def _state_code(state: str) -> float:
    """Breaker state as a number, because Prometheus stores float64 only.

    A string-valued label per state (the `orch_breaker_state{state="open"} 1`
    idiom) is the other convention and is better for alerting on a specific
    state. This encoding is used because the useful query here is "is anything
    degraded", and `max(orch_breaker_state) > 0` answers it without knowing the
    state names.
    """
    return {
        BreakerState.CLOSED.value: 0.0,
        BreakerState.HALF_OPEN.value: 1.0,
        BreakerState.OPEN.value: 2.0,
    }.get(state, -1.0)


async def _worker_families(redis: Any) -> list[MetricFamily]:
    """Rebuild the worker-sourced families out of the three shared hashes."""
    async with redis.pipeline(transaction=False) as pipe:
        pipe.hgetall(COUNTER_KEY)
        pipe.hgetall(HISTOGRAM_COUNT_KEY)
        pipe.hgetall(HISTOGRAM_SUM_KEY)
        counters_raw, bucket_raw, sum_raw = await pipe.execute()

    families: list[MetricFamily] = []

    grouped: dict[str, list[Sample]] = {}
    for field_name, value in (counters_raw or {}).items():
        name, labels = decode_field(field_name)
        grouped.setdefault(name, []).append(Sample(name, labels, float(value)))
    for name, samples in sorted(grouped.items()):
        families.append(
            MetricFamily(
                name,
                "counter",
                _WORKER_COUNTER_HELP.get(name, "Worker-sourced counter."),
                sorted(samples, key=lambda s: sorted(s.labels.items())),
            )
        )

    # Gauges live one key per worker and are SUMMED. Each key carries a TTL, so
    # a dead worker drops out of this scan instead of contributing forever.
    gauge_grouped: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
    worker_keys = [
        key
        async for key in redis.scan_iter(match=f"{GAUGE_KEY_PREFIX}:*", count=100)
        if not key.startswith(f"{GAUGE_KEY_PREFIX}:alive:")
    ]
    if worker_keys:
        async with redis.pipeline(transaction=False) as pipe:
            for key in worker_keys:
                pipe.hgetall(key)
            for raw in await pipe.execute():
                for field_name, value in (raw or {}).items():
                    name, labels = decode_field(field_name)
                    key_tuple = tuple(sorted(labels.items()))
                    bucket = gauge_grouped.setdefault(name, {})
                    bucket[key_tuple] = bucket.get(key_tuple, 0.0) + float(value)
    for name, values in sorted(gauge_grouped.items()):
        families.append(
            MetricFamily(
                name,
                "gauge",
                _WORKER_GAUGE_HELP.get(name, "Worker-sourced gauge, summed."),
                [Sample(name, dict(k), v) for k, v in sorted(values.items())],
            )
        )

    # Histograms: the per-bucket counts were flushed with an `le` label, which
    # has to come back off before the cumulative rule is applied - otherwise
    # `le` would be treated as an ordinary series-identifying label and every
    # bucket would render as its own series.
    per_metric: dict[str, dict[tuple[tuple[str, str], ...], dict[str, float]]] = {}
    for field_name, value in (bucket_raw or {}).items():
        name, labels = decode_field(field_name)
        bound = labels.pop("le", None)
        if bound is None:
            continue
        per_metric.setdefault(name, {}).setdefault(
            tuple(sorted(labels.items())), {}
        )[bound] = float(value)

    sums: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
    for field_name, value in (sum_raw or {}).items():
        name, labels = decode_field(field_name)
        sums.setdefault(name, {})[tuple(sorted(labels.items()))] = float(value)

    bounds = (*DEFAULT_BUCKETS, math.inf)
    labelled_bounds = [
        "+Inf" if math.isinf(b) else (str(int(b)) if float(b).is_integer() else repr(b))
        for b in bounds
    ]
    for name, by_labels in sorted(per_metric.items()):
        counts = {
            key: [by_bound.get(label, 0.0) for label in labelled_bounds]
            for key, by_bound in by_labels.items()
        }
        help_text = _WORKER_HISTOGRAMS.get(name, ("Worker-sourced histogram.",))[0]
        families.append(
            MetricFamily(
                name,
                "histogram",
                help_text,
                histogram_samples(name, bounds, counts, sums.get(name, {})),
            )
        )

    return families


async def _queue_families(redis: Any) -> list[MetricFamily]:
    """Backlog, pending and consumer-group lag, per lane.

    Lag is read from XINFO GROUPS rather than computed: Redis 7 tracks entries
    added but not yet delivered to the group, which is the number that
    distinguishes "workers are slow" from "workers are stuck". `backlog` alone
    cannot, because an entry sitting in the PEL of a dead consumer has left the
    backlog without being done.
    """
    backlog: list[Sample] = []
    pending: list[Sample] = []
    lag: list[Sample] = []

    async with redis.pipeline(transaction=False) as pipe:
        for stream in LANES:
            pipe.xlen(stream)
            pipe.xpending(stream, GROUP)
            pipe.xinfo_groups(stream)
        pipe.get(POISON_COUNTER_KEY)
        rows = await pipe.execute()

    poison = int(rows[-1] or 0)

    for index, (stream, lane) in enumerate(LANES.items()):
        length = rows[index * 3]
        raw_pending = rows[index * 3 + 1]
        groups = rows[index * 3 + 2]

        backlog.append(Sample("orch_queue_backlog", {"lane": lane}, float(length or 0)))
        in_flight = 0
        if isinstance(raw_pending, dict):
            in_flight = int(raw_pending.get("pending", 0))
        pending.append(Sample("orch_queue_pending", {"lane": lane}, float(in_flight)))

        group_lag = 0.0
        for group in groups or ():
            if group.get("name") == GROUP:
                # `lag` is None when Redis cannot compute it (after XTRIM or an
                # XSETID); 0 is the wrong substitute for "unknown", so it is
                # reported as NaN, which Prometheus treats as stale rather than
                # as a healthy zero.
                raw = group.get("lag")
                group_lag = float(raw) if raw is not None else float("nan")
        lag.append(Sample("orch_queue_lag", {"lane": lane}, group_lag))

    return [
        MetricFamily(
            "orch_queue_backlog",
            "gauge",
            "Entries published to a lane and not yet delivered to any consumer.",
            backlog,
        ),
        MetricFamily(
            "orch_queue_pending",
            "gauge",
            "Entries delivered to a consumer and not yet acknowledged.",
            pending,
        ),
        MetricFamily(
            "orch_queue_lag",
            "gauge",
            "Consumer-group lag as reported by Redis; NaN when uncomputable.",
            lag,
        ),
        # A counter that should read 0 forever. It is exported anyway because
        # the alternative - a worker quietly discarding entries it cannot parse
        # - is the failure this counter exists to make impossible to miss. Any
        # non-zero value means a producer is writing malformed tasks.
        MetricFamily(
            "orch_poison_entries_total",
            "counter",
            "Stream entries quarantined because they could not be parsed.",
            [Sample("orch_poison_entries_total", {}, float(poison))],
        ),
    ]


async def _breaker_and_rate_families(request: Request) -> list[MetricFamily]:
    """Breaker state and the adaptive limit, read from the keys the WORKERS own.

    The API holds no breaker or limiter of its own - it never calls a model. It
    can report them only because Step 9 and Step 10 put that state in Redis
    rather than in worker memory, which was done so three replicas would agree.
    Being observable from a process that is not a participant is the second
    dividend of that decision.
    """
    breaker_samples: list[Sample] = []
    rate_samples: list[Sample] = []

    breakers = getattr(request.app.state, "metrics_breakers", {})
    limiters = getattr(request.app.state, "metrics_limiters", {})

    for endpoint, breaker in breakers.items():
        try:
            verdict = await breaker.peek()
        except Exception:  # noqa: BLE001 - a scrape reports what it can
            continue
        breaker_samples.append(
            Sample(
                "orch_breaker_state",
                {"endpoint": endpoint},
                _state_code(verdict.state.value),
            )
        )

    for endpoint, limiter in limiters.items():
        try:
            rate_samples.append(
                Sample(
                    "orch_adaptive_rate_limit",
                    {"endpoint": endpoint},
                    float(await limiter.current()),
                )
            )
        except Exception:  # noqa: BLE001
            continue

    return [
        MetricFamily(
            "orch_breaker_state",
            "gauge",
            "Circuit breaker: 0 closed, 1 half-open, 2 open, -1 unknown.",
            breaker_samples,
        ),
        MetricFamily(
            "orch_adaptive_rate_limit",
            "gauge",
            "Requests per second the AIMD controller currently permits.",
            rate_samples,
        ),
    ]


async def _admission_families(request: Request) -> list[MetricFamily]:
    admission = request.app.state.admission
    stats = await admission.stats()
    return [
        MetricFamily(
            "orch_jobs_total",
            "counter",
            "Jobs seen at the edge, by admission decision.",
            [
                Sample("orch_jobs_total", {"decision": "admitted"},
                       float(stats["admitted_jobs"])),
                Sample("orch_jobs_total", {"decision": "shed"},
                       float(stats["shed_jobs"])),
            ],
        ),
        MetricFamily(
            "orch_admitted_pages_total",
            "counter",
            "Pages accepted or refused at the edge.",
            [
                Sample("orch_admitted_pages_total", {"decision": "admitted"},
                       float(stats["admitted_pages"])),
                Sample("orch_admitted_pages_total", {"decision": "shed"},
                       float(stats["shed_pages"])),
            ],
        ),
        MetricFamily(
            "orch_admission_shedding",
            "gauge",
            "1 while the edge is refusing work because the queue is too deep.",
            [Sample("orch_admission_shedding", {}, 1.0 if stats["shedding"] else 0.0)],
        ),
        MetricFamily(
            "orch_admission_transitions_total",
            "counter",
            "Watermark crossings: shedding started (trip) or stopped (recovery).",
            [
                Sample("orch_admission_transitions_total", {"kind": "trip"},
                       float(stats["trips"])),
                Sample("orch_admission_transitions_total", {"kind": "recovery"},
                       float(stats["recoveries"])),
            ],
        ),
        MetricFamily(
            "orch_queue_watermark",
            "gauge",
            "Configured admission watermarks, for comparison against depth.",
            [
                Sample("orch_queue_watermark", {"edge": "high"},
                       float(stats["high_watermark"])),
                Sample("orch_queue_watermark", {"edge": "low"},
                       float(stats["low_watermark"])),
            ],
        ),
    ]


async def _worker_liveness_family(redis: Any) -> MetricFamily:
    """How many workers flushed recently.

    Derived from the TTL'd heartbeat keys rather than from a registry, so it
    counts workers that are ALIVE rather than workers that once existed - the
    same reason the gauge keys expire.
    """
    alive = 0
    async for _ in redis.scan_iter(match=f"{GAUGE_KEY_PREFIX}:alive:*", count=100):
        alive += 1
    return MetricFamily(
        "orch_workers_reporting",
        "gauge",
        "Workers whose metrics flush is recent enough to still be counted.",
        [Sample("orch_workers_reporting", {}, float(alive))],
    )


def _local_families(request: Request) -> list[MetricFamily]:
    """This process's own metrics: SSE occupancy and /evaluate timings."""
    families: list[MetricFamily] = []

    slots = request.app.state.sse_slots
    stats = slots.stats()
    families.append(
        MetricFamily(
            "orch_sse_subscribers",
            "gauge",
            "SSE clients currently attached.",
            [Sample("orch_sse_subscribers", {}, float(stats["active"]))],
        )
    )
    families.append(
        MetricFamily(
            "orch_sse_subscribers_peak",
            "gauge",
            "High-water mark of concurrent SSE clients since start.",
            [Sample("orch_sse_subscribers_peak", {}, float(stats["peak"]))],
        )
    )
    families.append(
        MetricFamily(
            "orch_sse_refusals_total",
            "counter",
            "SSE connections refused because the subscriber cap was full.",
            [Sample("orch_sse_refusals_total", {}, float(stats["refused"]))],
        )
    )
    families.append(
        MetricFamily(
            "orch_sse_subscriber_limit",
            "gauge",
            "Configured concurrent SSE subscriber cap.",
            [Sample("orch_sse_subscriber_limit", {}, float(stats["limit"]))],
        )
    )

    registry: Registry | None = getattr(request.app.state, "metrics_registry", None)
    if registry is not None:
        families.extend(registry.families())
    return families


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    """Scrape endpoint. Degrades rather than failing.

    Each source is attempted independently and a failure drops that family
    instead of the response: metrics are how an operator finds out what is
    broken, so returning nothing at the moment something breaks is the least
    useful possible behaviour. Failures are logged, and
    `orch_metrics_scrape_errors` is itself exported so a silently degraded
    scrape is visible in the data rather than only in the logs.
    """
    redis = request.app.state.page_queue._redis  # noqa: SLF001 - see note below
    # Reaching into the queue for its client rather than taking a second one:
    # the pool is shared on purpose (Step 4 sized it once), and a scrape opening
    # its own connection every 15s would be a slow leak of the thing the pool
    # exists to bound.

    families: list[MetricFamily] = []
    errors = 0

    for label, coro in (
        ("queue", _queue_families(redis)),
        ("worker", _worker_families(redis)),
        ("breaker", _breaker_and_rate_families(request)),
        ("admission", _admission_families(request)),
    ):
        try:
            families.extend(await coro)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            log.warning("metrics_source_failed", source=label, error=str(exc))

    try:
        families.append(await _worker_liveness_family(redis))
    except Exception as exc:  # noqa: BLE001
        errors += 1
        log.warning("metrics_source_failed", source="liveness", error=str(exc))

    try:
        families.extend(_local_families(request))
    except Exception as exc:  # noqa: BLE001
        errors += 1
        log.warning("metrics_source_failed", source="local", error=str(exc))

    families.append(
        MetricFamily(
            "orch_metrics_scrape_errors",
            "gauge",
            "Metric sources that failed during this scrape.",
            [Sample("orch_metrics_scrape_errors", {}, float(errors))],
        )
    )

    settings = get_settings()
    families.append(
        MetricFamily(
            "orch_build_info",
            "gauge",
            "Static configuration, as labels on a constant 1.",
            [
                Sample(
                    "orch_build_info",
                    {
                        "worker_concurrency": str(settings.worker_concurrency),
                        "vlm_rps": str(settings.vlm_rps),
                        "ted_budget_ms": str(int(settings.eval_ted_budget_ms)),
                    },
                    1.0,
                )
            ],
        )
    )

    return Response(content=render(families), media_type=CONTENT_TYPE)
