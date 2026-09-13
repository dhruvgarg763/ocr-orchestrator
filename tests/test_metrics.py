"""Metrics: the format rules, and the cross-container aggregation.

The exposition format is the risk taken on by not using `prometheus_client`, so
the rules it would have enforced are asserted here instead - cumulative
buckets, a mandatory +Inf, one HELP/TYPE per family, escaping, and the special
float spellings. A violation of any of these does not raise; it makes a scraper
either reject the family or compute a wrong quantile, which is why they are
tested rather than trusted.

The aggregation half tests the property the whole design rests on: that the
fleet-wide counter never goes backwards when a worker restarts.
"""

from __future__ import annotations

import math

import pytest

from app.worker.metrics import WorkerMetrics
from common.metrics import (
    COUNTER_KEY,
    GAUGE_KEY_PREFIX,
    Counter,
    Gauge,
    Histogram,
    MetricFamily,
    Registry,
    Sample,
    decode_field,
    encode_field,
    render,
)

# ------------------------------------------------------------ format rules


def test_a_counter_renders_help_type_and_value() -> None:
    family = MetricFamily(
        "orch_thing_total", "counter", "How many things.", [Sample("orch_thing_total", {}, 3)]
    )

    assert render([family]) == (
        "# HELP orch_thing_total How many things.\n"
        "# TYPE orch_thing_total counter\n"
        "orch_thing_total 3\n"
    )


def test_labels_are_sorted_so_output_is_stable() -> None:
    """Two scrapes of identical state must produce identical text. Dict order
    is insertion order, so unsorted labels would make the output depend on which
    code path happened to touch the series first."""
    family = MetricFamily(
        "m", "gauge", "h", [Sample("m", {"b": "2", "a": "1"}, 1)]
    )

    assert 'm{a="1",b="2"} 1' in render([family])


def test_help_and_type_appear_exactly_once_per_family() -> None:
    """A repeated TYPE line makes a scraper reject the whole family, so many
    samples must share one header rather than each carrying its own."""
    family = MetricFamily(
        "m",
        "counter",
        "h",
        [Sample("m", {"k": "a"}, 1), Sample("m", {"k": "b"}, 2)],
    )

    text = render([family])

    assert text.count("# TYPE m counter") == 1
    assert text.count("# HELP m h") == 1
    assert text.count("\nm{") == 2


def test_an_empty_family_still_gets_metadata() -> None:
    """So a registered-but-currently-empty series is distinguishable from one
    that was never registered - the difference between "no errors" and "error
    tracking is broken"."""
    text = render([MetricFamily("m", "counter", "h", [])])

    assert "# TYPE m counter" in text
    assert "\nm " not in text


def test_label_values_are_escaped() -> None:
    r"""Quotes, backslashes and newlines in a label value would otherwise
    terminate the label block early and corrupt every following sample."""
    family = MetricFamily(
        "m", "gauge", "h", [Sample("m", {"k": 'a"b\\c\nd'}, 1)]
    )

    line = render([family]).strip().splitlines()[-1]

    assert line == 'm{k="a\\"b\\\\c\\nd"} 1'


def test_help_text_newlines_are_escaped() -> None:
    """An unescaped newline in HELP makes the next line look like a sample."""
    text = render([MetricFamily("m", "gauge", "line one\nline two", [])])

    assert "# HELP m line one\\nline two" in text
    assert len(text.strip().splitlines()) == 2


def test_the_special_floats_use_prometheus_spelling() -> None:
    """Python writes `inf` and `nan`; a scraper accepts neither."""
    family = MetricFamily(
        "m",
        "gauge",
        "h",
        [
            Sample("m", {"k": "pos"}, math.inf),
            Sample("m", {"k": "neg"}, -math.inf),
            Sample("m", {"k": "nan"}, math.nan),
        ],
    )

    text = render([family])

    assert 'm{k="pos"} +Inf' in text
    assert 'm{k="neg"} -Inf' in text
    assert 'm{k="nan"} NaN' in text


def test_output_ends_with_a_newline() -> None:
    assert render([MetricFamily("m", "gauge", "h", [Sample("m", {}, 1)])]).endswith("\n")


# ------------------------------------------------------------- histograms


def test_histogram_buckets_are_cumulative_and_end_with_inf() -> None:
    """THE histogram rule. `le="0.1"` must include everything counted by
    `le="0.05"`, and the +Inf bucket must equal _count. Non-cumulative buckets
    do not error - they make `histogram_quantile()` return plausible nonsense.
    """
    histogram = Histogram("h", "help", buckets=[0.05, 0.1, 1.0])
    for value in (0.01, 0.07, 0.5, 5.0):
        histogram.observe(value)

    samples = {
        (s.name, s.labels.get("le")): s.value for s in histogram.samples()
    }

    assert samples[("h_bucket", "0.05")] == 1  # 0.01
    assert samples[("h_bucket", "0.1")] == 2  # + 0.07
    assert samples[("h_bucket", "1")] == 3  # + 0.5
    assert samples[("h_bucket", "+Inf")] == 4  # + 5.0
    assert samples[("h_count", None)] == 4
    assert samples[("h_sum", None)] == pytest.approx(5.58)


def test_bucket_counts_never_decrease_across_the_rendered_series() -> None:
    """The invariant stated directly, over random-ish data, so a future change
    to the bucket walk cannot quietly break monotonicity."""
    histogram = Histogram("h", "help")
    for value in (0.001, 0.02, 0.3, 0.7, 1.2, 2.5, 9.0, 40.0):
        histogram.observe(value)

    running = [s.value for s in histogram.samples() if s.name == "h_bucket"]

    assert running == sorted(running), running


def test_a_value_on_a_bucket_boundary_lands_in_that_bucket() -> None:
    """`le` means less-than-or-EQUAL. An off-by-one here shifts every
    percentile by a bucket."""
    histogram = Histogram("h", "help", buckets=[0.1, 1.0])
    histogram.observe(0.1)

    samples = {s.labels.get("le"): s.value for s in histogram.samples() if s.name == "h_bucket"}

    assert samples["0.1"] == 1


def test_a_histogram_needs_at_least_one_bucket() -> None:
    with pytest.raises(ValueError, match="at least one bucket"):
        Histogram("h", "help", buckets=[])


# ------------------------------------------------------------- primitives


def test_a_counter_refuses_to_decrease() -> None:
    """Monotonicity is the one property a counter cannot lose: a decrease is how
    Prometheus detects a process restart, so a spurious one corrupts every rate
    over that window."""
    counter = Counter("c", "help")

    with pytest.raises(ValueError, match="refusing a decrease"):
        counter.inc(-1)


def test_draining_a_counter_resets_it() -> None:
    """`drain` is read-and-clear because the sink ships deltas. If it read
    without clearing, every flush would re-send the lifetime total."""
    counter = Counter("c", "help", ["k"])
    counter.inc(2, k="a")

    first = counter.drain()
    counter.inc(3, k="a")
    second = counter.drain()

    assert first == {(("k", "a"),): 2}
    assert second == {(("k", "a"),): 3}
    assert counter.drain() == {}


def test_wrong_labels_are_rejected() -> None:
    """A typo'd label name would otherwise create a second, near-identical
    series that silently splits the data."""
    counter = Counter("c", "help", ["endpoint"])

    with pytest.raises(ValueError, match="expects labels"):
        counter.inc(endpoint="vlm", oops="1")
    with pytest.raises(ValueError, match="expects labels"):
        counter.inc()


def test_a_gauge_moves_both_ways() -> None:
    gauge = Gauge("g", "help", ["w"])
    gauge.set(5, w="a")
    gauge.inc(2, w="a")
    gauge.dec(3, w="a")

    assert gauge.snapshot() == {(("w", "a"),): 4.0}


# ------------------------------------------------------- field encoding


def test_field_encoding_round_trips() -> None:
    field = encode_field("orch_m_total", {"b": "2", "a": "1"})

    assert decode_field(field) == ("orch_m_total", {"a": "1", "b": "2"})


def test_label_values_containing_delimiters_do_not_merge_series() -> None:
    """The reason the encoding is JSON rather than `k=v,k=v`. A value holding a
    comma or an equals sign would otherwise decode as a different label set -
    two distinct series quietly becoming one."""
    tricky = encode_field("m", {"k": "a,b=c"})
    plain = encode_field("m", {"k": "a", "b": "c"})

    assert tricky != plain
    assert decode_field(tricky) == ("m", {"k": "a,b=c"})


def test_encoding_is_order_independent() -> None:
    assert encode_field("m", {"a": "1", "b": "2"}) == encode_field(
        "m", {"b": "2", "a": "1"}
    )


# --------------------------------------------------- the aggregation claim


async def test_a_worker_flush_writes_counters_as_deltas(redis) -> None:
    """Two flushes must ADD, not overwrite - which is what makes the shared
    fleet total correct when three workers write to the same field."""
    metrics = WorkerMetrics(redis, worker="w1", gauge_ttl_s=30, prefetch_limit=4)

    metrics.model_requests.inc(endpoint="vlm", outcome="success")
    await metrics._sink.flush()  # noqa: SLF001
    metrics.model_requests.inc(2, endpoint="vlm", outcome="success")
    await metrics._sink.flush()  # noqa: SLF001

    field = encode_field(
        "orch_model_requests_total", {"endpoint": "vlm", "outcome": "success"}
    )
    assert float(await redis.hget(COUNTER_KEY, field)) == 3.0


async def test_two_workers_sum_into_one_fleet_counter(redis) -> None:
    a = WorkerMetrics(redis, worker="w1", gauge_ttl_s=30, prefetch_limit=4)
    b = WorkerMetrics(redis, worker="w2", gauge_ttl_s=30, prefetch_limit=4)

    a.retries.inc(2, endpoint="vlm")
    b.retries.inc(5, endpoint="vlm")
    await a._sink.flush()  # noqa: SLF001
    await b._sink.flush()  # noqa: SLF001

    field = encode_field("orch_model_retries_total", {"endpoint": "vlm"})
    assert float(await redis.hget(COUNTER_KEY, field)) == 7.0


async def test_a_restarted_worker_does_not_pull_the_fleet_counter_backwards(
    redis,
) -> None:
    """THE property the delta design exists for.

    A worker that flushed 10 and then restarts begins again from its own zero.
    With per-worker ABSOLUTE keys the summed total would drop from 10 to
    whatever the new process has counted so far - and Prometheus reads a
    fleet-wide decrease as a counter reset, mis-attributing every rate over the
    window. With deltas into a shared key the total only ever rises.
    """
    first = WorkerMetrics(redis, worker="w1", gauge_ttl_s=30, prefetch_limit=4)
    first.retries.inc(10, endpoint="vlm")
    await first._sink.flush()  # noqa: SLF001

    field = encode_field("orch_model_retries_total", {"endpoint": "vlm"})
    before = float(await redis.hget(COUNTER_KEY, field))

    # Same worker name, fresh object: exactly what a container restart looks
    # like to Redis.
    restarted = WorkerMetrics(redis, worker="w1", gauge_ttl_s=30, prefetch_limit=4)
    restarted.retries.inc(1, endpoint="vlm")
    await restarted._sink.flush()  # noqa: SLF001

    after = float(await redis.hget(COUNTER_KEY, field))

    assert before == 10.0
    assert after == 11.0
    assert after >= before


async def test_gauges_are_per_worker_and_expire(redis) -> None:
    """A dead worker's in-flight count must disappear rather than freeze, so
    gauge keys are per-worker and carry a TTL - absence of a heartbeat is the
    signal, the same as the lease renewal in app/worker/reaper.py."""
    metrics = WorkerMetrics(redis, worker="w9", gauge_ttl_s=30, prefetch_limit=8)
    metrics.in_flight.set(3, worker="w9")
    await metrics._sink.flush()  # noqa: SLF001

    key = f"{GAUGE_KEY_PREFIX}:w9"
    ttl = await redis.ttl(key)

    assert 0 < ttl <= 30
    field = encode_field("orch_worker_in_flight", {"worker": "w9"})
    assert float(await redis.hget(key, field)) == 3.0
    assert await redis.ttl(f"{GAUGE_KEY_PREFIX}:alive:w9") > 0


async def test_derived_counters_push_the_difference_not_the_total(redis) -> None:
    """`WorkerStats.outcomes` is a running total. Feeding it to a delta sink
    unchanged would re-add the lifetime figure on every flush - a worker up for
    an hour reporting tens of thousands of pages it never processed.
    """
    metrics = WorkerMetrics(redis, worker="w1", gauge_ttl_s=30, prefetch_limit=4)

    await metrics.flush(outcomes={"resumed": 5}, reaper={}, in_flight=0)
    await metrics.flush(outcomes={"resumed": 8}, reaper={}, in_flight=0)
    await metrics.flush(outcomes={"resumed": 8}, reaper={}, in_flight=0)

    field = encode_field("orch_page_attempts_total", {"outcome": "resumed"})
    assert float(await redis.hget(COUNTER_KEY, field)) == 8.0


async def test_a_backwards_total_is_clamped_rather_than_pushed_negative(
    redis,
) -> None:
    """A reset stats object would otherwise send a negative delta and make the
    fleet counter non-monotonic - the one thing it must never be."""
    metrics = WorkerMetrics(redis, worker="w1", gauge_ttl_s=30, prefetch_limit=4)

    await metrics.flush(outcomes={"resumed": 9}, reaper={}, in_flight=0)
    await metrics.flush(outcomes={"resumed": 2}, reaper={}, in_flight=0)

    field = encode_field("orch_page_attempts_total", {"outcome": "resumed"})
    assert float(await redis.hget(COUNTER_KEY, field)) == 9.0


async def test_terminal_pages_are_classified_for_the_zero_drop_metric(
    redis,
) -> None:
    """`degraded` counts as success: the page produced usable output at reduced
    fidelity, which the zero-drop guarantee treats as served. `handoff` is not
    terminal at all and must not appear."""
    metrics = WorkerMetrics(redis, worker="w1", gauge_ttl_s=30, prefetch_limit=4)

    await metrics.flush(
        outcomes={
            "resumed": 3,
            "degraded": 2,
            "failed": 1,
            "already_terminal": 4,
            "handoff": 7,
        },
        reaper={},
        in_flight=0,
    )

    async def total(result: str) -> float:
        raw = await redis.hget(
            COUNTER_KEY, encode_field("orch_pages_terminal_total", {"result": result})
        )
        return float(raw or 0)

    assert await total("success") == 5.0  # resumed + degraded
    assert await total("failed") == 1.0
    assert await total("duplicate") == 4.0


async def test_nothing_is_written_when_nothing_changed(redis) -> None:
    """A flush with no activity must not touch Redis at all, or an idle fleet
    would spend a round trip per worker per interval writing zeroes."""
    metrics = WorkerMetrics(redis, worker="w1", gauge_ttl_s=30, prefetch_limit=4)
    await metrics.flush(outcomes={}, reaper={}, in_flight=0)

    # The gauge key exists (in_flight and prefetch_limit are always reported),
    # but no counter field should have been created.
    assert await redis.exists(COUNTER_KEY) == 0


def test_the_registry_reports_every_registered_family() -> None:
    registry = Registry()
    registry.counter("c", "help")
    registry.gauge("g", "help")
    registry.histogram("h", "help")

    names = {family.name for family in registry.families()}
    kinds = {family.kind for family in registry.families()}

    assert names == {"c", "g", "h"}
    assert kinds == {"counter", "gauge", "histogram"}
