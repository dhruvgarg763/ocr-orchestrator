"""Retry classification and backoff tests.

Pure logic, no Redis and no network. Classification is a decision table and
backoff is a probability distribution, so both can be tested exhaustively and
fast - which matters, because a retry bug is invisible until production is on
fire.
"""

from __future__ import annotations

import random
from collections import Counter

import httpx
import pytest

from app.ratelimit.retry import (
    RETRYABLE_STATUSES,
    Disposition,
    backoff_delay_s,
    classify,
    retry_after_ms,
)
from app.ratelimit.token_bucket import RateLimitTimeout


def status_error(code: int, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://mock/v1/predict/vlm")
    response = httpx.Response(code, headers=headers or {}, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


# ------------------------------------------------------------ classification


@pytest.mark.parametrize("code", sorted(RETRYABLE_STATUSES))
def test_transient_statuses_are_retried(code: int) -> None:
    assert classify(status_error(code)) is Disposition.RETRY


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 415, 422, 501])
def test_permanent_statuses_are_not_retried(code: int) -> None:
    """Retrying a permanent error spends the budget and delays the inevitable.

    A malformed request is just as malformed the second time, and meanwhile it
    consumes capacity that a retryable page could have used.
    """
    assert classify(status_error(code)) is Disposition.PERMANENT


def test_429_is_retryable_because_the_server_said_later() -> None:
    assert classify(status_error(429)) is Disposition.RETRY


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectTimeout("timeout"),
        httpx.ReadTimeout("timeout"),
        httpx.ConnectError("refused"),
        httpx.RemoteProtocolError("reset"),
    ],
)
def test_transport_faults_are_retried(exc: Exception) -> None:
    """Safe ONLY because every request carries a deterministic Idempotency-Key.

    A timeout means the outcome is unknown - the work may have completed with
    the response lost. Without idempotency, retrying would double-charge for a
    3-second inference; with it, the retry is served from cache.
    """
    assert classify(exc) is Disposition.RETRY


def test_rate_limit_timeout_is_saturation_not_failure() -> None:
    """A distinct disposition, because the correct response is different.

    Saturation is systemic: retrying in place just pins the page while the
    worker could serve others. It must be requeued, not retried.
    """
    assert classify(RateLimitTimeout("vlm", 30.0)) is Disposition.SATURATED


def test_unknown_errors_are_treated_as_permanent() -> None:
    """Fail fast on the unrecognised.

    An unknown error retried three times is three times the log noise and a
    delayed diagnosis. A genuine bug should surface immediately.
    """
    assert classify(ValueError("something new")) is Disposition.PERMANENT


# ----------------------------------------------------------- Retry-After


def test_precise_millisecond_header_is_preferred() -> None:
    """RFC 9110 Retry-After allows only integer seconds.

    For a 10 rps limiter a 33ms wait rounds up to 1000ms, wasting 97% of the
    endpoint's capacity - so the mock sends both and we read the precise one.
    """
    exc = status_error(429, {"retry-after": "1", "x-retry-after-ms": "33"})
    assert retry_after_ms(exc) == 33.0


def test_falls_back_to_retry_after_seconds() -> None:
    assert retry_after_ms(status_error(429, {"retry-after": "2"})) == 2000.0


def test_http_date_retry_after_is_ignored_rather_than_crashing() -> None:
    """Retry-After may be an HTTP-date. We fall back to our own backoff rather
    than parsing a format an internal endpoint will never send."""
    assert retry_after_ms(status_error(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})) is None


def test_no_header_and_non_http_errors_yield_no_hint() -> None:
    assert retry_after_ms(status_error(500)) is None
    assert retry_after_ms(httpx.ConnectError("refused")) is None


# --------------------------------------------------------------- backoff


def test_delay_is_bounded_by_the_exponential_ceiling() -> None:
    """Full jitter draws uniformly over [0, ceiling]."""
    for attempt in range(5):
        ceiling_s = min(5_000, 200 * 2**attempt) / 1000
        for _ in range(200):
            delay = backoff_delay_s(attempt, base_ms=200, max_ms=5_000)
            assert 0 <= delay <= ceiling_s


def test_ceiling_is_capped_so_late_attempts_do_not_schedule_minutes() -> None:
    """Without a cap, attempt 10 would be 200ms * 2^10 = 3.4 minutes."""
    for _ in range(200):
        assert backoff_delay_s(10, base_ms=200, max_ms=5_000) <= 5.0


def test_mean_delay_grows_with_attempt_number() -> None:
    """Backing off is the point: each failure should ease pressure further."""
    rng = random.Random(1)
    means = [
        sum(backoff_delay_s(a, base_ms=100, max_ms=10_000, rng=rng) for _ in range(500)) / 500
        for a in range(4)
    ]
    assert means == sorted(means), means
    assert means[3] > means[0] * 3


def test_full_jitter_spreads_delays_across_the_whole_window() -> None:
    """Not merely random - uniformly spread.

    A distribution clustered near the ceiling would still synchronise clients
    into a convoy. Bucketing the draws shows every part of the window is used.
    """
    rng = random.Random(2)
    delays = [backoff_delay_s(3, base_ms=100, max_ms=10_000, rng=rng) for _ in range(2_000)]
    ceiling = 0.8  # 100ms * 2^3

    buckets = Counter(min(9, int(d / ceiling * 10)) for d in delays)
    assert len(buckets) == 10, f"only {len(buckets)} of 10 sub-ranges used"
    assert all(count > 100 for count in buckets.values()), buckets


def test_server_hint_acts_as_a_floor_not_a_replacement() -> None:
    """Sleeping less than the server asked guarantees another rejection.

    Extra jitter is still layered on top, otherwise every client handed the
    same Retry-After returns in the same instant - reintroducing the convoy
    that jitter exists to prevent.
    """
    rng = random.Random(3)
    delays = [
        backoff_delay_s(0, base_ms=200, max_ms=5_000, retry_after_ms_hint=1_000, rng=rng)
        for _ in range(500)
    ]

    assert all(d >= 1.0 for d in delays), f"min was {min(delays)}"
    assert max(delays) <= 1.2
    assert len(set(round(d, 4) for d in delays)) > 100, "hint collapsed the jitter"


def test_small_hint_does_not_shorten_a_large_backoff() -> None:
    """max(), not override: a 1ms hint must not undo attempt 4's backoff."""
    rng = random.Random(4)
    delays = [
        backoff_delay_s(4, base_ms=200, max_ms=5_000, retry_after_ms_hint=1, rng=rng)
        for _ in range(300)
    ]
    assert max(delays) > 1.0


def test_thundering_herd_is_measurably_reduced_by_full_jitter() -> None:
    """The reason jitter is not optional.

    60 clients, 4 attempts each, counting requests that a 10-concurrent server
    would reject. Measured:

        no jitter      peak 60 req/20ms    200 rejected
        equal jitter   peak 17 req/20ms     14 rejected
        FULL jitter    peak 12 req/20ms      5 rejected

    Without jitter every client that failed together retries at the same
    instant, recreating the overload that caused the failure - the retry storm
    becomes the outage.
    """
    clients, attempts, base_ms, window_ms, capacity = 60, 4, 200, 20, 10

    def rejected(delay_for) -> int:
        rng = random.Random(7)
        arrivals: list[float] = []
        for _ in range(clients):
            t = 0.0
            for attempt in range(attempts):
                t += delay_for(attempt, rng)
                arrivals.append(t)
        hist = Counter(int(t // window_ms) for t in arrivals)
        return sum(n - capacity for n in hist.values() if n > capacity)

    none = rejected(lambda a, _rng: base_ms * (2**a))
    full = rejected(lambda a, rng: rng.uniform(0, base_ms * (2**a)))

    assert full < none / 5, f"full jitter rejected {full}, no jitter {none}"
