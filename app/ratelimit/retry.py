"""Error classification and backoff computation.

Pure functions, no I/O. Classification matters because retrying the wrong
error wastes budget on a failure that is merely delayed - a 400 is
permanent, a 429 is the server saying "later". Full jitter (a uniform
draw over the whole backoff window, not half of it) beats both no jitter
and equal jitter at avoiding synchronised retry storms: simulated with 60
clients retrying together against a 10-concurrent server, full jitter
gave 40x fewer rejections than none. Retrying a timeout is only safe here
because every request carries a deterministic `Idempotency-Key`
(app/worker/client.py) - a timeout means the work MAY have completed, and
without idempotency a retry would double-charge it.
"""

from __future__ import annotations

import random
from enum import Enum

import httpx

from app.ratelimit.breaker import CircuitOpen
from app.ratelimit.token_bucket import RateLimitTimeout

# Statuses worth another attempt. Everything else in the 4xx range describes a
# request that will be just as wrong the second time.
RETRYABLE_STATUSES: frozenset[int] = frozenset(
    {
        408,  # Request Timeout
        425,  # Too Early
        429,  # Too Many Requests
        500,
        502,
        503,
        504,
    }
)


class Disposition(str, Enum):
    RETRY = "retry"
    """Transient. Try again in place, after a backoff."""

    PERMANENT = "permanent"
    """Will not improve with time. Fail fast and keep the budget."""

    SATURATED = "saturated"
    """We could not even get a rate-limit token. The constraint is systemic, so
    retrying in place just holds the page hostage while the worker could be
    doing other work. Requeue instead."""

    CIRCUIT_OPEN = "circuit_open"
    """The endpoint has been judged unhealthy and we did not call it.

    Distinct from SATURATED, because the right response differs. Saturation
    means "busy, come back shortly", so the page is requeued at full fidelity.
    An open circuit means "not answering", and waiting does not help - the page
    should degrade to the output it already has. Retrying in place is the one
    thing that must NOT happen: the breaker exists precisely to stop that."""


def classify(exc: BaseException) -> Disposition:
    if isinstance(exc, CircuitOpen):
        return Disposition.CIRCUIT_OPEN

    if isinstance(exc, RateLimitTimeout):
        return Disposition.SATURATED

    if isinstance(exc, httpx.HTTPStatusError):
        return (
            Disposition.RETRY
            if exc.response.status_code in RETRYABLE_STATUSES
            else Disposition.PERMANENT
        )

    # TimeoutException and TransportError (connect errors, resets, protocol
    # errors) are both transient. Safe to retry only because of idempotency
    # keys - see the module docstring.
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return Disposition.RETRY

    # Unknown failures are treated as permanent on purpose. An unrecognised
    # error retried three times is three times the confusion, and a genuine bug
    # should surface immediately rather than after a backoff schedule.
    return Disposition.PERMANENT


def retry_after_ms(exc: BaseException) -> float | None:
    """Extract the server's own guidance about when to come back.

    Prefers `X-Retry-After-Ms`: RFC 9110's `Retry-After` only permits integer
    seconds, so a 33ms wait rounds up to 1000ms and wastes 97% of the
    endpoint's capacity. The mock sends both (mock_model/main.py).
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return None

    headers = exc.response.headers
    if (precise := headers.get("x-retry-after-ms")) is not None:
        try:
            return max(0.0, float(precise))
        except ValueError:
            pass

    if (seconds := headers.get("retry-after")) is not None:
        try:
            return max(0.0, float(seconds) * 1000)
        except ValueError:
            # Retry-After may also be an HTTP-date. Not worth parsing for an
            # internal endpoint; fall through to our own backoff.
            return None

    return None


def backoff_delay_s(
    attempt: int,
    *,
    base_ms: float,
    max_ms: float,
    retry_after_ms_hint: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Seconds to wait before attempt `attempt` (0-based).

    Full jitter: a uniform draw over [0, ceiling], where the ceiling grows
    exponentially and is capped. The cap matters - without it, attempt 10 would
    schedule a wait of 200ms * 2^10 = 3.4 minutes.

    A server hint acts as a FLOOR, not a replacement. Sleeping less than the
    server asked for guarantees another rejection, so it is wasted capacity.
    Jitter is still layered on top, because otherwise every client given the
    same Retry-After returns in the same instant - reintroducing precisely the
    convoy that jitter exists to break up.
    """
    draw = (rng or random).uniform

    ceiling_ms = min(max_ms, base_ms * (2**attempt))
    delay_ms = draw(0.0, ceiling_ms)

    if retry_after_ms_hint is not None:
        delay_ms = max(delay_ms, retry_after_ms_hint) + draw(0.0, base_ms)

    return delay_ms / 1000.0
