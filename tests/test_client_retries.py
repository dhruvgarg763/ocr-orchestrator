"""ModelClient retry-loop tests, driven through a fake transport.

The retry loop's behaviour is only observable on the wire: how many requests it
issues, which headers it repeats, whether it honours Retry-After. A fake
transport makes all of that directly assertable, and keeps the tests fast -
exercising a 3-attempt backoff against the real mock would take seconds per
case.
"""

from __future__ import annotations

import time

import httpx
import pytest

from app.ratelimit.token_bucket import RateLimitTimeout
from app.worker.client import ModelClient, idempotency_key


class RecordingTransport(httpx.AsyncBaseTransport):
    """Replays a scripted sequence of responses and records every request."""

    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self._responses = responses
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._responses) - 1)
        outcome = self._responses[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def calls(self) -> int:
        return len(self.requests)


def ok(payload: dict | None = None) -> httpx.Response:
    return httpx.Response(200, json=payload or {"model": "m", "confidence": 0.9})


def err(code: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(code, headers=headers or {}, json={"detail": "nope"})


def build(
    transport: httpx.AsyncBaseTransport,
    *,
    max_attempts: int = 3,
    base_ms: float = 1.0,  # tiny, so tests are fast; the maths is tested in test_retry
    limiters: dict | None = None,
) -> ModelClient:
    return ModelClient(
        "http://mock",
        timeout_s=5,
        connect_timeout_s=1,
        max_connections=4,
        max_attempts=max_attempts,
        backoff_base_ms=base_ms,
        backoff_max_ms=20.0,
        limiters=limiters,
        transport=transport,
    )


# --------------------------------------------------------------- happy retry


async def test_transient_failure_is_retried_and_then_succeeds() -> None:
    """The page that would have been dropped before Step 8."""
    transport = RecordingTransport([err(500), ok({"text": "recovered"})])

    async with build(transport) as client:
        result = await client.vlm("job", 1)

    assert result == {"text": "recovered"}
    assert transport.calls == 2
    assert client.retries == 1
    assert client.retry_successes == 1


async def test_retry_reuses_the_same_idempotency_key() -> None:
    """The single most important property of the retry loop.

    A retry with a FRESH key would re-run the model. With the same key, an
    attempt whose response was merely lost in transit is served from the
    server's cache - which is what makes retrying a timeout safe at all.
    """
    transport = RecordingTransport([err(503), err(503), ok()])

    async with build(transport) as client:
        await client.vlm("job", 7)

    keys = {r.headers["idempotency-key"] for r in transport.requests}
    assert len(keys) == 1, f"retries used {len(keys)} different keys"
    assert keys.pop() == idempotency_key("job", 7, "vlm")


async def test_every_attempt_carries_trace_context() -> None:
    """A retry that loses its trace id is a retry you cannot debug."""
    transport = RecordingTransport([err(500), ok()])

    async with build(transport) as client:
        await client.layout("job", 2)

    assert all("traceparent" in r.headers for r in transport.requests)


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectError("connection refused"),
    ],
)
async def test_transport_faults_are_retried(failure: Exception) -> None:
    transport = RecordingTransport([failure, ok()])

    async with build(transport) as client:
        await client.vlm("job", 0)

    assert transport.calls == 2


# ------------------------------------------------------------ giving up


async def test_retries_are_bounded_by_max_attempts() -> None:
    """An unbounded retry loop against a down endpoint is a denial of service
    on your own dependency."""
    transport = RecordingTransport([err(500)])

    async with build(transport, max_attempts=3) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.vlm("job", 0)

    assert transport.calls == 3, "should attempt exactly max_attempts times"
    assert client.retries == 2, "2 retries after the initial attempt"
    assert client.retry_successes == 0


async def test_max_attempts_of_one_disables_retrying() -> None:
    """The config must be able to turn the feature off for A/B measurement."""
    transport = RecordingTransport([err(500)])

    async with build(transport, max_attempts=1) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.vlm("job", 0)

    assert transport.calls == 1


# ---------------------------------------------------------- not retried


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
async def test_permanent_errors_fail_on_the_first_attempt(code: int) -> None:
    """Fast failure keeps the budget for pages that can actually succeed."""
    transport = RecordingTransport([err(code)])

    async with build(transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.vlm("job", 0)

    assert transport.calls == 1, f"{code} must not be retried"


async def test_saturation_propagates_without_consuming_retries() -> None:
    """Waiting longer cannot conjure rate-limit capacity.

    RateLimitTimeout must reach the caller so the page is requeued - retrying
    in place would pin it while the worker could serve other pages. Note the
    request count is zero: we never even reached the network.
    """

    class Saturated:
        # `rate` is the AIMD setpoint the client now supplies per call. Accepted
        # and ignored here: this test is about saturation propagating, not about
        # rate control, and a double that rejects the real signature fails for
        # the wrong reason.
        async def acquire(
            self, *, max_wait_s: float, rate: float | None = None
        ) -> float:
            raise RateLimitTimeout("vlm", retry_after_s=max_wait_s)

    transport = RecordingTransport([ok()])

    async with build(transport, limiters={"vlm": Saturated()}) as client:
        with pytest.raises(RateLimitTimeout):
            await client.vlm("job", 0)

    assert transport.calls == 0
    assert client.retries == 0


# --------------------------------------------------------- Retry-After


async def test_server_retry_after_is_honoured_as_a_floor() -> None:
    """Returning before the server is ready guarantees another rejection."""
    transport = RecordingTransport(
        [err(429, {"retry-after": "1", "x-retry-after-ms": "120"}), ok()]
    )

    async with build(transport, base_ms=1.0) as client:
        started = time.monotonic()
        await client.vlm("job", 0)
        elapsed = time.monotonic() - started

    # Our own backoff base is 1ms, so anything past ~120ms came from the header.
    assert elapsed >= 0.12, f"waited only {elapsed * 1000:.0f}ms, hint was 120ms"
    assert elapsed < 1.0, "should use the precise ms header, not the 1s RFC value"


async def test_layout_and_vlm_use_distinct_keys_and_paths() -> None:
    """Sharing a key across stages would serve the VLM call the layout result."""
    transport = RecordingTransport([ok()])

    async with build(transport) as client:
        await client.layout("job", 3)
        await client.vlm("job", 3)

    paths = [r.url.path for r in transport.requests]
    keys = [r.headers["idempotency-key"] for r in transport.requests]
    assert paths == ["/v1/predict/layout", "/v1/predict/vlm"]
    assert keys[0] != keys[1]


async def test_request_body_carries_a_page_reference_not_page_bytes() -> None:
    """O(1) request bodies, whatever the document's size.

    This is the decision Step 11's memory arithmetic depends on. Admission
    control bounds Redis by counting pages at ~300 bytes each; if a page's
    CONTENT travelled with the task then a 100-page 100 MiB document would cost
    100 MiB of queue and the watermark would stop meaning anything.

    So the body carries a reference plus a bounded descriptor - geometry and an
    optionally-truncated text sample - and never the page bytes themselves.
    """
    transport = RecordingTransport([ok()])

    descriptor = {
        "page_index": 4,
        "width": 612.0,
        "height": 792.0,
        "rotation": 0,
        "text_sample": "a bounded sample",
        "text_chars": 926_956,
    }
    async with build(transport) as client:
        await client.vlm(
            "job", 4, page_ref="pdf:job#4", page_content=descriptor
        )

    import json

    body = json.loads(transport.requests[0].content)
    assert body["job_id"] == "job"
    assert body["page_index"] == 4
    assert body["page_ref"] == "pdf:job#4"
    assert body["page"] == descriptor

    # The descriptor records the page's true text length while carrying only a
    # sample of it, so size on the wire is independent of size on disk.
    assert body["page"]["text_chars"] > 900_000
    assert len(body["page"]["text_sample"]) < 100
    assert len(transport.requests[0].content) < 1024, "body must stay small"


async def test_request_body_omits_page_data_for_synthetic_jobs() -> None:
    """POST /jobs with a page count has no document behind it.

    The benchmark drives 1,000 pages that way rather than shipping 50 real
    PDFs, so the no-document path has to be a first-class case rather than an
    error.
    """
    transport = RecordingTransport([ok()])

    async with build(transport) as client:
        await client.vlm("job", 4)

    import json

    body = json.loads(transport.requests[0].content)
    assert body["page_ref"] is None
    assert body["page"] is None
