"""Request contract for the mock endpoints.

Only *requests* are modelled. Responses are returned as plain dicts on purpose:
pydantic serialisation would burn CPU on the hot path for 1,000 pages x 2 calls
with no benefit, since nothing downstream validates against a mock's schema.
Requests are validated because a malformed one should fail loudly and fast.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=128)
    page_index: int = Field(ge=0)

    page_ref: str | None = None
    """Where the page bytes live. The mock never reads it - and that is the
    point: passing a *reference* rather than the bytes is what keeps the
    orchestrator's memory O(1) per page instead of O(page size)."""

    text_hint: str | None = Field(default=None, max_length=4096)
    """Optional text sampled from the PDF, capped so a caller cannot use the
    mock as an unbounded memory sink."""


class ChaosRequest(BaseModel):
    """Body for POST /admin/chaos."""

    endpoint: str = Field(pattern="^(layout|vlm)$")
    status: int = Field(default=429, ge=200, le=599)
    """429 exercises backpressure, 500 exercises retries, 200 means
    latency-degradation only."""

    ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    seconds: float = Field(default=30.0, gt=0, le=600)
    """Bounded so a forgotten rule cannot wedge the mock indefinitely."""

    extra_latency_ms: float = Field(default=0.0, ge=0, le=60_000)
