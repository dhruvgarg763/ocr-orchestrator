"""Mock model server configuration.

These knobs let the mock act as a tunable adversary: the load tests dial failure
rates and rate limits up and down to drive the orchestrator into 429 storms,
latency spikes and breaker-open states on demand.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MockSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOCK_", env_file=".env", extra="ignore")

    log_level: str = "INFO"

    # --- Fast-Layout-Model: 50ms, 100 RPS, 2% failures --------------------
    layout_latency_ms: float = Field(default=50.0, ge=0)
    layout_rps: float = Field(default=100.0, gt=0)
    layout_burst: int = Field(default=100, gt=0)
    layout_failure_rate: float = Field(default=0.02, ge=0, le=1)

    # --- Heavy-VLM-Model: 1500-3000ms, 10 RPS, 5% failures ----------------
    vlm_latency_min_ms: float = Field(default=1_500.0, ge=0)
    vlm_latency_max_ms: float = Field(default=3_000.0, ge=0)
    vlm_rps: float = Field(default=10.0, gt=0)
    vlm_burst: int = Field(default=10, gt=0)
    vlm_failure_rate: float = Field(default=0.05, ge=0, le=1)

    # How long a response is replayable for a given Idempotency-Key.
    idempotency_ttl_s: float = Field(default=300.0, gt=0)

    # Hard cap on cached responses. Bounded by count as well as TTL, because a
    # TTL alone still allows unbounded growth within one TTL window.
    idempotency_max_entries: int = Field(default=20_000, gt=0)


@lru_cache(maxsize=1)
def get_mock_settings() -> MockSettings:
    return MockSettings()
