"""Unit tests for the mock server's primitives.

These are the pieces the whole orchestrator is tested *against*, so a bug here
would silently invalidate every downstream backpressure and idempotency result.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

import pytest

from mock_model.chaos import ChaosController
from mock_model.idempotency import IdempotencyStore
from mock_model.payloads import layout_payload, vlm_payload
from mock_model.ratelimit import TokenBucket


# --------------------------------------------------------------- token bucket


def test_bucket_starts_full_and_then_denies() -> None:
    bucket = TokenBucket(rate=10, burst=10)
    assert all(bucket.acquire() == 0.0 for _ in range(10))

    wait = bucket.acquire()
    # One token at 10/sec = 100ms.
    assert 0.09 < wait < 0.11


def test_bucket_refills_proportionally_to_elapsed_time() -> None:
    bucket = TokenBucket(rate=10, burst=10)
    for _ in range(10):
        bucket.acquire()

    time.sleep(0.5)
    assert bucket.remaining == 5


def test_bucket_refill_is_capped_at_burst() -> None:
    """The cap is what makes the limiter a limiter.

    Without it an idle client accumulates unbounded credit and can then dump it
    all at once, which is the exact spike a rate limit exists to prevent.
    """
    bucket = TokenBucket(rate=1000, burst=10)
    for _ in range(10):
        bucket.acquire()

    time.sleep(0.3)  # 300 tokens if uncapped
    assert bucket.remaining == 10


def test_bucket_sustained_rate_converges_on_configured_rate() -> None:
    bucket = TokenBucket(rate=50, burst=5)
    for _ in range(5):
        bucket.acquire()  # drain the burst so we measure steady state

    granted, start = 0, time.monotonic()
    while time.monotonic() - start < 1.0:
        if bucket.acquire() == 0.0:
            granted += 1
        else:
            time.sleep(0.001)

    assert 45 <= granted <= 55


@pytest.mark.parametrize("rate,burst", [(0, 10), (10, 0), (-1, 10), (10, -1)])
def test_bucket_rejects_invalid_config(rate: float, burst: int) -> None:
    with pytest.raises(ValueError):
        TokenBucket(rate=rate, burst=burst)


# ---------------------------------------------------------------------- chaos


def test_chaos_ratio_one_faults_everything() -> None:
    chaos = ChaosController()
    chaos.set("vlm", status=429, ratio=1.0, seconds=5)
    assert all(chaos.roll("vlm") is not None for _ in range(50))


def test_chaos_is_scoped_to_one_endpoint() -> None:
    chaos = ChaosController()
    chaos.set("vlm", status=429, ratio=1.0, seconds=5)
    assert chaos.roll("layout") is None


def test_chaos_rules_self_expire() -> None:
    """Time-boxing keeps a crashed test from wedging the mock for later tests."""
    chaos = ChaosController()
    chaos.set("vlm", status=429, ratio=1.0, seconds=0.2)
    assert chaos.roll("vlm") is not None

    time.sleep(0.3)
    assert chaos.roll("vlm") is None
    assert chaos.snapshot() == {}


# ---------------------------------------------------------------- idempotency


def test_idempotency_cache_is_bounded_by_entry_count() -> None:
    store = IdempotencyStore(ttl_s=60, max_entries=5)
    for i in range(20):
        store.put(f"k{i}", {"i": i})

    assert store.stats()["cached_responses"] == 5
    assert store.get("k0") is None      # oldest evicted
    assert store.get("k19") is not None  # newest retained


def test_idempotency_cache_expires_by_ttl() -> None:
    store = IdempotencyStore(ttl_s=0.2, max_entries=10)
    store.put("k", {"v": 1})
    assert store.get("k") == {"v": 1}

    time.sleep(0.3)
    assert store.get("k") is None


def test_duplicate_evidence_survives_lru_eviction() -> None:
    """The duplicate record must outlive cache pressure.

    It is kept outside the LRU precisely so that flushing the cache cannot
    destroy the evidence Module D depends on.
    """
    store = IdempotencyStore(ttl_s=60, max_entries=3)
    store.record_execution("dup")
    store.record_execution("dup")
    for i in range(50):
        store.record_execution(f"other{i}")

    assert store.stats()["duplicate_executions"] == {"dup": 2}


async def test_single_flight_coalesces_concurrent_duplicates() -> None:
    """REGRESSION: concurrent requests with one key must execute ONCE.

    A completed-result cache alone does not achieve this. Checking the cache and
    filling it are separated by the model's latency, so N requests arriving in
    that window all miss and all execute. Found live: 5 concurrent requests
    produced 5 executions before single-flight was added.
    """
    store = IdempotencyStore(ttl_s=60, max_entries=10)
    executions = 0

    async def request(key: str) -> str:
        nonlocal executions
        cached = store.get(key)
        if cached is not None:
            return cached

        is_leader, handle = store.begin(key)
        if not is_leader:
            return await store.join(handle)

        try:
            await asyncio.sleep(0.05)  # stands in for model latency
            executions += 1
            body = f"result-for-{key}"
        except BaseException as exc:
            store.abandon(key, exc)
            raise
        store.record_execution(key)
        store.put(key, body)
        store.finish(key, body)
        return body

    results = await asyncio.gather(*(request("same-key") for _ in range(10)))

    assert executions == 1
    assert results == ["result-for-same-key"] * 10
    assert store.stats()["duplicate_executions"] == {}
    assert store.stats()["in_flight"] == 0


async def test_single_flight_propagates_leader_failure_to_followers() -> None:
    """A follower must not report success for work that never succeeded."""
    store = IdempotencyStore(ttl_s=60, max_entries=10)

    async def leader() -> None:
        is_leader, _ = store.begin("k")
        assert is_leader
        await asyncio.sleep(0.02)
        store.abandon("k", RuntimeError("model exploded"))

    async def follower() -> str:
        await asyncio.sleep(0.01)  # arrive after the leader has claimed the key
        is_leader, handle = store.begin("k")
        assert not is_leader
        return await store.join(handle)

    leader_task = asyncio.create_task(leader())
    with pytest.raises(RuntimeError, match="model exploded"):
        await follower()
    await leader_task

    # And the handle is gone, so a later request can retry cleanly.
    assert store.stats()["in_flight"] == 0


# ------------------------------------------------------------------- payloads


def _digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def test_payloads_are_deterministic_for_the_same_page() -> None:
    assert _digest(vlm_payload("job-A", 3)) == _digest(vlm_payload("job-A", 3))
    assert _digest(layout_payload("job-A", 3)) == _digest(layout_payload("job-A", 3))


def test_payloads_differ_across_pages_and_jobs() -> None:
    assert _digest(vlm_payload("job-A", 3)) != _digest(vlm_payload("job-A", 4))
    assert _digest(vlm_payload("job-A", 3)) != _digest(vlm_payload("job-B", 3))


def test_payload_digest_is_stable_across_processes() -> None:
    """Seeded with hashlib, never builtin hash().

    hash() on str is salted per process by PYTHONHASHSEED, so a hash()-seeded
    generator returns different output after every restart - breaking
    determinism in exactly the crash-recovery scenario it is needed for. This
    hardcoded digest fails if anyone swaps hashlib for hash().
    """
    assert _digest(layout_payload("stable", 0)).startswith("5d87f7c2")


def test_vlm_tree_has_nested_structure_for_ted() -> None:
    """TED needs real depth; a flat list would reduce it to string distance."""
    found_depth = 0

    def walk(node: dict, depth: int) -> None:
        nonlocal found_depth
        found_depth = max(found_depth, depth)
        for child in node.get("children", []):
            walk(child, depth + 1)

    # Page 3 of job-A is known to contain a table (page -> table -> row -> cell).
    walk(vlm_payload("job-A", 3)["tree"], 0)
    assert found_depth >= 3
