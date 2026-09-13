"""Shared test fixtures.

State tests run against a REAL Redis rather than a fake. The whole point of
app/queue/state.py is that a Lua script executes atomically inside the Redis
server; a Python fake would reimplement that behaviour and therefore prove
nothing about the property under test. Testing the real thing is the only way a
concurrency claim is meaningful.

Isolation comes from using database 15 and flushing around every test, so the
suite can never disturb the application data in db 0.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from app.core import redis_client

# Inside docker the host is `redis`; from the host the port is published.
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest_asyncio.fixture
async def redis() -> Redis:
    """A clean Redis, or skip if one is not running.

    init_redis() is used rather than a bare client because register_script()
    resolves through the module-level client - so this also exercises the real
    initialisation path instead of a test-only shortcut.
    """
    try:
        client = await redis_client.init_redis(
            TEST_REDIS_URL, max_connections=16, socket_connect_timeout=1.0
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Redis unavailable at {TEST_REDIS_URL}: {exc}")

    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        # Also clears the cached Script handles, so the next test re-registers
        # against its own client instead of reusing a closed one.
        await redis_client.close_redis()


@pytest_asyncio.fixture
async def store(redis: Redis):
    """Page state store backed by the test database."""
    from app.queue.state import PageStateStore

    return PageStateStore(redis, ttl_s=60)


@pytest_asyncio.fixture
async def queue(redis: Redis):
    """Page queue with its consumer group already created."""
    from app.queue.streams import PageQueue

    q = PageQueue(redis)
    await q.ensure_group()
    return q



@pytest.fixture
def settings():
    """Default application settings, with the REAL reaper clocks.

    An earlier version of this fixture collapsed `reaper_min_idle_s` to 1ms so
    tests would not have to wait 30s for an entry to look abandoned. That made
    the suite flaky in both directions at once: whether a just-delivered entry
    had accumulated 1ms of idle time depended on how fast the test ran, so
    recovery cases silently found nothing to recover, and the concurrent-reaper
    case DID double-claim because 1ms of real time elapsed between the two
    XCLAIMs. Both results were properties of the fixture, not the code.

    Tests age entries explicitly instead (`abandon` in tests/test_reaper.py),
    which is deterministic and exercises the threshold that actually ships.
    """
    from app.config import Settings

    return Settings()
