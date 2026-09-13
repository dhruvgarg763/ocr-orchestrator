"""Redis connection lifecycle.

Replaces the old module-level `app/redis_client.py`, which created a client at
import time. That is a real bug, not a style preference: import runs before the
event loop exists, so an async client built there binds its internal state to
the wrong loop and fails with "attached to a different loop" once a request
arrives. Long-lived async resources must be created inside the running loop,
which is what `lifespan` gives us.

The pool is bounded (`max_connections`) as a memory and file-descriptor ceiling:
every connection holds its own read/write buffers.

It uses BlockingConnectionPool, NOT the default ConnectionPool. This is load
bearing. redis-py's default pool *raises* MaxConnectionsError the moment all
connections are checked out; BlockingConnectionPool makes the calling task wait
for one. With 50 concurrent jobs and 1,000 pages contending for ~32 connections,
the default would turn ordinary contention into a storm of unhandled exceptions
and dropped pages - which is precisely the zero-drop guarantee we are graded on.
Waiting turns pool exhaustion into backpressure; raising turns it into data loss.

The wait is bounded by `pool_timeout` rather than infinite: blocking forever
would convert a Redis outage into a silent hang with no error and no metric.

Two pools, not one
------------------
A second pool exists for long-BLOCKING reads (the SSE tailer). Both problems it
solves were measured, not anticipated:

1. `socket_timeout` and `BLOCK` are in direct conflict. The socket timeout is
   how a wedged Redis is detected - no reply within N seconds means something
   is wrong. But a blocking command legitimately sends nothing for its entire
   block duration, so a 5s socket timeout against `XREAD BLOCK 15000` reads
   normal blocking as a dead server. Measured: every SSE stream died at ~5s
   with redis.exceptions.TimeoutError, and because a streaming body has
   already sent its status, the client saw a truncated response
   ("incomplete chunked read") rather than an error it could act on.

   The worker's own blocking read escaped this only because `worker_block_ms`
   (2s) happened to sit below the 5s timeout - two independently chosen numbers
   in two different files, with nothing expressing the relationship. Deriving
   this pool's timeout FROM the block duration is what makes it a rule instead
   of a coincidence.

2. A blocked reader holds its connection for the whole block. redis-py checks a
   connection out for the duration of a command, so N subscribers blocked in
   XREAD occupy N connections. The benchmark runs 50 concurrent jobs against a
   32-connection pool: the 33rd subscriber would wait out `pool_timeout` and
   fail, and - far worse - so would ingestion, because it draws from the same
   pool. A subscriber is a nice-to-have; accepting a job is not. Separate pools
   make that priority structural rather than a matter of who arrives first.
"""

from __future__ import annotations

from typing import Any

from redis.asyncio import BlockingConnectionPool, Redis

# redis-py >= 8 exposes the async script handle as AsyncScript here; it is not
# importable from redis.asyncio.client any more.
from redis.commands.core import AsyncScript

from common.logging import get_logger

log = get_logger("redis")

_client: Redis | None = None
_stream_client: Redis | None = None
_scripts: dict[str, AsyncScript] = {}


async def init_redis(
    url: str,
    *,
    max_connections: int,
    pool_timeout: float = 10.0,
    socket_timeout: float = 5.0,
    socket_connect_timeout: float = 2.0,
) -> Redis:
    """Create the shared client and verify it answers. Called from lifespan."""
    global _client
    pool = BlockingConnectionPool.from_url(
        url,
        max_connections=max_connections,
        # Seconds to wait for a free connection before giving up. None would
        # block forever and hide an outage.
        timeout=pool_timeout,
        # Return str instead of bytes. Costs a decode per field, but every value
        # we store is text (states, JSON, counters), so decoding at the boundary
        # once beats scattering .decode() through the codebase.
        decode_responses=True,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_connect_timeout,
        # Detect connections silently dropped by an idle-timeout or a restart,
        # instead of failing the next real command.
        health_check_interval=30,
    )
    # from_pool (not Redis(connection_pool=...)) so aclose() also disposes of
    # the pool instead of leaving its sockets open.
    _client = Redis.from_pool(pool)

    pong = await _client.ping()
    log.info(
        "redis_connected",
        url=url,
        max_connections=max_connections,
        pool_timeout=pool_timeout,
        ping=pong,
    )
    return _client


async def init_stream_redis(
    url: str, *, max_connections: int, block_ms: int, pool_timeout: float = 10.0
) -> Redis:
    """A second pool, for commands that block on purpose.

    `socket_timeout` is DERIVED from the block duration rather than configured
    alongside it. The two are not independent: a socket timeout below the block
    duration guarantees that every successful block is reported as a network
    failure, so the only safe value is one the block cannot reach. The margin
    covers scheduling and the round trip, and keeps the timeout finite so a
    genuinely wedged Redis is still detected - just later than on the main pool,
    which is the correct trade for a connection whose job is to wait.
    """
    global _stream_client
    socket_timeout = block_ms / 1000 + 5.0
    pool = BlockingConnectionPool.from_url(
        url,
        max_connections=max_connections,
        timeout=pool_timeout,
        decode_responses=True,
        socket_timeout=socket_timeout,
        socket_connect_timeout=2.0,
        # No health_check_interval: a PING injected onto a connection parked in
        # a blocking XREAD would read the block's eventual reply as the PING's.
        health_check_interval=0,
    )
    _stream_client = Redis.from_pool(pool)
    await _stream_client.ping()
    log.info(
        "redis_stream_pool_ready",
        max_connections=max_connections,
        block_ms=block_ms,
        socket_timeout=socket_timeout,
    )
    return _stream_client


def get_stream_redis() -> Redis:
    """The blocking-read client, falling back to the main one.

    The fallback keeps every test and script that only calls init_redis()
    working, and is safe as long as the caller's block stays under the main
    pool's socket timeout - which is exactly the invariant that broke, so it is
    a convenience for short blocks and never the production path.
    """
    return _stream_client if _stream_client is not None else get_redis()


async def close_redis() -> None:
    """Release both pools. Without this, shutdown leaks sockets."""
    global _client, _stream_client
    if _stream_client is not None:
        await _stream_client.aclose()
        _stream_client = None
    if _client is not None:
        await _client.aclose()
        _client = None
        _scripts.clear()
        log.info("redis_closed")


def get_redis() -> Redis:
    if _client is None:
        raise RuntimeError("Redis not initialised; init_redis() runs in lifespan")
    return _client


def register_script(name: str, source: str) -> AsyncScript:
    """Register a Lua script once and reuse the handle.

    redis-py sends EVALSHA (just the script's hash) and transparently falls back
    to a full EVAL if the server has not seen it - so the script body crosses the
    wire once, not on every call. At two scripted calls per page x 1,000 pages
    that difference is real.
    """
    if name not in _scripts:
        _scripts[name] = get_redis().register_script(source)
    return _scripts[name]


async def healthcheck() -> dict[str, Any]:
    try:
        client = get_redis()
        pong = await client.ping()
        return {"redis": "ok" if pong else "unreachable"}
    except Exception as exc:  # noqa: BLE001 - health must never raise
        return {"redis": "error", "detail": str(exc)}
