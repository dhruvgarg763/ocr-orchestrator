"""Redis connection lifecycle.

Two `BlockingConnectionPool`s, not one, and not the default
`ConnectionPool`. `BlockingConnectionPool` makes a caller WAIT for a
connection instead of raising `MaxConnectionsError` - with 50 concurrent
jobs contending for ~32 connections, the default would turn ordinary
contention into unhandled exceptions and dropped pages. A second pool
exists for long-BLOCKING reads (the SSE tailer): a single `socket_timeout`
against `XREAD BLOCK 15000` reads normal blocking as a dead server
(measured: every SSE stream died at ~5s), and a blocked reader holds its
connection for the whole block, so N subscribers would starve ingestion
out of the same pool it needs to admit jobs. Clients are constructed
inside FastAPI's `lifespan`, not at import time - an async client built
before the event loop exists binds to the wrong loop and fails later with
"attached to a different loop".
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
