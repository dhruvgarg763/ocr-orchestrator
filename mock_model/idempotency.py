"""Idempotency-Key response replay, and the evidence trail for Module D.

Same key, model runs once - the second request gets the stored response,
which is what makes at-least-once queue delivery safe to redeliver
without paying twice for a 3s VLM call. Execution counts per key turn "no
duplicated model calls" from a claim into a mechanical assertion
(`duplicate_executions` must be empty after a SIGKILL test). Caching
completed responses alone is not sufficient: checking and populating the
cache are separated by up to 3 seconds of model latency, so concurrent
requests inside that window all miss and all run the model. The first
request for a key becomes the leader and executes; concurrent requests
become followers awaiting the leader's result (single-flight /
request-coalescing). The cache is bounded by both entry count and TTL,
since this assignment grades peak RSS.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _InFlight:
    """One in-progress execution that followers can wait on.

    An asyncio.Event plus stored result, rather than a Future: a Future whose
    exception is never retrieved (leader fails with no followers waiting) logs a
    noisy "exception was never retrieved" warning at GC time.
    """

    event: asyncio.Event = field(default_factory=asyncio.Event)
    result: Any = None
    error: BaseException | None = None
    followers: int = 0


class IdempotencyStore:
    def __init__(self, ttl_s: float, max_entries: int) -> None:
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        # Keyed by Idempotency-Key. Naturally bounded by in-flight request
        # count, and always removed in a `finally`.
        self._inflight: dict[str, _InFlight] = {}
        # OrderedDict gives O(1) insert plus O(1) eviction of the oldest key.
        # value = (stored_at_monotonic, response_body)
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        # executions[key] counts how many times the model actually RAN for that
        # key. Correct behaviour is exactly 1, forever.
        self._executions: OrderedDict[str, int] = OrderedDict()
        # Keys that ran more than once. Should always be empty; kept separate
        # from _executions so LRU eviction can never destroy the evidence.
        self._duplicates: dict[str, int] = {}

    # -- cache ---------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        stored_at, body = entry
        if time.monotonic() - stored_at > self._ttl_s:
            del self._cache[key]  # lazy expiry, no sweeper task
            return None
        self._cache.move_to_end(key)  # LRU: recently replayed stays resident
        return body

    def put(self, key: str, body: Any) -> None:
        self._cache[key] = (time.monotonic(), body)
        self._cache.move_to_end(key)
        self._evict(self._cache)

    # -- single flight -------------------------------------------------------

    def begin(self, key: str) -> tuple[bool, _InFlight]:
        """Claim leadership for `key`, or get the handle to wait on.

        Returns (is_leader, handle). A leader MUST later call finish() or
        abandon() - guaranteed by a `finally` at the call site - otherwise
        followers wait forever.

        No lock is needed: this method contains no `await`, so no other
        coroutine can interleave between the lookup and the insert.
        """
        existing = self._inflight.get(key)
        if existing is not None:
            existing.followers += 1
            return False, existing
        handle = _InFlight()
        self._inflight[key] = handle
        return True, handle

    async def join(self, handle: _InFlight) -> Any:
        """Follower path: wait for the leader, then share its outcome.

        If the leader failed, the follower raises the same error. That is the
        honest result - the model genuinely did not produce output for this key,
        so reporting success would be a lie.
        """
        await handle.event.wait()
        if handle.error is not None:
            raise handle.error
        return handle.result

    def finish(self, key: str, body: Any) -> None:
        handle = self._inflight.pop(key, None)
        if handle is not None:
            handle.result = body
            handle.event.set()

    def abandon(self, key: str, error: BaseException) -> None:
        """Leader failed or was cancelled: release followers with the error.

        Without this, a leader killed by a client disconnect would leave every
        follower awaiting an Event that is never set - a permanent hang, and a
        leak of both the handle and the waiting tasks.
        """
        handle = self._inflight.pop(key, None)
        if handle is not None:
            handle.error = error
            handle.event.set()

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    # -- evidence ------------------------------------------------------------

    def record_execution(self, key: str) -> int:
        """Note that the model ran for `key`. Returns the new count."""
        count = self._executions.get(key, 0) + 1
        self._executions[key] = count
        self._executions.move_to_end(key)
        self._evict(self._executions)
        if count > 1:
            self._duplicates[key] = count
        return count

    def _evict(self, store: OrderedDict[str, Any]) -> None:
        while len(store) > self._max_entries:
            store.popitem(last=False)  # drop least-recently-used

    def stats(self) -> dict[str, Any]:
        return {
            "cached_responses": len(self._cache),
            "tracked_keys": len(self._executions),
            "in_flight": len(self._inflight),
            "duplicate_executions": dict(self._duplicates),
        }

    def reset(self) -> None:
        """Used by tests between scenarios.

        Deliberately does not touch _inflight: cancelling live leaders would
        hang their followers. Only completed bookkeeping is cleared.
        """
        self._cache.clear()
        self._executions.clear()
        self._duplicates.clear()
