"""Page state machine, persisted in Redis.

Why Redis and not a dict
------------------------
Module D requires resuming from the exact uncompleted page after a SIGKILL. A
Python dict dies with the process, and with N worker replicas each process would
hold a different dict and no worker would know what the others finished.

Why each stage commits separately
---------------------------------
A page makes two model calls: layout (~50ms) then VLM (up to 3s). Committing
only at the end means a crash during the VLM stage loses the layout result too
and the retry re-runs both. Committing per stage makes every stage a checkpoint,
so the most work a crash can destroy is one stage.

Why transitions are a Lua script
--------------------------------
    state = await r.hget(key, "state")        # both workers read PENDING
    if state == "PENDING":
        await r.hset(key, "state", "RUNNING") # both workers write

Two workers interleave between the read and the write, both believe they own the
page, and the page is processed twice. Redis executes commands single-threaded
and runs a Lua script as one indivisible unit, so the compare-and-set below
cannot be interleaved by any other client.

Note the division of responsibility: Lua provides *atomicity*, Python holds the
*policy*. The transition table stays here where it is readable, diffable and
unit-testable; the script only enforces "apply this change if and only if the
current state is one of these".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from redis.asyncio import Redis

from app.core.redis_client import register_script
from common.logging import get_logger

log = get_logger("state")


class PageState(str, Enum):
    """str-valued so it serialises straight into a Redis hash field."""

    PENDING = "PENDING"
    LAYOUT_RUNNING = "LAYOUT_RUNNING"
    LAYOUT_DONE = "LAYOUT_DONE"
    VLM_RUNNING = "VLM_RUNNING"

    DONE = "DONE"
    """Full-fidelity result: layout + VLM both succeeded."""

    FALLBACK_DONE = "FALLBACK_DONE"
    """Degraded result: VLM retries exhausted or its breaker was open, so the
    page carries layout-only output with a low confidence flag. Still a success
    for the zero-drop guarantee - the page produced output."""

    FAILED = "FAILED"
    """Even the fallback failed. Terminal, counted, routed to the dead-letter
    stream. Never silent."""


class JobState(str, Enum):
    INGESTING = "INGESTING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# Single source of truth. Everything else is derived, so the table cannot drift
# out of sync with its own inverse.
TRANSITIONS: Mapping[PageState, frozenset[PageState]] = {
    PageState.PENDING: frozenset({PageState.LAYOUT_RUNNING, PageState.FAILED}),
    PageState.LAYOUT_RUNNING: frozenset(
        {
            PageState.LAYOUT_DONE,
            PageState.PENDING,  # retry, or reclaim after a worker died
            PageState.FAILED,
        }
    ),
    PageState.LAYOUT_DONE: frozenset(
        {
            PageState.VLM_RUNNING,
            PageState.FALLBACK_DONE,  # breaker open: skip the VLM entirely
        }
    ),
    PageState.VLM_RUNNING: frozenset(
        {
            PageState.DONE,
            PageState.LAYOUT_DONE,  # retry, or reclaim after a worker died
            PageState.FALLBACK_DONE,  # retries exhausted -> degrade
            # Should be rare once fallback exists: a VLM failure normally
            # degrades to FALLBACK_DONE because the layout result is already
            # committed. But the edge must be legal, or an unrecoverable VLM
            # error would strand the page in VLM_RUNNING forever with no
            # terminal state and no increment to done_count - so the job would
            # never report complete.
            PageState.FAILED,
        }
    ),
    PageState.DONE: frozenset(),
    PageState.FALLBACK_DONE: frozenset(),
    PageState.FAILED: frozenset(),
}

TERMINAL_STATES: frozenset[PageState] = frozenset(
    state for state, nxt in TRANSITIONS.items() if not nxt
)

RUNNING_STATES: frozenset[PageState] = frozenset(
    {PageState.LAYOUT_RUNNING, PageState.VLM_RUNNING}
)

# Where a page goes when its worker vanished mid-stage: back to the last
# committed checkpoint, never to the start.
RECLAIM_TARGET: Mapping[PageState, PageState] = {
    PageState.LAYOUT_RUNNING: PageState.PENDING,
    PageState.VLM_RUNNING: PageState.LAYOUT_DONE,
}

# Inverted table: which states may legally precede a given target. Derived, so
# callers name only where they are going and cannot accidentally widen the set
# of states they will accept - a classic source of state-machine bugs.
PREDECESSORS: Mapping[PageState, frozenset[PageState]] = {
    target: frozenset(src for src, dsts in TRANSITIONS.items() if target in dsts)
    for target in PageState
}


def can_transition(src: PageState, dst: PageState) -> bool:
    return dst in TRANSITIONS[src]


# KEYS[1] page hash, KEYS[2] job hash
# ARGV[1] new state
# ARGV[2] now (ms)
# ARGV[3] "1" if the new state is terminal
# ARGV[4] ttl in ms, to renew both keys on a successful transition
# ARGV[5] number of allowed source states
# ARGV[6 .. 5+n] the allowed source states
# ARGV[6+n ..]   extra field/value pairs to write in the same atomic step
# returns {ok, observed_state, done_count, total_pages, sweep_pages}
#   done_count/total_pages are -1 unless this call made the page terminal
#   sweep_pages > 0 asks the caller to renew the TTL on ALL of the job's pages
_TRANSITION_LUA = """
local new_state  = ARGV[1]
local now        = ARGV[2]
local terminal   = ARGV[3]
local ttl_ms     = tonumber(ARGV[4])
local n_allowed  = tonumber(ARGV[5])

local current = redis.call('HGET', KEYS[1], 'state')
if not current then
  return {0, 'MISSING', -1, -1, 0}
end

local allowed = 0
for i = 6, 5 + n_allowed do
  if current == ARGV[i] then
    allowed = 1
  end
end
if allowed == 0 then
  return {0, current, -1, -1, 0}
end

redis.call('HSET', KEYS[1], 'state', new_state, 'updated_at', now)

-- Extra fields are written in the SAME script, so a result and the state that
-- describes it can never disagree - no window where state says DONE but the
-- payload has not landed yet.
for i = 6 + n_allowed, #ARGV, 2 do
  redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1])
end

-- TTL is renewed by PROGRESS, not fixed at creation.
--
-- Set once in init_job and never touched, the TTL is a hard cap on a job's
-- lifetime rather than a cleanup policy - and it can therefore expire a job
-- that is still running. The job hash is the dangerous one: HINCRBY on a
-- missing key RECREATES it, so the line below would find no total_pages,
-- return -1, and `completed_job` could never be true again. The job would keep
-- completing pages correctly and never report complete, and every subscriber
-- would wait out sse_max_duration_s for an event that can no longer exist.
--
-- Renewing here makes result_ttl_s mean "expires after this much INACTIVITY",
-- which is what init_job's docstring always claimed it was for ("finished jobs
-- reclaim themselves"). A finished job stops transitioning, so its clock starts
-- at its last committed stage and it still reclaims itself on schedule.
--
-- Deliberately only on SUCCESS. A rejected transition - a redelivered page
-- that is already terminal, a lost reclaim race - must not extend anything, or
-- at-least-once redelivery would keep a finished job's records alive forever.
redis.call('PEXPIRE', KEYS[1], ttl_ms)
redis.call('PEXPIRE', KEYS[2], ttl_ms)

-- Renewing this page and the job hash is not sufficient on its own.
--
-- A page the job has NOT STARTED yet is never transitioned, so nothing above
-- ever touches it, and it keeps the fixed clock init_job gave it. When it
-- expires, its queue entry is still there - so a worker picks it up, finds the
-- hash gone, and `process_page` returns FAILED for "page state missing". The
-- worker acks it, and because there is no hash there is nothing to make
-- terminal and no HINCRBY to count: done_count can never reach total_pages and
-- the job hangs, exactly as if the job hash had expired. Measured by
-- tests/test_state_ttl.py, which failed at page 2 of 6 with observed='MISSING'
-- when only this page and the job hash were renewed.
--
-- Sweeping every page on EVERY transition would fix it and cost far too much:
-- four transitions per page over a 100-page job is 400 scripts, so 40,000
-- PEXPIREs per job of pure bookkeeping churn.
--
-- So the sweep is CONDITIONAL, and gated on its OWN marker rather than on a
-- key's remaining TTL.
--
-- The first attempt gated it on `PTTL` of the job hash, which never fired: the
-- job hash is renewed by every transition of every page, so its remaining TTL
-- is permanently near-full and says nothing about how long an UNSTARTED page
-- has been sitting. The two clocks are unrelated, and the test still failed at
-- page 2 of 6 with the check in place.
--
-- `pages_renewed_at` measures the thing that actually matters: wall-clock time
-- since the last sweep. Half the TTL is the threshold, so unstarted pages are
-- refreshed with at least half their lifetime still in hand, and the sweep
-- runs about twice per TTL period regardless of how many transitions occur.
--
-- The marker is written HERE, inside the same atomic script that decides to
-- sweep, so N workers transitioning pages concurrently cannot all be told to
-- sweep: exactly one sees the stale marker and claims the work.
local sweep = 0
local last_sweep = tonumber(redis.call('HGET', KEYS[2], 'pages_renewed_at')) or 0
if tonumber(now) - last_sweep > ttl_ms / 2 then
  sweep = tonumber(redis.call('HGET', KEYS[2], 'total_pages')) or 0
  redis.call('HSET', KEYS[2], 'pages_renewed_at', now)
end

local done  = -1
local total = -1
if terminal == '1' then
  done = redis.call('HINCRBY', KEYS[2], 'done_count', 1)
  -- Returned from the SAME script that incremented the counter, so exactly one
  -- caller can ever observe done == total. Read outside it, a second worker
  -- completing a page between the HINCRBY and the HGET could see the same
  -- equality and publish a duplicate job.complete - or, with the reads in the
  -- other order, neither could see it and the stream would never close.
  total = tonumber(redis.call('HGET', KEYS[2], 'total_pages')) or -1
end

return {1, new_state, done, total, sweep}
"""


@dataclass(frozen=True)
class TransitionResult:
    ok: bool
    """True if this caller performed the transition."""

    observed: str
    """State found in Redis. 'MISSING' if the page hash does not exist."""

    done_count: int
    """Terminal pages completed for the job after this call, else -1."""

    total_pages: int = -1
    """The job's page count, read in the same script as the increment above.
    -1 unless this call made the page terminal."""

    requested: PageState = PageState.PENDING

    @property
    def already_applied(self) -> bool:
        """Rejected because the page is *already* in the requested state.

        Benign: a duplicate delivery or a retried command. Distinguishing this
        from a genuinely illegal transition is what lets callers stay quiet
        about the former and shout about the latter.
        """
        return not self.ok and self.observed == self.requested.value

    @property
    def missing(self) -> bool:
        return self.observed == "MISSING"

    @property
    def completed_job(self) -> bool:
        """True only for the ONE transition that made the job's last page
        terminal.

        Uniqueness comes from HINCRBY: Redis is single threaded, so of N
        concurrent workers finishing the last N pages exactly one receives the
        return value equal to `total_pages`. That is what lets `job.complete` be
        published exactly once with no extra coordination, no lock and no
        "have I already sent it?" flag.
        """
        return self.ok and self.total_pages > 0 and self.done_count >= self.total_pages


def job_key(job_id: str) -> str:
    return f"job:{job_id}"


def page_key(job_id: str, page_index: int) -> str:
    return f"job:{job_id}:page:{page_index}"


class PageStateStore:
    def __init__(self, redis: Redis, *, ttl_s: int) -> None:
        self._redis = redis
        self._ttl_s = ttl_s

    @property
    def _script(self) -> Any:
        # Registered lazily: register_script needs a live client, which only
        # exists after lifespan startup.
        return register_script("page_transition", _TRANSITION_LUA)

    # ------------------------------------------------------------------ setup

    async def init_job(
        self, job_id: str, total_pages: int, **meta: str | int
    ) -> None:
        """Create the job hash and every page hash in one atomic round trip.

        MULTI/EXEC matters here: a half-initialised job whose page 40 hash is
        missing would have workers failing with MISSING forever. Pipelining also
        turns 101 round trips into one, which is the difference between a
        100-page ingest taking ~5ms and ~500ms - and time-to-first-page is
        graded at under 200ms.

        A TTL is set on every key so finished jobs reclaim themselves. Without
        it, Redis memory grows monotonically with total jobs ever submitted.

        The TTL set HERE is only the initial one. `transition` renews it on
        every committed stage, so it bounds inactivity rather than total job
        lifetime - a job that is still making progress cannot age out from
        under itself. See the PEXPIRE calls in _TRANSITION_LUA for why the job
        hash expiring mid-flight is corrupting rather than merely lossy.
        """
        now_ms = int(time.time() * 1000)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(
                job_key(job_id),
                mapping={
                    "job_id": job_id,
                    "total_pages": total_pages,
                    "done_count": 0,
                    "state": JobState.RUNNING.value,
                    "created_at": now_ms,
                    # Seeded so the first transition does not sweep
                    # immediately: every page hash was just given a full TTL
                    # one line below.
                    "pages_renewed_at": now_ms,
                    **{k: str(v) for k, v in meta.items()},
                },
            )
            pipe.expire(job_key(job_id), self._ttl_s)
            for index in range(total_pages):
                key = page_key(job_id, index)
                pipe.hset(
                    key,
                    mapping={
                        "page_index": index,
                        "state": PageState.PENDING.value,
                        "attempts": 0,
                        "updated_at": now_ms,
                    },
                )
                pipe.expire(key, self._ttl_s)
            await pipe.execute()

        log.info("job_initialised", job_id=job_id, total_pages=total_pages)

    # ------------------------------------------------------------ transitions

    async def transition(
        self,
        job_id: str,
        page_index: int,
        to: PageState,
        *,
        allowed_from: Iterable[PageState] | None = None,
        fields: Mapping[str, str | int | float] | None = None,
    ) -> TransitionResult:
        """Atomically move a page to `to`, if it is currently in a legal state.

        `allowed_from` defaults to the derived predecessor set, which is what you
        almost always want. Pass it explicitly only to narrow the set further -
        for example a reclaim that must apply *only* to VLM_RUNNING even though
        LAYOUT_DONE also legally precedes itself.
        """
        sources = tuple(allowed_from if allowed_from is not None else PREDECESSORS[to])
        if not sources:
            raise ValueError(f"{to.value} is unreachable: no legal predecessor")

        argv: list[str] = [
            to.value,
            str(int(time.time() * 1000)),
            "1" if to in TERMINAL_STATES else "0",
            # Renewed on every committed stage, so the TTL bounds INACTIVITY
            # rather than total job lifetime. See the script.
            str(self._ttl_s * 1000),
            str(len(sources)),
            *(s.value for s in sources),
        ]
        for name, value in (fields or {}).items():
            argv.extend((name, str(value)))

        ok, observed, done_count, total_pages, sweep_pages = await self._script(
            keys=[page_key(job_id, page_index), job_key(job_id)],
            args=argv,
        )

        if sweep_pages and int(sweep_pages) > 0:
            # Deliberately driven from Python rather than looped inside the
            # script. Lua would have to BUILD the page keys from a job id, and
            # keys a script was not given are exactly what Redis Cluster
            # forbids - it cannot route a script whose key set is not declared
            # up front. Here the keys stay explicit, and a cluster deployment
            # only needs a hash tag on the shared `job:{id}` prefix to colocate
            # them. One pipelined round trip, a couple of times per job.
            await self._renew_job_pages(job_id, int(sweep_pages))
        result = TransitionResult(
            ok=bool(ok),
            observed=observed,
            done_count=int(done_count),
            total_pages=int(total_pages),
            requested=to,
        )

        if result.ok and to in TERMINAL_STATES and result.total_pages < 0:
            # The job hash was missing when HINCRBY ran, so HINCRBY recreated it
            # without total_pages. `completed_job` can never be true for this
            # job again, so it will process every remaining page correctly and
            # never report complete.
            #
            # TTL renewal above is what stops this happening; this is the alarm
            # for if it ever does, because the symptom otherwise is a job that
            # simply hangs at 99% with nothing in the logs. Not raised: the page
            # itself committed fine, and failing it would turn a reporting
            # problem into a data-loss one.
            log.error(
                "job_hash_vanished_mid_flight",
                job_id=job_id,
                page_index=page_index,
                done_count=result.done_count,
                detail="total_pages unreadable; job.complete can no longer fire",
            )

        if not result.ok and not result.already_applied:
            # A genuine rejection. Expected during a reclaim race (another
            # worker got there first); a bug anywhere else.
            log.info(
                "transition_rejected",
                job_id=job_id,
                page_index=page_index,
                requested=to.value,
                observed=observed,
            )
        return result

    async def _renew_job_pages(self, job_id: str, total_pages: int) -> None:
        """Push every page hash's expiry back out, including unstarted ones.

        Called only when the transition script reports the job's clock has run
        below half, so this is amortised to roughly twice per job rather than
        once per stage. Best-effort: the job and the transitioning page have
        already been renewed by the script, so a failure here costs a later
        retry of the same sweep and never the transition that triggered it.
        """
        try:
            async with self._redis.pipeline(transaction=False) as pipe:
                for index in range(total_pages):
                    pipe.pexpire(page_key(job_id, index), self._ttl_s * 1000)
                await pipe.execute()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "page_ttl_sweep_failed",
                job_id=job_id,
                total_pages=total_pages,
                error=type(exc).__name__,
            )

    async def bump_attempts(self, job_id: str, page_index: int) -> int:
        """HINCRBY is atomic, so concurrent retries cannot lose a count."""
        return int(await self._redis.hincrby(page_key(job_id, page_index), "attempts", 1))

    # ---------------------------------------------------------------- reading

    async def get_page(self, job_id: str, page_index: int) -> dict[str, str]:
        return await self._redis.hgetall(page_key(job_id, page_index))

    async def get_job(self, job_id: str) -> dict[str, str]:
        return await self._redis.hgetall(job_key(job_id))

    async def progress(self, job_id: str) -> tuple[int, int]:
        """(done, total) in one round trip.

        Reads the job's own counter rather than scanning 100 page hashes:
        done_count is maintained by the same atomic script that performs the
        terminal transition, so it cannot drift or double-count.
        """
        done, total = await self._redis.hmget(
            job_key(job_id), "done_count", "total_pages"
        )
        return int(done or 0), int(total or 0)

    async def page_states(self, job_id: str, total_pages: int) -> list[str]:
        """All page states in one round trip, for status endpoints and tests."""
        async with self._redis.pipeline(transaction=False) as pipe:
            for index in range(total_pages):
                pipe.hget(page_key(job_id, index), "state")
            return await pipe.execute()
