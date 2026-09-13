"""Per-job result stream: what the SSE endpoint tails.

A separate stream per job (`stream:out:{job_id}`), not one shared output
stream. Two reasons, both load bearing:

  * a subscriber wants one job's events, and `XREAD` has no server-side filter.
    On a shared stream a client watching a 5-page job would have to read and
    discard every event of the other 49 concurrent jobs to find its own.
  * MAXLEN is per stream. Shared, one busy job's events would evict a quiet
    job's, so the quiet job's client would lose events it never had a chance to
    read. Per job, a slow client can only ever hurt itself.

Why MAXLEN here but NOT on the task stream
------------------------------------------
`stream:pages` deliberately has no MAXLEN (see streams.py): trimming discards
the OLDEST entries, and the oldest entry in a task queue is unprocessed work -
a memory bound that eats your jobs.

Here the opposite holds. These entries are a *notification* of state that is
already durably committed to the page hash, so the stream is a cache, not the
record. Dropping the oldest event costs a client a re-read of
`GET /jobs/{id}`; dropping the oldest task would lose a page. Different data,
different trimming policy - and this is the distinction to make if asked why
one stream is capped and the other is not.

Why `seq` is assigned inside the XADD
-------------------------------------
The obvious version is two commands:

    seq = await r.incr(seq_key)          # worker A gets 7, worker B gets 8
    await r.xadd(stream, {"seq": seq})   # B's XADD lands first

Now the stream holds seq 8 before seq 7. A client that orders by `seq` but
dedupes by entry id sees seq 7 arrive "late" and can never tell whether it is
a reordering or a gap it should resync over. Assigning the counter and
appending the entry in one script makes seq order and stream order the same
order, by construction, with no coordination between workers.

That matters because `seq` and the entry id do different jobs and neither can
replace the other:

    entry id   an opaque RESUME CURSOR. Redis-assigned, usable directly as
               `XREAD` start, which is exactly what SSE `Last-Event-ID` needs.
    seq        a dense APPLICATION ORDER. Comparable, so a client can detect a
               hole; countable, so it can tell "I have 41 of 41 events".

An entry id cannot be used for the second job - `1738-0` to `1740-0` tells you
nothing about whether an entry existed in between - and seq cannot be used for
the first, because `XREAD` does not accept it.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis

from app.core.redis_client import register_script
from common.logging import get_logger

log = get_logger("results")

# Two per page plus one job event. A client never needs to interpret these
# strings, but a human reading a log or a curl transcript does, so they are
# dotted names rather than integers.
PAGE_PARTIAL = "page.partial"
"""Layout committed. A real, usable result - blocks, types, reading order -
and the reason time-to-first-page can be under 200ms while the VLM takes 3s."""

PAGE_FINAL = "page.final"
"""The page reached a terminal state: DONE, FALLBACK_DONE or FAILED. One event
type for all three, carrying `state`, so a client has exactly one "this page is
settled" case to handle rather than three."""

JOB_COMPLETE = "job.complete"
"""Every page is terminal. Closes the stream."""

# Synthesised by the SSE layer per connection and never stored, because they
# describe THIS subscriber's view rather than anything that happened to the job.
STREAM_OPEN = "stream.open"
GAP = "stream.gap"

_ID = re.compile(r"\A\d+-\d+\Z")


def out_stream(job_id: str) -> str:
    return f"stream:out:{job_id}"


def seq_key(job_id: str) -> str:
    return f"stream:out:{job_id}:seq"


def parse_entry_id(raw: str | None) -> str | None:
    """Validate a client-supplied stream id, or return None.

    `Last-Event-ID` is an arbitrary request header: it arrives from whatever the
    client last saw, from a proxy, or from a hand-written curl. Feeding it
    unchecked to `XREAD` turns a malformed header into a 500, so an id that is
    not `<ms>-<seq>` is treated as "no cursor" and the subscriber gets a
    snapshot instead.
    """
    if not raw:
        return None
    raw = raw.strip()
    return raw if _ID.match(raw) else None


def id_tuple(entry_id: str) -> tuple[int, int]:
    """Stream ids compare as NUMBER PAIRS, never as strings.

    Lexicographically `"10-0" < "9-0"`, so a string comparison reports that
    every id in millisecond 10 predates millisecond 9. Gap detection is built
    on exactly this comparison, so getting it wrong would make the gap check
    fire at random and silently stop firing when it mattered.
    """
    ms, _, seq = entry_id.partition("-")
    return int(ms), int(seq or 0)


# KEYS[1] result stream, KEYS[2] seq counter
# ARGV[1] maxlen, ARGV[2] ttl seconds, ARGV[3..] field/value pairs
# returns {entry_id, seq}
_PUBLISH_LUA = """
local seq = redis.call('INCR', KEYS[2])

local args = {'XADD', KEYS[1], 'MAXLEN', '~', ARGV[1], '*', 'seq', seq}
for i = 3, #ARGV do
  args[#args + 1] = ARGV[i]
end
local id = redis.call(unpack(args))

-- Both keys, every publish. The stream is a cache of committed state, so it
-- must reclaim itself even for a job nobody ever subscribed to; and the
-- counter must outlive the stream it numbers, or a late publish after the
-- counter expired would restart at 1 and hand out seqs the client has seen.
redis.call('EXPIRE', KEYS[1], ARGV[2])
redis.call('EXPIRE', KEYS[2], ARGV[2])

return {id, seq}
"""


@dataclass(frozen=True)
class ResultEvent:
    entry_id: str
    """Resume cursor. Goes out as the SSE `id:` field."""

    seq: int
    event: str
    data: dict[str, Any]

    def sse(self) -> str:
        """Render as an SSE frame.

        `seq` and `stream_id` are merged INTO the payload as well as being sent
        as the SSE `id:`. A JavaScript `EventSource` exposes `event.lastEventId`
        but a plain HTTP client reading the body usually parses only `data:`, so
        duplicating them costs ~30 bytes and means every kind of client can
        order and dedupe without special-casing the framing.
        """
        body = json.dumps(
            {**self.data, "seq": self.seq, "stream_id": self.entry_id},
            separators=(",", ":"),
        )
        # The blank line terminates the frame. Without it the client buffers
        # forever and time-to-first-page is infinite - the single easiest way to
        # get SSE wrong.
        return f"id: {self.entry_id}\nevent: {self.event}\ndata: {body}\n\n"


def _decode(entry_id: str, fields: dict[str, str]) -> ResultEvent:
    return ResultEvent(
        entry_id=entry_id,
        seq=int(fields.get("seq", 0)),
        event=fields.get("event", ""),
        data=json.loads(fields.get("data") or "{}"),
    )


class ResultPublisher:
    """Write side. Lives in the worker, next to the state transitions."""

    def __init__(self, redis: Redis, *, maxlen: int, ttl_s: int) -> None:
        self._redis = redis
        self._maxlen = maxlen
        self._ttl_s = ttl_s

    @property
    def _script(self) -> Any:
        return register_script("result_publish", _PUBLISH_LUA)

    async def publish(
        self, job_id: str, event: str, payload: dict[str, Any]
    ) -> ResultEvent:
        """Append one event. One round trip, atomic in seq.

        The payload is stored as a single JSON field rather than spread across
        stream fields. Stream fields are flat strings, so a nested layout result
        would have to be flattened and reassembled; and a schema change would
        otherwise mean readers that have to cope with two field layouts.
        """
        data = {"job_id": job_id, "emitted_at_ms": int(time.time() * 1000), **payload}
        entry_id, seq = await self._script(
            keys=[out_stream(job_id), seq_key(job_id)],
            args=[
                self._maxlen,
                self._ttl_s,
                "event",
                event,
                "data",
                json.dumps(data, separators=(",", ":")),
            ],
        )
        return ResultEvent(entry_id=entry_id, seq=int(seq), event=event, data=data)


class ResultReader:
    """Read side. Lives in the API, behind the SSE endpoint."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def window(self, job_id: str) -> tuple[str | None, str | None]:
        """Oldest and newest surviving entry ids, or (None, None) if empty.

        This is what makes MAXLEN safe to expose rather than something clients
        discover by losing data. `window_lo` is the honest statement "I cannot
        replay anything older than this", and it is what the gap check below
        compares a resume cursor against.
        """
        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.xrange(out_stream(job_id), count=1)
            pipe.xrevrange(out_stream(job_id), count=1)
            first, last = await pipe.execute()
        lo = first[0][0] if first else None
        hi = last[0][0] if last else None
        return lo, hi

    async def gap_after(self, job_id: str, last_id: str) -> bool:
        """Has the entry the client wants to resume from been trimmed away?

        THE failure this endpoint has to handle explicitly. `XREAD` from a
        trimmed id does not error - it returns the entries that still exist, so
        a client that was disconnected while 200 events went past resumes
        cleanly, misses a hundred pages, and is never told. Silent data loss
        that looks exactly like success.

        A cursor strictly older than the oldest surviving entry means at least
        one event between them is gone. The subscriber is sent an explicit `gap`
        event telling it to re-read `GET /jobs/{id}`, which is authoritative
        because page state - unlike this stream - is never trimmed.
        """
        lo, _ = await self.window(job_id)
        if lo is None:
            # Nothing retained. Either no event has been published yet, or the
            # whole stream expired - neither is a gap we can prove, and crying
            # gap at a client that has simply subscribed early would be noise.
            return False
        return id_tuple(last_id) < id_tuple(lo)

    async def history(
        self, job_id: str, *, after: str | None = None, count: int = 500
    ) -> list[ResultEvent]:
        """Events already in the stream, oldest first.

        `count` is a bound, not a page size: it exists so a subscriber to a job
        with a large retained window cannot make the API materialise the whole
        window at once. MAXLEN keeps the real figure well under it.
        """
        start = f"({after}" if after else "-"
        entries = await self._redis.xrange(out_stream(job_id), min=start, count=count)
        return [_decode(entry_id, fields) for entry_id, fields in entries]

    async def tail(
        self, job_id: str, *, last_id: str, block_ms: int, count: int
    ) -> list[ResultEvent]:
        """Block until events arrive after `last_id`, or the timeout expires.

        Blocking rather than polling: Redis wakes us the instant a worker
        publishes, which is what keeps time-to-first-page a property of the
        pipeline rather than of a poll interval.

        An empty list means the block expired with nothing new - the caller's
        cue to send a heartbeat and check whether the client is still there.
        """
        response = await self._redis.xread(
            streams={out_stream(job_id): last_id}, count=count, block=block_ms
        )
        if not response:
            return []
        return [
            _decode(entry_id, fields)
            for _stream, entries in response
            for entry_id, fields in entries
        ]
