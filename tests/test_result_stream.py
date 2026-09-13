"""Result stream mechanics: sequencing, trimming, and the gap it creates.

The two properties worth a test here are the ones that fail silently rather
than loudly:

  * `seq` must agree with stream order, or a client cannot distinguish a
    reordering from a hole
  * a resume cursor older than the retained window must be DETECTED, because
    XREAD's own behaviour in that case is to succeed while losing data
"""

from __future__ import annotations

import asyncio
import json

import pytest
from redis.asyncio import Redis

from app.queue.results import (
    JOB_COMPLETE,
    PAGE_FINAL,
    PAGE_PARTIAL,
    ResultEvent,
    ResultPublisher,
    ResultReader,
    id_tuple,
    out_stream,
    parse_entry_id,
    seq_key,
)


@pytest.fixture
def publisher(redis: Redis) -> ResultPublisher:
    return ResultPublisher(redis, maxlen=256, ttl_s=60)


@pytest.fixture
def reader(redis: Redis) -> ResultReader:
    return ResultReader(redis)


# ------------------------------------------------------------------ sequencing


async def test_seq_starts_at_one_and_increments(publisher: ResultPublisher) -> None:
    seqs = [
        (await publisher.publish("j", PAGE_PARTIAL, {"page_index": i})).seq
        for i in range(5)
    ]
    assert seqs == [1, 2, 3, 4, 5]


async def test_publish_returns_a_usable_entry_id(
    publisher: ResultPublisher, reader: ResultReader
) -> None:
    """The returned id must be the real stream id, not a local invention - the
    SSE layer hands it straight back to XREAD as a resume cursor."""
    event = await publisher.publish("j", PAGE_PARTIAL, {"page_index": 0})

    assert await reader.history("j", after=event.entry_id) == []
    assert [e.entry_id for e in await reader.history("j")] == [event.entry_id]


async def test_seq_order_matches_stream_order_under_concurrency(
    publisher: ResultPublisher, reader: ResultReader
) -> None:
    """The reason the counter lives inside the XADD script.

    Concurrent publishers must produce a stream whose entries are in ascending
    seq order. If they are not, a client that orders by seq cannot tell a
    reordering from a missing event - so it can never safely decide it has
    everything.
    """
    await asyncio.gather(
        *(publisher.publish("j", PAGE_FINAL, {"page_index": i}) for i in range(60))
    )

    seqs = [e.seq for e in await reader.history("j")]

    assert seqs == sorted(seqs), "stream order disagrees with seq order"
    assert seqs == list(range(1, 61)), "a seq was skipped or reused"


async def test_naive_incr_then_xadd_inverts_under_concurrency(
    redis: Redis, reader: ResultReader
) -> None:
    """The bug the Lua script exists to prevent, demonstrated rather than
    asserted.

    Two commands, so another publisher can land its XADD between this one's
    INCR and its own XADD. The counter is still perfectly unique - INCR is
    atomic - but uniqueness was never the property we needed. ORDER was.

    Skipped rather than failed if no inversion happens to occur: this is a race,
    so the honest test is "when it does interleave, it inverts", and a run that
    happened not to interleave has not disproved anything.
    """

    async def naive(page: int) -> None:
        seq = await redis.incr(seq_key("j"))
        await redis.xadd(out_stream("j"), {"seq": seq, "event": "x", "data": "{}"})

    await asyncio.gather(*(naive(i) for i in range(60)))

    seqs = [e.seq for e in await reader.history("j")]

    assert sorted(seqs) == list(range(1, 61)), "INCR itself is atomic; seqs are unique"
    if seqs == sorted(seqs):
        pytest.skip("no interleaving occurred this run; the race is not disproved")
    inversions = sum(1 for a, b in zip(seqs, seqs[1:]) if b < a)
    assert inversions > 0


async def test_stream_ids_compare_as_numbers_not_strings() -> None:
    """REGRESSION GUARD for the gap check.

    Lexicographically "10-0" < "9-0", so a string comparison decides that
    millisecond 10 predates millisecond 9. The gap check is exactly this
    comparison, so a string compare would make it fire at random and stop
    firing when it mattered.
    """
    assert "10-0" < "9-0", "string ordering really is wrong here"
    assert id_tuple("10-0") > id_tuple("9-0")
    assert id_tuple("5-2") > id_tuple("5-1")
    assert id_tuple("7") == (7, 0), "a bare millisecond is a valid id form"


# ------------------------------------------------------------------- framing


def test_sse_frame_carries_id_event_and_a_blank_terminator() -> None:
    """The blank line is what tells a client the frame is complete. Omit it and
    the client buffers forever, so time-to-first-page becomes infinite."""
    frame = ResultEvent("1-0", 3, PAGE_PARTIAL, {"page_index": 7}).sse()

    assert frame.startswith("id: 1-0\nevent: page.partial\ndata: ")
    assert frame.endswith("\n\n")
    assert "\n\n" not in frame[:-2], "a premature blank line would split the frame"


def test_sse_payload_duplicates_seq_and_id_into_the_body() -> None:
    """A browser reads the framing; a plain HTTP client usually parses only
    `data:`. Duplicating costs ~30 bytes and lets both order and dedupe."""
    frame = ResultEvent("1-0", 3, PAGE_FINAL, {"page_index": 7}).sse()
    body = json.loads(frame.split("data: ", 1)[1].strip())

    assert body == {"page_index": 7, "seq": 3, "stream_id": "1-0"}


@pytest.mark.parametrize("raw", [None, "", "  ", "$", "abc", "1-", "-1", "1-2-3", "x-y"])
def test_a_malformed_resume_cursor_is_treated_as_absent(raw: str | None) -> None:
    """`Last-Event-ID` is an arbitrary request header. Handing it unchecked to
    XREAD turns a bad header into a 500; treating it as "no cursor" gives the
    client a snapshot, which is what it wanted anyway."""
    assert parse_entry_id(raw) is None


@pytest.mark.parametrize("raw", ["1-0", "1738000000000-12", " 1-0 "])
def test_a_wellformed_cursor_is_kept(raw: str) -> None:
    assert parse_entry_id(raw) == raw.strip()


# -------------------------------------------------------- window, trim, gap


async def test_window_is_empty_before_anything_is_published(
    reader: ResultReader,
) -> None:
    assert await reader.window("nothing-here") == (None, None)


async def test_window_reports_oldest_and_newest(
    publisher: ResultPublisher, reader: ResultReader
) -> None:
    first = await publisher.publish("j", PAGE_PARTIAL, {"page_index": 0})
    await publisher.publish("j", PAGE_PARTIAL, {"page_index": 1})
    last = await publisher.publish("j", PAGE_FINAL, {"page_index": 0})

    assert await reader.window("j") == (first.entry_id, last.entry_id)


async def test_maxlen_trims_the_oldest_events(redis: Redis, reader: ResultReader) -> None:
    """Trimming is safe HERE - and only here - because these entries notify
    about state that is durably committed elsewhere. The task stream has no
    MAXLEN for the opposite reason: its oldest entry is unprocessed work."""
    tight = ResultPublisher(redis, maxlen=5, ttl_s=60)
    for i in range(200):
        await tight.publish("j", PAGE_FINAL, {"page_index": i})

    surviving = await reader.history("j")

    assert len(surviving) < 200, "MAXLEN did not trim"
    # `MAXLEN ~` trims whole radix nodes, so the real length sits at or above
    # the figure. Asserting equality here would be asserting Redis's internal
    # node size, which is not ours to depend on.
    assert surviving[-1].data["page_index"] == 199, "the NEWEST must always survive"


async def test_a_cursor_inside_the_window_is_not_a_gap(
    publisher: ResultPublisher, reader: ResultReader
) -> None:
    first = await publisher.publish("j", PAGE_PARTIAL, {"page_index": 0})
    await publisher.publish("j", PAGE_FINAL, {"page_index": 0})

    assert await reader.gap_after("j", first.entry_id) is False


async def test_a_trimmed_cursor_is_detected_as_a_gap(
    redis: Redis, reader: ResultReader
) -> None:
    """THE failure this module exists to catch.

    XREAD from a trimmed id does not error. It returns the entries that still
    exist, so a client disconnected while 200 events went past resumes
    cleanly, is permanently missing a run of pages, and is never told - data
    loss that is indistinguishable from success. Only an explicit comparison
    against the retained window turns it into something reportable.
    """
    tight = ResultPublisher(redis, maxlen=5, ttl_s=60)
    stale = await tight.publish("j", PAGE_PARTIAL, {"page_index": 0})
    for i in range(1, 200):
        await tight.publish("j", PAGE_FINAL, {"page_index": i})

    assert await reader.gap_after("j", stale.entry_id) is True

    # And the thing being guarded against: XREAD is perfectly happy.
    resumed = await reader.tail("j", last_id=stale.entry_id, block_ms=10, count=500)
    assert resumed, "XREAD returns data from a trimmed cursor without complaint"
    assert resumed[0].seq > 2, "which is precisely the silent hole"


async def test_an_empty_stream_is_not_reported_as_a_gap(reader: ResultReader) -> None:
    """A client that subscribes before the first page commits has lost nothing.
    Crying gap at it would make the signal meaningless."""
    assert await reader.gap_after("never-published", "1-0") is False


# ----------------------------------------------------------------- reading


async def test_history_excludes_the_cursor_itself(
    publisher: ResultPublisher, reader: ResultReader
) -> None:
    """`after` is exclusive, or a resuming client is re-sent the last event it
    already has - and would double-count it against total_pages."""
    a = await publisher.publish("j", PAGE_PARTIAL, {"page_index": 0})
    b = await publisher.publish("j", PAGE_FINAL, {"page_index": 0})

    assert [e.entry_id for e in await reader.history("j", after=a.entry_id)] == [
        b.entry_id
    ]


async def test_history_is_bounded_by_count(
    publisher: ResultPublisher, reader: ResultReader
) -> None:
    for i in range(50):
        await publisher.publish("j", PAGE_FINAL, {"page_index": i})

    assert len(await reader.history("j", count=10)) == 10


async def test_tail_returns_nothing_when_idle(reader: ResultReader) -> None:
    """An empty list is the caller's cue to heartbeat and check for a
    disconnected client - the only moment it can."""
    assert await reader.tail("quiet", last_id="0", block_ms=20, count=10) == []


async def test_tail_wakes_on_a_publish(
    publisher: ResultPublisher, reader: ResultReader
) -> None:
    """Blocking, not polling: first-page latency must be a property of the
    pipeline, not of a poll interval."""

    async def publish_soon() -> None:
        await asyncio.sleep(0.05)
        await publisher.publish("j", PAGE_PARTIAL, {"page_index": 3})

    asyncio.create_task(publish_soon())
    events = await reader.tail("j", last_id="0", block_ms=3_000, count=10)

    assert [e.data["page_index"] for e in events] == [3]


async def test_payload_survives_the_round_trip(
    publisher: ResultPublisher, reader: ResultReader
) -> None:
    """Stored as one JSON field, so nested layout output needs no flattening."""
    layout = {"blocks": [{"bbox": [1, 2, 3, 4], "type": "text"}], "reading_order": [0]}
    await publisher.publish("j", PAGE_PARTIAL, {"page_index": 2, "layout": layout})

    (event,) = await reader.history("j")

    assert event.data["layout"] == layout
    assert event.data["job_id"] == "j"
    assert event.data["emitted_at_ms"] > 0


# --------------------------------------------------------------------- hygiene


async def test_both_keys_carry_a_ttl(redis: Redis, publisher: ResultPublisher) -> None:
    """The stream is a cache, so it must reclaim itself even for a job nobody
    subscribed to. The counter needs a TTL for a sharper reason: if it expired
    while the stream lived, a later publish would restart at seq 1 and reissue
    numbers a client had already seen."""
    await publisher.publish("j", JOB_COMPLETE, {"total_pages": 1})

    assert 0 < await redis.ttl(out_stream("j")) <= 60
    assert 0 < await redis.ttl(seq_key("j")) <= 60


async def test_streams_are_per_job(publisher: ResultPublisher, reader: ResultReader) -> None:
    """Per job, not shared. XREAD has no server-side filter, so on a shared
    stream a 5-page job's subscriber would read and discard every event of the
    other 49 concurrent jobs - and one busy job's events would evict a quiet
    job's, losing data the quiet client never had a chance to see."""
    await publisher.publish("a", PAGE_FINAL, {"page_index": 0})
    await publisher.publish("b", PAGE_FINAL, {"page_index": 0})

    assert len(await reader.history("a")) == 1
    assert (await publisher.publish("b", PAGE_FINAL, {"page_index": 1})).seq == 2
