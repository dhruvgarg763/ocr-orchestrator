"""End-to-end verification against a running stack.

    docker compose up -d --build
    python scripts/verify.py

Checks every behaviour claimed so far and exits non-zero if any fails, so it can
also serve as a smoke gate in CI. Unit-level properties live in tests/; this
script verifies the assembled system over real HTTP and real Redis.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import sys
import time
from typing import Any

import httpx
from redis.asyncio import Redis

# Overridable so this runs from EITHER side of the container boundary. From the
# host the ports are published; from inside `docker compose exec api` only the
# api is on localhost and the other two are reachable by service name. Hardcoding
# the host form made the script silently unrunnable in-container, which matters
# because that is where it has to run when the host has no dependencies
# installed - and "ModuleNotFoundError: redis" is an unhelpful way to discover
# that.
API = os.getenv("VERIFY_API", "http://localhost:8000")
MOCK = os.getenv("VERIFY_MOCK", "http://localhost:8001")
REDIS_URL = os.getenv("VERIFY_REDIS_URL", "redis://localhost:6379/0")

# Duplicated from app.queue.streams rather than imported, on purpose: this
# script is a black box over the running stack and imports nothing from `app`,
# so it stays runnable from inside a container that has the code and from a
# host that does not. The cost is that a rename here is a second edit; the
# benefit is that the script cannot accidentally pass by sharing a bug with the
# implementation it is checking.
POISON_COUNTER_KEY = "metrics:poison_entries_total"

_passed = 0
_failed: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    global _passed
    if ok:
        _passed += 1
        print(f"  [PASS] {label}" + (f"  ({detail})" if detail else ""))
    else:
        _failed.append(label)
        print(f"  [FAIL] {label}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'-' * 72}\n{title}\n{'-' * 72}")


async def counts(client: httpx.AsyncClient) -> dict[str, Any]:
    return (await client.get(f"{MOCK}/admin/call-counts")).json()


async def wait_for_job(
    client: httpx.AsyncClient, job_id: str, timeout_s: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status = (await client.get(f"{API}/jobs/{job_id}")).json()
        if status.get("complete"):
            return status
        await asyncio.sleep(0.5)
    return status


# ---------------------------------------------------------------- 1. health


async def verify_health(client: httpx.AsyncClient) -> None:
    section("1. Services reachable (Step 1-4)")

    api = await client.get(f"{API}/health")
    check("api /health returns 200 ok", api.status_code == 200 and api.json()["status"] == "ok")
    check(
        "api reports Redis dependency explicitly",
        api.json().get("redis") == "ok",
        json.dumps(api.json()),
    )
    check(
        "every response carries X-Trace-Id",
        bool(api.headers.get("x-trace-id")),
        api.headers.get("x-trace-id", "")[:16],
    )

    mock = await client.get(f"{MOCK}/health")
    check("mock-model /health returns 200 ok", mock.status_code == 200)


# ------------------------------------------------------- 2. trace propagation


async def verify_tracing(client: httpx.AsyncClient) -> None:
    section("2. Trace context propagation (Step 2)")

    supplied = "a" * 32
    adopted = await client.get(
        f"{API}/health",
        headers={"traceparent": f"00-{supplied}-{'b' * 16}-01"},
    )
    check(
        "inbound traceparent is ADOPTED, not replaced",
        adopted.headers.get("x-trace-id") == supplied,
        adopted.headers.get("x-trace-id", ""),
    )

    garbage = await client.get(f"{API}/health", headers={"traceparent": "not-valid"})
    tid = garbage.headers.get("x-trace-id", "")
    check(
        "malformed traceparent is rejected and a fresh id generated",
        len(tid) == 32 and tid != "not-valid",
        tid[:16],
    )


# --------------------------------------------------------- 3. rate limiting


async def verify_rate_limiting(client: httpx.AsyncClient) -> None:
    section("3. Token-bucket rate limiting (Step 3)")

    await client.post(f"{MOCK}/admin/reset")
    await client.delete(f"{MOCK}/admin/chaos")

    responses = await asyncio.gather(
        *(
            client.post(f"{MOCK}/v1/predict/vlm", json={"job_id": "verify-rl", "page_index": i})
            for i in range(30)
        )
    )
    codes = collections.Counter(r.status_code for r in responses)
    check(
        "30 concurrent VLM calls produce 429s (limit is 10 rps)",
        codes.get(429, 0) > 0,
        dict(codes),
    )
    check(
        "successes are bounded near the burst size, not all 30",
        codes.get(200, 0) <= 15,
        f"{codes.get(200, 0)} succeeded",
    )

    rejected = next((r for r in responses if r.status_code == 429), None)
    if rejected is not None:
        check("429 carries Retry-After", bool(rejected.headers.get("retry-after")))
        check(
            "429 carries precise X-Retry-After-Ms (RFC seconds are too coarse)",
            rejected.headers.get("x-retry-after-ms") is not None,
            f"{rejected.headers.get('retry-after')}s vs {rejected.headers.get('x-retry-after-ms')}ms",
        )
        check(
            "429 reports bucket state",
            rejected.headers.get("x-ratelimit-limit") is not None,
            f"limit={rejected.headers.get('x-ratelimit-limit')} remaining={rejected.headers.get('x-ratelimit-remaining')}",
        )


# ---------------------------------------------------------- 4. idempotency


async def verify_idempotency(client: httpx.AsyncClient) -> None:
    section("4. Idempotency and single-flight coalescing (Step 3)")

    await client.post(f"{MOCK}/admin/reset")
    key = "verify:page-1:vlm"

    responses = await asyncio.gather(
        *(
            client.post(
                f"{MOCK}/v1/predict/vlm",
                json={"job_id": "verify-idem", "page_index": 1},
                headers={"Idempotency-Key": key},
            )
            for _ in range(5)
        )
    )
    data = await counts(client)
    vlm = data["endpoints"].get("vlm", {})

    check("all 5 concurrent duplicates returned 200", all(r.status_code == 200 for r in responses))
    check(
        "the model executed exactly ONCE for 5 identical requests",
        vlm.get("executions") == 1,
        f"executions={vlm.get('executions')} replays={vlm.get('replays')}",
    )
    check(
        "no duplicate executions recorded",
        data["idempotency"]["duplicate_executions"] == {},
        json.dumps(data["idempotency"]["duplicate_executions"]),
    )
    bodies = {json.dumps(r.json(), sort_keys=True) for r in responses}
    check("all 5 callers received identical output", len(bodies) == 1)

    started = time.monotonic()
    await client.post(
        f"{MOCK}/v1/predict/vlm",
        json={"job_id": "verify-idem", "page_index": 1},
        headers={"Idempotency-Key": key},
    )
    replay_ms = (time.monotonic() - started) * 1000
    check(
        "a replay is far cheaper than a 1500-3000ms inference",
        replay_ms < 500,
        f"{replay_ms:.0f}ms",
    )


# ---------------------------------------------------------------- 5. chaos


async def verify_chaos(client: httpx.AsyncClient) -> None:
    section("5. Chaos injection (Step 3)")

    await client.post(f"{MOCK}/admin/reset")
    await client.post(
        f"{MOCK}/admin/chaos",
        json={"endpoint": "vlm", "status": 429, "ratio": 1.0, "seconds": 3},
    )
    active = (await client.get(f"{MOCK}/admin/chaos")).json()["chaos"]
    check("chaos rule is active on vlm", "vlm" in active, json.dumps(active))

    forced = await client.post(f"{MOCK}/v1/predict/vlm", json={"job_id": "c", "page_index": 0})
    check("vlm forced to 429 by chaos", forced.status_code == 429)

    untouched = await client.post(
        f"{MOCK}/v1/predict/layout", json={"job_id": "c", "page_index": 0}
    )
    # Asserted on the chaos counter, NOT on a 200.
    #
    # This check used to require status == 200 and failed intermittently once
    # the stack had three busy workers: layout has a 2% injected failure rate
    # and a 100 rps token bucket shared with every worker, so a 500 or a 429
    # here is entirely legitimate and says nothing about chaos scoping. The
    # property under test is "the vlm chaos rule did not fault layout", and the
    # mock counts exactly that per endpoint - so the counter is the assertion
    # and the status code is only reported.
    layout_stats = (await counts(client))["endpoints"].get("layout", {})
    check(
        "layout is unaffected: chaos is endpoint-scoped",
        layout_stats.get("chaos_rejections", 0) == 0,
        f"chaos_rejections=0, status={untouched.status_code}",
    )

    await asyncio.sleep(3.2)
    expired = (await client.get(f"{MOCK}/admin/chaos")).json()["chaos"]
    check(
        "chaos rule SELF-EXPIRES (a crashed test cannot wedge the mock)",
        expired == {},
        json.dumps(expired),
    )
    await client.delete(f"{MOCK}/admin/chaos")


# ------------------------------------------------------------ 6. end to end


async def verify_end_to_end(client: httpx.AsyncClient) -> str:
    section("6. End-to-end job through the queue and worker (Step 5)")

    await client.post(f"{MOCK}/admin/reset")
    pages = 4

    accepted = await client.post(f"{API}/jobs", json={"pages": pages, "filename": "verify.pdf"})
    check("POST /jobs returns 202 Accepted (work queued, not performed)", accepted.status_code == 202)
    body = accepted.json()
    job_id = body["job_id"]
    check("response includes a trace_id for correlation", bool(body.get("trace_id")))

    depth = (await client.get(f"{API}/queue/depth")).json()
    check(
        f"all {pages} pages were enqueued",
        depth["stream_length"] >= pages - 1,
        json.dumps(depth),
    )

    status = await wait_for_job(client, job_id, timeout_s=90)
    check(
        f"job completed all {pages} pages",
        status.get("complete") is True,
        f"done={status.get('done')}/{status.get('total_pages')} {status.get('state_counts')}",
    )
    check(
        "every page reached DONE (no failures, no stragglers)",
        status.get("state_counts", {}).get("DONE") == pages,
        json.dumps(status.get("state_counts")),
    )

    data = await counts(client)
    check(
        f"exactly {pages} layout executions - one per page",
        data["endpoints"].get("layout", {}).get("executions") == pages,
        f"{data['endpoints'].get('layout', {}).get('executions')}",
    )
    check(
        f"exactly {pages} VLM executions - one per page",
        data["endpoints"].get("vlm", {}).get("executions") == pages,
        f"{data['endpoints'].get('vlm', {}).get('executions')}",
    )

    page = (await client.get(f"{API}/jobs/{job_id}/pages/0")).json()
    check("page stores its layout result", "layout" in page)
    check("page stores its VLM result", "vlm" in page)
    check(
        "stored VLM result contains a nested tree (for Module C later)",
        "tree" in json.loads(page.get("vlm", "{}")),
    )

    final_depth = (await client.get(f"{API}/queue/depth")).json()
    # Assert on the depth fields, not on exact dict equality: /queue/depth also
    # reports the admission watermarks now, and an exact match would break every
    # time the payload gains a field.
    check(
        "queue fully drained: XACK + XDEL leave nothing behind",
        all(
            final_depth.get(k) == 0
            for k in ("stream_length", "pending", "backlog")
        ),
        json.dumps(final_depth),
    )
    return job_id


# --------------------------------------------------------- 6b. admission


async def verify_admission(client: httpx.AsyncClient) -> None:
    section("6b. Admission control refuses explicitly, never silently (Step 11)")

    stats = (await client.get(f"{API}/admission")).json()
    check("admission stats endpoint responds", "high_watermark" in stats, str(stats))
    check(
        "low watermark sits strictly below high (hysteresis exists)",
        stats["low_watermark"] < stats["high_watermark"],
        f"{stats['low_watermark']} < {stats['high_watermark']}",
    )
    check(
        "a healthy system sheds nothing",
        stats["shed_jobs"] == 0 and not stats["shedding"],
        f"shed_jobs={stats['shed_jobs']} shedding={stats['shedding']}",
    )
    check(
        "admitted work is counted, so the admitted/shed ratio is knowable",
        stats["admitted_jobs"] > 0 and stats["admitted_pages"] > 0,
        f"{stats['admitted_jobs']} jobs / {stats['admitted_pages']} pages",
    )

    # An over-sized job must be refused as VALIDATION (413), not as capacity -
    # otherwise it would pollute the shed counters with a malformed request.
    before = stats["shed_jobs"]
    oversized = await client.post(f"{API}/jobs", json={"pages": 100_000})
    after = (await client.get(f"{API}/admission")).json()["shed_jobs"]
    check(
        "an over-sized job is 413 (validation), not 503 (capacity)",
        oversized.status_code == 413,
        f"HTTP {oversized.status_code}",
    )
    check(
        "a validation refusal is not counted against capacity",
        after == before,
        f"shed_jobs {before} -> {after}",
    )


# ------------------------------------------------------ 6c. PDF ingestion


def _synth_pdf(pages: int, kb_per_page: int = 1) -> bytes:
    import sys

    sys.path.insert(0, ".")
    from bench.make_pdf import build

    return build(pages, kb_per_page)


async def verify_pdf_ingestion(client: httpx.AsyncClient) -> None:
    section("6c. PDF ingestion is bounded and validates at the edge (Step 12)")

    async def post_raw(body: bytes):
        return await client.post(
            f"{API}/jobs/stream",
            content=body,
            headers={"Content-Type": "application/pdf"},
        )

    # A real multi-page document must be accepted and counted from the file,
    # not from anything the client claims.
    ok = await post_raw(_synth_pdf(7, kb_per_page=8))
    check(
        "a real PDF is accepted and its pages counted from the file",
        ok.status_code == 202 and ok.json()["pages"] == 7,
        f"HTTP {ok.status_code} {ok.text[:80]}",
    )

    # Every one of these used to be, or could be, a silent acceptance.
    for label, body, expected in (
        ("a 0-page PDF", _synth_pdf(0), 400),
        ("a non-PDF body", b"plainly not a pdf" * 20, 400),
        ("an empty body", b"", 400),
        ("a document over max_pages", _synth_pdf(150), 413),
    ):
        response = await post_raw(body)
        check(
            f"{label} is refused with {expected}",
            response.status_code == expected,
            f"HTTP {response.status_code}",
        )

    # Rejections must not be counted as capacity pressure, or the shed
    # statistics stop meaning what they claim.
    stats = (await client.get(f"{API}/admission")).json()
    check(
        "validation refusals are not counted against capacity",
        stats["shed_jobs"] == 0,
        f"shed_jobs={stats['shed_jobs']}",
    )

    # Drain the job this section created before returning. Without this its
    # pages keep completing during LATER sections, and the redelivery check -
    # which compares VLM execution counts before and after one replay - sees
    # those extra calls and fails. A verification section has to leave no work
    # in flight behind it.
    if ok.status_code == 202:
        job_id = ok.json()["job_id"]
        for _ in range(60):
            state = (await client.get(f"{API}/jobs/{job_id}")).json()
            if state.get("complete"):
                break
            await asyncio.sleep(1)
        check(
            "the ingested document completes every page",
            state.get("complete") and state["done"] == 7,
            f"done={state.get('done')}/{state.get('total_pages')}",
        )
        check(
            "no page was left unhandled",
            state.get("state_counts", {}).get("FAILED", 0) == 0,
            str(state.get("state_counts")),
        )

    # Both ingestion paths exist: multipart for convenience, raw for memory.
    schema = (await client.get(f"{API}/openapi.json")).json()["paths"]
    check(
        "both ingestion endpoints are exposed",
        "/jobs/upload" in schema and "/jobs/stream" in schema,
        ", ".join(sorted(k for k in schema if "jobs" in k)),
    )


# ----------------------------------------------------- 7. redelivery safety


async def verify_streaming(client: httpx.AsyncClient) -> None:
    section("6d. SSE streams pages as they land, out of order (Step 13)")

    started = time.perf_counter()
    response = await client.post(f"{API}/jobs", json={"pages": 8})
    job_id = response.json()["job_id"]

    frames: list[dict] = []
    first_page_ms = None
    first_final_ms = None
    heartbeats = 0

    async with client.stream("GET", f"{API}/jobs/{job_id}/stream") as stream:
        content_type = stream.headers.get("content-type", "")
        buffer = ""
        async for chunk in stream.aiter_text():
            buffer += chunk
            while "\n\n" in buffer:
                raw, buffer = buffer.split("\n\n", 1)
                if not raw.strip():
                    continue
                frame: dict = {}
                for line in raw.splitlines():
                    if line.startswith(":"):
                        heartbeats += 1
                    elif line.startswith("id: "):
                        frame["id"] = line[4:]
                    elif line.startswith("event: "):
                        frame["event"] = line[7:]
                    elif line.startswith("data: "):
                        frame["data"] = json.loads(line[6:])
                if "event" not in frame:
                    continue
                frames.append(frame)
                if frame["event"].startswith("page.") and first_page_ms is None:
                    first_page_ms = (time.perf_counter() - started) * 1000
                if frame["event"] == "page.final" and first_final_ms is None:
                    first_final_ms = (time.perf_counter() - started) * 1000
                if frame["event"] == "job.complete":
                    break
            if frames and frames[-1].get("event") == "job.complete":
                break

    check(
        "content-type is text/event-stream",
        content_type.startswith("text/event-stream"),
        content_type,
    )

    partials = [f for f in frames if f["event"] == "page.partial"]
    finals = [f for f in frames if f["event"] == "page.final"]
    check(
        "every page is announced twice: partial then final",
        len(partials) == 8 and len(finals) == 8,
        f"{len(partials)} partial / {len(finals)} final",
    )
    check(
        "a partial is explicitly incomplete and a final explicitly complete",
        all(f["data"]["complete"] is False for f in partials)
        and all(f["data"]["complete"] is True for f in finals),
    )
    check(
        "every page index appears exactly once in each phase",
        sorted(f["data"]["page_index"] for f in partials) == list(range(8))
        and sorted(f["data"]["page_index"] for f in finals) == list(range(8)),
    )

    # THE point of the two-phase design: the fast stage is what the client
    # waits for, not the 1.5-3s one.
    check(
        "first page arrives before the first FINAL (the whole reason for two phases)",
        first_page_ms is not None
        and first_final_ms is not None
        and first_page_ms < first_final_ms,
        f"first page {first_page_ms:.0f}ms vs first final {first_final_ms:.0f}ms",
    )
    check(
        f"time-to-first-page under 200ms (measured {first_page_ms:.0f}ms)",
        first_page_ms is not None and first_page_ms < 200,
        f"{first_page_ms:.0f}ms",
    )

    stored = [f for f in frames if "id" in f]
    seqs = [f["data"]["seq"] for f in stored]
    check(
        "seq is dense and monotonic, so a client can PROVE it saw everything",
        seqs == sorted(seqs) and seqs == list(range(1, len(seqs) + 1)),
        f"{len(seqs)} events",
    )
    check(
        "the SSE id matches the stream_id in the payload",
        all(f["data"]["stream_id"] == f["id"] for f in stored),
    )
    check(
        "stream.open carries total_pages and the retained window",
        frames[0]["event"] == "stream.open"
        and frames[0]["data"]["total_pages"] == 8
        and "window_lo" in frames[0]["data"],
        str(frames[0].get("data"))[:120],
    )
    check(
        "connection-scoped frames carry no id (they must not become a cursor)",
        all("id" not in f for f in frames if f["event"] == "stream.open"),
    )
    check(
        "job.complete closes the stream",
        frames[-1]["event"] == "job.complete"
        and frames[-1]["data"]["done"] == 8,
        str(frames[-1].get("data"))[:120],
    )

    # Resume: replay from a mid-stream cursor must not redeliver what is seen.
    #
    # Ids compare as NUMBER PAIRS, never as strings. Lexicographically
    # "10-0" < "9-0", so a string comparison decides that millisecond 10
    # predates millisecond 9 - and this assertion would then pass or fail
    # depending on where the clock happened to be when the run started.
    def as_pair(entry_id: str) -> tuple[int, int]:
        ms, _, seq = entry_id.partition("-")
        return int(ms), int(seq or 0)

    cursor = stored[len(stored) // 2]["id"]
    seen_after: list[str] = []
    async with client.stream(
        "GET", f"{API}/jobs/{job_id}/stream", headers={"Last-Event-ID": cursor}
    ) as stream:
        buffer = ""
        done = False
        async for chunk in stream.aiter_text():
            buffer += chunk
            while "\n\n" in buffer:
                raw, buffer = buffer.split("\n\n", 1)
                for line in raw.splitlines():
                    if line.startswith("id: "):
                        seen_after.append(line[4:])
                    elif line == "event: job.complete":
                        done = True
            if done:
                break

    check(
        "Last-Event-ID resumes strictly AFTER the cursor, redelivering nothing",
        bool(seen_after) and all(as_pair(i) > as_pair(cursor) for i in seen_after),
        f"{len(seen_after)} events after cursor {cursor}",
    )

    unknown = await client.get(f"{API}/jobs/no-such-job-xyz/stream")
    check(
        "an unknown job is 404 before the body starts, not an error event in a 200",
        unknown.status_code == 404,
        str(unknown.status_code),
    )

    slots = (await client.get(f"{API}/streams")).json()
    check(
        "subscriber slots are bounded and released after each stream",
        slots["active"] == 0 and slots["limit"] > 0 and slots["refused"] == 0,
        str(slots),
    )

    depth = (await client.get(f"{API}/queue/depth")).json()
    check(
        "queue depth counts BOTH lanes (priority lane is not invisible)",
        depth["stream_length"] >= 0 and "backlog" in depth,
        str(depth),
    )


async def verify_redelivery(client: httpx.AsyncClient, job_id: str) -> None:
    section("7. At-least-once redelivery is harmless (Step 4-5)")

    before = (await counts(client))["endpoints"]
    before_layout = before.get("layout", {}).get("executions", 0)
    before_vlm = before.get("vlm", {}).get("executions", 0)

    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        # Re-publish a task for a page that is already DONE, exactly as the
        # queue would after a worker died before acknowledging it.
        await redis.xadd(
            "stream:pages",
            {
                "job_id": job_id,
                "page_index": 0,
                "enqueued_at_ms": int(time.time() * 1000),
                "trace_id": "verify-redelivery",
            },
        )
    finally:
        await redis.aclose()

    await asyncio.sleep(4)

    after = (await counts(client))["endpoints"]
    check(
        "redelivered page triggered NO additional layout call",
        after.get("layout", {}).get("executions", 0) == before_layout,
        f"{before_layout} -> {after.get('layout', {}).get('executions', 0)}",
    )
    check(
        "redelivered page triggered NO additional VLM call",
        after.get("vlm", {}).get("executions", 0) == before_vlm,
        f"{before_vlm} -> {after.get('vlm', {}).get('executions', 0)}",
    )
    status = (await client.get(f"{API}/jobs/{job_id}")).json()
    check(
        "done_count did not double-count the redelivered page",
        status["done"] == status["total_pages"],
        f"done={status['done']}/{status['total_pages']}",
    )
    depth = (await client.get(f"{API}/queue/depth")).json()
    check("redelivered entry was acknowledged and removed", depth["stream_length"] == 0)


# ----------------------------------------------------- 8. admission control


async def verify_limits(client: httpx.AsyncClient) -> None:
    section("11. Input validation and edge rejection (Step 5)")

    too_many = await client.post(f"{API}/jobs", json={"pages": 100_000})
    check(
        "oversized job rejected at the edge with 413",
        too_many.status_code == 413,
        f"status={too_many.status_code}",
    )

    invalid = await client.post(f"{API}/jobs", json={"pages": 0})
    check("pages=0 rejected by schema validation (422)", invalid.status_code == 422)

    missing = await client.get(f"{API}/jobs/does-not-exist")
    check("unknown job returns 404", missing.status_code == 404)

    bad_body = await client.post(f"{MOCK}/v1/predict/layout", json={"page_index": 0})
    check("mock rejects a request missing job_id (422)", bad_body.status_code == 422)


def _page_tree(blocks: int) -> dict:
    """A document tree of the shape mock_model.payloads emits: page -> blocks,
    tables holding rows holding cells. Depth 4, ~50 nodes at blocks=9 - which is
    exactly the size the graded metric names."""
    children = []
    y = 40.0
    for i in range(blocks):
        if i % 3 == 2:
            children.append({
                "type": "table",
                "bbox": [40, y, 500, 90],
                "children": [
                    {"type": "row", "children": [
                        {"type": "cell", "text": f"r{r}c{c}"} for c in range(3)]}
                    for r in range(3)
                ],
            })
        else:
            children.append({
                "type": ["paragraph", "title", "figure"][i % 3],
                "bbox": [40, y, 500, 60],
                "text": f"block {i} body text for scoring",
            })
        y += 70
    return {"type": "page", "bbox": [0, 0, 595, 842], "children": children}


def _caterpillar(spine: int) -> dict:
    """Deep spine shedding a leaf per level: depth AND leaves both large, which
    is the shape that reaches Zhang-Shasha's quartic worst case."""
    node = {"type": "tip"}
    for _ in range(spine):
        node = {"type": "spine", "children": [{"type": "leaf"}, node]}
    return node


async def verify_metrics(client: httpx.AsyncClient) -> None:
    section("10. Prometheus /metrics and the TED cost calibration (Step 18)")

    response = await client.get(f"{API}/metrics")
    body = response.text
    check("GET /metrics returns 200", response.status_code == 200,
          f"status={response.status_code}")
    check("content type is the Prometheus exposition format",
          "text/plain" in response.headers.get("content-type", ""),
          response.headers.get("content-type", ""))
    if response.status_code != 200:
        return

    lines = [ln for ln in body.splitlines() if ln and not ln.startswith("#")]
    types = [ln for ln in body.splitlines() if ln.startswith("# TYPE")]
    names = {ln.split()[2] for ln in types}

    check(f"families are exported ({len(types)} TYPE lines, {len(lines)} samples)",
          len(types) >= 15, f"{len(types)} families")
    check("no family declares TYPE twice (a scraper rejects the family)",
          len(types) == len(set(types)), f"{len(types)} vs {len(set(types))} unique")

    for required in (
        "orch_queue_backlog",
        "orch_queue_pending",
        "orch_breaker_state",
        "orch_adaptive_rate_limit",
        "orch_admitted_pages_total",
        "orch_workers_reporting",
        "orch_sse_subscribers",
    ):
        check(f"exports {required}", required in names)

    check("the scrape reported no failing sources",
          "orch_metrics_scrape_errors 0" in body,
          next((ln for ln in lines if ln.startswith("orch_metrics_scrape_errors")), "?"))

    # Worker-sourced families only appear once a worker has flushed, which is
    # what proves the cross-container aggregation actually works rather than the
    # API reporting only what it can see locally.
    workers = next(
        (ln for ln in lines if ln.startswith("orch_workers_reporting")), "0 0"
    )
    reporting = float(workers.rsplit(" ", 1)[1])
    check(f"workers are flushing metrics into Redis ({int(reporting)} reporting)",
          reporting >= 1, f"reporting={reporting}")

    # Histogram buckets must be cumulative and end at +Inf, or
    # histogram_quantile() returns plausible nonsense rather than an error.
    bucket_lines = [ln for ln in lines if "_bucket{" in ln]
    if bucket_lines:
        series: dict[str, list[tuple[str, float]]] = {}
        for ln in bucket_lines:
            head, value = ln.rsplit(" ", 1)
            name = head.split("{")[0]
            labels = head.split("{", 1)[1].rstrip("}")
            le = head.split('le="')[1].split('"')[0]
            # Keyed by name AND the other labels: cumulativeness holds WITHIN a
            # series, not across them. Grouping by name alone concatenates
            # endpoint="layout" (ending at 1071) with endpoint="vlm" (starting
            # at 1) and reports a false violation - which is exactly what this
            # check did on its first run.
            others = ",".join(
                part for part in labels.split(",") if not part.startswith("le=")
            )
            series.setdefault(f"{name}{{{others}}}", []).append((le, float(value)))
        monotonic = True
        has_inf = True
        for name, points in series.items():
            values = [v for _, v in points]
            monotonic = monotonic and values == sorted(values)
            has_inf = has_inf and any(le == "+Inf" for le, _ in points)
        check(f"histogram buckets are cumulative ({len(series)} series)", monotonic)
        check("every histogram has a +Inf bucket", has_inf)

    # --- the calibration self-check promised in app/config.py
    #
    # `eval_ted_work_per_ms` translates a work-unit cap into a latency budget,
    # and the rate DRIFTS: the same container measured 3,500/ms once and
    # 2,216/ms weeks later, at which point a tree under the cap took 134 ms
    # against a 100 ms budget. A constant calibrated once is a latent bug, so
    # the achieved rate is measured here and an optimistic setting fails.
    probe = {
        "type": "page",
        "children": [
            {
                "type": "table",
                "bbox": [0, 0, 10, 10],
                "children": [
                    {"type": "row", "children": [
                        {"type": "cell", "text": f"r{r}c{c}"} for c in range(3)]}
                    for r in range(3)
                ],
            }
            for _ in range(6)
        ],
    }
    measured = await client.post(
        f"{API}/evaluate", json={"predicted": probe, "truth": probe}
    )
    if measured.status_code == 200:
        payload = measured.json()
        work = payload["structure"]["ted_work"]
        ted_ms = payload["timings_ms"]["ted_ms"]
        achieved = work / ted_ms if ted_ms > 0 else 0.0
        configured = float(
            next(
                (
                    ln.split('ted_budget_ms="')[1].split('"')[0]
                    for ln in lines
                    if ln.startswith("orch_build_info")
                ),
                "100",
            )
        )
        # The cap is budget x rate, so an optimistic rate means an admitted
        # request can miss the budget. Compare the achieved rate against the
        # configured one with 25% tolerance for a single noisy sample.
        from app.config import get_settings

        expected = get_settings().eval_ted_work_per_ms
        check(
            f"TED cost calibration is not optimistic "
            f"(achieved {achieved:.0f} work/ms vs configured {expected})",
            achieved >= expected * 0.75,
            f"achieved={achieved:.0f} configured={expected} budget={configured:.0f}ms",
        )
        check(f"the probe itself met the budget ({ted_ms:.1f} ms)",
              ted_ms < configured, f"ted_ms={ted_ms:.2f}")


async def verify_evaluate(client: httpx.AsyncClient) -> None:
    section("9. Evaluation endpoint and the graded tree-diff latency (Step 15-17)")

    truth = _page_tree(9)
    predicted = json.loads(json.dumps(truth))
    predicted["children"][0]["text"] = "blockk 0 body text for scoring"  # 1 char
    predicted["children"][0]["bbox"] = [42, 41, 495, 61]

    response = await client.post(
        f"{API}/evaluate", json={"predicted": predicted, "truth": truth}
    )
    check("POST /evaluate returns 200", response.status_code == 200,
          f"status={response.status_code}")
    if response.status_code != 200:
        return

    body = response.json()
    nodes = body["structure"]["truth_nodes"]
    ted_ms = body["timings_ms"]["ted_ms"]

    check(f"the tree is document-shaped and ~50 nodes ({nodes} nodes, depth "
          f"{body['structure']['truth_depth']})",
          40 <= nodes <= 60 and body["structure"]["truth_depth"] == 4,
          f"nodes={nodes}")

    # THE GRADED METRIC.
    check(f"GRADED: tree diff < 100ms for a {nodes}-node tree "
          f"({ted_ms:.2f} ms, {100 / ted_ms:.0f}x headroom)", ted_ms < 100.0,
          f"ted_ms={ted_ms:.2f}")

    check("all three metrics are reported",
          {"cer", "wer"} <= body["text"].keys()
          and "mean_iou" in body["boxes"]
          and "tree_edit_distance" in body["structure"])

    # One text error and one box shift, identical structure: the three metrics
    # must not triple-count it.
    check("a text+box error does not move the structure metric "
          f"(cer={body['text']['cer']:.4f}, iou={body['boxes']['mean_iou']:.4f}, "
          f"ted={body['structure']['tree_edit_distance']})",
          body["text"]["cer"] > 0
          and body["boxes"]["mean_iou"] < 1.0
          and body["structure"]["tree_edit_distance"] == 0)

    # The cost guard, on a tree that is small but expensively shaped.
    hostile = _caterpillar(100)
    started = time.perf_counter()
    refused = await client.post(
        f"{API}/evaluate", json={"predicted": hostile, "truth": hostile}
    )
    refusal_ms = (time.perf_counter() - started) * 1000
    check(f"a 201-node caterpillar (~22s of CPU) is refused with 413 "
          f"in {refusal_ms:.0f} ms", refused.status_code == 413,
          f"status={refused.status_code}")

    # Same node count, different shape: the page tree is served, the
    # caterpillar is not. A cap on node count alone could not tell them apart.
    cat_50 = _caterpillar(24)
    cat_response = await client.post(
        f"{API}/evaluate", json={"predicted": cat_50, "truth": cat_50}
    )
    check("the cap discriminates by SHAPE, not size: a ~49-node caterpillar is "
          "refused while a ~46-node page tree is served",
          cat_response.status_code == 413,
          f"status={cat_response.status_code}")

    # The event loop must stay responsive while a comparison is in flight.
    admitted = _caterpillar(20)
    heavy = asyncio.create_task(
        client.post(f"{API}/evaluate", json={"predicted": admitted, "truth": admitted})
    )
    await asyncio.sleep(0)
    ping_started = time.perf_counter()
    health = await client.get(f"{API}/health")
    ping_ms = (time.perf_counter() - ping_started) * 1000
    await heavy
    check(f"/health stays responsive during a comparison ({ping_ms:.0f} ms)",
          health.status_code == 200 and ping_ms < 2000, f"ping_ms={ping_ms:.0f}")


async def verify_reaper(client: httpx.AsyncClient) -> None:
    section("8. Orphaned work is reclaimed (Step 14)")

    # A real SIGKILL is scripts/prove_recovery.sh - it needs the Docker CLI and
    # two container lifecycles, which does not belong in a verification pass.
    # What IS checked here is the mechanism a dead worker depends on, driven
    # against the live stack: an entry pending under a consumer name that no
    # process owns, backdated past the idle threshold, must be reclaimed and the
    # page must finish.
    settings_view = (await client.get(f"{API}/queue/consumers")).json()
    check(
        "reaper is enabled and reports its own threshold",
        settings_view.get("reaper_enabled") is True
        and settings_view.get("min_idle_ms", 0) > 0,
        f"min_idle_ms={settings_view.get('min_idle_ms')}",
    )
    check(
        "no orphaned entries on a healthy stack",
        settings_view.get("orphaned") == 0,
        f"orphaned={settings_view.get('orphaned')}",
    )

    # Live workers hold entries; they must NEVER show as stale, because lease
    # renewal keeps resetting their idle clocks. This is the check that would
    # catch a broken or stopped renewal loop - a failure mode whose only other
    # symptom is healthy pages being reclaimed and recomputed for no reason.
    response = await client.post(f"{API}/jobs", json={"pages": 12})
    job_id = response.json()["job_id"]
    await asyncio.sleep(3)
    during = (await client.get(f"{API}/queue/consumers")).json()
    depth_during = (await client.get(f"{API}/queue/depth")).json()
    check(
        "entries held by LIVE workers are never stale (leases renewing)",
        during.get("orphaned") == 0,
        f"pending={depth_during['pending']} stale={during.get('orphaned')}",
    )
    await wait_for_job(client, job_id, timeout_s=90)

    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        # A page that a vanished worker was holding mid-VLM. Written directly
        # because the alternative - killing a container from inside a
        # verification script - is not something this script should be able to
        # do.
        probe = await client.post(f"{API}/jobs", json={"pages": 3})
        probe_id = probe.json()["job_id"]

        # Put a page in a dead consumer's pending list, deterministically.
        #
        # The creation and the claim MUST be one atomic server-side step, and
        # getting this wrong twice is what makes it worth explaining.
        #
        # Attempt 1 - XREADGROUP as "ghost-worker" - loses a race it cannot
        # win: three live workers sit blocked on this stream, so one takes the
        # entry within microseconds of the XADD and the ghost reads nothing.
        # Measured: 0 entries every time.
        #
        # Attempt 2 - XADD, then XCLAIM FORCE, asserting on XCLAIM's own
        # JUSTID return so there was no second round trip to race. It still
        # failed intermittently (once in three live runs), because the race was
        # never about the PEL. `PageQueue.ack()` is XACK *plus XDEL*: when a
        # live worker finishes the probe page it DELETES the stream entry. And
        # XCLAIM FORCE on an id that no longer exists in the stream cannot
        # create a PEL entry for it - it is a no-op returning empty. The
        # sequence was:
        #
        #     XADD -> (worker reads, completes, XACK+XDEL) -> XCLAIM FORCE -> []
        #
        # So the assertion was racing the entry's EXISTENCE, not its ownership,
        # and moving the assertion earlier could not fix that.
        #
        # A Lua script fixes it properly. Redis runs it atomically, so no
        # worker's XREADGROUP can interleave between the XADD and the XCLAIM.
        # Once the entry sits in ghost-worker's PEL it is no longer deliverable
        # by `XREADGROUP >` at all - `>` returns only entries never delivered
        # to the group - so the only route back to a live worker is the
        # reaper's XAUTOCLAIM, which is exactly the mechanism under test.
        # IDLE backdates the entry in the same step, replacing a 30s sleep.
        now_ms = int(time.time() * 1000)
        ghost_ids = await redis.eval(
            """
            local id = redis.call('XADD', KEYS[1], '*',
                'job_id', ARGV[1], 'page_index', ARGV[2],
                'enqueued_at_ms', ARGV[3], 'first_enqueued_at_ms', ARGV[3],
                'trace_id', 'verify-reaper')
            return redis.call('XCLAIM', KEYS[1], ARGV[4], 'ghost-worker', 0,
                id, 'IDLE', 120000, 'JUSTID', 'FORCE')
            """,
            1,
            "stream:pages",
            probe_id,
            "1",
            str(now_ms),
            "workers",
        )
        check(
            "an entry can be left pending under a name no process owns",
            bool(ghost_ids),
            f"{len(ghost_ids)} entries held by ghost-worker",
        )
        if ghost_ids:
            # NOT checked here: that /queue/consumers transiently REPORTS a
            # non-zero orphan count. That assertion was tried and is
            # unfalsifiable against a live stack - three replicas each scan
            # every reaper_interval_s, so the expected time to reclaim is a
            # couple of seconds and can be microseconds, which is faster than
            # the round trip that would observe it. It failed with
            # "orphaned=0" precisely because the mechanism worked too well.
            #
            # The two halves are asserted separately instead: the check above
            # proves the entry really was pending under a name no process owns
            # (returned by the atomic script itself), and the check below
            # proves it went to zero. A count sampled somewhere between them
            # adds flakiness and no information.
            #
            # Because the claim is atomic with the XADD, the entry was never
            # deliverable by `XREADGROUP >` - so unlike the two earlier
            # versions of this check, no live worker can settle it and the ONLY
            # route to zero is the reaper's XAUTOCLAIM. That makes the
            # assertion below a test of the reaper specifically, rather than of
            # "something eventually cleaned this up".

            # The reaper scans every reaper_interval_s on every replica.
            deadline = time.monotonic() + 60
            remaining = len(ghost_ids)
            while time.monotonic() < deadline:
                await asyncio.sleep(2)
                view = (await client.get(f"{API}/queue/consumers")).json()
                remaining = view.get("orphaned", 0)
                if remaining == 0:
                    break
            check(
                "the reaper reclaimed every orphaned entry",
                remaining == 0,
                f"{len(ghost_ids)} orphaned -> {remaining} remaining",
            )

        status = await wait_for_job(client, probe_id, timeout_s=120)
        check(
            "the job whose pages were orphaned still reaches 100%",
            bool(status.get("complete")),
            f"{status.get('done')}/{status.get('total_pages')}",
        )
        states = status.get("state_counts", {})
        check(
            "no page is left in a *_RUNNING state after recovery",
            not any(name.endswith("_RUNNING") for name in states),
            str(states),
        )

        # The zero-drop invariant, stated as it is everywhere else in this
        # build: a page may not be non-terminal AND absent from the queue.
        depth = (await client.get(f"{API}/queue/depth")).json()
        check(
            "queue drains to empty, leaving nothing unreachable",
            depth["stream_length"] == 0,
            str(depth),
        )

        # ------------------------------------------------------------------
        # Poison entry: one malformed task must not take the pipeline down.
        #
        # REGRESSION, found live. `_parse` runs inside `read()`, which is
        # upstream of every per-task try/except in the worker, so a KeyError on
        # one entry's fields did not fail that page - it propagated out of
        # Worker.run() and exited the process. All three replicas read the same
        # entry and died identically, and a crashed worker never XACKs, so the
        # entry was still waiting on restart. Observed:
        #
        #     worker-1/2/3  restarts=0 state=exited
        #     queue: stream_length=24, pending=0, backlog=24
        #
        # A permanent outage with 24 pages stranded, from one bad message.
        # This is checked against the live stack rather than only in unit
        # tests because the unit test cannot observe the part that mattered -
        # that the three real worker PROCESSES are still running afterwards.
        # ------------------------------------------------------------------
        before = int(await redis.get(POISON_COUNTER_KEY) or 0)
        await redis.xadd(
            "stream:pages",
            {
                # Deliberately missing `enqueued_at_ms` - the exact entry that
                # caused the outage.
                "job_id": "verify-poison",
                "page_index": 0,
                "first_enqueued_at_ms": int(time.time() * 1000),
                "trace_id": "verify-poison",
            },
        )
        deadline = time.monotonic() + 30
        after = before
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            after = int(await redis.get(POISON_COUNTER_KEY) or 0)
            if after > before:
                break
        check(
            "a malformed entry is quarantined and counted, not swallowed",
            after == before + 1,
            f"poison counter {before} -> {after}",
        )

        # The property the unit tests cannot express: the workers are still
        # ALIVE. A job submitted after the poison entry has to be picked up,
        # which is only possible if no replica exited.
        post = await client.post(f"{API}/jobs", json={"pages": 2})
        post_status = await wait_for_job(client, post.json()["job_id"], timeout_s=120)
        check(
            "the workers survive a poison entry and keep processing jobs",
            bool(post_status.get("complete")),
            f"{post_status.get('done')}/{post_status.get('total_pages')} after poison",
        )
        depth = (await client.get(f"{API}/queue/depth")).json()
        check(
            "the poison entry is settled, not left to be redelivered forever",
            depth["stream_length"] == 0 and depth["pending"] == 0,
            str(depth),
        )
    finally:
        await redis.aclose()


async def main() -> int:
    print("=" * 72)
    print("Sarvam Vision Orchestrator - system verification")
    print("=" * 72)

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            await client.get(f"{API}/health")
        except Exception as exc:  # noqa: BLE001
            print(f"\nCannot reach {API}: {exc}")
            print("Start the stack first:  docker compose up -d --build")
            return 2

        await verify_health(client)
        await verify_tracing(client)
        await verify_rate_limiting(client)
        await verify_idempotency(client)
        await verify_chaos(client)
        job_id = await verify_end_to_end(client)
        await verify_admission(client)
        await verify_pdf_ingestion(client)
        await verify_streaming(client)
        await verify_redelivery(client, job_id)
        await verify_reaper(client)
        await verify_evaluate(client)
        await verify_metrics(client)
        await verify_limits(client)

    print("\n" + "=" * 72)
    total = _passed + len(_failed)
    print(f"{_passed}/{total} checks passed")
    if _failed:
        print("\nFAILED:")
        for name in _failed:
            print(f"  - {name}")
    print("=" * 72)
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
