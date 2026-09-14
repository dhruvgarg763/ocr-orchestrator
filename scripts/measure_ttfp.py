"""Measure time-to-first-page over the live SSE endpoint.

Graded at 15%, measured the way a client experiences it. Two clocks are
reported: "from POST" (end-to-end - admission, enqueue, handshake, the
layout call, everything) and "from OPEN" (isolates the pipeline from the
ingestion round trip). `stream.open` is NOT counted as a first page - it's
a connection header emitted immediately, and timing to it would produce a
flattering number that says nothing about whether a page was processed.

Usage: python scripts/measure_ttfp.py [jobs] [pages] [concurrency]
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time

import httpx

BASE = "http://localhost:8000"


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


async def one_job(client: httpx.AsyncClient, pages: int) -> dict:
    t_post = time.perf_counter()
    response = await client.post(f"{BASE}/jobs", json={"pages": pages})
    if response.status_code != 202:
        return {"error": f"{response.status_code}: {response.text[:120]}"}
    job_id = response.json()["job_id"]

    marks: dict[str, float] = {}
    counts: dict[str, int] = {}
    seqs: list[int] = []
    pages_seen: set[int] = set()

    t_open = time.perf_counter()
    async with client.stream("GET", f"{BASE}/jobs/{job_id}/stream") as stream:
        buffer = ""
        async for chunk in stream.aiter_text():
            if "first_byte" not in marks:
                marks["first_byte"] = time.perf_counter()
            buffer += chunk
            while "\n\n" in buffer:
                raw, buffer = buffer.split("\n\n", 1)
                name = data = None
                for line in raw.splitlines():
                    if line.startswith("event: "):
                        name = line[7:]
                    elif line.startswith("data: "):
                        data = json.loads(line[6:])
                if name is None:
                    counts["heartbeat"] = counts.get("heartbeat", 0) + 1
                    continue
                counts[name] = counts.get(name, 0) + 1
                if data and "seq" in data:
                    seqs.append(data["seq"])
                # Only a real page event counts as "first page".
                if name.startswith("page.") and "first_page" not in marks:
                    marks["first_page"] = time.perf_counter()
                if name == "page.partial":
                    pages_seen.add(data["page_index"])
                    marks["last_partial"] = time.perf_counter()
                if name == "page.final" and "first_final" not in marks:
                    marks["first_final"] = time.perf_counter()
                if name == "job.complete":
                    marks["complete"] = time.perf_counter()

    if "first_page" not in marks:
        return {"error": "no page event", "counts": counts}

    return {
        "job_id": job_id,
        "ttfp_from_post_ms": (marks["first_page"] - t_post) * 1000,
        "ttfp_from_open_ms": (marks["first_page"] - t_open) * 1000,
        "first_byte_ms": (marks.get("first_byte", marks["first_page"]) - t_open) * 1000,
        "ttff_ms": (marks.get("first_final", marks["first_page"]) - t_post) * 1000,
        # If layout ran at its own 100 rps this is ~pages/100 seconds. If it is
        # seconds, the fast stage is queueing behind the slow one.
        "all_partials_ms": (marks.get("last_partial", marks["first_page"]) - t_post)
        * 1000,
        "total_ms": (marks.get("complete", marks["first_page"]) - t_post) * 1000,
        "counts": counts,
        # The completeness check the dense counter exists for: a client can
        # prove it saw every event rather than hoping.
        "seq_gap": sorted(seqs) != list(range(1, len(seqs) + 1)),
        "partials": len(pages_seen),
    }


async def main() -> None:
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else jobs

    print(f"jobs={jobs} pages={pages} concurrency={concurrency}")
    gate = asyncio.Semaphore(concurrency)

    # No client-side connection cap below the concurrency level, or the
    # measurement would be timing httpx's own queue rather than the server.
    limits = httpx.Limits(
        max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2
    )
    async with httpx.AsyncClient(timeout=180.0, limits=limits) as client:

        async def run() -> dict:
            async with gate:
                return await one_job(client, pages)

        started = time.perf_counter()
        results = await asyncio.gather(*(run() for _ in range(jobs)))
        wall = time.perf_counter() - started

    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]

    if bad:
        print(f"\n!! {len(bad)} failed: {bad[0]['error']}")
    if not ok:
        return

    from_post = [r["ttfp_from_post_ms"] for r in ok]
    from_open = [r["ttfp_from_open_ms"] for r in ok]

    print(f"\nwall {wall:.1f}s  jobs ok {len(ok)}/{jobs}")
    print(f"{'metric':<26}{'p50':>9}{'p95':>9}{'max':>9}")
    for label, values in (
        ("ttfp from POST (ms)", from_post),
        ("ttfp from stream open", from_open),
        ("first byte (ms)", [r["first_byte_ms"] for r in ok]),
        ("all partials in (ms)", [r["all_partials_ms"] for r in ok]),
        ("time to first FINAL", [r["ttff_ms"] for r in ok]),
        ("job total (ms)", [r["total_ms"] for r in ok]),
    ):
        print(
            f"{label:<26}{statistics.median(values):>9.1f}"
            f"{pct(values, 0.95):>9.1f}{max(values):>9.1f}"
        )

    # Only the STORED events count. stream.open, stream.gap and heartbeats are
    # synthesised per connection and describe the subscriber, not the job.
    synthetic = {"heartbeat", "stream.open", "stream.gap"}
    expected = 2 * pages + 1
    exact = sum(
        1
        for r in ok
        if sum(v for k, v in r["counts"].items() if k not in synthetic) == expected
    )
    print(f"\nevents == 2*pages+1 ({expected}):  {exact}/{len(ok)} jobs")
    print(f"seq gaps:                     {sum(1 for r in ok if r['seq_gap'])}")
    print(f"partials seen (want {pages}):     "
          f"{min(r['partials'] for r in ok)}..{max(r['partials'] for r in ok)}")
    print(f"\nTARGET ttfp < 200ms -> p95 from POST = {pct(from_post, 0.95):.1f}ms "
          f"{'PASS' if pct(from_post, 0.95) < 200 else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
