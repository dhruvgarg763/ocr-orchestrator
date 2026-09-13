"""A job that takes longer than its own `result_ttl_s` must still complete.

Run against a stack started with a deliberately tiny TTL:

    ORCH_RESULT_TTL_S=5 docker compose up -d --force-recreate --scale worker=3
    docker compose exec -T api python /tmp/ttl.py <pages> <ttl_s>

Why this needs a live stack rather than only a unit test
-------------------------------------------------------
The unit test drives `transition` directly, so it proves the script renews what
it says it renews. It cannot prove the renewal covers every key the REAL
pipeline depends on while a job is in flight - the job hash, every page hash
including ones no worker has claimed yet, and the result stream that SSE reads
from. Those are touched by different code paths (the state script, the
conditional page sweep, and results.py's own EXPIRE), and the only way to know
they agree is to run a job that outlives the TTL and watch for the one event
that requires all of them: `job.complete`.

The failure this detects
------------------------
Before TTL renewal, the job hash expired partway through and `HINCRBY` silently
recreated it without `total_pages`, so `completed_job` could never be true
again. Every page kept processing correctly and the job simply never reported
complete - a hang at 99% with nothing in the logs, and an SSE client waiting out
`sse_max_duration_s` for an event that could no longer exist.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx

BASE = "http://localhost:8000"


async def main() -> None:
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    ttl_s = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    async with httpx.AsyncClient(timeout=300.0) as client:
        started = time.perf_counter()

        response = await client.post(f"{BASE}/jobs", json={"pages": pages})
        response.raise_for_status()
        job_id = response.json()["job_id"]
        print(f"submitted {job_id} with {pages} pages")

        saw_complete = False
        finals = 0
        async with client.stream("GET", f"{BASE}/jobs/{job_id}/stream") as stream:
            buffer = ""
            async for chunk in stream.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    raw, buffer = buffer.split("\n\n", 1)
                    event = None
                    for line in raw.splitlines():
                        if line.startswith("event: "):
                            event = line[7:]
                    if event == "page.final":
                        finals += 1
                    elif event == "job.complete":
                        saw_complete = True
                if saw_complete:
                    break

        elapsed = time.perf_counter() - started
        status = (await client.get(f"{BASE}/jobs/{job_id}")).json()

        print(f"elapsed            {elapsed:.1f}s")
        print(f"page.final events  {finals}/{pages}")
        print(f"job.complete       {saw_complete}")
        print(f"status             {json.dumps(status)}")

        failures = []

        # THE GUARD, and the reason it is here rather than assumed.
        #
        # The first run of this script used a 20s TTL and a job that finished in
        # 10.3s. It printed PASS, and the PASS was worthless: the job never
        # reached its own expiry, so nothing about renewal was exercised and a
        # completely unrenewed build would have passed identically. A test whose
        # scenario cannot produce the failure is not evidence, so the scenario
        # itself is now asserted before the result is believed.
        if elapsed <= ttl_s:
            failures.append(
                f"VACUOUS: the job finished in {elapsed:.1f}s without reaching "
                f"its {ttl_s:.0f}s TTL, so renewal was never exercised. Raise "
                "the page count or lower ORCH_RESULT_TTL_S."
            )

        if not saw_complete:
            failures.append(
                "no job.complete: the job hash almost certainly expired "
                "mid-flight, so total_pages became unreadable"
            )
        if finals != pages:
            failures.append(f"{finals} page.final events for {pages} pages")
        if not status.get("complete"):
            failures.append(f"status never reported complete: {status}")
        if status.get("done") != pages:
            failures.append(f"done={status.get('done')} expected {pages}")

        print()
        for line in failures:
            print(f"FAIL  {line}")
        if not failures:
            print(
                f"PASS  a {elapsed:.1f}s job completed and closed its stream on a "
                f"{ttl_s:.0f}s TTL - it outlived its own records "
                f"{elapsed / ttl_s:.1f}x over"
            )
        raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
