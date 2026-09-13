"""Measure throughput and failure rate for a single job.

    python scripts/measure.py --pages 100

Deliberately small: the graded benchmark (50 concurrent jobs, 1,000 pages, peak
RSS sampling) arrives in Step 18. This exists to make one thing visible now -
raising worker concurrency without client-side rate limiting buys throughput and
pays for it in dropped pages.
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx

API = "http://localhost:8000"
MOCK = "http://localhost:8001"


async def run(pages: int, timeout_s: float) -> int:
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(f"{MOCK}/admin/reset")
        await client.delete(f"{MOCK}/admin/chaos")

        started = time.monotonic()
        accepted = await client.post(f"{API}/jobs", json={"pages": pages})
        accepted.raise_for_status()
        job_id = accepted.json()["job_id"]
        print(f"job {job_id}: {pages} pages submitted")

        deadline = started + timeout_s
        status: dict = {}
        last_done = -1
        while time.monotonic() < deadline:
            status = (await client.get(f"{API}/jobs/{job_id}")).json()
            done = status["done"]
            if done != last_done:
                elapsed = time.monotonic() - started
                depth = (await client.get(f"{API}/queue/depth")).json()
                print(
                    f"  t+{elapsed:5.1f}s  done={done:>4}/{pages}"
                    f"  backlog={depth['backlog']:>4} in_flight={depth['pending']:>3}"
                    f"  {status['state_counts']}"
                )
                last_done = done
            if status.get("complete"):
                break
            await asyncio.sleep(1.0)

        elapsed = time.monotonic() - started
        counts = (await client.get(f"{MOCK}/admin/call-counts")).json()

    states = status.get("state_counts", {})
    done_ok = states.get("DONE", 0)
    failed = states.get("FAILED", 0) + states.get("FALLBACK_DONE", 0)
    stuck = pages - done_ok - failed

    print()
    print("=" * 62)
    print(f"  wall time          : {elapsed:.1f}s")
    # GOODPUT, not throughput. done_count counts every page that reached a
    # terminal state, and FAILED is terminal - so a "throughput" figure built on
    # it rewards failing faster. At concurrency=64 that metric read 15.98
    # pages/sec against a 10 pages/sec ceiling: impossible, and the giveaway
    # that it was measuring the wrong thing. Only successful pages count.
    print(f"  goodput (DONE/s)   : {done_ok / elapsed:.2f} pages/sec")
    print(f"  terminal states/s  : {status.get('done', 0) / elapsed:.2f} pages/sec"
          f"  <- includes failures; not a success metric")
    print(f"  VLM ceiling        : 10.00 pages/sec")
    print(f"  useful utilisation : {done_ok / elapsed / 10 * 100:.0f}%")
    print("-" * 62)
    print(f"  DONE               : {done_ok:>4} / {pages}")
    print(f"  FAILED             : {failed:>4} / {pages}   <-- dropped pages")
    if stuck:
        print(f"  neither            : {stuck:>4} / {pages}   <-- stranded (bug!)")
    print(f"  success rate       : {done_ok / pages * 100:.1f}%")
    print("-" * 62)
    for name in ("layout", "vlm"):
        stat = counts["endpoints"].get(name, {})
        requests = stat.get("requests", 0)
        limited = stat.get("rate_limited", 0)
        share = (limited / requests * 100) if requests else 0.0
        print(
            f"  {name:<7} requests={requests:>5} executions={stat.get('executions', 0):>5}"
            f" 429s={limited:>5} ({share:.0f}%) 5xx={stat.get('injected_failures', 0):>4}"
        )
    print("=" * 62)
    return 0 if not stuck else 1


def cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    return asyncio.run(run(args.pages, args.timeout))


if __name__ == "__main__":
    raise SystemExit(cli())
