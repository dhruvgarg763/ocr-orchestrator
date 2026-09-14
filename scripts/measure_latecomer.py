"""Time-to-first-page for a job that arrives INTO a busy system.

The case scripts/measure_ttfp.py can't test, since there every job starts
at t=0 into an empty system. Two suspected defects, measured separately:
(A) a single-page job on an idle system - it goes entirely to the lead
lane, but the dispatch loop's blocking read watches only the main stream,
so nothing wakes an idle worker; (B) any job arriving while every slot is
occupied - the lead probe only runs when capacity > 0, so a saturated
system doesn't poll the lane at all.

Usage: python scripts/measure_latecomer.py [mode]   mode: idle | busy | both
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time

import httpx

BASE = "http://localhost:8000"

SETTLE_S = 18.0
"""Fixed, so every configuration is measured in the same system state."""

SAMPLES = 30
"""A real p95 needs enough samples that it is not simply the maximum.

At n=12, int(0.95 * 12) = 11 selects the last element, so the reported "p95"
was the max of 12 - a far stricter statistic than intended, and one that swings
wildly between runs. At n=30 the p95 is the 29th sample and the max is
reported separately, so an outlier is visible as an outlier instead of being
read as the tail."""


async def ttfp(client: httpx.AsyncClient, pages: int) -> float:
    """Submit one job, tail it, return ms from POST to the first page event."""
    started = time.perf_counter()
    response = await client.post(f"{BASE}/jobs", json={"pages": pages})
    response.raise_for_status()
    job_id = response.json()["job_id"]

    async with client.stream("GET", f"{BASE}/jobs/{job_id}/stream") as stream:
        buffer = ""
        async for chunk in stream.aiter_text():
            buffer += chunk
            while "\n\n" in buffer:
                raw, buffer = buffer.split("\n\n", 1)
                for line in raw.splitlines():
                    if line.startswith("event: page."):
                        return (time.perf_counter() - started) * 1000
    return float("nan")


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def report(label: str, samples: list[float]) -> None:
    clean = [s for s in samples if s == s]
    if not clean:
        print(f"{label:<28} no samples")
        return
    print(
        f"{label:<28} p50={statistics.median(clean):7.1f}ms "
        f"p95={pct(clean, 0.95):7.1f}ms max={max(clean):7.1f}ms n={len(clean)}"
    )


async def measure_idle(client: httpx.AsyncClient) -> None:
    print("\n--- A. idle system: does page count change first-page latency? ---")
    print("A 1-page job is ENTIRELY lead-lane, so nothing lands in the main")
    print("stream to wake a worker blocked there.\n")

    for pages in (1, 2, 20):
        samples = []
        for _ in range(6):
            samples.append(await ttfp(client, pages))
            await asyncio.sleep(0.4)
        report(f"{pages}-page job", samples)


async def saturate(client: httpx.AsyncClient, jobs: int, pages: int) -> None:
    """Fill the system with VLM-stage work, without tailing any of it."""
    await asyncio.gather(
        *(client.post(f"{BASE}/jobs", json={"pages": pages}) for _ in range(jobs))
    )


async def measure_busy(client: httpx.AsyncClient) -> None:
    print("\n--- B. latecomer into a saturated system ---")
    print("Background load first, then wait for every slot to be holding a")
    print("VLM-stage page, then submit ONE new job and time its first page.\n")

    await saturate(client, jobs=40, pages=20)

    # A FIXED settle time, not a depth threshold.
    #
    # The obvious condition - wait until `pending` reaches N - is not
    # comparable across configurations, because `pending` is
    # delivered-but-unacked and is therefore bounded by the slots available to
    # the MAIN lane, which is exactly what worker_lead_reserve changes. A
    # reserve=12 run would never reach the same threshold as a reserve=0 run,
    # so each config would be measured in a different system state and the
    # comparison would be meaningless. A fixed wait puts every config at the
    # same point on the same workload; the depth is printed so that the states
    # being compared are visible rather than assumed.
    await asyncio.sleep(SETTLE_S)
    depth = (await client.get(f"{BASE}/queue/depth")).json()
    print(f"settled after {SETTLE_S}s: {depth}")

    samples = []
    for _ in range(SAMPLES):
        samples.append(await ttfp(client, 5))
        await asyncio.sleep(1.2)
    report("latecomer 5-page job", samples)

    single = []
    for _ in range(SAMPLES):
        single.append(await ttfp(client, 1))
        await asyncio.sleep(1.2)
    report("latecomer 1-page job", single)


async def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    limits = httpx.Limits(max_connections=80, max_keepalive_connections=80)
    async with httpx.AsyncClient(timeout=180.0, limits=limits) as client:
        if mode in ("idle", "both"):
            await measure_idle(client)
        if mode in ("busy", "both"):
            await measure_busy(client)


if __name__ == "__main__":
    asyncio.run(main())
