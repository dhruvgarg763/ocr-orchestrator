"""In-container half of the SIGKILL idempotency proof.

Driven by scripts/prove_recovery.sh, which owns the `docker kill` (the
Docker CLI isn't available inside the api container). Asserts the two
claims Module D makes: every page reaches a terminal state with no
`*_RUNNING` left behind, and the mock's per-key execution count shows
zero duplicates - an exact count, not a heuristic, since it only
increments on success.

Usage: python scripts/prove_recovery.py <phase> [args]
  arm <jobs> <pages>   reset counters, submit, wait for work to be in flight
  watch <deadline_s>   wait for every job to complete, then assert
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx

BASE = "http://localhost:8000"
MOCK = "http://mock-model:8001"

STATE_FILE = "/tmp/prove_recovery.json"
"""Carries job ids between the `arm` and `watch` phases, because the shell
driver kills a container between them and each phase is its own process."""


async def arm(client: httpx.AsyncClient, jobs: int, pages: int) -> None:
    """Reset the evidence, submit the work, and wait until it is really moving.

    The wait matters. Killing a worker before any page has been claimed proves
    nothing at all - there would be no pending entries to orphan, so the job
    would complete by the ordinary path and the test would pass without
    exercising a single line of recovery code. So this blocks until pages are
    actually in flight.
    """
    await client.post(f"{MOCK}/admin/reset")

    job_ids = []
    for _ in range(jobs):
        response = await client.post(f"{BASE}/jobs", json={"pages": pages})
        response.raise_for_status()
        job_ids.append(response.json()["job_id"])

    deadline = time.monotonic() + 30
    pending = 0
    while time.monotonic() < deadline:
        depth = (await client.get(f"{BASE}/queue/depth")).json()
        pending = depth["pending"]
        if pending >= 8:
            break
        await asyncio.sleep(0.25)
    else:
        raise SystemExit(f"work never got in flight (pending={pending})")

    with open(STATE_FILE, "w") as handle:
        json.dump({"jobs": job_ids, "pages": pages}, handle)

    print(f"armed: {jobs} jobs x {pages} pages, pending={pending}")


async def watch(client: httpx.AsyncClient, deadline_s: float) -> None:
    with open(STATE_FILE) as handle:
        state = json.load(handle)
    job_ids: list[str] = state["jobs"]
    pages: int = state["pages"]
    expected = len(job_ids) * pages

    started = time.monotonic()
    deadline = started + deadline_s
    done = 0
    statuses: list[dict] = []
    while time.monotonic() < deadline:
        responses = await asyncio.gather(
            *(client.get(f"{BASE}/jobs/{job_id}") for job_id in job_ids)
        )
        statuses = [r.json() for r in responses]
        done = sum(int(s["done"]) for s in statuses)
        if done >= expected:
            break
        await asyncio.sleep(1.0)

    elapsed = time.monotonic() - started
    print()
    print(f"completed {done}/{expected} pages in {elapsed:.1f}s")

    # ---- (a) every page terminal, nothing left running, nothing queued ----
    #
    # Read from each job's state_counts rather than by fetching all 60 pages:
    # the aggregate is what the invariant is about, and the endpoint already
    # computes it. A page in ANY non-terminal state is a failure here - a
    # *_RUNNING page after recovery means a claim was left behind, and a PENDING
    # one with an empty queue is the stranded-page shape exactly.
    terminal = {"DONE", "FALLBACK_DONE", "FAILED"}
    by_state: dict[str, int] = {}
    for status in statuses:
        for name, count in status["state_counts"].items():
            by_state[name] = by_state.get(name, 0) + int(count)
    stuck = {n: c for n, c in by_state.items() if n not in terminal}

    depth = (await client.get(f"{BASE}/queue/depth")).json()
    consumers = (await client.get(f"{BASE}/queue/consumers")).json()

    print("page states:", by_state)
    print("queue depth:", depth)
    print("orphans still stale:", consumers["orphaned"])

    # ---- (b) no stage was ever computed twice --------------------------
    counts = (await client.get(f"{MOCK}/admin/call-counts")).json()
    duplicates = counts["idempotency"]["duplicate_executions"]
    endpoints = counts["endpoints"]
    print("\nmock endpoint counters:")
    for name, stats in endpoints.items():
        print(
            f"  {name:<8} requests={stats['requests']:<6} "
            f"executions={stats['executions']:<6} replays={stats['replays']:<6} "
            f"injected_failures={stats['injected_failures']}"
        )
    print("duplicate executions:", duplicates or "{} (none)")

    failures = []
    if done < expected:
        failures.append(f"job did not reach 100%: {done}/{expected}")
    if stuck:
        failures.append(f"pages left non-terminal: {stuck}")
    if depth["stream_length"] != 0:
        failures.append(f"queue not drained: {depth}")
    if duplicates:
        failures.append(f"duplicate model executions: {duplicates}")

    # The STRONGEST form of claim (b), and the one that does not depend on any
    # counter the mock keeps about itself.
    #
    # `expected` pages were submitted and N of them were reclaimed after the
    # kill, so those N were delivered at least twice. If stage-level idempotency
    # were broken - if a reclaimed page re-ran a stage it had already committed -
    # executions would EXCEED the page count, by exactly the number of pages
    # whose stage was redone. So `executions <= expected` per stage is the
    # assertion, and equality is the interesting case: every page computed once,
    # no more and no fewer.
    #
    # Note why <= and not ==: a page whose layout fails permanently lands in
    # FAILED with no successful layout execution at all, which is a legitimate
    # outcome that would make a == assertion flaky for reasons unrelated to
    # idempotency. Retries do not inflate this - the mock counts an execution
    # only after the call SUCCEEDS, so a 500-then-success is one execution under
    # the same key.
    for name, stats in endpoints.items():
        if int(stats["executions"]) > expected:
            failures.append(
                f"{name} ran {stats['executions']} times for {expected} pages: "
                "a committed stage was recomputed"
            )

    # Secondary evidence, reported rather than asserted. `replays` counts
    # requests served from the idempotency cache or coalesced onto an in-flight
    # leader - redeliveries that WOULD have become duplicate model calls. It is
    # NOT asserted because it is only non-zero when the dying worker's in-flight
    # call happened to complete at the mock before the socket closed; a SIGKILL
    # usually cancels it instead, so the reclaimed page's call is a first
    # execution rather than a replay. Both are correct, and only the executions
    # bound above distinguishes correct from broken.
    replays = sum(int(v["replays"]) for v in endpoints.values())

    print()
    for line in failures:
        print(f"FAIL  {line}")
    if not failures:
        print("PASS  every page terminal, job at 100%, zero duplicate executions")
        print(f"      no stage exceeded {expected} executions for {expected} pages")
        print(f"      ({replays} redeliveries absorbed by the idempotency cache)")
    raise SystemExit(1 if failures else 0)


async def main() -> None:
    phase = sys.argv[1]
    async with httpx.AsyncClient(timeout=60.0) as client:
        if phase == "arm":
            await arm(client, int(sys.argv[2]), int(sys.argv[3]))
        elif phase == "watch":
            await watch(client, float(sys.argv[2]))
        else:
            raise SystemExit(f"unknown phase {phase!r}")


if __name__ == "__main__":
    asyncio.run(main())
