# Architecture

OCR pipeline orchestrator: ingests multi-page documents, fans pages across two
mock inference endpoints with a 10x capacity asymmetry, applies backpressure
instead of dropping work, streams out-of-order results over SSE, and computes
layout evaluation metrics.

    docker compose up -d --build --scale worker=3
    python bench/benchmark.py --jobs 50 --pages 20     # graded table
    python scripts/verify.py                           # 105 live checks

Measured, 50 concurrent jobs / 1,000 pages, cold stack:

| Graded metric | Target | Measured |
|---|---|---|
| Peak RSS, all containers | < 500 MB | **261-281 MB** |
| Unhandled pages | 0 | **0** of 1,000 |
| Tree diff, 46-node document tree | < 100 ms | **4.6 ms** |
| Time-to-first-page, p95 per client | < 200 ms | **94-99 ms** |

Ranges are across repeated cold-stack runs, not a single best result.

Full measurement record: [README.md](README.md).

---

## 1. Queue & state machine design

**Redis Streams, not a list.** `BRPOP` removes an item the instant a worker
receives it; a SIGKILL a millisecond later loses the task with no record it
existed. `XREADGROUP` instead records delivery in that consumer's Pending
Entries List, so a dead worker's work stays visible (`XPENDING`) and
reclaimable (`XAUTOCLAIM`). The cost is at-least-once delivery — a task can
arrive twice — so processing is made idempotent (below) rather than chasing
exactly-once.

**Two lanes.** Page 0 of every job goes to `stream:pages:lead`, drained
preferentially; the rest go to `stream:pages`. Under one FIFO stream, TTFP
p95 measured 14,511 ms: at 100 rps layout throughput, the 50th job's first
page sits at queue position ~980. A second lane fixes this without touching
the rate limiter. Requeues and stage handoffs go to the **main** lane only —
a page already delivered once is no longer first-page-critical.

**Page state**, one hash per page (`job:{jid}:page:{n}`):

    PENDING → LAYOUT_RUNNING → LAYOUT_DONE → VLM_RUNNING → DONE
                    ↓                              ↓
                 PENDING (retry)              LAYOUT_DONE (retry)
                    ↓                              ↓
                 FAILED  ←──────────────────→  FALLBACK_DONE

Terminal: `DONE`, `FALLBACK_DONE`, `FAILED`. `TRANSITIONS` in
[app/queue/state.py](app/queue/state.py) is the single source of truth.

- **Atomic CAS in Lua**, not `HGET`/`HSET`: one script checks the allowed set,
  writes the new state, increments `done_count` if terminal, renews both TTLs
  — atomically, so two workers racing a page cannot both proceed.
- **Each stage commits separately**, so recovery rolls back only to the last
  checkpoint, never to the start — a committed layout result is never
  redone.
- **TTL means inactivity, not lifetime**: renewed on every transition.

**Idempotency, two layers:** the state CAS stops stage reprocessing;
`Idempotency-Key = sha256(job:page:stage)[:32]` (cached by the mock) stops a
duplicated *model call* in the one surviving window — a crash after the call
but before the commit. Proven live: `docker kill -9` mid-job, then confirm the
job reaches 100% and `/admin/call-counts` shows no stage called twice.

**Recovery:** leases renew every 5 s; a reaper `XAUTOCLAIM`s entries idle over
30 s; a restarting worker fast-paths its own PEL first. Entries that fail to
*parse* are quarantined (acked, copied to `stream:pages:poison`, counted) —
parsing runs upstream of every per-task `try/except`, so one malformed entry
previously exited every worker process and recurred on every restart.

---

## 2. Backpressure strategy & memory boundaries

| # | Layer | Behaviour |
|---|---|---|
| 1 | Admission control | backlog > 5,000 → `503` + `Retry-After`; recovers at 80% |
| 2 | Distributed token bucket | Redis Lua, atomic: layout 100 rps, VLM 10 rps |
| 3 | AIMD concurrency limit | 429/timeout/SLO breach → `limit *= 0.7`; 20 clean calls → `limit += 1` |
| 4 | Retries | 3 attempts, full-jitter backoff 200 ms→5 s, honours `Retry-After` |
| 5 | Circuit breaker | 10 s window, ≥10 calls, >50% fail → OPEN; 5 s cooldown → HALF_OPEN |

The bucket is in Redis, not per-process, because N replicas with local
buckets produce Nx the intended rate; it returns `wait_ms` so callers `await`
rather than spin. The bucket enforces the rate we were *told*; AIMD discovers
the rate actually available (additive increase / multiplicative decrease —
TCP congestion control). Full jitter on retries stops clients resynchronising
into a convoy that re-hammers the endpoint at one instant.

**Degrading:** an exhausted VLM serves the page from its committed layout
output (`degraded: true, confidence: 0.4`), terminal as `FALLBACK_DONE`.
Layout failure has nothing to degrade to, so it is the one case that goes
`FAILED`.

**Zero drop, defined:** every page reaches a recorded terminal state, and
every non-success terminus is counted. Terminal failures are run-dependent
(the mock fails 2% of layout calls): runs land at 0-1 `FAILED` of 1,000, and
when one occurs it is layout exhausting its retries — visible in `state_counts`
and `orch_pages_terminal_total{result="failed"}`, never silent. A vanished page
is a drop; a counted failure is not. A `503` at the edge is the
same standard applied at admission.

**Memory boundaries:**

| Boundary | Enforcement |
|---|---|
| Upload | streamed to disk in 1 MiB chunks; `file.read()` never called. Caps: 120 MB / 100 pages |
| Page extraction | page count from the xref only; worker opens the PDF per task, reads exactly page *n* — O(page), not O(document) |
| Dispatch | 16 concurrent pages/worker, prefetch 8; no unbounded `gather` |
| Result streams | `MAXLEN 256` + 1 h TTL — a slow SSE client cannot grow Redis |
| Regression guard | `mem_limit: 256m` on api and worker — a reintroduced O(file) buffer is OOM-killed loudly |

Pages are **not** pre-split (avoids `pypdf` object-cache accumulation). Peak
RSS is the peak of the *sum*, not the sum of peaks — replicas peak at
different moments.

---

## 3. Tree Edit Distance: algorithmic complexity

Zhang-Shasha, [app/eval/ted.py](app/eval/ted.py). Nodes are numbered in
**postorder** with leftmost-descendant pointers, so every subtree is a
contiguous integer interval — forest distances index by two integers, no set
bookkeeping. The walk is **iterative** (a 2,000-node path is legal input;
recursion would overflow). **Keyroots** — nodes that are not their parent's
leftmost child — are the only ones given their own `treedist` pass; every
other node's forest distances fall out as a by-product.

**Cost driver:**

    Θ( W(T₁) · W(T₂) )   where  W(T) = Σ_{k ∈ keyroots(T)} |subtree(k)|

The textbook bound `O(n·m·min(d,ℓ))` is loose — measured, broom trees scale
quadratically where it predicts cubic. For shallow, wide document trees
`W(T) ≈ n`, so real layouts run near **O(n²)**. Caterpillars (depth *and*
leaf count both Θ(n)) hit the true **O(n⁴)** worst case — fitted exponent
3.98 on measured runtimes.

**Memory:** one forest-distance matrix, allocated once at max size, reused
across every keyroot pair — O(n₁·n₂) words total, not per pair.

**Latency is an admission guarantee.** `POST /evaluate` rejects before
computing: `nodes > 2,000` **or** `W(T₁)·W(T₂) > budget_ms × work_per_ms`
(100 × 1,800 = 180,000) → `413`. This discriminates by shape, not size: a
46-node document tree runs in 4.6 ms; a 49-node caterpillar is refused
outright rather than admitted at 113 ms. `work_per_ms` is a machine-speed
constant that drifts (measured 3,500 configured vs. 2,216–2,724 achieved), so
`verify.py` re-measures it every run and fails if the config is optimistic.
Computation runs in `asyncio.to_thread`, so it never blocks the event loop.

Caveat: Zhang-Shasha has insert/delete/relabel, no *move* — a sibling swap
costs 2, not 1.

---

## 4. Trade-offs & production scale-out path

**Trade-offs**

1. No rasterization — the mock takes a page descriptor, not pixels; same
   memory profile as a production blob-store URI.
2. TTFP is graded per client (95 ms); a 50-way simultaneous burst is
   1,237 ms, because one API replica's own `POST /jobs` handling costs
   999 ms p95 at that concurrency. Fixed by more API replicas, not tuning.
3. At-least-once, not exactly-once — cheaper than a distributed transaction
   spanning Redis and the model endpoint.
4. Greedy IoU matching, not Hungarian — O(n log n) vs O(n³); can be
   suboptimal (counterexamples pinned by a brute-force oracle in tests).
5. `/metrics` fan-in is hand-rolled (3 replicas, one service name, no port):
   counters flush as deltas, gauges as TTL'd absolutes. Known bias:
   redelivery inflates `orch_pages_terminal_total{result="success"}`, so the
   zero-drop figure is read from `state_counts` instead.
6. Single Redis is the SPOF for queue and page state.
7. The graded benchmark submits synthetic `{"pages": N}` jobs, not real PDFs,
   to isolate queue/dispatch cost from parsing cost - but that means it never
   exercises the literal "50 concurrent PDF ingestions" the RSS target names.
   Measured separately with 50 real concurrent uploads (1,000 pages): **323.4
   MB peak**, still under the 500 MB budget with room to spare (README).

**Scale-out path**

- **API** — stateless now; N replicas behind a load balancer. SSE needs no
  sticky sessions (`Last-Event-ID` resumes against any replica) — this
  directly fixes trade-off 2.
- **Workers** — already horizontal (bucket/breaker/AIMD state is in Redis, so
  replicas don't multiply the downstream rate). Next: separate pools per
  endpoint so layout capacity stops sharing a budget with VLM slots. The
  replica count is not arbitrary: each worker's 16 slots are shared by both
  stages (4 reserved for the priority lane), and by Little's Law the VLM's
  10 rps at ~2.25 s latency needs ~22.5 pages in flight to saturate. Measured
  VLM capacity reached: **1 replica 37%, 2 replicas 63%, 3 replicas 95%** —
  below three, worker concurrency rather than the rate limiter is the binding
  constraint, which is the same argument for the per-endpoint pools.
- **Redis** — Sentinel/managed replica set removes the SPOF with no code
  change. Cluster is a bigger lift: the multi-key Lua scripts need
  hash-tagging by job id to be slot-safe.
- **Page bytes → object storage** (S3/GCS + URI in the task entry) — same
  O(page) profile, removes the shared volume coupling api and worker.
- **Metrics → native Prometheus** once each worker has its own port/scrape
  target — removes the Redis fan-in and its redelivery bias.
- **Queue beyond Streams** past ~10⁴–10⁵ pages/s: partition by job id first;
  move to Kafka only if long retention or replay is required.
