# Sarvam Vision Orchestrator

High-throughput OCR pipeline orchestrator: ingests multi-page document jobs,
schedules page-level tasks across heterogeneous mock inference endpoints with
adaptive backpressure, streams out-of-order results over SSE, and computes
deterministic layout evaluation metrics.

## Run

```bash
docker compose up --build
```

| Service | Port | Purpose |
|---|---|---|
| `api` | 8000 | Ingestion, SSE streaming, evaluation |
| `worker` | - | Consumes page tasks; scale with `--scale worker=3` |
| `mock-model` | 8001 | Simulated inference engines |
| `redis` | 6379 | Queue + page state |

Design document: **[ARCHITECTURE.md](ARCHITECTURE.md)** - queue and state
machine, backpressure and memory boundaries, TED complexity, trade-offs and
scale-out path. This file is the long-form measurement record behind it.

## Orchestrator API

| Endpoint | Purpose |
|---|---|
| `POST /jobs` | Submit a synthetic job (`{"pages": N}`) - a page count, no document. Returns `202` with `job_id` and `trace_id`. |
| `POST /jobs/stream` | Ingest a PDF as a **raw request body**. The memory-optimal path. |
| `POST /jobs/upload` | Ingest a PDF as **multipart** (`-F file=@doc.pdf`). What a browser sends. |
| `GET /jobs/{id}/stream` | **SSE**: `page.partial` / `page.final` / `job.complete`, resumable via `Last-Event-ID` |
| `GET /jobs/{id}` | Progress and per-state page counts |
| `GET /jobs/{id}/pages/{n}` | One page's state and stored model output |
| `POST /evaluate` | **Module C**: compare an extracted document tree against ground truth. Returns CER/WER, IoU and TED plus per-metric timings |
| `GET /queue/depth` | `backlog` (undelivered) vs `pending` (in flight), across both lanes |
| `GET /admission` | Admitted vs shed jobs and pages, and whether shedding is active |
| `GET /queue/consumers` | Per-lane consumer ownership, and what the reaper considers abandoned |
| `GET /streams` | SSE subscriber occupancy, peak and refusals |
| `GET /metrics` | Prometheus exposition: queue depth, worker lag, retries, 429s, breaker, in-flight, latency histograms |
| `GET /health` | Liveness plus dependency status |

```bash
# synthetic job, then watch its pages arrive out of order
job=$(curl -sX POST localhost:8000/jobs -H 'Content-Type: application/json'   -d '{"pages":8}' | python -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
curl -N localhost:8000/jobs/$job/stream

# a real document
curl -X POST localhost:8000/jobs/stream --data-binary @doc.pdf
```

`page.partial` lands in ~50ms carrying layout output; `page.final` upgrades the
same `page_index` when the VLM returns 1.5-3s later. Measured
time-to-first-page: **94-99ms** at p95 per client (`bench/benchmark.py`,
1,000 pages).

## Queue design

Redis Stream `stream:pages` with consumer group `workers`. `XREADGROUP`
delivers an entry *and* records it in the consumer's Pending Entries List, so an
unacknowledged entry survives a worker crash and is reclaimable. `BRPOP` on a
list would delete the task on delivery and lose it.

That gives at-least-once delivery, so a page can arrive twice. The atomic state
CAS plus the `Idempotency-Key` header make redelivery a no-op instead of a
duplicate inference.

The task stream has **no `MAXLEN`** - trimming by length discards the oldest
entries, which in a task queue are unprocessed work. Entries are `XDEL`ed once
acknowledged instead.

## Verification

```bash
docker compose up -d --build --scale worker=3    # 4 services, 3 worker replicas
python scripts/verify.py                         # 105 live system checks
python -m pytest -q                              # 567 unit tests
```

`scripts/verify.py` exercises the assembled system over real HTTP and real
Redis: rate limiting, idempotent coalescing, chaos injection, a full job through
the queue, and at-least-once redelivery safety. It exits non-zero on any
failure, so it also works as a CI smoke gate.

The unit suite needs Redis on `localhost:6379` and uses database 15, flushed
around every test, so it cannot disturb application data.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
./.venv/Scripts/python.exe -m pytest -q

# Executable proofs of specific bugs that were found and fixed:
python -m pytest -k "single_flight_coalesces" -v   # idempotency TOCTOU race
python -m pytest -k "naive_check_then_act" -v      # naive Redis claim over-claims
python -m pytest -k "atomic_claim_survives" -v     # control: same window, correct
```

### Scope: what is complete

Every module in the assignment is implemented and every graded metric is
measured rather than asserted. The deliberate simplifications - no
rasterization, greedy box matching, single Redis - are named in
[ARCHITECTURE.md](ARCHITECTURE.md#4-trade-offs) rather than left to be found.

Crash recovery is complete as of Step 14 and covers both shapes: a worker
**restarting** under the same name, and a worker being **replaced** by a
container with a new one. Both verified live with `docker kill -9` mid-flight -
see [Crash recovery](#crash-recovery-leases-and-the-reaper).

A `429` storm is handled: classified retries with full jitter (Step 8), a shared
circuit breaker, and degradation to layout-only output with a low confidence
flag once a page is genuinely out of budget (Step 9). Try it:

```bash
curl -X POST localhost:8001/admin/chaos -H 'Content-Type: application/json' \
  -d '{"endpoint":"vlm","status":429,"ratio":1.0,"seconds":120}'
curl -X POST localhost:8000/jobs -H 'Content-Type: application/json' -d '{"pages":5}'
curl -N localhost:8000/jobs/<job_id>/stream
# then confirm the terminus, which is the actual assertion:
curl -s localhost:8000/jobs/<job_id> | python -m json.tool
```

Pages reach `FALLBACK_DONE`, not `FAILED`, and every non-`DONE` terminus is
counted rather than silently swallowed.

**The chaos window has to be long, and that is not padding.** A page holds out
for full VLM fidelity until `final_attempt`: `age_s >= degrade_after_s` (90s),
or the requeue backstop is spent (`app/worker/main.py`). A SHORT storm
therefore degrades nothing - the page retries, the storm ends, the VLM answers
normally, and the page lands `DONE`. Measured: a 30 second window produced
**zero** `FALLBACK_DONE` out of 10 pages, while 120 seconds produced 5 of 5.
That is degradation behaving as a last resort rather than a first response,
which is intended - but it makes a 20-30 second reproduction misleading, so the
window above is deliberately longer than the hold-out budget.

## Build status

| Step | Module | State |
|---|---|---|
| 1 | Orientation | done |
| 2 | Config, JSON logging, trace context | done |
| 3 | Mock model as tunable adversary | done |
| 4 | Redis page state + atomic state machine | done |
| 5 | Redis Streams queue + end-to-end worker | done |
| 6 | Bounded-dispatch concurrency | done |
| 7 | Distributed token bucket (Redis Lua) | done |
| 8 | Classified retries, full-jitter backoff | done |
| 9 | Circuit breaker + degraded fallback | done |
| 10 | AIMD adaptive rate control | done |
| 11 | Admission control at the edge | done |
| 12 | Chunked upload + lazy page extraction | done |
| 13 | SSE two-phase streaming + priority lane | done |
| 14 | Lease renewal + orphan reaper, SIGKILL idempotency proof | done |
| 15 | CER/WER: two-row edit distance, grapheme segmentation | done |
| 16 | Bounding-box IoU + greedy matching | done |
| 17 | Tree Edit Distance (Zhang-Shasha), iterative + memory-bounded | done |
| 18 | `POST /evaluate`: Module C wired, thread-offloaded, cost-capped | done |
| 19 | Prometheus `/metrics` + graded load benchmark | done |
| 20 | [ARCHITECTURE.md](ARCHITECTURE.md) design document | done |

### Concurrency model: bounded dispatch

The worker never reads more tasks than it has free capacity to run:

```
capacity = worker_concurrency - len(in_flight)
tasks    = await queue.read(count=min(worker_prefetch, capacity))
```

The usual alternative is to read a batch and guard execution with a semaphore.
That bounds concurrent *execution* but not what the process *holds*: by the time
the semaphore is consulted, N coroutines exist (~3.2 KB each, measured) and N
tasks have already moved into this consumer's Pending Entries List, so a crash
leaves all N to reclaim. Bounded dispatch leaves the backlog in Redis - durable,
observable via `XLEN`, and not our memory.

Measured goodput against concurrency, single worker, 60-page job, **no
client-side rate limiting yet**:

| concurrency | goodput | success | VLM 429s |
|---|---|---|---|
| 4 | 1.57 pages/s | 91.7% | 0% |
| 8 | 2.96 pages/s | 91.7% | 0% |
| 16 | **4.74 pages/s** | 81.7% | 13% |
| 32 | 4.58 pages/s | 31.7% | 67% |
| 64 | 3.85 pages/s | 33.3% | 65% |

Goodput peaks near 16 and then *declines* - classic congestion collapse, with
the endpoint's capacity spent on requests that get rejected. Measured with
goodput (`DONE`/sec), not terminal states/sec: `done_count` includes `FAILED`,
so a naive throughput metric rewards failing faster and read 16 pages/s against
a 10 pages/s ceiling.

Even the safe settings top out at 91.7%, because the residual loss is the
transient 5% failure rate and retries do not exist yet. No value of this knob
reaches the zero-drop target - that needs the mechanisms in Steps 7-9.

### Rate limiting: a shared token bucket in Redis

The mock's limiter is in-process, which is correct for a single server
enforcing its own published limit. A *client*-side limiter cannot work that
way - each replica would enforce 10 rps for an aggregate of N x 10. Measured:

| replicas | requests/sec sent | limit |
|---|---|---|
| 1 | 9 | 10 |
| 3 | 27 | 10 |
| 5 | 45 | 10 |

The error is proportional to how well you have scaled, so the bug appears
exactly when the system starts succeeding. Three things therefore live in
Redis, enforced by one Lua script:

1. the token count,
2. the read-modify-write (atomic, so concurrent claimants cannot overdraw),
3. **the clock** - via `redis.call('TIME')`, not a client-supplied timestamp.

Point 3 is load-bearing. With a client clock, a replica running 3 seconds ahead
was granted a full burst of 10 requests against an already-empty bucket: its
clock lead minted tokens from nothing. NTP steps, VM resumes and drifted hosts
all produce exactly that.

The script **reserves** a slot rather than reporting a wait to be re-polled.
When no token is free it deducts the cost anyway, driving the balance negative,
and returns the delay until that debt clears; the caller sleeps once and
proceeds. The negative balance *is* the queue. Polling instead makes every
waiter wake on the same computed delay and all but `rate` of them get denied
again - measured on a 10 rps bucket:

| waiters | grants | Redis calls (polling) | Redis calls (reserving) |
|---|---|---|---|
| 48 | 48 | 751 | 48 |
| 150 | 150 | 9,063 | 150 |
| 400 | 313 | 61,692 | 400 |

199 round trips per grant became 1. This is what Guava's
`RateLimiter.acquire()` does, and it is still a token bucket - the balance
simply carries a debt, bounded to `-(max_wait * rate)`.

Jitter is a small **absolute** spread (25ms), not a fraction of the wait.
Reservations already sequence callers 1/rate apart, so proportional jitter only
delayed the queue (8.4 grants/sec against a 10 rps limit) and reordered it (136
FIFO inversions in 313 grants).

Effect on the same 3-replica configuration:

| | before | after |
|---|---|---|
| goodput | 4.00 pages/s | **7.54 pages/s** |
| success rate | 18.0% | **95.0%** |
| VLM 429s | **81%** | **0%** |

And concurrency is now decoupled from overload - goodput stays flat at ~7.3
pages/s from 24 up to 192 effective concurrency, where previously raising
concurrency collapsed the success rate:

| effective concurrency | goodput | success | VLM 429s |
|---|---|---|---|
| 24 | 7.09 pages/s | 89.0% | 0 |
| 96 | 7.33 pages/s | 92.0% | 1 |
| 192 | 7.37 pages/s | 93.0% | 1 |

The residual failures are the transient 5xx rate, plus the occasional 429 that
slips through because a client limiter can never be perfectly in step with the
server's independent bucket. Both need retries (Step 8).

### Retries: classification, backoff, jitter

Errors are classified before anything is retried, because retrying the wrong
error spends the budget and merely delays the failure:

| Error | Disposition |
|---|---|
| `429`, `408`, `425`, `500`-`504` | retry, honouring `Retry-After` |
| other `4xx` | permanent - a malformed request stays malformed |
| timeouts, connection errors | retry |
| `RateLimitTimeout` | **requeue** - saturation is systemic, not page-specific |

Retrying a timeout is only safe because every attempt reuses the same
deterministic `Idempotency-Key`: if the original request landed and only the
response was lost, the retry is served from the server's cache instead of
re-running a 3-second inference. Retries and idempotency are a matched pair.

Backoff uses **full jitter** - a uniform draw over the whole exponential
window. Without jitter, every client that failed together retries in the same
instant and recreates the overload that caused the failure. Simulated with 60
clients and 4 attempts each, counting requests a 10-concurrent server would
reject:

| strategy | peak req/20ms | rejected |
|---|---|---|
| no jitter | 60 | 200 |
| equal jitter | 17 | 14 |
| **full jitter** | **12** | **5** |

Result on 100 pages, 3 replicas: **100/100 DONE, 0 failures, 8.10 pages/s**
(81% of the VLM ceiling). Nine transient 5xx occurred and retries recovered all
of them.

### What retries do NOT fix

Under a sustained 80% `429` storm, 40 pages produced **23 failures**. That is
the maths, not a bug: with three attempts, P(all fail) = 0.8^3 = 51%.

Raising the attempt budget is the wrong fix - reaching 99% success at an 80%
per-attempt failure rate needs 21 attempts, i.e. 21x the load on an endpoint
that is already rejecting everything:

| target success | attempts needed |
|---|---|
| 90% | 11 |
| 99% | 21 |
| 99.9% | 31 |

The answer is to stop asking, and to answer with something degraded instead.

### Circuit breaker + degraded fallback

```
         failure ratio >= threshold
CLOSED ------------------------------> OPEN
   ^                                     |  cooldown elapsed
   |  N consecutive probe successes      v
   +--------------- HALF_OPEN <----------+
                        |  any probe fails
                        +--> OPEN (cooldown restarts)
```

State lives in Redis, so one replica's discovery of an outage stops all of
them - with per-process breakers the blast radius scales with replica count,
the same failure mode as an in-process rate limiter.

`HALF_OPEN` rather than closing on a timer: a timer-close sends full production
load at a service that may still be dead, re-killing it. Probing answers the
same question with 2 requests. Probe slots are counted in Redis so N replicas
cannot each send "just one".

`min_volume` guards against noise - 1 failure out of 2 requests reads as 50%
and would open the circuit on a single unlucky call. The window rolls, so an
old outage is forgotten rather than keeping the ratio elevated forever.

### Degrading is a last resort, not a first response

By the time the VLM runs, the layout result is already committed, and layout
output is usable on its own: bounding boxes, block types, reading order. So an
unavailable VLM can produce `FALLBACK_DONE` with `degraded: true,
confidence: 0.4` - output was produced, only fidelity was lost. Note this needs
no extra call to "fall back to Fast-Layout-Model": that result is in hand.

But *when* to degrade matters more than the mechanism. The spec says fall back
"if retries are exhausted", and a page that arrives while the VLM's circuit is
open has exhausted nothing - it never made the call. Degrading there is both
off-spec and wasteful:

| policy on an open circuit | `DONE` | `FALLBACK_DONE` | `FAILED` |
|---|---|---|---|
| Step 8: retries only, no fallback | 17 | - | **23** |
| degrade immediately | 8 | 32 | 0 |
| **hold out, degrade at the deadline** | **40** | **0** | **0** |

Same 40-page job, same 80% `429` storm for 40s. Degrading on the first open
circuit threw away 32 pages' worth of quality that would almost all have
succeeded: a page gets ~8 deliveries of 3 attempts during a 40s outage, putting
P(never succeeding) near 0.5%.

So a page holds out for full fidelity until `degrade_after_s` (90s default),
then accepts a degraded result. Holding out costs no time-to-first-page -
layout output streams as soon as it lands, and only the VLM upgrade waits.

The backstop is what keeps zero-drop true rather than turning it into "the job
hangs". With a 70s total outage and a 12s budget, all 20 pages degraded and the
job completed:

| outage vs budget | result |
|---|---|
| 40s outage, 90s budget | **40/40 `DONE`**, full fidelity |
| 70s outage, 12s budget | **20/20 `FALLBACK_DONE`**, job completes |

The asymmetry between stages is deliberate: **layout cannot degrade, because
layout IS the fallback.** A page with no layout result has nothing to fall back
to, so layout failure is the only remaining path to `FAILED`.

A note on bounding by age rather than by attempt count: a requeue *count* is
the wrong unit for a time-based policy, and using it as one caused the same bug
three times - its wall-clock meaning depends on how fast saturation is detected
and how long the pause is. `max_requeues` is now derived from the time budget
(`effective_max_requeues`) so age is what fires, leaving the count as the pure
runaway-loop safety net it was meant to be.

### Page state machine

```
PENDING -> LAYOUT_RUNNING -> LAYOUT_DONE -> VLM_RUNNING -> DONE
              |                  |               |
              |                  |               +-> FALLBACK_DONE  (degraded, confidence 0.4)
              +-> PENDING        +-> FALLBACK_DONE   (breaker open)
              +-> FAILED             (retry / reclaim)
```

Each stage commits separately, so every stage is a checkpoint: a crash during
the VLM call resumes at `LAYOUT_DONE` and never re-runs the layout call.
Transitions are a single Lua script, which Redis executes atomically - a
read-check-write from the client would let two workers claim the same page.

---

## Mock model API

Deliberately built as a *tunable adversary*: it enforces real rate limits,
injects faults on demand, and reports enough internal state to prove the
orchestrator behaved correctly.

### Inference

Both endpoints take the same body. Note it carries a page **reference**, never
page bytes - that is what keeps orchestrator memory O(1) per page.

```jsonc
{ "job_id": "abc", "page_index": 0, "page_ref": "/data/abc.pdf#0", "text_hint": null }
```

| Endpoint | Latency | Rate limit | Failure rate | Returns |
|---|---|---|---|---|
| `POST /v1/predict/layout` | 50 ms | 100 RPS | 2% | bounding boxes + reading order |
| `POST /v1/predict/vlm` | 1500-3000 ms | 10 RPS | 5% | text, markdown tables, key-values, layout tree |

Send an `Idempotency-Key` header to make a call replay-safe. The model executes
**once** per key; concurrent duplicates are coalesced onto the first caller
(single-flight), and later duplicates are served from a bounded TTL cache.

On `429` the response carries:

| Header | Meaning |
|---|---|
| `Retry-After` | Integer seconds (RFC 9110). Too coarse for a 10 RPS limit. |
| `X-Retry-After-Ms` | Exact wait in milliseconds. Preferred by our client. |
| `X-RateLimit-Limit` / `-Burst` / `-Remaining` | Current bucket state |

Rejections are cheap by design - a `429` returns in ~8 ms versus ~1900 ms for a
served VLM request, so backpressure reaches the orchestrator promptly instead of
after a full inference.

### Admin

| Endpoint | Purpose |
|---|---|
| `POST /admin/chaos` | Force faults: `{endpoint, status, ratio, seconds, extra_latency_ms}` |
| `GET /admin/chaos` | Show active rules |
| `DELETE /admin/chaos?endpoint=vlm` | Clear rules |
| `GET /admin/call-counts` | Per-endpoint counters, idempotency evidence, bucket state |
| `POST /admin/reset` | Clear counters between test scenarios |

Chaos rules are time-boxed (max 600 s) so a crashed test cannot wedge the mock.

```bash
# 80% of VLM calls return 429 for the next 30 seconds
curl -X POST localhost:8001/admin/chaos -H 'Content-Type: application/json' \
  -d '{"endpoint":"vlm","status":429,"ratio":0.8,"seconds":30}'

# Degrade latency without failing (drives the adaptive limiter down)
curl -X POST localhost:8001/admin/chaos -H 'Content-Type: application/json' \
  -d '{"endpoint":"vlm","status":200,"ratio":1.0,"seconds":30,"extra_latency_ms":2000}'
```

`GET /admin/call-counts` is the evidence endpoint for the crash-recovery
requirement. `executions` counts calls where the model actually ran, `replays`
counts calls served from the idempotency cache, and `duplicate_executions` must
always be empty - including after a worker is `SIGKILL`ed mid-job.

## Configuration

All limits are env-tunable, so load tests can drive the system into states the
defaults never reach. Orchestrator vars use the `ORCH_` prefix, the mock uses
`MOCK_`. See [app/config.py](app/config.py) and
[mock_model/config.py](mock_model/config.py).

## Layout

```
app/          orchestrator: API, config
common/       shared observability: tracing, JSON logging, ASGI middleware
mock_model/   mock inference engines: rate limiting, chaos, idempotency, payloads
tests/        unit tests
```

### Adaptive rate control (AIMD)

Everything up to Step 9 enforces limits we were *told about*. The bucket sends
10 rps because the spec says 10 rps; if the endpoint can really only serve 4,
six requests per second are destroyed and the bucket never learns, because it
has no feedback loop.

AIMD is that loop - TCP congestion control applied to a dispatcher:

    congestion (429 / timeout / p95 breach)   rate = max(floor, rate x 0.7)
    a streak of N successes                   rate = min(advertised, rate + 1)

**Why the asymmetry.** Too high means congestion collapse - pages destroyed,
endpoint degraded - so the exit must be fast: multiplicative decrease reaches
safety in O(log) steps. Too low merely means slower than optimal, so probing
can be gentle. Both-multiplicative overshoots on every probe (1,2,4,8,16 blows
past a true limit of 10) giving a large-amplitude sawtooth that spends half its
life in overload; both-additive needs nine steps of -1 to escape a collapse
from 10 to 1, destroying pages the whole way down.

**One-sided, unlike TCP.** Textbook AIMD has no ceiling because bandwidth is
unknown. We know the published limit, so `max_rate` is the advertised rate:
this controller detects capacity *below* spec and recovers to spec rather than
hunting for capacity above it.

**The refractory period.** Without it a single congestion event floors the
rate. At the moment of a cut there are ~rate x latency requests already in
flight, admitted at the old rate and about to fail against the same overloaded
endpoint; compounding a decrease per failure gives 0.7^22 = 0.0004 of the
original rate. Measured: 1.0 rps (the floor) without the guard versus 7.0 rps
with it, from one event. TCP's equivalent rule is one reduction per RTT.

**5xx is deliberately NOT a congestion signal.** The mock's 5% baseline failure
rate has nothing to do with load, and counting it would trigger a decrease
about as often as a success streak earns an increase. Simulated over 120s of
traffic:

| baseline failures | rate with 5xx excluded | counted as congestion | throughput lost |
|---|---|---|---|
| 2% | 10.00 | 7.54 | 25% |
| 5% | 10.00 | 3.33 | **67%** |
| 10% | 10.00 | 2.78 | 72% |

Sustained 5xx is the breaker's job. This controller answers "how fast may I
go", not "is it alive".

#### Result: when the advertised limit is wrong

The mock's real capacity set to 3 rps while the orchestrator is still told 10,
45 pages, 3 replicas:

| | adaptive off | adaptive on |
|---|---|---|
| rate discovered | - | **2.68** (true capacity 3.0) |
| 429s | 74 | **29** |
| requests sent | 119 | **79** |
| wasted traffic | 76 (64%) | **35 (44%)** |
| circuit-breaker trips | 42 | **22** |
| full-fidelity DONE | 43 | **44** |
| wall time | 42s | 47s |

It found the real limit within 11% of truth, cut wasted traffic by a third and
breaker trips by half, and produced *more* full-fidelity pages from *fewer*
requests. The cost is 12% wall time, spent converging.

#### Two floors, because the two signals are not equally trustworthy

A latency breach is the signal a token bucket and a failure-ratio breaker are
both blind to, and the assignment names it explicitly. But it is *ambiguous*:
slowness caused by our load is relieved by backing off, while slowness from the
endpoint's own GC pause or slow dependency is not - and throttling then discards
throughput for nothing. Latency alone cannot tell the two apart.

Measured, honestly: the mock caps rate but not concurrency, so at 13s injected
latency it still serves its full 10 rps. Treating that as definitive and
flooring the rate cost **56% wall time** (98s vs 63s) to save 17% of wasted
requests - a bad trade.

So the floors differ. A 429 or timeout is the endpoint saying directly that we
are too fast: back off to `aimd_min_rate` (1 rps). A latency breach stops at
half the advertised rate - enough to relieve an endpoint we might be
overloading, without paying 90% of our throughput on evidence we cannot
attribute.

The better long-term signal is latency relative to the *minimum observed* - a
queueing-delay estimate, as TCP Vegas and Netflix's adaptive-concurrency
limiter use - since queueing delay is the part actually caused by our own load.
An absolute SLO is used here because it is derived from the published contract
and keeps meaning the same thing, whereas a learned baseline that drifts upward
during a long outage stops detecting it (the boiling-frog failure).

#### Why rate and not concurrency

The plan called for an adaptive *concurrency* limit. Rate was chosen instead:
we already own a correct, tested, cross-replica rate limiter whose Lua takes
`rate` as an argument, so AIMD becomes a controller on its setpoint rather than
a new distributed primitive. A distributed concurrency semaphore needs leases -
a worker SIGKILLed while holding a permit leaks it forever without expiry - for
a benefit largely already in hand, since bounded dispatch caps in-flight work
at `concurrency x replicas` regardless of latency. The latency-adaptation that
concurrency control would give for free is recovered explicitly by the p95
signal above.

### Admission control at the edge

Everything in Steps 7-10 pushes back *downstream* and protects the model
endpoints. None of it protects this service from ingestion, because all of it
acts after the work is already queued. A client can POST a 100-page job in
~5ms, offering ~20,000 pages/sec of arrival against the VLM's 10 pages/sec of
service. When lambda > mu is sustained, queue length grows without bound - that
is arithmetic, not a tuning problem.

Our own RSS is already bounded by bounded dispatch (Step 6). **Redis** is the
unbounded resource: a queued page costs a stream entry plus a state hash,
measured at ~300 bytes. Offering 40,000 pages to one slow worker:

| | queued | Redis growth | shed |
|---|---|---|---|
| watermark off (50,000) | 39,998 | **+11.3 MiB** | 0 |
| watermark on (5,000) | 6,000 | **+1.8 MiB** | 340 jobs |

Nothing stops the first row: the client can keep posting, and at ~300
bytes/page a million queued pages is ~300MB of Redis with no natural limit.

#### "Isn't a 503 a dropped job?"

The graded metric is "0% **unhandled** failed jobs". A 503 carrying
`Retry-After` is handled: synchronous, explicit, counted, machine-readable.
Compare the alternative - accept the job, return 202, then OOM-kill the worker
and lose every in-flight page. The client was *told* the work was accepted and
it silently never happens. That is a dropped job.

You cannot have both unbounded admission and bounded memory. The limit exists
either way; the only choice is whether the refusal is explicit and early or an
OOM kill nobody is told about.

Which makes the real guarantee stronger, not weaker. Not "we never say no",
but: **once admitted, a page is never dropped.** Admission is the boundary;
inside it, zero drop - which is exactly what Steps 8 and 9 earned.

Where else you could push back, and why not: *slowing the accept* makes
backpressure indistinguishable from a network fault and holds connections open;
*refusing at TCP level* gives the client no Retry-After and nothing to
diagnose; *spilling the queue to disk* moves the wall further out without
removing it. A typed refusal with a retry interval is the only option the
client can act on correctly.

#### Hysteresis

`XLEN` is the signal, because `ack()` does XACK **and** XDEL - so stream length
is exactly the pages admitted but not yet settled. It is *derived*, not
counted: a separate counter would have to be decremented on completion, and a
worker crashing in between would leak it upward forever, permanently tightening
admission. A crashed worker's tasks stay in the PEL, where they still correctly
count.

Two watermarks, not one. With a single threshold the system flaps on every page
completion - at the mark it refuses, one page drains so it admits, the next job
pushes it back over. Measured recovery through a 300/240 pair:

```
depth=263  POST -> 503      still shedding, though below the HIGH mark
depth=253  POST -> 503
depth=245  POST -> 503
depth=236  POST -> 202      crossed the LOW mark, recovered
```

A single-threshold design would have admitted at 299 and oscillated. `Retry-After`
is derived from the backlog and the AIMD controller's **discovered** rate, not
the advertised one: if the VLM has degraded to 2.7 rps the same backlog takes
4x as long to clear, and quoting a figure based on 10 rps just brings the client
back too early to be refused again. It is jittered for Step 8's reason - fifty
clients handed an identical `Retry-After: 60` return in the same instant and
recreate the overload.

#### The check and the enqueue must be serialised

`evaluate()` is atomic, but that is not sufficient: it reads `XLEN`, and `XLEN`
does not move until the caller enqueues - a separate round trip. Concurrent
requests that check before any of them enqueues all admit against the same
depth.

Unserialised, this is severe rather than marginal. At the assignment's own
benchmark concurrency (50 concurrent POSTs of 20 pages, watermark 400):

| trial | admitted | watermark | overshoot |
|---|---|---|---|
| 1 | 920 pages | 400 | 130% |
| 2 | **1,000 pages, 0 shed** | 400 | **150%** |
| 3 | 940 pages | 400 | 135% |

Trial 2 is the damning one - the watermark did nothing. A bound that can be
exceeded 2.5x is not a bound, and "memory footprint must remain bounded under
sustained load" is a graded requirement.

The fix is a process-local lock across check -> state init -> enqueue, so every
check observes all prior enqueues:

| | before | after |
|---|---|---|
| overshoot (3 trials) | 130-150% | **0%, 0%, 0%** |
| admitted vs 400 watermark | 920-1,000 | **exactly 400** |
| 50 POSTs wall time | 934ms | **351-487ms** |

It is *faster*, which is not a coincidence: a refusal skips the state-init and
enqueue an admission performs, so shedding earlier makes the batch finish
sooner. Overload protection should make overload cheap.

With N API replicas a residual overshoot returns, bounded by
`(N-1) x max_pages` - a provable constant, unlike the unserialised bound of
"however many requests a client chooses to send at once". Closing it across
replicas would mean folding the depth check, the state init and the enqueue
into one Lua script: correct, but it couples three subsystems, and is only
worth it if the API needs to scale horizontally.

### Chunked upload and lazy page extraction

The requirement is explicit - "PDF stream processing must handle page splitting
on-the-fly without buffering entire 100MB PDF byte arrays in memory" - against a
500 MB RSS budget under 50 concurrent ingestions. The naive version misses it by
an order of magnitude:

```python
content = await file.read()               # 100 MB resident, right here
reader = PdfReader(io.BytesIO(content))   # + the parsed object graph
for page in reader.pages:
    ...; pages.append(buf.getvalue())     # + a second full copy
```

O(file) x 2-3 resident; at 50 x 100 MB that is 5-15 GB. Three independent
decisions each remove one O(file) term: 1 MiB chunks straight to disk, page
count from the xref alone, and open/extract/close one page at a time.

Underneath all three: **the queue carries a reference, not bytes.** That is what
keeps Step 11's arithmetic true - admission bounds Redis by counting pages at
~300 bytes each, and page bytes on the queue would make a 100-page 100 MiB
document cost 100 MiB of Redis.

#### The pypdf trap

`PdfReader` must be handed an open file object, never a path. The path form
looks *more* idiomatic and is what every example shows, but pypdf does:

```python
if isinstance(stream, (str, Path)):
    with open(stream, "rb") as fh:
        stream = BytesIO(fh.read())      # the ENTIRE file, resident
```

| | resident growth, 100 MiB file |
|---|---|
| `PdfReader(str(path))` | **+100.4 MiB** |
| `PdfReader(open(path,'rb'))` | **+0.0 MiB** |
| + `extract_text()` on one page | +5.5 MiB |

The first implementation used the path form. Because extraction is per-page it
slurped 100 MiB a hundred times, and measured **worse than the naive version it
replaced** - 420 MiB peak against 301 MiB. After the fix, 19.5 MiB.

#### Text extraction is off by default

`extract_text()` is the most expensive thing in the pipeline on adversarial
input, on both axes. Per page, on a document whose pages carry ~1 MB of text:

| | resident | CPU |
|---|---|---|
| text ON | 18.1 MiB | **3038 ms** |
| text OFF | 1.6 MiB | 26 ms |

3 seconds per page is slower than the VLM it feeds, so enabling it moves the
bottleneck off the 10 rps endpoint and onto our own CPU - and 18.1 MiB x 48
concurrent extractions is 864 MiB, over budget on that term alone. Truncating
the *output* saves nothing, because the whole content stream must be walked to
produce any text at all, so the knob disables the work. Nothing in the pipeline
needs the text: the mock produces its own output and `/evaluate` compares
client-supplied trees.

#### Two ingestion endpoints, because the receive path dominates

FastAPI parses a whole multipart body *before* the endpoint runs, so with
`UploadFile` the request is already fully received - spooled to a temp file,
through the parser's buffers - before the handler sees it. Consuming
`request.stream()` takes chunks from the transport straight to disk.

Measured at 50 concurrent 20 MiB uploads (1,000 MiB offered):

| endpoint | api peak | total, all containers | per in-flight upload |
|---|---|---|---|
| `/jobs/upload` (multipart) | 268.2 MiB | 425.1 MiB | 5.36 MiB |
| `/jobs/stream` (raw body) | **125.7 MiB** | **298.5 MiB** | **2.51 MiB** |

Both pass the 500 MB budget; only the raw path has headroom, and it is what the
benchmark uses. Multipart is kept because it is what a browser or `curl -F`
sends.

A single 100 MiB / 100-page upload completed in 1.1s and grew the API by
7.7 MiB - O(chunk), not O(file). 1,000 pages across 50 jobs finished with
**1000 DONE, 0 FAILED**, and every source document deleted afterwards (uploads
are the one resource with no TTL, so completion triggers an explicit delete).

### SSE streaming and time-to-first-page

`GET /jobs/{job_id}/stream` tails a per-job Redis Stream (`stream:out:{job_id}`)
with `XREAD BLOCK` and emits Server-Sent Events. SSE rather than WebSockets
because the traffic is one-directional, so plain HTTP keeps status codes for
errors, `Last-Event-ID` for resume, proxies and `curl` - a WebSocket would add a
protocol upgrade, its own liveness scheme and its own reconnect story to buy a
direction never used.

#### Each page is announced twice, because 200ms < one VLM call

Time-to-first-page is graded at under 200ms. A VLM call takes 1.5-3s. A stream
emitting one event per *finished* page cannot satisfy both: the target is
smaller than a single call, so it is an arithmetic constraint, not a tuning
problem. So a page is announced when its **fast** stage commits and again when
its **heavy** stage does:

| event | when | payload |
|---|---|---|
| `page.partial` | layout committed, ~50ms | layout blocks, `complete: false` |
| `page.final` | terminal state | VLM output + confidence, or `degraded: true`, `complete: true` |
| `job.complete` | last page terminal | `total_pages`, `done` |

Same `page_index`, so a client upgrades in place. This is the same per-stage
checkpoint that makes crash recovery cheap (Step 4) - the commit that bounds
lost work is the commit that makes this event possible.

Measured, one client, 10 sequential 20-page jobs:

| | p50 | p95 |
|---|---|---|
| time to first **page** (`page.partial`) | 72.7 ms | **145.0 ms** |
| time to first **final** (what a one-phase stream would give) | - | 1927.6 ms |

**13x**, and the target is met with headroom. Both numbers come from the same
run, so the comparison is not against a reconstruction.

#### Time-to-first-page is a FAIRNESS problem, not a latency one

Under a single FIFO task stream the graded benchmark shape - 50 concurrent jobs,
1,000 pages - measured a p95 time-to-first-page of **14,511 ms**. Not tuning:
1,000 layout calls at the endpoint's 100 rps is a 10 second floor, and in FIFO
order the 50th job's first page sits at queue position ~980, so it is laid out
*last*, behind 979 pages belonging to clients already being served. Every job
waits for every earlier job's entire document before seeing anything.

Two changes fixed it, both reusing machinery that already existed:

1. **A priority lane.** Page 0 of each job goes to `stream:pages:lead`, read
   preferentially. One page per job is all the metric needs, which is what keeps
   the lane small enough to drain inside the layout endpoint's burst. Requeues
   and handoffs go to the main lane - admitting retries would let a saturated
   endpoint fill the priority lane with work that cannot run.
2. **A stage handoff.** A page releases its worker slot after layout instead of
   holding it for the next 1.5-3s on a 10 rps endpoint. Without this the two
   stages share a slot, so the *fast* stage inherits the *slow* stage's
   queueing: with 48 slots and 1,000 pages every slot parks on a VLM token and
   later pages are never even read off the queue. Safe because `LAYOUT_DONE` is
   already a durable checkpoint and already a legal resume point, so the
   redelivered page runs only the VLM stage - no new state, no new recovery path.

50 concurrent jobs / 1,000 pages, three trials of the final configuration:

| | before | after |
|---|---|---|
| p95 time-to-first-page | 14,511 ms | **853 - 1,246 ms** |
| p95 time to *all* partials | 14,768 ms | **10,654 - 11,717 ms** |
| wall clock | 108 s | 109 - 118 s |
| pages/sec | 9.26 | 8.5 - 9.2 |
| correctness | 41/41 events, 0 seq gaps | 41/41 events, 0 seq gaps |

A **15x** improvement in first-page latency at no throughput cost. The
all-partials figure now sits close to its 10s arithmetic floor (1,000 layouts at
100 rps), i.e. the layout stage finally runs at its own rate instead of the
VLM's. Zero degradations, zero saturations and zero requeues across the run, so
none of this was bought with fidelity.

**p95 under a 50-way simultaneous burst does not reach 200ms, and cannot on one
API replica.** Measured terms: 50 concurrent `POST /jobs` alone is 279ms at p95
on an idle system (3,331 pages/sec admitted), so the 50th client's job is not
even *accepted* until ~300ms. Stated plainly rather than reported against a
friendlier definition. Per-client TTFP - the figure a single client experiences
- is 145ms at p95. Multiple API replicas are the fix.

#### The bugs this step surfaced

**A blocking read and a socket timeout are in direct conflict.** The pool's 5s
`socket_timeout` fired before `XREAD BLOCK 15000` returned, so *every* stream
died at ~5s with `redis.exceptions.TimeoutError` - and because a streaming body
has already sent its 200, the client saw `incomplete chunked read`: a transport
error for a server-side fault. The worker's own blocking read escaped this only
because `worker_block_ms` (2s) happened to sit below 5s, two independently
chosen numbers in two different files with nothing expressing the relationship.
The SSE pool now *derives* its socket timeout from the block duration.

**A blocked reader holds its connection for the whole block.** 50 concurrent
subscribers against a 32-connection pool would starve *ingestion*, which matters
more than any subscriber. A second pool, sized from the subscriber cap, makes
that priority structural rather than a matter of who arrives first - and
`sse_max_subscribers` refuses the 65th with a counted 503 + `Retry-After`
instead of a pool timeout nobody is told about. Verified at peak 50 subscribers,
0 refused.

**`MAXLEN` silently breaks `Last-Event-ID`.** `XREAD` from a trimmed id does not
error - it returns whatever still exists, so a client disconnected while 200
events went past resumes cleanly, is permanently missing a run of pages, and is
never told. The endpoint compares the cursor against the retained window and
emits an explicit `stream.gap` naming `GET /jobs/{id}`, which is authoritative
because page state is never trimmed.

**`event` is a reserved kwarg in structlog.** `log.warning("msg", event=...)`
raises `TypeError: got multiple values for argument 'event'` - which happened
*inside* the handler whose entire job was to swallow publish failures, so the
error escaped and turned a page whose result was already committed into
`FAILED`. The rule earned: a handler that exists to guarantee nothing escapes
must contain nothing that can throw.

**`XREADGROUP` applies `COUNT` per stream.** Measured: one call over two lanes
at `COUNT 3` returns 6 entries, and every one lands in the consumer's PEL, so
the over-read cannot be discarded - it would break bounded dispatch's "resident
tasks <= worker_concurrency". Each lane gets its own budget, and the main
budget is computed from what the lead lane actually left (an even split gives
1 + 1 = 2 at `count=1`).

**`ack()` has no default stream.** `XACK` against a stream that never held the
entry returns 0 rather than raising, so the entry stays in the PEL and the
reaper later redelivers a finished page. Making the argument required turned a
one-word mistake into a type error; the tests themselves had made it, and only a
depth assertion off by one caught it.

#### The latecomer: a job arriving into a busy system

The benchmark above submits every job at t=0 into an empty system. That
structurally cannot test the more realistic case: the system is already
saturated with VLM-stage work and a new client shows up. Measured separately -
one job arriving into ~800 queued pages with ~40 in flight - it exposed two
defects the burst benchmark had hidden completely.

**A 20x latency cliff at exactly `pages == 1`.** The dispatch loop's blocking
read watched the *main* stream only, so a lead-lane `XADD` could not wake an
idle worker. Multi-page jobs hid it entirely, because their pages 1..n-1 land in
the main stream and do the waking - but a one-page job is *entirely* lead-lane,
so nothing woke the worker and its only page waited out `worker_block_ms`.
Measured on an **idle** system, i.e. the best case:

| job size | before | after |
|---|---|---|
| 1 page | 1,499 ms (max 2,840) | **76.5 ms** |
| 2 pages | 74.1 ms | 70.0 ms |
| 20 pages | 83.8 ms | 87.6 ms |

The wait now watches both lanes in one call, with a halved per-stream `COUNT` to
keep the read bound exact. Latency is flat across page counts.

**A job arriving while every slot is occupied waits for a slot, not for a
model.** The lead probe only runs when `capacity > 0`, so while a worker is at
capacity the priority lane is not polled at all. Fixed by capping what the main
lane may hold (`worker_lead_reserve`), which keeps a slot free for a first page
at all times:

| latecomer into a saturated system | `reserve=0` | `reserve=4` |
|---|---|---|
| 5-page job, p50 | 245.1 ms | **108.3 ms** |
| 5-page job, max | 626.0 ms | **196.3 ms** |
| 50-job benchmark wall clock | 109 - 118 s | 109.8 / 112.7 s |

2.3x at p50, 3.2x at the tail, the tail under target, and no throughput cost.

#### Why the reservation was first measured as worthless

An earlier A/B concluded the reservation did nothing, and that was an artefact
of the experiment rather than a property of the feature. It submitted all 50
jobs at t=0 into an empty system - so every slot was free when the lead lane
filled, the lane drained instantly, and a reservation had nothing to do. The
scenario it exists for was never run. Two lessons, the second sharper than the
first: a benchmark that starts from an idle system cannot measure steady state,
and *"no measurable effect" is a claim about the measurement* until the scenario
is shown capable of producing the effect.

The same A/B also mismeasured the cost, because the implementation had a bug.
When free capacity fell to equal the reservation, the main budget hit zero and
the worker blocked on the **empty** lead lane for a full `worker_block_ms` while
hundreds of main-lane pages waited - in-flight pages fell from 48 to 2 and
throughput dropped ~40%. That regression is what made a reserved lane look like
a throughput tax, and it is why the knob was wrongly defaulted off. The lane was
never the cost; waiting on the wrong thing was. The worker now returns
immediately in that branch and waits on its in-flight tasks instead - the event
that actually matters is a slot freeing, and that needs no timer and no poll.
It is provably not a spin: the branch is reachable only when
`capacity <= lead_reserve < worker_concurrency`, so the in-flight set is
necessarily non-empty.

#### Three identifiers, three jobs

| | purpose |
|---|---|
| `stream_id` (SSE `id:`) | opaque **resume cursor**, fed straight back to `XREAD` |
| `seq` | dense per-job counter: **comparable** (detect a hole) and **countable** (prove completeness) |
| `page_index` | the **reassembly** key |

An entry id cannot do `seq`'s job - `1738-0` to `1740-0` says nothing about
whether an entry existed between them - and `seq` cannot do the cursor's,
because `XREAD` does not accept it. `seq` is assigned *inside* the `XADD` script:
as two commands, a concurrent publisher can land its `XADD` between another's
`INCR` and its own, so the stream would hold seq 8 before seq 7 and a client
ordering by `seq` could never distinguish a reordering from a gap. Demonstrated
in `tests/test_result_stream.py`, not asserted.

Pages are **not** reordered on the way out: buffering until page 2 arrives would
idle the client on the slowest page in the job, which is the head-of-line
blocking per-page streaming exists to avoid. `MAXLEN ~ 256` caps a job's stream
(exactly `2 x pages + 1` events, so a client connecting at the very end can
still replay a whole 100-page job), and the task stream deliberately has none -
its oldest entry is unprocessed work, so trimming it would eat jobs.


### Crash recovery: leases and the reaper

The spec asks for a SIGKILLed worker to "resume execution from the exact
uncompleted page without re-running already completed pages or duplicating
downstream model calls." That is four mechanisms, not one:

| Layer | Covers |
|---|---|
| Per-stage commits | a crash costs **one stage**, not the page. `LAYOUT_DONE` is durable and is a legal resume point. |
| `read_own_pending` | a worker **restarting** under the same name. The consumer name is the container hostname, so the new process finds its own PEL and resumes with no idle timeout. |
| **The reaper** | a worker **replaced** rather than restarted. A new container has a new hostname, so the dead consumer's PEL belongs to a name that will never return. |
| `Idempotency-Key` | a duplicate *delivery* never becomes a duplicate model *call*, which is what makes the three above safe to be aggressive. |

Row three is the hole Step 14 closes, and it is not a rare one -
`--force-recreate`, a rescheduled pod and a scale-down from 3 replicas to 2 all
produce it. Those entries are skipped by `XREADGROUP >` because they were
already delivered, and no live consumer will ever ack them: the pages are
**non-terminal and unreachable**, the same zero-drop violation as the stranding
bugs below, reached by a third route.

#### Why idle time needs a lease to mean anything

Idle time in a Pending Entries List measures time since **delivery**, not time
since progress. On its own it is a poor liveness signal, because a perfectly
healthy task can be held for a long time: three attempts of (up to
`rate_limit_max_wait_s` waiting for a token, plus up to `http_timeout_s` in the
call) plus backoff is ~130s. Thresholding above that makes recovery slower than
the page deadline it is meant to beat.

So each worker renews the leases on the entries it holds, every
`lease_renew_interval_s`, with one `XCLAIM ... JUSTID` per lane carrying every
in-flight id. Idle time then means *"this worker has stopped checking in"* - an
actual liveness signal - and the threshold can be 30s (six missed renewals)
instead of 130s+.

`JUSTID` is load-bearing rather than an optimisation: plain `XCLAIM` increments
`delivery_count`, so a page legitimately held for 90s would look as though it
had been delivered 18 times, poisoning the one counter that distinguishes a
genuinely redelivered page from a slow one.

Renewal runs in its own task, not in the dispatch loop. The dispatch loop parks
in `asyncio.wait` whenever the worker is at capacity - exactly when it holds the
most entries and has the most to lose - so a renewal folded into it would stop
firing under the load that makes it matter.

#### Two bugs the obvious implementation introduces

**`XAUTOCLAIM` cannot exclude yourself.** It selects purely on idle time, and a
worker's own task parked on a VLM token is indistinguishable by idle time from a
dead worker's. The reaper would roll back a page its own process is actively
working on. Owner filtering is not expressible in `XAUTOCLAIM`, so the
implementation is `XPENDING ... IDLE` to list candidates, filter out self, then
`XCLAIM` - one extra round trip, paid on an idle path once per interval.

**Reclaiming in place makes two workers share one entry id.** `XACK` is
*group*-scoped, not consumer-scoped. If A is alive but slow and still holds
entry `X`, and B reclaims `X` in place, then A finishes, sees the page is no
longer its own, and acks - deleting the only queue entry for a page B is
mid-flight on. If B then dies, the page is stranded, and the recovery mechanism
would have manufactured the exact failure it exists to prevent.

So the reaper **requeues rather than dispatches**: it rolls the page back to its
last committed checkpoint, appends a *fresh* entry to the main lane, and acks the
original. Nobody ever shares an entry id, A's later ack is a harmless no-op on an
id that no longer exists, and the fresh entry travels the ordinary dispatch path -
so this needed **no pipeline changes at all**. A first delivery of a page sitting
at `PENDING` or `LAYOUT_DONE` is already the normal resume path from Step 4.

#### Why concurrent reapers need no leader election

Every replica reaps. That looks like a read-modify-write race - two reapers both
see an entry as idle, both claim it, the page is requeued twice - but
`XCLAIM`'s `min-idle-time` argument is **mandatory and conditional**: an entry
whose idle clock has been reset below it is not claimed and does not appear in
the reply. The first claimant resets idle to 0, so every other claimant's
`XCLAIM` atomically returns nothing. Passing the scan's own threshold turns the
claim into a compare-and-set on idle time.

Even if a duplicate did slip through it would be wasteful rather than
corrupting: the state CAS admits exactly one claimant, and the deterministic
`Idempotency-Key` coalesces a second call for the same `(job, page, stage)` onto
the first. The conditional claim is what makes it not happen; those two are what
make it survivable if it ever did.

#### A reclaim spends an attempt; a handoff does not

Three requeue reasons, charged differently on purpose:

- **stage handoff** - the page's own success. Charging it would mean every page spent an attempt on its happy path.
- **saturation** - a property of the endpoint. Charging it would degrade pages for arriving during a busy period.
- **reclaim** - weak evidence about *that page*. The innocent cause is an unrelated SIGKILL; the guilty one is that the page is what killed the worker. An uncounted reclaim would let a poison page cycle through every replica indefinitely, so it **is** charged, and such a page eventually degrades to `FALLBACK_DONE` or `FAILED` and is reported.

#### The proof

`scripts/prove_recovery.sh` runs **both** scenarios, because a bare `docker kill`
followed by `docker start` reuses the container and therefore only exercises
`read_own_pending` - a demo that omitted the replacement case would report
success while Step 14 went entirely untested. 3 jobs x 20 pages, one worker of
three SIGKILLed mid-flight, 12 pages orphaned:

| | restart (own-pending) | replacement (reaper) |
|---|---|---|
| pages completed | **60/60** in 11.2s | **60/60** in 17.3s |
| page states | all `DONE` | all `DONE` |
| queue depth after | 0 | 0 |
| layout executions (60 pages) | **60** | **60** |
| vlm executions (60 pages) | **60** | **60** |
| duplicate executions | **0** | **0** |
| recovered by own-pending | 1 | 0 |
| recovered by the reaper | 0 | **12 pages** |

The attribution row is the point. Both scenarios print "60/60 pages", so without
counting *which* mechanism fired, a reaper that silently did nothing while
own-pending quietly handled everything would look identical to a working one.

`executions == 60` for 60 pages is the strong form of the no-duplicate-calls
claim, and it does not rely on any counter the mock keeps about itself: 12 pages
were delivered at least twice, so if stage-level idempotency were broken,
executions would exceed the page count by exactly the number of pages whose
committed stage was recomputed. Retries do not inflate it - the mock counts an
execution only after a call *succeeds*, so a 500-then-success is one execution
under the same key.

#### Known window

A duplicate call is still possible in one window: a worker that dies *after* the
model returns but *before* the state commit lands. The reclaiming worker finds
the page at its previous checkpoint and calls again. This is bounded rather than
eliminated - the mock's `Idempotency-Key` cache absorbs it within
`idempotency_ttl_s`, so the second call is a replay rather than a second
execution, and it shows in the counters as `replays` rather than `executions`.
Making it truly exactly-once would need the model call and the state commit in
one distributed transaction, which no HTTP endpoint offers; at-least-once plus
idempotency is the standard trade, and the one the spec's own wording ("without
duplicating downstream model calls") describes.


### TTL bounds inactivity, not a job's lifetime

`init_job` puts a TTL on the job hash and every page hash so finished jobs
reclaim themselves — without it Redis grows monotonically with every job ever
submitted. Set once at creation and never renewed, though, that same TTL is also
a hard cap on how long a job may *take*, and a job that exceeds it does not fail
cleanly. It corrupts, because:

```
HINCRBY on a missing key CREATES it.
```

So when the job hash expires mid-flight, the next terminal transition recreates
it holding only `done_count`. `total_pages` is then unreadable, the script
returns `-1`, and `completed_job` can never be true again. Every remaining page
still processes perfectly and the job **never reports complete** — a silent hang
at 99%, with an SSE client waiting out `sse_max_duration_s` for an event that can
no longer exist.

This needs no crash to reach. A long job under sustained backlog, a wide chaos
window, or simply `result_ttl_s` set below what a large job takes will do it.
`page_deadline_s` does not help: it bounds one page's wait, not the job's wall
clock.

#### The fix, and the part I got wrong first

`_TRANSITION_LUA` now `PEXPIRE`s both the page hash and the job hash on every
**successful** transition, so `result_ttl_s` means "expires after this much
inactivity" — which is what `init_job`'s docstring always claimed. Only on
success, deliberately: at-least-once delivery redelivers finished pages
routinely, and renewing on a *rejected* transition would keep a completed job's
records alive forever, turning the cleanup policy back into the leak it existed
to prevent.

Renewing the transitioning page and the job hash turned out **not** to be
enough, and the regression test caught it — failing at page 2 of 6 with
`observed='MISSING'`. A page the job has not **started** yet is never
transitioned, so nothing ever touches it and it keeps the original fixed clock.
When it expires its queue entry is still there, so a worker claims it, finds no
hash, and `process_page` returns `FAILED` for "page state missing". The worker
acks it — and with no hash there is nothing to make terminal and no `HINCRBY` to
count, so `done_count` can never reach `total_pages`. Same silent hang, reached
from the other end.

Sweeping every page on every transition would fix that and cost far too much:
four transitions per page over a 100-page job is 400 scripts, so **40,000
`PEXPIRE`s per job** of pure bookkeeping churn. So the sweep is conditional, and
gated on its own marker:

- the script keeps `pages_renewed_at` on the job hash
- when `now - pages_renewed_at > ttl_ms / 2`, it returns `sweep_pages` and stamps the marker
- the caller then renews all page TTLs in one pipelined round trip

That bounds the sweep to roughly twice per TTL period per job however many
transitions occur. The marker is read *and* written inside the same atomic
script, so N workers transitioning different pages concurrently cannot all be
told to sweep — exactly one sees the stale marker and claims the work.

My first gate was `PTTL` on the job hash, and it never fired once. The job hash
is renewed by every transition of every page, so its remaining TTL is
permanently near-full and says nothing about how long an *unstarted* page has
been sitting. The two clocks are unrelated, and the test still failed at page 2
with that check in place.

The sweep is driven from Python rather than looped inside Lua on purpose: Lua
would have to **build** the page keys from a job id, and keys a script was not
given are exactly what Redis Cluster forbids — it cannot route a script whose
key set is not declared up front. Keeping them explicit means a cluster
deployment only needs a hash tag on the shared `job:{id}` prefix to colocate
them.

#### Proof

`scripts/prove_ttl_renewal.py`, run against a stack with a deliberately tiny
TTL. A 100-page job on a **5 second** `result_ttl_s`:

| | with renewal | renewal disabled |
|---|---|---|
| elapsed | 15.6s (**3.1x the TTL**) | killed after 600s |
| `page.final` events | **100/100** | — |
| `job.complete` | **yes** | **never arrived** |
| Redis keys left | job complete, reclaimed on schedule | `DBSIZE 2` — every job and page hash expired |
| pages | all `DONE` | all 100 returned `MISSING` |

The disabled-renewal column is the reason the script exists: the stream never
closed, so the run hung until it was killed, and every one of the 100 pages was
dropped with no terminal state and no record it had existed.

The script also asserts its own scenario before believing its result. The first
version used a 20s TTL against a job that finished in 10.3s and printed `PASS` —
a worthless pass, since the job never reached its expiry and a completely
unrenewed build would have passed identically. It now fails with `VACUOUS` if
`elapsed <= ttl_s`.

#### The residual, named

A page whose hash is missing is still handled as `FAILED` with an ack, which
cannot increment `done_count` — so if a hash is ever lost to something other
than an unrenewed TTL (a `maxmemory` eviction, an operator `FLUSHDB`), the job
hangs rather than reporting a shortfall. `transition` now logs
`job_hash_vanished_mid_flight` when a terminal transition succeeds but
`total_pages` reads back `-1`, which makes the narrower version of that case
loud instead of silent. Making it fully self-healing would mean reconstructing
`total_pages` from a source outside the expiring keyspace; the renewal above is
what stops it arising in the first place.

### CER/WER: two rows, and three ways to publish a wrong number

`app/eval/text.py`. Character and word error rate are the same normalised edit
distance over different tokenisations, so the module is one dynamic program and
some thin wrappers. The algorithm is textbook; everything that took work is
around it.

**The memory optimisation.** `d[i][j]` reads row `i-1` and itself, never row
`i-2`, so the full matrix is memory written once and never read again - and it
is the only term that grows quadratically. Keeping two rows makes space
O(min(m,n)) with time unchanged at O(mn). Measured with `tracemalloc` at a
steady 36-39 bytes per cell:

| n | full matrix | two rows | ratio |
|---|---|---|---|
| 1,600 | 100.03 MB | 0.131 MB | 764x |
| 5,000 (projected) | ~900 MB | ~0.36 MB | ~2500x |

A dense OCR page is a few thousand characters, so the matrix version needs more
than this service's entire 500 MB RSS allowance for one `/evaluate` call.
Two measurement lessons, both of which had to be measured rather than reasoned:
a hand estimate of "~200 MB" for n=5000 turned out to be the cost of the bare
list slots with no int objects at all, and any doubling test below n=257 is
measuring CPython's small-int cache rather than the algorithm - a naive
250-vs-500 comparison appears to grow 8.9x instead of 2x.

**A "character" is not a Python code point.** `len("क्षि") == 4`. Indic scripts
build one written character from a base consonant, a virama and vowel signs,
each its own code point, so scoring CER over `str` penalises one misread
conjunct up to 4x while a misread Latin letter costs 1x - and inflates the
denominator at the same time. `graphemes()` segments first (an approximation of
UAX #29 covering combining marks, Indic conjuncts and ZWJ/ZWNJ; the gaps are
named in the docstring). Measured on हिन्दी - 6 code points, 2 written
characters - with its conjunct dropped:

| unit | denominator | CER |
|---|---|---|
| grapheme (default) | 2 | **0.500** |
| code point | 6 | 0.333 |

Half the word is wrong, and only one of those numbers says so. `unit="codepoint"`
is retained because published baselines and tooling are code-point based and a
number you cannot reproduce is not a comparison. ASCII is unaffected either way,
which is asserted.

**Rates do not average.** Corpus CER is `sum(errors) / sum(lengths)`, not the
mean of per-page rates. On 50 clean 11-character pages plus one 2-character page
scored 3.5:

| | value |
|---|---|
| micro-average (what "CER" means) | **0.0127** |
| mean of per-page rates | 0.0686 (5.4x worse) |

So the scoring functions return a `TextScore` carrying numerator and denominator
separately and `aggregate()` is the only supported way to combine them; a bare
float cannot be combined correctly. This also keeps an empty-reference page
(`rate == inf`, deliberately, since there is nothing to be wrong about) from
turning a whole corpus report into `inf`. Relatedly, CER is **not** bounded by
1.0 - 3 reference characters against 150 hallucinated ones is CER 50, and
clamping it would erase the most useful signal the metric carries.

**Two things deliberately not done.** Myers' bit-parallel algorithm computes the
identical distance 146x faster (7.5 ms vs 1097 ms on a 2,000-character pair) and
is not used: a 146x speedup built from bitwise carry-propagation tricks is not
worth defending under questioning when the two-row DP is already fast enough
for the spec's own numbers - a dense page pair costs ~1.1 s of CPU either way,
which just means an `/evaluate` handler belongs in a worker thread rather than
on the event loop. Affix stripping was measured as a cheaper mitigation and
rejected at 1.2x on a realistic page and 1.0x on a noisy one.

Also not built: a substitution/insertion/deletion breakdown. The spec asks for
"standard Levenshtein edit distance" - the rate, not its decomposition - and an
earlier draft of this module carried an `edit_counts` DP variant (four ints per
cell instead of one, ~2.5-3x the memory and ~2x the time of the plain distance)
that nothing in the codebase consumed. It was cut: not in the spec, not on the
scoring table, and not free to carry for zero credit.

**Verified:** 30 tests in `tests/test_text_metrics.py`, including 4,000 random
pairs cross-checked against a naive full-matrix oracle over a 3-letter alphabet
(small on purpose - frequent coincidental matches are what exercise the diagonal
branch and the tie-breaking).

---

### Bounding-box IoU: one sign bug, and an assignment problem

`app/eval/iou.py`. The overlap arithmetic is ten lines. What needed thought was
a sign bug that produces confident wrong answers, the fact that a single IoU
number for a page requires solving a matching problem first, and two different
denominators that both get called "mean IoU".

**The clamp is load-bearing.** IoU is separable per axis, and `max(0, ...)` on
each overlap is not defensive tidiness. For two boxes that miss **diagonally**,
both overlaps come out negative - and negative x negative is positive. Two
10x10 boxes offset by 20 produce a phantom intersection of `-10 x -10 = 100`
against a union of `100 + 100 - 100 = 100`:

| | reported IoU |
|---|---|
| correct (clamped) | 0.0 |
| unclamped | **1.0** - a perfect overlap, for boxes that do not touch |

It only misbehaves when the boxes miss on *both* axes; a one-axis miss gives a
negative product that any smoke test catches. That asymmetry is why the bug
survives casual testing, so `tests/test_iou.py` pins it against a deliberately
unclamped implementation rather than trusting the clamp to stay there. The
other classic - union as `area_a + area_b`, double-counting the shared region -
is pinned the same way (0.25 instead of 0.333 on a half-overlap).

**Matching is a separate problem from scoring.** With N predictions and M
ground truths there is no "the" IoU, only N x M pairwise values; reporting one
number means first deciding which prediction corresponds to which ground
truth. `match_boxes` is greedy on descending IoU, O(NM log NM). The optimal
alternative (Hungarian, O(n^3)) maximises the global sum and greedy is only a
1/2-approximation in general - so the question is what it loses on inputs that
are *real rectangles*, since an IoU matrix from geometry is constrained (1-IoU
is a proper metric). Measured against a brute-force bitmask-DP optimal matcher:

| input | IoU-sum suboptimal | worst relative loss | **TP count** suboptimal | worst deficit |
|---|---|---|---|---|
| document layouts (a column of blocks) | 33 / 4,000 | 42.0% | **0 / 4,000** | - |
| pure random boxes | 116 / 4,000 | 31.2% | **0 / 4,000** | - |
| tight clustered boxes | 1,957 / 6,000 | 42.5% | 273 / 6,000 | 2 boxes |

The column that matters is the true-positive count, because precision and
recall depend on nothing else - and on document-shaped input greedy never lost
one. Greedy is kept on that evidence, not on the usual hand-wave. Hungarian was
also rejected for a second reason: maximising the global sum can prefer a
sub-threshold pair to an above-threshold one, which would break the
threshold-ordering property below.

Both limitations are pinned as literal coordinates found by search, so they
live in the suite rather than in an interviewer's notes:

    predictions down, ground truths across
               G1      G2      G3
        P1   0.6023  0.3026  0.4266
        P2   0.6778  0.5221  0.2713

Greedy takes the largest single pair, `(P2,G1)` at 0.6778, consuming the only
ground truth P1 overlaps above threshold; P1 settles for 0.4266 and is not a
true positive. **1 TP, recall 1/3.** The optimal matching declines that pair:
`(P1,G1)` + `(P2,G2)`, both above threshold, **2 TPs.** A second pinned case
loses **48.6% of the obtainable IoU sum** (0.2784 vs 0.5411).

**Thresholding after matching is safe, and that is not obvious.**
`match_boxes` pairs everything it can and `score_boxes` applies IoU>=0.5
afterwards. Refusing sub-threshold matches up front sounds safer but yields the
identical true-positive set, because greedy visits pairs in descending IoU, so
every above-threshold pair is considered before any below-threshold one - a
weak match can only ever pair boxes already left over. Doing it in this order
gets both numbers from one matching: a box overlapping at 0.49 contributes 0.49
to spatial quality while still counting as a miss for detection. There is a
test asserting the two orders agree, including on the counterexample above.

A weak match is counted as **both** a false positive and a false negative - the
box was emitted (unearned output) and the ground truth was not found (a miss).
Counting it once would make precision and recall disagree about how many boxes
exist.

**Two denominators, both called "mean IoU."** The same trap as
`mean(per_page_cer)` in Step 15. One perfect box against ten ground truths:

| | value |
|---|---|
| `mean_matched_iou` (over matches only) | **1.0** |
| `mean_iou` (over ground truths, misses = 0) | **0.1** |

Nine misses, and one of those numbers is a flawless score. Both are exposed,
`mean_iou` is the honest one, and `aggregate()` micro-averages across pages by
summing counts rather than averaging rates - one dense page of ten good boxes
plus one sparse page holding a single miss gives 0.909 micro versus 0.5 macro.
Mixed thresholds are refused: true positives at 0.5 and at 0.75 answer
different questions and their sum answers neither.

**Coordinate convention:** `(x, y, w, h)` as specified, `area = w*h`, no "+1".
PASCAL VOC historically used `w = x2 - x1 + 1`; COCO does not. A negative
extent is **rejected, not clamped** - it is structurally impossible for a
rectangle and almost always means corner coordinates were passed as
`(x, y, w, h)`, and clamping would report that caller bug as "the model found
nothing". `Box.from_xyxy` exists because that same confusion is *undetectable*
when the corners are positive: there is a test showing `(100,100,150,160)`
misread as xywh yields a valid box 8x too large, with no error anywhere.

**Verified:** 37 tests in `tests/test_iou.py`, including the brute-force
optimal matcher as an oracle (exponential, hence small inputs only - which is
itself the reason the shipped code is greedy).

---

### Tree Edit Distance: Zhang-Shasha, and where its worst case really is

`app/eval/ted.py`. Structural layout drift in node inserts, deletes and
relabels. This is the graded metric with a latency target - **under 100 ms for a
50-node document tree** - so the implementation's shape is load-bearing.

**Why a tree needs its own algorithm.** Deleting a character removes it;
deleting a *node* promotes its children to its parent, so dropping a `table`
leaves its `row`s attached to `page`. Comparing serialisations would score that
promotion as free. And not every correspondence is legal: a valid mapping is
one-to-one and preserves both sibling order (reading order is what we are
measuring) and ancestry - which is what stops "this cell escaped its table and
became a title" being priced as a cheap relabel.

**The idea that makes it tractable.** Number nodes in postorder and let `l(i)`
be the leftmost leaf below `i`. Then **subtree(i) is exactly the contiguous
range `[l(i), i]`** - every subtree becomes an interval of integers, so the DP
is indexed by two ints rather than by sets of nodes. The recurrence:

    fd[i][j] = min( fd[i-1][j]         + 1,            delete i
                    fd[i][j-1]         + 1,            insert j
                    fd[i-1][j-1] + relabel(i,j)        if both are trees
                    fd[l(i)-1][l(j)-1] + treedist[i][j]  otherwise )

The fourth branch is where ancestry is enforced: when the forests are not
single trees, subtree(i) matches subtree(j) as an **indivisible unit** via the
already-computed `treedist[i][j]`. An alignment crossing a subtree boundary is
not rejected - it is inexpressible.

**Keyroots.** `LR-keyroots = { k : no k' > k shares l(k) }` - the root plus
every node with a left sibling. A non-keyroot is a leftmost child, so it shares
`l()` with its parent and its `treedist` row is filled during the parent's pass
for free. `#keyroots <= #leaves`; running the forest DP over all n1 x n2 pairs
instead would repeat that work once per ancestor.

**Complexity, and a mistake worth recording.** The textbook bound is
`O(|T1||T2| min(d1,l1) min(d2,l2))`, but it hides the driver and is loose on
real shapes. The exact cost is `Theta(W(T1) . W(T2))` where
`W(T) = sum over keyroots of |subtree(k)|`:

| shape | W/n | total |
|---|---|---|
| page -> blocks -> table -> row -> cell (ours) | ~2.0 | **O(n^2)** |
| broom: sqrt(n) arms of length sqrt(n) | ~1.9 | **O(n^2)** |
| caterpillar | ~n/4 | **O(n^4)** |

It is tempting to argue O(n^4) is unreachable because depth and leaves cannot
both be large - a path has depth n and *one* leaf, a star has n leaves and depth
one - capping `min(d,l)` at ~sqrt(n) and the whole thing at O(n^3). **That
argument is wrong**, and it was in this project's own plan notes until it got
measured. A broom does not disprove it either: the broom is quadratic, because
W/n stays at ~1.9 and the `n.min(d,l)` bound is simply loose there.

The shape that breaks it is a **caterpillar** - a deep right-leaning spine where
every spine node also sheds a leaf, making depth *and* leaves both `Theta(n)` at
once. Every spine node is then a keyroot and `W = Theta(n^2)`. Measured exponent
on caterpillar-vs-caterpillar: **3.52, 3.75, 3.98** over n = 13..81. O(n^4) is
real.

**The graded metric, measured in the container** (Python 3.12, the deployment
target):

| input | n | depth | latency | headroom vs 100 ms |
|---|---|---|---|---|
| document tree | 46 | 4 | **4.97 ms** | **21.2x** |
| document tree | 61 | 4 | 9.30 ms | 13.7x |
| caterpillar | 33 | 17 | 19.21 ms | 5.2x |
| caterpillar | 49 | 25 | **89.81 ms** | **1.1x** |

Measured exponent on page trees: 1.98, 2.39 over n = 121..481 - quadratic, as
predicted. The uncomfortable row is the last one: **latency depends on tree
SHAPE, not node count**, and a 49-node adversarial tree very nearly misses the
graded budget. `Tree.keyroot_weight` is exposed as the hook for that, because it
is O(keyroots) to compute and separates the two shapes *before* any DP runs -
W=132 for the 46-node page tree versus W=625 for the 49-node caterpillar. Since
work is the product `W1.W2`, a self-comparison scales as `W^2`, and
`625^2/132^2 = 22x` is exactly the ~18x observed in wall clock. Nothing calls it
yet; the defence against a hostile tree is a node cap at the request boundary,
which arrives with the endpoint.

**Memory.** O(|T1|.|T2|), two tables. `treedist` is irreducible - the fourth
branch reads arbitrary entries, so it cannot be rolled into two rows the way
`app/eval/text.py` does. `forestdist` is allocated **once** at full size and
reused by every keyroot pair instead of being reallocated per pair
(`O(keyroots^2)` matrix allocations). Reuse is safe because each pass writes its
own base row and column before reading, and the loop reads only cells written in
that same pass - there is a test that runs a large pair, then a small one, and
checks the small answer is untouched.

**No recursion.** `parse_tree` walks with an explicit stack. CPython's limit is
1000, so a recursive postorder would `RecursionError` on a deep tree - a crash,
in a service whose Module D requirement is bounded behaviour. Tested at 5,000
deep. (The remaining risk is not ours: `json.loads` uses a recursive C scanner
and raises before this module is reached; that belongs to edge validation.)

**No move operation**, which shapes how to read the number. `page(title,
figure)` vs `page(figure, title)` costs **2, not 1**: mapping each child to
itself is free but inverts their order and is illegal, leaving two relabels or a
delete plus an insert. So a merely *reordered* document scores as though it were
rewritten. Defensible - reading order genuinely is part of the structure - but
the metric cannot distinguish "moved" from "replaced". RTED/APTED variants have
a move; nothing here asks for one.

**Labels are the node `type`** - not text, not bbox. Text would re-measure CER;
bbox would re-measure IoU. Three metrics should answer three questions rather
than summing three views of one error; `label_key` is a parameter for callers
who disagree.

**Verified:** 32 tests in `tests/test_ted.py`, against **three independent
oracles**: a path-shaped tree reduces TED to `levenshtein` from Step 15 (itself
checked against a full-matrix oracle), a star reduces to `levenshtein` over its
children plus the root relabel, and a brute-force enumeration of **every legal
mapping** - one-to-one, order- and ancestry-preserving - straight from the
definition. Oracles 1-2 cover long inputs in degenerate shapes; oracle 3 covers
arbitrary shapes at small size. One test expectation was wrong on first write
(the sibling-swap cost) and brute force is what caught it.

---

### POST /evaluate: the graded tree-diff metric, made reachable

`app/api/evaluate.py`. The three algorithms were correct and fast before this
existed and it did not matter, because **a fast function nothing can call is
worth nothing**: the graded metric is measured by "Automated Test Execution",
and `curl -X POST /evaluate` returned **404**. That was the gate on 15% of the
score, not the algorithm's speed.

**One tree in, three metrics out.** The spec asks for an endpoint that "takes an
extracted document output tree and compares it against a ground-truth JSON
schema", and one tree suffices for all three metrics because the nodes already
carry everything - `type` is the label TED compares, `bbox` is the box IoU
matches, `text` is what CER/WER scores. Accepting three separate payloads would
let them disagree about the same document, which is a bug the caller should not
be able to express.

The metric independence is observable from outside, and asserted: a text error
plus a shifted box, with structure untouched, moves CER and IoU while TED stays
at **0**. That is the "labels are `type` only" decision from Step 17 paying off -
one OCR slip charged once, not three times.

**Everything runs in `asyncio.to_thread`.** All three metrics are synchronous
CPU-bound Python, and this process also serves every SSE stream and the graded
TTFP metric. Inline, a 22-second comparison would stall every in-flight stream,
so an awkward tree posted by one client would fail an unrelated graded metric for
everyone. One hop rather than three, because three would pay the handoff cost
without buying isolation.

What the thread does *not* buy is stated too: the GIL means a CPU-bound thread
still contends with the loop, so throughput degrades even though nothing blocks.
Measured: a 47-node tree costs ~5 ms alone and up to **58 ms** with four
competing CPU threads. The test asserts *ordering* rather than a latency
threshold - a trivial request issued after a heavy one must finish first - which
is the actual property and does not go flaky on slow CI.

**The cost cap is a latency guarantee, and getting there took a correction.**
Zhang-Shasha is O(n^4) in the worst case, reachable with a few kilobytes of
boring JSON, and nothing inside the DP can defend against it - by the time it is
running the cost is committed. But the cost is *predictable*:
`Tree.keyroot_weight` is O(keyroots) and the pass is `Theta(W1 x W2)`, so both
trees are parsed, weights multiplied, and an unservable request refused with
**413 before a single DP cell is touched**.

The first cap was a flat 3.5M work units (~1 s), chosen to stop CPU bombs. It
did - and it **admitted a 49-node caterpillar that then took 113 ms**, over the
graded budget, on a 49-node payload. So the cap is now *derived* from
`eval_ted_budget_ms` (defaulted to the graded 100 ms) times a measured
`eval_ted_work_per_ms` of 3,500, making admission imply the budget:

| tree | nodes | depth | W² | result |
|---|---|---|---|---|
| document | 46 | 4 | 17,424 | 200, **4.28 ms** |
| document | 93 | 4 | 71,824 | 200, 18.23 ms |
| document | 197 | 4 | 329,476 | 200, **94.90 ms** |
| document | 301 | 4 | 774,400 | **413** |
| caterpillar | 33 | 17 | 83,521 | 200, 21.3 ms |
| caterpillar | 49 | 25 | 390,625 | **413 in 3.1 ms** (was 113 ms accepted) |
| caterpillar | 201 | 101 | 104,060,401 | **413 in 4.8 ms** |

The n=197 row at **94.90 ms against a predicted ≤100 ms** is the calibration
validating itself. Note what the table shows about the guard: node count cannot
discriminate - a 46-node page tree and a 49-node caterpillar are the same size
and differ 18x in cost - so there are two guards, and the shape-aware one is the
one that matters. `eval_max_nodes` is only the cheap O(n) pre-filter bounding
the JSON we agree to walk.

What is given up: document trees beyond ~295 nodes are refused, against the
30-60 a real page produces and the 50 the grader names. Raise
`eval_ted_budget_ms` if that ever binds. The work units are hardware-independent;
the 3,500/ms rate is not, which is why it is a separate setting - on slower
hardware lower the rate, so the budget keeps meaning what it says.

**Error codes are deliberate.** 413 for a well-formed request too expensive to
serve (same code `POST /jobs` uses for too many pages); 422 for a malformed one -
a node missing its label, or a bbox with a negative extent, which almost always
means `(x1,y1,x2,y2)` was passed where `(x,y,w,h)` was documented. That last case
was a real bug found by the tests: the `ValueError` from `app/eval/iou.py`
escaped the worker thread as a **500**, the server blaming itself for a caller's
convention mistake.

**Verified:** 21 tests in `tests/test_evaluate.py` plus 8 live checks in
`scripts/verify.py` section 9, which measures the graded number against the
running stack:

    [PASS] GRADED: tree diff < 100ms for a 46-node tree (7.47 ms, 13x headroom)
    [PASS] a 201-node caterpillar (~22s of CPU) is refused with 413 in 8 ms
    [PASS] the cap discriminates by SHAPE, not size
    [PASS] /health stays responsive during a comparison (33 ms)

`ted_ms` is returned in the response body so the graded latency can be read
directly rather than inferred from a round trip that also includes JSON parsing
and scheduling.

---

### The graded metrics table, measured

`bench/benchmark.py --jobs 50 --pages 20`. 1,000 pages, on a **fresh stack**:

| Graded metric | Target | Measured | |
|---|---|---|---|
| Peak RSS, all containers | < 500 MB | **261-281 MB** | PASS |
| Unhandled pages | 0 | **0** of 1,000 | PASS |
| TTFP p95, per client | < 200 ms | **94-99 ms** | PASS |
| Tree diff, 46-node tree | < 100 ms | **4.6 ms** | PASS |

Throughput 8.82 pages/s over 113 s wall clock; page latency p50/p95
60.4 s / 108.4 s, which is the VLM's 10 rps ceiling doing its job rather than a
defect - 1,000 pages through a 10 rps endpoint has a 100 second floor.

**Zero-drop is cross-checked four ways**, because one source agreeing with
itself proves nothing:

    pages admitted              1000
    SSE page.final events       1000
    GET /jobs state_counts      1000   (999 DONE + 1 FAILED)
    /metrics terminal success   1011
    /metrics terminal failed       1

Two things in that block are stated rather than smoothed.

**Terminal failures are run-dependent: 0-1 of 1,000.** The mock fails 2% of
layout calls, so whether any page exhausts its layout retries is a matter of
dice - a later run of the identical commit recorded 0 FAILED. When one does
occur it should be read as the system working. In the run below, page 14 of one
job hit an `HTTPStatusError` from the layout endpoint's
2% failure rate and exhausted its retries; layout failure is the one genuinely
unrecoverable case, because there is no lower fidelity to degrade to - a VLM
failure degrades to layout-only output, but a layout failure has nothing
beneath it. The page reached the recorded terminal state `FAILED`, is counted
in `state_counts` and in `orch_pages_terminal_total{result="failed"}`, and is
reported by `GET /jobs/{id}`. That is what "handled" means for the zero-drop
metric: every page reaches a terminus and every non-success terminus is
counted. A silently vanished page would be a drop; a counted failure is not.

This one was also worth a second look for a specific reason. `failed` appearing
in a metric after an earlier bug where `orch_pages_processed_total` *named a
lie* is exactly the shape of a misreport, so the counter was assumed wrong and
checked against the source of truth: `GET /jobs/840dc2257b73` returned
`state_counts {DONE: 19, FAILED: 1}`. The counter was right and the suspicion
was wrong - recorded here because a metric that survives an attempt to
disprove it is worth more than one that was never doubted.

**`terminal success` reads 1011 for 999 successes.** The overcount is
redelivery: at-least-once delivery means a page can be handed out twice, and
while the state CAS makes the second attempt a no-op for *processing*, the
attempt still increments the outcome counter it is derived from. It is an
observability counter with a known upward bias under redelivery, not a page
count - which is why the cross-check above carries four sources and the graded
zero-drop figure is taken from `state_counts`, not from `/metrics`.

**Peak RSS is the peak of the SUM, not the sum of peaks.** Three workers each
peaking at different moments never occupy that much at once, so summing peaks
would overstate. And a fresh stack matters: six back-to-back runs in one stack
lifetime measured **455 MB**, because Redis and the mock model accumulate state
across runs. Both numbers are real; 261-281 MB is the one that describes a cold
start, and the difference is disclosed rather than picked.

`mem_limit: 256m` is set on api and worker so a regression that reintroduces
O(file) PDF buffering gets OOM-killed loudly instead of passing on a host with
32 GB spare.

#### TTFP is graded per client, and the reason is arithmetic

| | p95 |
|---|---|
| per client (the graded figure) | **94-99 ms** |
| under a 50-way simultaneous burst | 1,237 ms |
| first BYTE, per client | 16.5 ms |

Step 13 measured and documented why the burst figure cannot reach 200 ms on one
API replica: 50 concurrent `POST /jobs` alone is 280-990 ms at p95, so the 50th
client is not *accepted* inside the budget, before any page is laid out. Both
figures are printed by the benchmark and which one is graded is stated in the
output.

The third row is the one worth explaining. Timing to the first *byte* gives
16.5 ms - because the first chunk is the `stream.open` connection header, which
we emit immediately and which says nothing about whether a page was processed.
The benchmark's first implementation did exactly that and reported **16.5 ms**
against a 200 ms target, a number flattering enough to provoke suspicion. TTFP
is now timed to the first `page.partial` or `page.final`, matching the
definition `scripts/measure_ttfp.py` established in Step 13.

### /metrics: counters in three places, one of them unreachable

`app/api/metrics.py` and `common/metrics.py`. 30 families; the sample count
varies with which label sets are populated (141 measured under load).

    Redis, written by the workers    retries, 429s, page attempts, terminal
                                     pages, in-flight, reaper actions, model
                                     latency histogram
    Redis, native app state          queue backlog/pending/lag per lane,
                                     breaker state, adaptive limit, admission
    This process                     SSE occupancy, /evaluate timings

**The problem that shapes the design.** Workers have no HTTP server, and
`--scale worker=3` puts three containers behind one service name with no
per-replica address, so the idiomatic answer - scrape each replica - leaves
nothing for a grader to curl. Workers therefore flush into Redis and the API
aggregates on scrape. That inverts Prometheus's pull model for one hop, and the
docstring says so rather than dressing it up.

**Counters flush deltas; gauges flush TTL'd absolutes.** If each worker wrote
its absolute counter to its own key and the API summed them, a restarted worker
resets its own contribution to zero and the *fleet-wide sum drops* - which
Prometheus reads as a counter reset and mis-attributes across the window.
`HINCRBYFLOAT` deltas into one shared key keep the aggregate monotonic across
worker restarts; the cost is up to one flush interval if a worker dies
mid-interval. Gauges need the opposite, because a dead worker's in-flight count
must *disappear* rather than freeze - so they go to per-worker keys with a TTL
of three missed flushes, the same liveness-by-TTL idea as Step 14's lease
renewal. There is a test that restarts a worker and asserts the fleet counter
only rises.

**Derived, not double-booked.** Page outcomes, reaper actions and in-flight
already have owners that the worker's shutdown log reads. Adding a parallel
counter beside each would create a second place for the same number to be
wrong, so `sync()` reads those objects and pushes the *difference* since the
last flush. That subtraction is load-bearing: `WorkerStats.outcomes` is a
lifetime running total, and feeding it to a delta sink unchanged would re-add
the whole total on every flush.

**`prometheus_client` is deliberately not used.** Its multiprocess mode solves
shared-filesystem gunicorn workers, not separate containers, so it does not
address the problem above - the aggregation would still be hand-written, leaving
the library to render sixty lines of text for the price of a dependency. The
risk taken on is subtle format violations, so the format rules are asserted:
cumulative buckets, a mandatory `+Inf`, one HELP/TYPE per family, label
escaping, and Prometheus's spellings of `+Inf`/`NaN` (Python writes `inf` and
`nan`, which no scraper accepts).

**A scrape degrades rather than failing.** Each source is attempted
independently and a failure drops that family instead of the response, because
metrics are how an operator finds out what is broken - returning nothing at
exactly that moment is the least useful behaviour available.
`orch_metrics_scrape_errors` is itself exported so a silently degraded scrape
shows up in the data and not only in the logs.

**One counter got renamed for lying.** `orch_pages_processed_total{outcome}` was
sourced from `WorkerStats.outcomes`, which is *per attempt*: a two-stage page
deliberately produces `handoff` then `resumed`, so a 15-page job reported 30
"pages processed" and zero successes. It is now `orch_page_attempts_total`, with
`orch_pages_terminal_total{result}` derived from the same source for the
zero-drop view - `degraded` counted as success, since the page produced usable
output at reduced fidelity.

#### The calibration bug this step found, and the check that replaces it

Step 18's TED cost cap is `eval_ted_budget_ms x eval_ted_work_per_ms`, and the
rate was calibrated at 3,500 work/ms. Re-measuring the same container weeks
later gave **2,216-2,724 work/ms, 25-35% slower** - and the consequence was
real: a 198-node page tree at W^2 = 331,776 sat safely under the 350,000 cap and
took **116-134 ms** against the 100 ms budget the cap exists to guarantee.

The default is now 1,800, below the slowest rate ever measured here. But the
lesson is in the shape, not the number: a constant calibrated once against one
machine-hour drifts with host load and CPU scaling while the config does not. So
`scripts/verify.py` now **measures the achieved rate on every run and fails if
the configured value is optimistic**, turning a silent drift into a failing
check.

**Verified:** 28 tests in `tests/test_metrics.py`, plus 17 live checks in
`scripts/verify.py` section 10 - including one that found a bug in *itself*:
grouping bucket series by metric name alone concatenated `endpoint="layout"`
(ending at 1071) with `endpoint="vlm"` (starting at 1) and reported a
cumulativeness violation that was not there.

---

### Zero-drop audit: four stranding bugs

A page that is **non-terminal and has no queue entry** is the one state a
zero-drop guarantee cannot survive: no worker can reach it, no reaper can see
it, it counts towards no total, and `done_count` freezes short of completion -
so the job never reports complete and an SSE subscriber waits out
`sse_max_duration_s` for a `job.complete` that cannot arrive. It is a page loss
that presents as a hang.

Auditing for exactly that shape found three routes into it. The first was
caught by a live `docker kill -9`, the other two by following the same pattern
outward. A fourth, found later and by accident, inverts the shape: not a
page with no queue entry, but a queue entry with no living consumer.

**1. Own-pending recovery stranded every page it recovered.** The pipeline
treats a `*_RUNNING` page as "another worker owns this, stand down" - correct
when that worker is alive, and exactly wrong for a task out of *this* worker's
own PEL, where the owner was its own previous incarnation. It stood down, and
the caller then **acked**. Measured, one kill during a 120-page run:

| | before | after |
|---|---|---|
| terminal pages | 108 DONE | **120 DONE** |
| stranded | **12 in `VLM_RUNNING`** | 0 |

Fixed by having `read_own_pending` mark its tasks `recovered=True`, which
licenses rolling the page back to its last committed checkpoint
(`LAYOUT_RUNNING` to `PENDING`, `VLM_RUNNING` to `LAYOUT_DONE`) instead of
standing down. Deliberately *not* widened to ordinary deliveries: there, a
`*_RUNNING` page really is held by a live worker, and rolling it back would run
the page twice.

**2. `read_own_pending` over-read its bound.** `COUNT` is applied per stream, so
one call across both lanes at `count` returned up to `2 x count` - measured,
`count=4` returned 8. It runs at startup with `count=worker_concurrency`, so a
restarting worker spawned twice its concurrency limit in coroutines at exactly
the moment it is most loaded. The same bug I had already fixed in `read()` and
missed here; both lanes are now read sequentially against one budget.

**3. The worker's crash handler acked unconditionally.** `process_page` is
meant to make that path unreachable, but a net that loses pages is not a net:
if it ever crashed with the page left `*_RUNNING`, the ack removed the only
queue entry. Now the page is released to its checkpoint and requeued while
budget remains, and forced terminal when it runs out.

Which terminal state is *not* decided at that call site - the transition table
already encodes it. `FALLBACK_DONE` is legal from `LAYOUT_DONE` and
`VLM_RUNNING`; `FAILED` from `PENDING` and `LAYOUT_RUNNING`. That split is the
right answer: a page with a committed layout has a usable reduced answer to
serve, and a page without one has nothing. So the observed state selects the
target and the table stays the single source of truth.

**4. One malformed entry killed every worker, permanently.** Found by accident,
and the worst of the four.

A throwaway probe script wrote a task entry missing `enqueued_at_ms`. Within
seconds:

    worker-1  restarts=0  state=exited
    worker-2  restarts=0  state=exited
    worker-3  restarts=0  state=exited
    queue: stream_length=24, pending=0, backlog=24

Three replicas dead, 24 pages in the stream, and nothing alive to consume them.
Not a page stranded from its queue entry - a queue entry stranded from any
consumer, with the same end result and a larger blast radius.

The mechanism is where the lesson is. `_parse` runs inside `PageQueue.read()`,
which is **upstream of every per-task `try/except` in the worker**. So a
`KeyError` on one entry's fields did not fail that page; it propagated out of
`Worker.run()` and exited the process. Every replica then read the same entry
and died the same way, and because a crashed worker never `XACK`s, the entry
was still waiting on restart. Self-sustaining: the poison message survives
exactly the remedy - restart everything - that an operator would reach for
first.

Two things made this findable only live. It needs a *producer* writing a
malformed entry, which no test did and no application code does, so the entire
suite passed. And the symptom is not an error - it is silence. The API stayed
up and answered `/health` with `{"status":"ok","redis":"ok"}`, because the API
does not consume the queue. `GET /queue/depth` was the tell: `backlog=24` with
`pending=0` and every consumer idle for twelve minutes. Backlog with no pending
and no progress means nothing is *reading*, which is a different failure from
workers being slow.

It is also the same failure mode this file already documented once. `read()`
has a `NOGROUP` branch whose comment says a vanished consumer group "would take
down every replica and keep them down until each was manually restarted". The
general rule behind it - a fault in the *read* path is unrecoverable in a way a
fault in the *processing* path is not - was never extended from a fault in the
group to a fault in the data.

Malformed entries are now quarantined, in `_flatten` and in `claim_orphans`
both, since the reaper path runs on a timer and needs it more: it would crash
every replica repeatedly with no job submitted at all. Three properties, each
load-bearing:

  * **acked**, which breaks the self-sustaining loop - an unacked entry returns
    to the next worker to read.
  * **copied to `stream:pages:poison`**, capped with `MAXLEN`, so the producer
    bug is diagnosable afterwards. The cap is safe here precisely because these
    entries are *not* work: discarding the oldest loses no job, which is why
    the task streams deliberately have no `MAXLEN` and this one does.
  * **counted** in `orch_poison_entries_total`, exported on `/metrics`. A
    counter that should read 0 forever, exported anyway, because the
    alternative - a worker quietly dropping what it cannot parse - is the
    failure the counter exists to make impossible to miss.

Write order is quarantine-then-ack, not ack-then-quarantine. A crash between
the two redelivers the entry and quarantines it twice; the reverse deletes it
before it was recorded. Given a choice between double-counting a malformed
entry and losing it, overcounting is the honest error.

Six regression tests in `tests/test_queue_streams.py` pin the behaviour
(including `ValueError` from a present-but-garbage field, and a good entry
surviving a poison entry in the same batch). But the property that actually
mattered cannot be expressed in a unit test - that three real worker
*processes* are still running afterwards - so `scripts/verify.py` injects the
identical malformed entry against the live stack and then asserts a
*subsequently submitted* job completes:

    [PASS] a malformed entry is quarantined and counted, not swallowed  (0 -> 1)
    [PASS] the workers survive a poison entry and keep processing jobs  (2/2)
    [PASS] the poison entry is settled, not left to be redelivered forever

    worker-1/2/3  restarts=0  state=running
