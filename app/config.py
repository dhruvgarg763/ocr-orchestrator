"""Orchestrator configuration.

Every limit is env-tunable and nothing is hardcoded at a call site. This is not
tidiness: during the benchmark you tune concurrency, prefetch and rate limits to
hit the RSS and throughput targets, and rebuilding an image per experiment is a
non-starter. It also lets the load tests drive the system into states (tiny
watermark, zero retries) that defaults would never reach.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORCH_", env_file=".env", extra="ignore")

    log_level: str = "INFO"
    redis_url: str = "redis://redis:6379/0"
    redis_max_connections: int = Field(default=32, gt=0)
    """Bounded pool: a memory and file-descriptor ceiling."""

    redis_pool_timeout: float = Field(default=10.0, gt=0)
    """Seconds a task waits for a free connection before failing. Waiting (via
    BlockingConnectionPool) turns pool exhaustion into backpressure; the default
    raising pool would turn it into dropped pages."""

    mock_base_url: str = "http://mock-model:8001"

    # --- HTTP client -------------------------------------------------------
    # A bounded pool is itself a backpressure lever: once all connections are in
    # use, further calls wait rather than piling up sockets.
    http_timeout_s: float = Field(default=10.0, gt=0)
    http_connect_timeout_s: float = Field(default=2.0, gt=0)
    http_max_connections: int = Field(default=64, gt=0)

    # --- Client-side rate limits (mirror the mock's advertised limits) -----
    # Enforced by a shared Redis token bucket, not per process: N replicas with
    # N in-process buckets would permit N x the intended rate.
    layout_rps: float = Field(default=100.0, gt=0)
    layout_burst: int = Field(default=100, gt=0)
    vlm_rps: float = Field(default=10.0, gt=0)
    vlm_burst: int = Field(default=10, gt=0)

    # A client-side limiter can never be perfectly in step with the server's.
    # Both run the same algorithm over independent state, so boundary timing
    # lets the occasional request through: measured at 1 stray 429 per 100
    # pages at 192-way concurrency, versus 81% before the limiter existed.
    #
    # Two ways to close the gap. Setting the client rate to ~95% of the
    # server's trades throughput for headroom. Retries (Step 8) absorb the
    # stragglers at no throughput cost, which is why the rates are left equal
    # here - and it is the reason a client limiter does not remove the need for
    # retries, only the need for them to be the primary defence.
    rate_limit_max_wait_s: float = Field(default=30.0, gt=0)
    """How long a call may wait for a token before raising RateLimitTimeout.
    Bounded so a wedged downstream surfaces as an error with a metric instead of
    tasks piling up in memory forever."""

    rate_limit_jitter_ms: float = Field(default=25.0, ge=0)
    """Absolute jitter added to a reserved wait, in milliseconds.

    Absolute, not fractional. The limiter reserves slots rather than polling, so
    callers are already sequenced 1/rate apart and do not need scattering;
    scaling a long wait by a fraction would delay the whole queue (measured:
    8.4 grants/sec against a 10 rps limit) and reorder it (136 FIFO inversions
    in 313 grants). A few milliseconds only keeps adjacent slots off the same
    timer tick."""

    rate_limit_bucket_ttl_s: int = Field(default=300, gt=0)
    """Idle buckets expire rather than accumulating one key per endpoint
    forever. Expiry is safe: a missing bucket starts full."""

    # --- Retry policy ------------------------------------------------------
    max_attempts: int = Field(default=3, ge=1)
    """In-process attempts per stage, including the first. Transient faults
    resolve in milliseconds, so a small budget with backoff covers them; a
    larger one mostly delays a genuine failure."""

    backoff_base_ms: int = Field(default=200, gt=0)
    """Base of the exponential ceiling, and the width of the extra jitter added
    on top of a server-supplied Retry-After."""

    backoff_max_ms: int = Field(default=5_000, gt=0)
    """Cap on the ceiling. Without it, attempt 10 would schedule 200ms * 2^10 =
    3.4 minutes."""

    page_deadline_s: float = Field(default=300.0, gt=0)
    """Total wall-clock budget for one page, measured from when it FIRST
    entered the system and preserved across requeues.

    This, not an attempt count, is what bounds systemic requeueing. Saturation
    is a property of the endpoint, so charging a page's attempt budget for a
    system-wide condition would drop pages whose only mistake was arriving
    during a busy period - which violates the zero-drop requirement. A deadline
    gives every page a fair share of wall-clock time no matter how many times
    congestion bounced it.

    300s against a ~10 pages/sec ceiling means roughly 3,000 pages of backlog
    may clear ahead of a given page before it gives up."""

    degrade_after_s: float = Field(default=90.0, gt=0)
    """How long a page holds out for full VLM fidelity before accepting a
    layout-only result.

    The quality-versus-latency knob, and the reason degrading is a last resort.
    The spec says to fall back only once retries are exhausted, so a page that
    never reached the VLM - because its circuit was open - waits for recovery
    instead. Measured on an 80% rejection storm: degrading on the first open
    circuit produced 32 layout-only pages out of 40, where waiting leaves
    P(never succeeding) near 0.5%.

    90s comfortably outlasts a breaker cooldown cycle (5s) and the kind of
    transient outage the storm test simulates, while bounding how long a job
    can appear stalled. Bounded either way: `page_deadline_s` and
    `max_requeues` are the outer limits."""

    saturation_pause_s: float = Field(default=0.5, ge=0)
    """How long ONE worker slot pauses after ONE saturation, before taking new
    work.

    Required because the limiter reserves rather than polls, so it reports
    saturation in ~2ms instead of after a 30s wait. Without a pause the requeue
    loop spins: measured at 99 events/sec from 4 slots, consuming a 50-requeue
    backstop in 2 seconds and generating ~6,000 Redis ops/sec across 48 slots.

    Deliberately NOT an endpoint-wide or worker-wide pause. Those would idle the
    100 rps layout endpoint because the 10 rps one is busy, starving the layout
    results that both the degraded fallback and time-to-first-page depend on.
    The pause is also capped by the projected time to the next free slot, since
    sleeping past that wastes capacity."""

    max_requeues: int = Field(default=50, ge=1)
    """Absolute floor for the runaway-loop backstop. Read via
    `effective_max_requeues`, never directly.

    A requeue COUNT is the wrong unit for a time-based policy, and using it as
    one has now caused the same bug three times: its wall-clock meaning depends
    on how fast saturation is detected and how long the pause is. At
    saturation_pause_s=0.5 a budget of 50 expires after 25s - shorter than the
    90s hold-out, so pages degraded 4s before the endpoint recovered
    (observed: attempt=49 at age_s=36.3, while degrade_after_s never fired).

    So it is derived rather than trusted. Age is the operative bound; this only
    exists to catch a loop where age somehow fails to advance."""

    # --- Adaptive rate control (AIMD) --------------------------------------
    adaptive_enabled: bool = Field(default=True)
    """Off switch, so the benchmark can A/B the controller's contribution
    without rebuilding an image."""

    aimd_min_rate: float = Field(default=1.0, gt=0)
    """Floor. MUST be > 0: at rate 0 no token is ever granted, so the
    controller could never observe a success and could never climb back out -
    a self-inflicted deadlock."""

    aimd_decrease_factor: float = Field(default=0.7, gt=0, lt=1)
    """Multiplicative decrease. Strictly below 1 or it is not a decrease.

    0.7 reaches the floor from 10 rps in ~7 events; TCP's 0.5 is more
    aggressive than we need, because our ceiling is a known published limit
    rather than an unknown bandwidth, so the distance to travel is small."""

    aimd_increase_step: float = Field(default=1.0, gt=0)
    """Additive increase, in rps. Additive is the whole point: a multiplicative
    increase overshoots capacity on every probe, producing a large-amplitude
    sawtooth that spends half its time in overload."""

    aimd_increase_after: int = Field(default=20, ge=1)
    """Consecutive successes required per increase step.

    Sets the recovery slope. At 20, climbing 1 -> 10 rps takes 180 successes;
    at ~10 rps that is around 18s. Lower recovers faster but re-probes an
    endpoint that may still be sick; higher is gentler but leaves throughput on
    the table after the outage has passed."""

    aimd_refractory_ms: float = Field(default=1_000.0, ge=0)
    """Minimum gap between decreases.

    Without it a single congestion event collapses the rate to the floor: at
    the moment of a cut there are ~rate x latency requests already in flight,
    admitted at the OLD rate and about to fail against the same overloaded
    endpoint. Compounding one decrease per failure gives 0.7^22 = 0.0004 of the
    original rate from one event. TCP's equivalent rule is one reduction per
    RTT - a reduction should not be re-applied before its effect is
    observable."""

    vlm_latency_slo_ms: float = Field(default=4_500.0, ge=0)
    """p95 above this counts as congestion even with zero rejections.

    This is the signal the token bucket and the failure-ratio breaker are both
    blind to, and the assignment names it explicitly ("or latency spikes").

    4500ms is 1.5x the top of the VLM's advertised 1500-3000ms band. An
    absolute SLO is used rather than a learned baseline on purpose: a baseline
    that adapts to sustained slowness stops detecting it - the boiling-frog
    failure - whereas a fixed threshold derived from the published contract
    keeps meaning the same thing. 0 disables the check."""

    layout_latency_slo_ms: float = Field(default=500.0, ge=0)
    """10x the layout model's advertised 50ms. Wider in relative terms than the
    VLM's because 50ms is small enough that scheduling noise alone moves it."""

    aimd_latency_samples: int = Field(default=100, ge=1)
    """Ring buffer depth for the p95 estimate. Bounded, so the memory cost is
    O(1) per endpoint rather than growing with traffic."""

    aimd_latency_floor_fraction: float = Field(default=0.5, gt=0, le=1)
    """Floor for LATENCY-triggered decreases, as a fraction of the advertised
    rate. Distinct from `aimd_min_rate`, which applies to 429s and timeouts.

    The two signals carry different weights of evidence. A 429 is the endpoint
    saying directly that we are too fast, which justifies backing off to the
    hard floor. A latency breach is ambiguous: slowness caused by our load is
    relieved by backing off, but slowness from their own GC pause or slow
    dependency is not, and throttling then discards throughput for nothing.
    Latency alone cannot tell the two apart.

    Measured: the mock caps rate but not concurrency, so at 13s latency it
    still serves its full 10 rps. Flooring the rate on that signal cut a
    60-page job to 34 pages in the window where adaptive=off finished all 60.
    Half the advertised rate is the compromise - relieve a possibly-overloaded
    endpoint, without paying 90% of our throughput on evidence we cannot
    attribute.

    The better long-term signal is latency relative to the MINIMUM observed
    (a queueing-delay estimate, as TCP Vegas and Netflix's adaptive-concurrency
    limiter do) rather than an absolute SLO, since queueing delay is the part
    actually caused by our own load."""

    aimd_latency_min_samples: int = Field(default=20, ge=1)
    """No p95 verdict below this many samples. A p95 over 3 samples is not a
    p95, and two slow warm-up calls should not throttle a healthy system."""

    # --- Circuit breaker ---------------------------------------------------
    breaker_window_s: float = Field(default=10.0, gt=0)
    """Rolling window over which the failure ratio is computed. Counters that
    never reset would remember an hour-old outage and keep the ratio elevated
    forever; only recent history predicts the next request."""

    breaker_min_volume: int = Field(default=10, ge=1)
    """Minimum requests in the window before the ratio can trip the breaker.
    Without it, 1 failure out of 2 requests reads as 50% and opens the circuit
    on noise."""

    breaker_failure_ratio: float = Field(default=0.5, gt=0, le=1)
    """Failure fraction that opens the circuit."""

    breaker_cooldown_s: float = Field(default=5.0, gt=0)
    """How long the circuit stays open before admitting probes."""

    breaker_max_probes: int = Field(default=3, ge=1)
    """Concurrent probes allowed while HALF_OPEN, counted in Redis so that N
    replicas cannot each send "just one" and collectively flood a recovering
    endpoint."""

    breaker_probe_successes: int = Field(default=2, ge=1)
    """Consecutive probe successes required to close the circuit. More than one,
    so a single lucky response does not readmit full production load."""

    # --- Worker ------------------------------------------------------------
    worker_concurrency: int = Field(default=16, gt=0)
    """Hard ceiling on pages in flight in one worker. Enforced by *bounded
    dispatch* - the worker never reads more tasks than it has free capacity -
    so resident tasks are <= this value no matter how deep the queue is.

    Measured goodput vs this setting, with no client-side rate limiting
    (60-page job, single worker):

        conc   goodput      success   vlm 429s
           4   1.57 p/s     91.7%       0%
           8   2.96 p/s     91.7%       0%
          16   4.74 p/s     81.7%      13%
          32   4.58 p/s     31.7%      67%
          64   3.85 p/s     33.3%      65%

    Goodput peaks near 16 and then falls: past that point the endpoint's
    capacity is spent on requests that get rejected. Note that even the safe
    settings only reach 91.7%, because the remaining loss is the 5% transient
    failure rate and there are no retries yet - so no value of this knob
    satisfies the zero-drop requirement. The correct value is not a constant
    anyway; the adaptive limiter discovers it at runtime."""

    worker_prefetch: int = Field(default=8, gt=0)
    """Max tasks fetched per XREADGROUP call, capped further by free capacity.
    Larger batches mean fewer round trips; smaller batches mean less work
    sitting in this worker's PEL if it dies."""

    worker_lead_reserve: int = Field(default=4, ge=0)
    """Dispatch slots the main lane may NOT occupy, held for the priority lane.

    Bounded dispatch only reads when a slot is free. After the stage handoff
    those slots fill with VLM-stage pages parked on a 10 rps endpoint's token,
    so a newly arriving job's first page waits for a SLOT rather than for the
    100 rps endpoint it actually needs - and while a worker is at capacity the
    priority lane is not polled at all. Capping what the main lane may hold
    keeps a slot available for a first page at all times.

    Measured: one job arriving INTO a system already saturated with 800 pages
    of VLM-stage work (~40 pages in flight, ~760 queued), time to its first
    page event:

        reserve   p50       max
        0         245.1 ms  626.0 ms
        4         108.3 ms  196.3 ms

    2.3x at p50, 3.2x at the tail, and the tail lands under the 200ms target.
    Throughput is unaffected: the 50-job / 1,000-page benchmark runs 109.8s and
    112.7s at reserve=4 against 109-118s at reserve=0.

    WHY THIS WAS FIRST MEASURED AS WORTHLESS
    ----------------------------------------
    An earlier A/B concluded there was no benefit, and that conclusion was an
    artefact of the experiment rather than a property of the feature. It
    submitted all 50 jobs at t=0 into an EMPTY system - so every slot was free
    when the lead lane filled, the lane drained instantly, and a reservation had
    nothing to do. The scenario it exists for, a latecomer arriving into a busy
    system, was never tested. Two lessons, and the second is the sharper one:
    a benchmark that starts from an idle system cannot measure steady state, and
    "no measurable effect" is a claim about the measurement until the scenario
    is shown to be able to produce the effect.

    That first A/B also mismeasured the COST, because the implementation had a
    bug. When free capacity fell to equal the reservation the main budget hit
    zero, and the worker then blocked on the (empty) lead lane for a full
    worker_block_ms while hundreds of main-lane pages waited - in-flight pages
    fell from 48 to 2 and throughput dropped ~40%. The lane was never the tax;
    waiting on the wrong thing was. The worker now returns immediately in that
    case and waits on its in-flight tasks instead, which is the event that
    actually matters (a slot freeing) and needs no timer.

    Carved OUT of worker_concurrency, so the resident-task bound is unchanged:
    this partitions the existing budget rather than growing it. Small on purpose
    - the lane holds one entry per job, and a lead task runs only the ~50ms
    layout stage before handing off, so a few slots turn over fast enough to
    absorb a burst of new jobs. Set to 0 to disable.

    SWEEP RESULTS, kept because the shape of the curve is the useful part.
    Latecomer into a system already holding ~700 queued pages, n=30 samples,
    5-page job, with worker_lead_poll_ms=100:

        reserve   p50       p95        notes
        0         268.5 ms  476.3 ms
        4          71.7 ms  274.0 ms   default
        8          71.5 ms  135.2 ms   best measured
        10         91.2 ms  379.0 ms
        12        223.5 ms  783.2 ms   throughput collapses: 82s vs 46s wall

    The collapse at 12 is predictable rather than surprising. Little's law puts
    the slots needed to saturate a 10 rps endpoint with ~2.25s calls at about
    10 x 2.25 = 23; reserve=12 leaves the main lane (16-12) x 3 = 12, well
    under that, so pages/sec drops and the resulting backlog makes LATENCY
    worse too. reserve=8 leaves 24, just above the floor - which is exactly why
    it is the best point on the curve and also why it has no margin.

    Left at 4 deliberately. 8 measures better on this scenario but sits one
    slot above the Little's-law floor, so any change that lengthens a VLM call
    - a slower endpoint, a higher latency SLO, a smaller replica count - pushes
    it into the collapse seen at 12. 4 keeps 36 main slots against a floor of
    23, which is margin worth having for a latency figure that already passes
    at p95. Revisit with 8 if the VLM's real latency is ever pinned down."""

    worker_lead_poll_ms: int = Field(default=100, ge=0)
    """How long to wait on the priority lane when the main lane is at its cap.

    This is the difference between "a first page starts when one arrives" and
    "a first page starts when some unrelated page happens to finish".

    When `worker_lead_reserve` slots are free but the main lane may take no
    more, the reserved slots are the only ones that can act - so the worker
    must wait on the LEAD LANE, not on its in-flight tasks. Returning
    immediately instead meant the lane was re-probed only when an unrelated
    VLM call completed: a ~94ms mean with a long tail, measured as a p95
    time-to-first-page of 333-406ms for a job arriving into a busy system, at
    every reserve value tried.

    Bounded, and NOT set to worker_block_ms, because the two failure modes sit
    on opposite sides of this number:

      too short   the lane is polled rather than awaited, burning round trips
      too long    capacity freed DURING the wait goes unused, because the
                  worker is parked in Redis and cannot refill the main lane.
                  At the full 2s this cost ~40% throughput - in-flight pages
                  fell from 48 to 2 with 760 pages queued.

    100ms sits where the second cost is ~2%: a main slot frees every ~94ms, so
    an average wait of 50ms against a 2.25s call is 2250/(2250+50) = 97.8%
    utilisation - while a first page now waits on its own arrival instead of on
    someone else's completion."""

    worker_block_ms: int = Field(default=2_000, gt=0)
    """How long XREADGROUP blocks when the queue is empty. Bounded so the loop
    can notice a shutdown request; 0 would block forever."""

    # --- Crash recovery: leases and the reaper (Step 14) -------------------
    reaper_enabled: bool = True
    """Whether workers reclaim page tasks orphaned by a worker that vanished.

    `read_own_pending` already covers a worker RESTARTING, because the consumer
    name is the container hostname and so survives the process. It cannot cover
    a worker being REPLACED - `--force-recreate`, a rescheduled pod, a scale-down
    from 3 replicas to 2 - because the new process has a new name and the dead
    consumer's Pending Entries List belongs to nobody. Those entries are skipped
    by `XREADGROUP >` (already delivered) and held by no live consumer, so the
    pages are non-terminal AND unreachable: the same zero-drop violation the
    own-pending fix closed, reached by a different route."""

    reaper_interval_s: float = Field(default=15.0, gt=0)
    """How often a worker scans for orphaned entries.

    Every replica runs it; no leader election. Two reapers racing the same entry
    is harmless by construction - see `PageQueue.claim_orphans`, where XCLAIM's
    mandatory min-idle-time makes the claim itself conditional on the idle clock
    not having been reset, so the loser's claim atomically returns nothing."""

    reaper_min_idle_s: float = Field(default=30.0, gt=0)
    """Idle time after which a pending entry is considered abandoned.

    This number is only defensible because of `lease_renew_interval_s` below.
    Idle time in a PEL measures time since DELIVERY, not time since progress,
    so on its own it is a poor liveness signal: a task legitimately held through
    3 attempts of (up to rate_limit_max_wait_s waiting for a token + up to
    http_timeout_s in the call) plus backoff can be idle for ~130s while
    perfectly healthy. Thresholding above that would make recovery slower than
    the page deadline it is supposed to beat.

    Lease renewal changes what the clock measures. A live worker resets the idle
    time of its own in-flight entries every `lease_renew_interval_s`, so idle
    time becomes "this worker has stopped renewing" - an actual liveness signal.
    30s is 6 consecutive missed renewals, which tolerates a long GC pause, a
    Redis blip or a briefly overloaded event loop without reclaiming a page
    somebody is still working on."""

    reaper_batch: int = Field(default=64, gt=0)
    """Orphans reclaimed per scan, per lane.

    Bounded for the same reason prefetch is: a worker that died holding 16
    pages, times N replicas, is a finite burst - but an operator flushing a
    consumer group or a mass eviction is not, and an unbounded scan would turn
    recovery into its own outage. The cursor is the idle clock itself, so
    whatever is skipped this scan is still eligible on the next one."""

    lease_renew_interval_s: float = Field(default=5.0, gt=0)
    """How often a worker resets the idle clock on its own in-flight entries.

    One XCLAIM per lane per interval, carrying every in-flight id at once, so
    the cost is O(1) round trips rather than O(pages). JUSTID is used
    deliberately: it resets idle time WITHOUT incrementing delivery_count, so
    renewal stays invisible to retry accounting. Without that, a page held for
    90s would appear to have been delivered 18 times.

    Must stay well below `reaper_min_idle_s`, or a healthy worker reaps itself.
    """

    # --- Ingestion & memory bounds -----------------------------------------
    data_dir: str = "/data"
    """Where uploaded documents live. Shared volume between api and worker:
    the API streams the file in, the workers read single pages back out."""

    max_pages: int = Field(default=100, gt=0)
    """Validation bound from the spec ("up to 100 pages per job"). Enforced
    against the PDF's real page count after upload, not just the declared one."""

    max_upload_mb: int = Field(default=120, gt=0)
    """Refused MID-STREAM, not after writing. Content-Length is advisory - a
    client can omit it under chunked encoding or simply lie - so checking it is
    not a bound. 120 leaves headroom over the spec's 100MB reference size."""

    upload_chunk_bytes: int = Field(default=1 << 20, gt=0)  # 1 MiB
    """Read granularity for uploads. This IS the resident memory for the upload
    path: `read(chunk)` keeps one chunk live, while `read()` with no argument
    un-spools Starlette's temp file into one contiguous 100MB bytes object."""

    orphan_sweep_interval_s: float = Field(default=120.0, gt=0)
    """How often a worker checks for documents whose job no longer exists.

    Deleting on completion only fires when a worker acks the LAST page of a
    job, so anything that stops a job completing strands its upload - a page in
    a dead worker's pending list, a kill between the final ack and the delete,
    or Redis state expiring mid-flight. Uploads are the only resource here with
    no TTL of their own."""

    orphan_sweep_age_s: float = Field(default=900.0, gt=0)
    """Minimum document age before it may be considered orphaned.

    Must stay comfortably above `page_deadline_s`, or the sweeper could race a
    job that is merely slow. The real test is Redis liveness - the job hash
    being gone means nothing can reference the file - and this only avoids
    looking at documents whose job could still be running."""

    pdf_text_sample_chars: int = Field(default=0, ge=0)
    """Characters of page text to sample, or 0 to skip extraction entirely.

    Defaults to OFF because `extract_text()` is the single most expensive thing
    in the pipeline on adversarial input, on both axes. Measured per page on a
    100 MiB / 100-page document whose pages carry ~1 MB of text each:

        text ON    18.1 MiB resident   3038 ms CPU
        text OFF    1.6 MiB resident     26 ms CPU

    3 seconds per page is slower than the VLM it feeds, so enabling it would
    move the bottleneck off the 10 rps endpoint and onto our own CPU. And
    18.1 MiB x 48 concurrent extractions is 864 MiB - over the 500 MB budget on
    that term alone.

    Truncating the OUTPUT does not help: the whole content stream has to be
    walked to produce any text at all, so the knob has to be able to disable
    the work rather than shrink the result.

    Nothing in the pipeline needs the text - the mock model produces its own
    output, and /evaluate compares client-supplied trees - so geometry alone is
    a faithful page-level extraction. The knob exists for the case where real
    text matters, with the cost stated rather than discovered later."""

    stage_handoff: bool = True
    """Release the worker slot between the layout and VLM stages.

    The two stages differ in capacity by 10x (100 rps vs 10 rps). Holding one
    slot across both makes the fast stage inherit the slow stage's queueing:
    with 48 slots and 100 pages, every slot parks on a VLM token and pages
    49-100 are never read off the queue, so their layout waits on an endpoint
    it does not use. Measured, 5 jobs x 20 pages, p95 time-to-first-page:

        handoff OFF   4901 ms
        handoff ON     (see README)

    The cost is one extra XADD+XACK per page - a page crosses the queue twice -
    which is why this is a switch rather than an assumption. Turn it off and
    the pipeline still works, just with the fast stage queued behind the slow
    one."""

    # --- Result streams ----------------------------------------------------
    result_stream_maxlen: int = Field(default=256, gt=0)
    """Cap on a per-job result stream. The bound on what a slow SSE client can
    cost Redis.

    Sized from the event arithmetic rather than picked: a page emits exactly two
    events (page.partial, page.final - both gated on a successful state
    transition, so a requeue does not add more), plus one job.complete. At the
    100-page limit that is 201, so 256 lets a client that connects at the very
    END of a job still replay every event of it. Set below that and a late
    subscriber silently starts mid-job, which is the failure the gap check
    exists to at least make visible.

    Note this is `MAXLEN ~` - approximate. Redis trims whole radix nodes, so the
    real length can sit somewhat above the figure; exact trimming costs more CPU
    per XADD to enforce a bound that is already an estimate of a memory budget."""

    result_ttl_s: int = Field(default=3_600, gt=0)

    sse_block_ms: int = Field(default=15_000, gt=0)
    """How long a tailing XREAD blocks before we surface to send a heartbeat.

    Blocking, not polling: Redis wakes the reader the instant a worker
    publishes, so first-page latency is a property of the pipeline and not of a
    poll interval. The bound exists because a blocked reader cannot notice its
    own client disconnecting - so this is really the worst-case delay in
    reclaiming a dead subscriber's Redis connection."""

    sse_batch: int = Field(default=64, gt=0)
    """Events per tail read. Bounds what one iteration materialises."""

    sse_history_limit: int = Field(default=512, gt=0)
    """Cap on the catch-up replay a new subscriber can trigger. Above
    result_stream_maxlen so the replay is normally complete, and finite so a
    misconfigured maxlen cannot make one request read an unbounded window."""

    sse_retry_ms: int = Field(default=2_000, gt=0)
    """Reconnect delay advertised to browsers via the SSE `retry:` field.

    EventSource defaults to 3s and, crucially, to the SAME 3s for every client -
    so an API restart has all of them return in one instant. This is set
    explicitly and jittered by the clients' own scheduling; the value matters
    less than being the one who chose it."""

    sse_max_subscribers: int = Field(default=64, gt=0)
    """Concurrent SSE connections this replica will serve.

    A subscriber is not free: it holds a Redis connection for the whole time it
    is blocked in XREAD, so "how many clients can subscribe" is a resource
    question, not a preference. Left uncapped, the 65th client would wait out
    `pool_timeout` and fail with a truncated body.

    Sized above the benchmark's 50 concurrent jobs with headroom for a client
    reconnecting before its old connection has been reaped. Refusing at the door
    with a 503 and a Retry-After is the same argument as Step 11's watermark: an
    explicit, counted refusal beats a resource exhaustion nobody is told about.

    The stream pool is sized from THIS value, so raising it raises both."""

    sse_max_duration_s: float = Field(default=900.0, gt=0)
    """Hard lifetime for one SSE connection.

    A stream is not allowed to live forever: a job that never completes - a page
    stranded in a dead worker's pending list - would otherwise pin a connection
    and a Redis socket indefinitely. Safe to enforce only because resume works:
    the client reconnects with Last-Event-ID and loses nothing."""

    # --- Admission control -------------------------------------------------
    queue_high_watermark: int = Field(default=5_000, gt=0)
    """Pages admitted-but-unsettled above which POST /jobs returns 503.

    The outermost memory bound. Our own RSS is already bounded by bounded
    dispatch, but each queued page costs ~500 bytes of Redis (a stream entry
    plus a state hash), so this caps Redis at roughly 2.5MB of queued work.

    5,000 pages is ~8 minutes of backlog at the VLM's 10 rps - deep enough to
    absorb the 1,000-page benchmark and a burst on top, shallow enough that a
    client refused here is being told something true about wait times rather
    than being throttled arbitrarily."""

    queue_low_watermark_fraction: float = Field(default=0.8, gt=0, lt=1)
    """Recovery mark, as a fraction of the high watermark. Hysteresis.

    With a single threshold the system flaps on every page completion: at the
    mark it refuses, one page drains so it admits, the next job pushes it back
    over. Clients would see an unpredictable mix of 202s and 503s with no
    stable signal. Two marks make it a Schmitt trigger - trip at high, recover
    only at low - for the same reason a thermostat does not switch at a single
    temperature.

    Must be strictly below 1: at 1 the two marks coincide and the hysteresis
    disappears."""

    retry_after_cap_s: int = Field(default=300, gt=0)
    """Ceiling on the Retry-After we quote. "Come back in two hours" is not
    actionable, and a client that waits that long has effectively been dropped
    rather than deferred."""

    retry_after_jitter: float = Field(default=0.2, ge=0)
    """Fractional jitter on Retry-After.

    Step 8's lesson, applied at the edge: fifty clients handed an identical
    `Retry-After: 60` all return in the same instant and recreate the overload
    that caused the refusal. Jitter turns that convoy back into a queue."""


    @property
    def effective_max_requeues(self) -> int:
        """Runaway-loop ceiling, derived so it cannot bind before the deadline.

        Derived from whichever AGE threshold actually fires - the smaller of
        degrade_after_s and page_deadline_s - then doubled for headroom. That
        makes age the operative bound in the default configuration and leaves
        this as the pure safety net it was meant to be.

        An explicitly configured value is honoured as-is: if an operator or a
        test sets ORCH_MAX_REQUEUES, they mean it, and silently overriding it
        would be worse than the bug this guards against. pydantic records which
        fields were supplied, so "explicit" is knowable rather than guessed.
        """
        if "max_requeues" in self.model_fields_set:
            return self.max_requeues

        budget_s = min(self.degrade_after_s, self.page_deadline_s)
        per_requeue_s = max(self.saturation_pause_s, 0.05)
        return max(self.max_requeues, int(budget_s / per_requeue_s) * 2)

    # ---------------------------------------------------- observability (18)

    metrics_flush_interval_s: float = Field(default=5.0, gt=0)
    """How often a worker pushes its counters into Redis for GET /metrics.

    Workers have no HTTP server and `--scale worker=3` gives them no per-replica
    address, so they cannot be scraped; they flush instead. See
    common/metrics.py for why counters go as deltas and gauges as TTL'd
    absolutes.

    5s against a typical 15s Prometheus scrape means a scrape is at most one
    flush stale, which is inside the resolution a 15s scrape has anyway.
    Lowering it buys nothing a scraper can see and costs a Redis round trip per
    worker per interval; raising it past the scrape interval would make
    consecutive scrapes return identical values and flat-line every rate."""

    # ------------------------------------------------------- evaluation (C)

    eval_max_nodes: int = Field(default=2000, gt=0)
    """Largest tree POST /evaluate will accept, per side.

    A document page is 30-60 nodes; the graded target is 50. 2,000 leaves room
    for a pathologically dense real page while refusing anything that is
    obviously not a page. This is the cheap, legible half of the guard - it
    bounds the JSON we will walk - but it is NOT what bounds the compute, because
    tree edit distance cost depends on SHAPE, not node count: a 49-node
    caterpillar costs 18x a 46-node page tree. See `eval_max_ted_work`."""

    eval_ted_budget_ms: float = Field(default=100.0, gt=0)
    """The latency this endpoint PROMISES for any comparison it accepts.

    Defaulted to the graded target - "Tree Diff Calculation Latency < 100 ms for
    a 50-node document structure tree" - deliberately, so the guard is a
    guarantee rather than merely a catastrophe filter. The cost of a
    Zhang-Shasha pass is Theta(W1 x W2) and both weights are O(keyroots) to
    compute, so a request that cannot be served inside this budget is knowable
    in advance and is refused with 413 instead of being started.

    That distinction was measured, not assumed. An earlier cap of 3.5M work
    units (~1 s) stopped the real CPU bombs but ADMITTED a 49-node caterpillar
    at W^2 = 390,625, which then took 113 ms - over the graded budget, on a
    payload of 49 nodes. Sizing the cap from the budget rejects it.

    What is given up is the ability to score an enormous DOCUMENT-shaped tree:
    real page trees have W ~ 2n, so this budget admits them up to roughly 295
    nodes, against the 30-60 a page actually produces and the 50 the grader
    asks about. Raise the budget if a legitimate document ever needs more."""

    eval_ted_work_per_ms: int = Field(default=1800, gt=0)
    """Work units per millisecond, measured on the reference container.

        n=46  page tree     W^2 =      17,424       4.97 ms   3,506/ms
        n=49  caterpillar   W^2 =     390,625      89.81 ms   4,349/ms
        n=101 caterpillar   W^2 =   6,765,201   1,427.60 ms   4,739/ms
        n=201 caterpillar   W^2 = 104,060,401  21,925.80 ms   4,746/ms

    3,500 was the slowest rate observed WHEN FIRST MEASURED, and that turned out
    not to be conservative at all. Re-measuring on the same container weeks
    later - after a Docker Desktop restart, on a differently-loaded host - gave
    2,216-2,724 work/ms, 25-35% slower. The consequence was real: a 198-node
    page tree with W^2 = 331,776 sat safely under the 350,000 cap and took
    116-134 ms, over the budget the cap exists to guarantee.

    So the default is 1,800 - below the slowest rate ever measured here, with
    margin. The cost is that the admitted document tree shrinks from ~295 nodes
    to ~215, which is still four times the graded size and well past any real
    page.

    The lesson is in the shape of the setting, not the number: a constant
    calibrated once against one machine-hour is a latent bug, because the rate
    drifts with host load, CPU scaling and container limits while the config
    does not. `scripts/verify.py` therefore MEASURES the achieved rate on every
    run and fails if this value is optimistic, which turns a silent drift into
    a failing check. The WORK is hardware-independent; this rate is not."""

    @property
    def eval_max_ted_work(self) -> int:
        """Ceiling on W(T1) x W(T2), derived so the budget is enforceable."""
        return int(self.eval_ted_budget_ms * self.eval_ted_work_per_ms)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached so config is parsed once per process, not per request."""
    return Settings()
