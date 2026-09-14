"""Job ingestion and status.

Three ways in - `POST /jobs` (page count, no document, for the
1,000-page benchmark), `/jobs/stream` (raw PDF body, memory-optimal),
`/jobs/upload` (multipart) - all converging on `_admit_and_enqueue` so
admission is stated and enforced exactly once. Order inside `create_job`
is deliberate and not swappable: 413 (cheap request validation) before
503 (a capacity check, so a malformed job isn't misfiled as "shed for
capacity"), and `init_job` (state rows) before `enqueue` (tasks), since a
worker could otherwise pick up a page before its state row exists. The
capacity check and the enqueue that satisfies it run under one
process-local lock - without it, concurrent requests admit against the
same pre-enqueue depth (measured: 130-150% overshoot at 50-way
concurrency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.api.admission import AdmissionController
from app.config import get_settings
from app.pdf.splitter import (
    InvalidPdf,
    UploadTooLarge,
    count_pages,
    delete_document,
    pdf_path,
    stream_body_to_disk,
    stream_to_disk,
)
from app.queue.state import PageStateStore
from app.queue.streams import PageQueue
from common.logging import get_logger
from common.tracing import get_trace_id

log = get_logger("jobs")

router = APIRouter()


class CreateJobRequest(BaseModel):
    pages: int = Field(ge=1, description="Pages in the document")
    filename: str | None = Field(default=None, max_length=256)


def _store(request: Request) -> PageStateStore:
    return request.app.state.state_store


def _queue(request: Request) -> PageQueue:
    return request.app.state.page_queue


def _admission(request: Request) -> AdmissionController:
    return request.app.state.admission


async def _ingest_document(
    request: Request,
    job_id: str,
    dest: Path,
    size: int,
    filename: str,
    trace_id: str,
) -> dict[str, Any]:
    """Everything after the bytes have landed: count, validate, admit, enqueue.

    Shared by both upload endpoints, which differ only in how the bytes arrive.
    """
    settings = get_settings()

    try:
        # Blocking parse, so off the event loop: reading a 100 MiB file's xref
        # would otherwise stall every other request in this process, including
        # the SSE streams whose time-to-first-page is graded.
        pages = await anyio.to_thread.run_sync(count_pages, dest)
    except InvalidPdf as exc:
        delete_document(settings.data_dir, job_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if pages > settings.max_pages:
        delete_document(settings.data_dir, job_id)
        raise HTTPException(
            status_code=413,
            detail=f"{pages} pages exceeds the limit of {settings.max_pages}",
        )

    try:
        verdict = await _admit_and_enqueue(
            request, job_id, pages, filename=filename, trace_id=trace_id
        )
    except HTTPException:
        # Shed: nothing references the document now, so it must not be left on
        # disk. Redis state expires by TTL; a file does not.
        delete_document(settings.data_dir, job_id)
        raise

    log.info("upload_accepted", job_id=job_id, pages=pages, bytes=size)
    return {
        "job_id": job_id,
        "pages": pages,
        "bytes": size,
        "status": "accepted",
        "queue_depth": verdict.depth,
        "trace_id": trace_id,
    }


async def _admit_and_enqueue(
    request: Request,
    job_id: str,
    pages: int,
    *,
    filename: str,
    trace_id: str,
) -> Any:
    """Admission, state init and enqueue - the shared core of both endpoints.

    One implementation because the invariants are the same either way, and two
    copies of a lock-ordering rule is one copy too many.
    """
    async with request.app.state.admission_lock:
        verdict = await _admission(request).evaluate(pages)
        if not verdict.admitted:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "queue_full",
                    "message": (
                        "The queue is at capacity. This job was not accepted; "
                        "retry after the interval below."
                    ),
                    "queue_depth": verdict.depth,
                    "high_watermark": _admission(request).high_watermark,
                    "retry_after_s": verdict.retry_after_s,
                },
                # RFC 9110 Retry-After, so a client does not have to parse the
                # body to behave correctly. Jittered, or fifty refused clients
                # all return in the same instant and recreate the overload.
                headers={"Retry-After": str(verdict.retry_after_s)},
            )

        # Ordering inside the lock is still state-before-work: a worker must
        # never dequeue a page whose state row does not exist yet.
        await _store(request).init_job(
            job_id, pages, filename=filename, trace_id=trace_id
        )
        await _queue(request).enqueue_pages(job_id, pages, trace_id=trace_id)

    return verdict


@router.post("/jobs", status_code=202)
async def create_job(req: CreateJobRequest, request: Request) -> dict[str, Any]:
    """Accept a job and return immediately.

    202, not 200: the work has been queued, not performed. Blocking until all
    pages finished would take minutes for a 100-page document and would hold a
    connection open the whole time.
    """
    settings = get_settings()

    # 413 first: a VALIDATION bound, and cheap. This job could never be served
    # at any queue depth, so evaluating capacity for it would be wasted work -
    # and worse, would count it in the shed statistics as though it were a
    # capacity problem rather than a malformed request.
    if req.pages > settings.max_pages:
        raise HTTPException(
            status_code=413,
            detail=f"{req.pages} pages exceeds the limit of {settings.max_pages}",
        )

    # 503 second: a CAPACITY bound. Downstream backpressure cannot help here,
    # because it all acts after the work is queued - a client can offer ~20,000
    # pages/sec against the VLM's 10. Unbounded admission and bounded memory
    # cannot both hold; the only choice is whether the refusal is explicit and
    # early or an OOM kill nobody is told about.
    job_id = uuid4().hex[:12]
    trace_id = get_trace_id()

    # The capacity check and the enqueue that satisfies it must not be
    # separated, or concurrent requests each admit against the same pre-enqueue
    # depth. Measured at the assignment's benchmark concurrency (50 concurrent
    # POSTs of 20 pages, watermark 400): 130-150% overshoot, and one trial
    # admitted 1,000 pages against a 400 limit while shedding nothing.
    #
    # The lock spans all three steps so each check observes every prior
    # enqueue. It costs little: ingestion was already serialised by Redis's
    # single thread, so this makes the ordering explicit rather than adding
    # contention.
    verdict = await _admit_and_enqueue(
        request, job_id, req.pages, filename=req.filename or "", trace_id=trace_id
    )

    log.info(
        "job_accepted", job_id=job_id, pages=req.pages, queue_depth=verdict.depth
    )
    return {
        "job_id": job_id,
        "pages": req.pages,
        "status": "accepted",
        "queue_depth": verdict.depth,
        # Returned so a client (or the benchmark) can correlate its request with
        # every log line this job produces across all three services.
        "trace_id": trace_id,
    }


@router.post("/jobs/stream", status_code=202)
async def stream_job(request: Request) -> dict[str, Any]:
    """Ingest a PDF as a RAW request body. The memory-optimal path.

    Identical to /jobs/upload once the bytes have landed; the difference is how
    they arrive, and that difference dominates the API's memory profile.

    FastAPI parses a whole multipart body before the endpoint runs, so
    `UploadFile` means the request is already fully received - spooled to a temp
    file, through the parser's buffers - before the handler sees it. Consuming
    `request.stream()` instead takes chunks from the transport straight to disk.

    Measured at 50 concurrent 20 MiB uploads (1,000 MiB offered):

        endpoint                   api peak    total    per upload
        /jobs/upload (multipart)   268.2 MiB   425.1 MiB   5.36 MiB
        /jobs/stream (raw body)    125.7 MiB   298.5 MiB   2.51 MiB

    Both pass the 500 MB budget; only the raw path has headroom.

    Both are kept. Multipart is what a browser or `curl -F` sends and is the
    friendlier interface; the raw path is what a client streaming a large
    document should use, and is what the benchmark uses.
    """
    settings = get_settings()
    job_id = uuid4().hex[:12]
    trace_id = get_trace_id()
    dest = pdf_path(settings.data_dir, job_id)

    try:
        size = await stream_body_to_disk(
            request.stream(),
            dest,
            max_bytes=settings.max_upload_mb * 1024 * 1024,
        )
    except UploadTooLarge as exc:
        raise HTTPException(
            status_code=413, detail=f"upload exceeds {settings.max_upload_mb} MB"
        ) from exc

    return await _ingest_document(request, job_id, dest, size, "", trace_id)


@router.post("/jobs/upload", status_code=202)
async def upload_job(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Ingest a real PDF. The memory-bounded path.

    Order is forced by the memory budget, not by taste:

      1. stream to disk in bounded chunks - the file is never resident
      2. count pages from the xref - O(objects), not O(bytes)
      3. validate the page count
      4. admission + state + enqueue, under the lock, as for a synthetic job

    Counting has to happen before admission, because admission needs the page
    count and the only trustworthy source of it is the file itself - a declared
    count would let a client claim 1 page and enqueue 100.

    The cost of that ordering is that a shed job has already been written to
    disk. That is accepted: the alternative is trusting the client's count,
    which would make the watermark unenforceable. The file is deleted on the
    refusal path.
    """
    settings = get_settings()
    job_id = uuid4().hex[:12]
    trace_id = get_trace_id()
    dest = pdf_path(settings.data_dir, job_id)

    try:
        size = await stream_to_disk(
            file,
            dest,
            chunk_bytes=settings.upload_chunk_bytes,
            max_bytes=settings.max_upload_mb * 1024 * 1024,
        )
    except UploadTooLarge as exc:
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds {settings.max_upload_mb} MB",
        ) from exc

    return await _ingest_document(
        request, job_id, dest, size, file.filename or "", trace_id
    )


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict[str, Any]:
    store = _store(request)
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job")

    total = int(job["total_pages"])
    done = int(job.get("done_count", 0))
    states = await store.page_states(job_id, total)

    counts: dict[str, int] = {}
    for state in states:
        counts[state or "MISSING"] = counts.get(state or "MISSING", 0) + 1

    return {
        "job_id": job_id,
        "total_pages": total,
        # done_count is maintained by the same atomic script that performs each
        # terminal transition, so it cannot drift from the page states.
        "done": done,
        "complete": done >= total,
        "state_counts": counts,
        "filename": job.get("filename") or None,
    }


@router.get("/jobs/{job_id}/pages/{page_index}")
async def get_page(job_id: str, page_index: int, request: Request) -> dict[str, Any]:
    """Single page detail, including its stored model output."""
    page = await _store(request).get_page(job_id, page_index)
    if not page:
        raise HTTPException(status_code=404, detail="unknown page")
    return page


@router.get("/queue/depth")
async def queue_depth(request: Request) -> dict[str, Any]:
    """Backlog vs pending, plus where both sit against the watermark.

    Distinct signals: rising `backlog` means workers cannot keep up with
    ingestion; rising `pending` means workers are stalled or dying.
    """
    depth = await _queue(request).depth()
    admission = _admission(request)
    return {
        **depth,
        "high_watermark": admission.high_watermark,
        "low_watermark": admission.low_watermark,
    }


@router.get("/queue/consumers")
async def queue_consumers(request: Request) -> dict[str, Any]:
    """Per-lane consumer ownership, and the reaper's own view of what is stale.

    Exists to make the crash-recovery claim checkable rather than asserted: it
    shows the dead worker's pending entries under a consumer name that will
    never return, and shows them going to zero once reclaimed.
    """
    settings = get_settings()
    return {
        **await _queue(request).consumers(
            min_idle_ms=int(settings.reaper_min_idle_s * 1000)
        ),
        "reaper_enabled": settings.reaper_enabled,
        "reaper_interval_s": settings.reaper_interval_s,
    }


@router.get("/admission")
async def admission_stats(request: Request) -> dict[str, Any]:
    """Admitted versus shed, counted.

    "0% unhandled" is only a meaningful claim if every refusal is recorded, so
    these counters are the evidence for it: a shed job appears here and in the
    logs, never nowhere.
    """
    return await _admission(request).stats()
