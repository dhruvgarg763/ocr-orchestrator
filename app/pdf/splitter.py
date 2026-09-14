"""Bounded PDF ingestion: O(page) resident memory, never O(file).

Three decisions remove the O(file) term: uploads stream to disk in 1 MiB
chunks (never a size-less `read()`), page count comes from the xref/page
tree only (no content-stream parsing), and each page is opened, extracted
and closed independently rather than pre-split - pypdf's object cache
would otherwise accumulate across the whole document.

`PdfReader` MUST receive an open file handle, not a path: passing a path
makes pypdf read the entire file into memory internally
(`_initialize_stream`), measured at +100.4 MiB on a 100 MiB file vs +0.0
MiB for an open handle. The task queue carries a (job_id, page_index)
reference, never page bytes, so a queue entry's size is independent of the
document's - which is what keeps admission control's memory arithmetic
honest. The mock receives a page descriptor (dimensions, a bounded text
sample), not a rendered image; the production shape is a blob-store URI,
same O(page) profile.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import anyio
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from common.logging import get_logger

log = get_logger("pdf")

PAGE_REF_PREFIX = "pdf"


class UploadLike(Protocol):
    """Just the slice of Starlette's UploadFile this module needs.

    Narrow on purpose: it keeps the streaming logic testable without
    constructing a real multipart request.
    """

    filename: str | None

    async def read(self, size: int = -1) -> bytes: ...


class UploadTooLarge(Exception):
    """Raised mid-stream once the byte budget is exceeded.

    Mid-stream matters. Checking Content-Length is advisory (a client can lie or
    omit it under chunked encoding), and checking the size after writing means a
    10 GB upload has already filled the disk before being refused.
    """

    def __init__(self, limit_bytes: int) -> None:
        super().__init__(f"upload exceeds {limit_bytes} bytes")
        self.limit_bytes = limit_bytes


class InvalidPdf(Exception):
    """Not a readable PDF, or encrypted. A client error, not a server fault."""


@dataclass(frozen=True)
class PageContent:
    """What one page contributes to a model request.

    Deliberately small and bounded. `text` is truncated because
    `extract_text()` on a dense table page can return hundreds of kilobytes,
    and with 48 pages in flight that would be a per-page cost multiplied by
    concurrency - the exact shape of leak this step exists to remove.
    """

    page_index: int
    width: float
    height: float
    rotation: int
    text: str
    text_chars: int
    """Length BEFORE truncation, so the sample can be recognised as a sample."""

    def as_payload(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "rotation": self.rotation,
            "text_sample": self.text,
            "text_chars": self.text_chars,
        }


def page_ref(job_id: str, page_index: int) -> str:
    """Opaque handle for "page n of this job's document".

    A reference rather than content, so the queue entry's size does not depend
    on the page's.
    """
    return f"{PAGE_REF_PREFIX}:{job_id}#{page_index}"


_SAFE_JOB_ID = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")


def pdf_path(data_dir: str, job_id: str) -> Path:
    """Resolve a job's document path, rejecting anything that could escape.

    Job ids are generated from `uuid4().hex`, so in practice they are always
    hex - but "the caller happens to pass safe input" is not a boundary. This
    function composes a filesystem path from a string, which makes it a
    traversal sink, and `pdf_path("/data", "../../etc/passwd")` cheerfully
    returned `/data/../../etc/passwd.pdf` before this check existed.

    Validated rather than sanitised: a job id is a known-shape token, so an
    unexpected one is a bug to surface, not a string to repair.

    The character class is what makes it safe, not the length: no separators
    and no dots means nothing can escape `data_dir`. It was hex-only at first,
    which was too strict - readable ids are legitimate, and rejecting them made
    every page of such a job crash on a path question that has nothing to do
    with whether the page can be processed.
    """
    if not _SAFE_JOB_ID.match(job_id):
        raise ValueError(f"unsafe job id: {job_id!r}")
    return Path(data_dir) / f"{job_id}.pdf"


async def stream_to_disk(
    upload: UploadLike,
    dest: Path,
    *,
    chunk_bytes: int,
    max_bytes: int,
) -> int:
    """Copy an upload to `dest` a chunk at a time. Returns bytes written.

    Never calls `read()` without a size. That single difference is what makes
    resident memory O(chunk) instead of O(file): `read()` with no argument
    un-spools Starlette's temporary file into one contiguous bytes object.

    The write happens on a worker thread. A 1 MiB write into page cache is
    fast, but it is still a blocking syscall on the event loop, and with 50
    concurrent uploads those add up against the 200ms time-to-first-page
    budget - a request cannot be served while the loop is blocked.

    A partial file is removed on failure. Leaving it behind would let a
    repeatedly-aborted upload fill the disk with unreferenced fragments, and a
    later reader would see a truncated PDF rather than a missing one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    try:
        async with await anyio.open_file(dest, "wb") as out:
            while True:
                chunk = await upload.read(chunk_bytes)
                if not chunk:
                    break

                written += len(chunk)
                if written > max_bytes:
                    # Refuse mid-stream, before the bytes reach the disk.
                    raise UploadTooLarge(max_bytes)

                await out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise

    log.info("upload_stored", dest=str(dest), bytes=written, chunk_bytes=chunk_bytes)
    return written


async def stream_body_to_disk(chunks: Any, dest: Path, *, max_bytes: int) -> int:
    """Write an async iterator of body chunks to `dest`. Returns bytes written.

    The difference from `stream_to_disk` is WHERE the buffering happens, and it
    turns out to be the dominant term in the API's memory profile.

    FastAPI parses a whole multipart body BEFORE the endpoint function runs, so
    with `UploadFile` the request is already fully received - spooled to a
    temporary file, through the multipart parser's own buffers - and the chunked
    read here is a second copy out of that temp file. Chunking still prevents
    materialising the document as one contiguous bytes object (measured: 7.7 MiB
    of growth for a single 100 MiB upload), but the per-request overhead of the
    parser belongs to Starlette, not to us.

    Consuming `request.stream()` skips the parser entirely: chunks arrive from
    the transport and go straight to disk. Measured at 50 concurrent 20 MiB
    uploads (1,000 MiB offered):

        endpoint                     api peak   per in-flight upload
        /jobs/upload  (multipart)    268.2 MiB          5.36 MiB
        /jobs/stream  (raw body)     125.7 MiB          2.51 MiB

    Total across all containers went from 425.1 MiB to 298.5 MiB against the
    500 MB budget - both pass, but only one has headroom. The residual 2.51 MiB
    is transport and uvicorn receive buffering, not this module.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    try:
        async with await anyio.open_file(dest, "wb") as out:
            async for chunk in chunks:
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    # Mid-stream, before the bytes reach the disk.
                    raise UploadTooLarge(max_bytes)
                await out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise

    log.info("upload_streamed", dest=str(dest), bytes=written)
    return written


def count_pages(path: Path) -> int:
    """Page count from the xref and page tree alone.

    An OPEN FILE OBJECT, never a path - see the module docstring. With a handle
    pypdf seeks to the trailer, reads the cross-reference table and resolves the
    page tree lazily; measured growth on a 100 MiB file is 0.0 MiB against
    100.4 MiB for the path form.

    The handle must stay open for the reader's whole lifetime, since the reader
    seeks against it on demand rather than holding a copy.

    strict=False so a slightly malformed but readable file is accepted; real
    PDFs violate the spec constantly, and refusing them would be a worse
    failure than tolerating them.
    """
    try:
        with open(path, "rb") as handle:
            reader = PdfReader(handle, strict=False)
            if reader.is_encrypted:
                # Every subsequent page read would fail, so fail once, clearly,
                # rather than 100 times inside the workers.
                raise InvalidPdf("encrypted PDFs are not supported")

            pages = len(reader.pages)
            if pages < 1:
                # A zero-page document is not processable, and accepting one
                # was an observed bug rather than a hypothetical: the job was
                # admitted with total_pages=0, immediately reported
                # `complete: true` having done nothing, enqueued no pages, and
                # therefore never reached the worker path that deletes the
                # source file - so the upload leaked on disk permanently.
                raise InvalidPdf("PDF contains no pages")
            return pages
    except InvalidPdf:
        raise
    except (PdfReadError, OSError, ValueError, KeyError) as exc:
        raise InvalidPdf(f"unreadable PDF: {type(exc).__name__}: {exc}") from exc


def extract_page(path: Path, page_index: int, *, text_limit: int) -> PageContent:
    """Open, take exactly one page, close.

    The reader is local, so pypdf's object cache is freed on return. Holding
    one reader across many extractions would let that cache accumulate the
    whole document, which is the O(file) term this step removes.

    Called once per page per attempt, so it re-reads the xref each time. That
    is the deliberate CPU-for-memory trade described in the module docstring.

    `text_limit=0` skips text extraction entirely. Truncating the OUTPUT does
    not reduce the cost - extract_text has to walk the page's whole content
    stream to produce any text at all - so the knob has to be able to turn the
    work off, not just shrink the result.
    """
    try:
        with open(path, "rb") as handle:
            reader = PdfReader(handle, strict=False)
            if page_index < 0 or page_index >= len(reader.pages):
                raise InvalidPdf(f"page {page_index} out of range")

            page = reader.pages[page_index]
            box = page.mediabox
            width, height = float(box.width), float(box.height)

            try:
                rotation = int(page.get("/Rotate", 0) or 0)
            except (TypeError, ValueError):
                # A malformed /Rotate must not cost us the page. The spec says
                # it is an integer multiple of 90, but real files carry strings
                # and junk, and geometry is what the layout stage actually
                # needs - `/Rotate (ninety)` used to raise here and fail an
                # otherwise perfectly readable page.
                rotation = 0

            text = ""
            if text_limit > 0:
                try:
                    text = page.extract_text() or ""
                except Exception:  # noqa: BLE001
                    # A page whose content stream cannot be decoded still has
                    # valid geometry, and the layout stage only needs geometry.
                    # Losing the text sample beats failing the page.
                    text = ""

        return PageContent(
            page_index=page_index,
            width=width,
            height=height,
            rotation=rotation,
            # Truncated deliberately: see PageContent.
            text=text[:text_limit],
            text_chars=len(text),
        )
    except InvalidPdf:
        raise
    except (PdfReadError, OSError, ValueError, KeyError) as exc:
        raise InvalidPdf(f"cannot read page {page_index}: {exc}") from exc


async def extract_page_async(
    path: Path, page_index: int, *, text_limit: int
) -> PageContent:
    """`extract_page` on a worker thread.

    pypdf is synchronous and CPU-bound; running it inline would block the event
    loop for the whole parse, stalling every other page in flight in the same
    worker. With worker_concurrency pages per process that is the difference
    between concurrent extraction and serial extraction.
    """
    return await anyio.to_thread.run_sync(
        lambda: extract_page(path, page_index, text_limit=text_limit)
    )


def orphan_candidates(data_dir: str, *, min_age_s: float) -> list[str]:
    """Job ids whose document is older than `min_age_s`.

    The backstop for a leak the happy path cannot cover. Deleting on completion
    only fires when a worker acknowledges the LAST page of a job, so anything
    that stops a job completing leaves its upload behind - a page stuck in a
    dead worker's pending list, a job whose Redis state expired mid-flight, or
    a process killed between the final ack and the delete.

    Age-gated so a document is never considered while its job could still be
    running: the caller passes a threshold safely beyond the page deadline.
    Returning candidates rather than deleting them keeps the Redis liveness
    check - "does this job still exist?" - with the caller that owns the
    connection.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        return []

    now = time.time()
    candidates: list[str] = []
    for entry in directory.glob("*.pdf"):
        try:
            if now - entry.stat().st_mtime < min_age_s:
                continue
        except FileNotFoundError:
            # Raced with a completion delete. Nothing to do - that is the
            # outcome this function wanted anyway.
            continue
        if _SAFE_JOB_ID.match(entry.stem):
            candidates.append(entry.stem)
    return candidates


def delete_document(data_dir: str, job_id: str) -> bool:
    """Remove a job's source file once every page is terminal.

    Uploads are the one unbounded resource left: Redis state expires by TTL,
    but a file does not. Without this, disk grows monotonically with every job
    ever ingested.
    """
    try:
        path = pdf_path(data_dir, job_id)
    except ValueError:
        # An id that cannot name a file cannot have one to delete.
        return False

    try:
        os.remove(path)
        log.info("document_deleted", job_id=job_id)
        return True
    except FileNotFoundError:
        return False
