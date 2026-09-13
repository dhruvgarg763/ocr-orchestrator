"""PDF ingestion tests: correctness, and the memory bound itself.

The memory tests measure RSS rather than asserting on code shape, because the
thing under test is a resident-bytes property and the first implementation of
this module passed every structural test while using MORE memory than the naive
version it replaced. Only a measurement caught that.

The decisive test is `test_reader_must_be_given_a_handle_not_a_path`: pypdf
slurps the whole file when handed a path, which is both the idiomatic-looking
form and a 100 MiB regression waiting to happen.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import psutil
import pytest

from app.pdf.splitter import (
    InvalidPdf,
    UploadTooLarge,
    count_pages,
    delete_document,
    extract_page,
    extract_page_async,
    page_ref,
    pdf_path,
    stream_body_to_disk,
    stream_to_disk,
)

# Must stay well under the 100 MiB benchmark file: these run on every commit.
PAGES = 12
KB_PER_PAGE = 256  # -> ~3 MiB, big enough that slurping is measurable


def build_pdf(pages: int = PAGES, kb_per_page: int = KB_PER_PAGE) -> bytes:
    """Reuse the benchmark generator so tests and benchmark agree on shape."""
    import sys

    sys.path.insert(0, ".")
    from bench.make_pdf import build

    return build(pages, kb_per_page)


@pytest.fixture(scope="module")
def pdf_bytes() -> bytes:
    return build_pdf()


@pytest.fixture
def pdf_file(pdf_bytes: bytes, tmp_path: Path) -> Path:
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf_bytes)
    return path


class FakeUpload:
    """Starlette's UploadFile, reduced to the two members that matter.

    `read()` refuses to serve an unbounded request, which is the whole point:
    a call with no size is the O(file) bug this module exists to avoid, and a
    double that silently allowed it would let the bug back in.
    """

    def __init__(self, data: bytes, filename: str = "doc.pdf") -> None:
        self.filename = filename
        self._data = data
        self._pos = 0
        self.reads: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise AssertionError(
                "read() with no size makes resident memory O(file); "
                "stream_to_disk must always pass a bound"
            )
        self.reads.append(size)
        chunk = self._data[self._pos : self._pos + size]
        self._pos += size
        return chunk


def rss_mib() -> float:
    return psutil.Process().memory_info().rss / 1048576


# --------------------------------------------------------------- upload path


async def test_upload_is_written_in_bounded_chunks(
    pdf_bytes: bytes, tmp_path: Path
) -> None:
    """Never `read()` without a size. That one difference is the memory bound.

    `read()` with no argument un-spools Starlette's temporary file into a
    single contiguous bytes object - the whole document resident. The fake
    upload raises if asked, so this asserts the property rather than trusting
    the implementation.
    """
    dest = tmp_path / "out.pdf"
    upload = FakeUpload(pdf_bytes)

    written = await stream_to_disk(
        upload, dest, chunk_bytes=64 * 1024, max_bytes=50 * 1024 * 1024
    )

    assert written == len(pdf_bytes)
    assert dest.read_bytes() == pdf_bytes
    assert all(size == 64 * 1024 for size in upload.reads), "chunk size must be honoured"
    assert len(upload.reads) > 10, "a multi-MiB file should take many reads"


async def test_oversized_upload_is_refused_mid_stream(tmp_path: Path) -> None:
    """Refused while streaming, not after.

    Content-Length is advisory - a client can omit it under chunked encoding or
    simply lie - so checking it is not a bound. Checking the size after writing
    means a 10 GB upload has already filled the disk.
    """
    dest = tmp_path / "big.pdf"
    limit = 256 * 1024
    upload = FakeUpload(b"x" * (4 * 1024 * 1024))

    with pytest.raises(UploadTooLarge):
        await stream_to_disk(
            upload, dest, chunk_bytes=64 * 1024, max_bytes=limit
        )

    # Bytes read stops shortly after the limit, rather than consuming all 4 MiB.
    assert sum(upload.reads) <= limit + 64 * 1024 * 2, "kept reading past the limit"


async def test_a_refused_upload_leaves_no_partial_file(tmp_path: Path) -> None:
    """A retried-and-refused upload must not accumulate fragments on disk.

    Worse than the disk cost: a later reader would find a truncated PDF rather
    than a missing one, which is a much more confusing failure.
    """
    dest = tmp_path / "partial.pdf"

    with pytest.raises(UploadTooLarge):
        await stream_to_disk(
            FakeUpload(b"x" * (2 * 1024 * 1024)),
            dest,
            chunk_bytes=64 * 1024,
            max_bytes=128 * 1024,
        )

    assert not dest.exists()


async def test_upload_memory_is_independent_of_file_size(tmp_path: Path) -> None:
    """The bound that the RSS metric actually rests on.

    Resident growth should track the CHUNK, not the document, so a 100 MiB
    upload costs the same as a 1 MiB one.
    """
    big = build_pdf(pages=4, kb_per_page=2048)  # ~8 MiB
    gc.collect()
    before = rss_mib()

    await stream_to_disk(
        FakeUpload(big),
        tmp_path / "big.pdf",
        chunk_bytes=1024 * 1024,
        max_bytes=100 * 1024 * 1024,
    )

    growth = rss_mib() - before
    size_mib = len(big) / 1048576
    assert growth < size_mib / 2, (
        f"grew {growth:.1f} MiB for a {size_mib:.1f} MiB upload - "
        "resident memory is tracking the file, not the chunk"
    )


# ----------------------------------------------------------------- counting


def test_page_count_comes_from_the_xref(pdf_file: Path) -> None:
    assert count_pages(pdf_file) == PAGES


def test_reader_must_be_given_a_handle_not_a_path(pdf_file: Path) -> None:
    """REGRESSION, and the most important test in this file.

    pypdf's `_initialize_stream` does this when handed a path or a Path (still
    true as of pypdf 6.18.1, re-checked by reading the source):

        with open(stream, "rb") as fh:
            stream = BytesIO(fh.read())        # the ENTIRE file, resident

    So `PdfReader(str(path))` costs O(file) while `PdfReader(open(path,'rb'))`
    costs nothing. Measured on the 100 MiB benchmark document: +100.4 MiB
    versus +0.0 MiB.

    The first version of this module used the path form. Because extraction is
    per-page it slurped the file once per page, and measured WORSE than the
    naive read-everything implementation it replaced - 420 MiB peak against
    301 MiB. The path form is also what every pypdf example shows, which is
    what makes it worth pinning: nothing about it looks wrong.

    Why tracemalloc here, when the rest of this file measures RSS
    ------------------------------------------------------------
    This test previously measured RSS and asserted the path form grew by at
    least half the file size. It never ran - `psutil` was missing from
    requirements.txt, so the whole module was a collection error that went
    unnoticed - and the first time it did run, it FAILED, reporting 0.0 MiB of
    growth for the path form.

    The claim was not stale; the instrument was too coarse. RSS is what the
    process has obtained from the kernel, not what Python has allocated, so a
    3 MiB BytesIO is satisfied from arenas the preceding parse had already
    freed and RSS never moves. The original +100.4 MiB figure was taken on a
    100 MiB file, where the allocation dwarfs any arena the process could be
    holding - which is exactly why it was visible there and invisible here.

    tracemalloc counts Python-level allocation directly and is immune to
    allocator reuse, so it resolves the same property at a file size small
    enough to run on every commit: measured 0.06 MiB for the handle form
    against 3.05 MiB for the path form - the file size, to two decimal places.
    RSS stays the instrument for the end-to-end per-page bounds below, where
    resident bytes are the actual property under test.
    """
    import tracemalloc

    from pypdf import PdfReader

    size_mib = pdf_file.stat().st_size / 1048576
    assert size_mib > 2, "file must be large enough for slurping to be visible"

    gc.collect()
    tracemalloc.start()
    with open(pdf_file, "rb") as handle:
        reader = PdfReader(handle, strict=False)
        assert len(reader.pages) == PAGES
        handle_peak = tracemalloc.get_traced_memory()[1] / 1048576
    tracemalloc.stop()
    del reader
    gc.collect()

    tracemalloc.start()
    reader = PdfReader(str(pdf_file), strict=False)
    assert len(reader.pages) == PAGES
    path_peak = tracemalloc.get_traced_memory()[1] / 1048576
    tracemalloc.stop()
    del reader
    gc.collect()

    assert path_peak > size_mib * 0.8, (
        f"expected the path form to allocate ~{size_mib:.1f} MiB, saw "
        f"{path_peak:.2f} MiB - if pypdf stopped slurping, the docstring above "
        "and app/pdf/splitter.py's rationale both need revisiting"
    )
    assert handle_peak < size_mib * 0.2, (
        f"handle form allocated {handle_peak:.2f} MiB against a "
        f"{size_mib:.1f} MiB file - it should be O(1), not O(file)"
    )

def test_counting_does_not_hold_the_document_resident(pdf_file: Path) -> None:
    """A page COUNT needs the xref and page tree, not the content streams."""
    size_mib = pdf_file.stat().st_size / 1048576
    gc.collect()
    before = rss_mib()

    for _ in range(5):
        count_pages(pdf_file)

    growth = rss_mib() - before
    assert growth < size_mib / 2, f"grew {growth:.1f} MiB over 5 counts"


def test_a_non_pdf_is_a_client_error(tmp_path: Path) -> None:
    """400, not 500. Garbage in a request body is the caller's mistake."""
    junk = tmp_path / "not.pdf"
    junk.write_bytes(b"this is plainly not a PDF" * 100)

    with pytest.raises(InvalidPdf):
        count_pages(junk)


def test_a_truncated_pdf_is_rejected_rather_than_half_read(
    pdf_bytes: bytes, tmp_path: Path
) -> None:
    """A file cut off mid-write has no valid trailer, so the xref cannot be
    found. Better to refuse the job than to enqueue pages that do not exist."""
    truncated = tmp_path / "cut.pdf"
    truncated.write_bytes(pdf_bytes[: len(pdf_bytes) // 3])

    with pytest.raises(InvalidPdf):
        count_pages(truncated)


# --------------------------------------------------------------- extraction


def test_extract_page_returns_real_geometry(pdf_file: Path) -> None:
    content = extract_page(pdf_file, 3, text_limit=0)

    assert content.page_index == 3
    assert content.width == pytest.approx(612.0)
    assert content.height == pytest.approx(792.0)
    assert content.rotation == 0


def test_extract_page_rejects_an_out_of_range_index(pdf_file: Path) -> None:
    with pytest.raises(InvalidPdf):
        extract_page(pdf_file, PAGES + 5, text_limit=0)
    with pytest.raises(InvalidPdf):
        extract_page(pdf_file, -1, text_limit=0)


def test_text_sample_is_truncated_but_the_true_length_is_kept(
    pdf_file: Path,
) -> None:
    """The sample must be recognisable AS a sample.

    Reporting only the truncated length would make a 900,000-character page
    indistinguishable from a 2,048-character one, and any downstream consumer
    would silently treat the fragment as the whole page.
    """
    content = extract_page(pdf_file, 0, text_limit=512)

    assert len(content.text) <= 512
    assert content.text_chars > 512, "this fixture's pages carry far more text"
    assert content.as_payload()["text_chars"] == content.text_chars


def test_text_extraction_can_be_disabled(pdf_file: Path) -> None:
    """`text_limit=0` must skip the WORK, not just shrink the result.

    Truncating the output saves nothing: `extract_text()` has to walk the whole
    content stream to produce any text at all. Measured per page on a dense
    document, 3038 ms with text against 26 ms without - slower than the VLM it
    feeds, which would move the bottleneck onto our own CPU.
    """
    with_text = time.monotonic()
    extract_page(pdf_file, 0, text_limit=2048)
    with_text = time.monotonic() - with_text

    without = time.monotonic()
    content = extract_page(pdf_file, 0, text_limit=0)
    without = time.monotonic() - without

    assert content.text == ""
    assert content.text_chars == 0
    assert without < with_text, (
        f"disabling text saved nothing ({without * 1000:.1f}ms vs "
        f"{with_text * 1000:.1f}ms) - the flag is shrinking output, not "
        "skipping work"
    )


def test_repeated_extraction_does_not_accumulate(pdf_file: Path) -> None:
    """The reason extraction is per-page rather than pre-split.

    Holding one reader across many pages lets pypdf's `resolved_objects` cache
    accumulate the whole document, which is the O(file) term this module
    exists to remove. A local reader per call is freed on return, so resident
    memory stays flat however many pages are processed.
    """
    size_mib = pdf_file.stat().st_size / 1048576
    gc.collect()
    before = rss_mib()

    for _ in range(3):
        for index in range(PAGES):
            extract_page(pdf_file, index, text_limit=0)

    growth = rss_mib() - before
    assert growth < size_mib, (
        f"grew {growth:.1f} MiB over {3 * PAGES} extractions of a "
        f"{size_mib:.1f} MiB file"
    )


async def test_async_extraction_does_not_block_the_event_loop(
    pdf_file: Path,
) -> None:
    """pypdf is synchronous and CPU-bound.

    Run inline it would hold the loop for the whole parse, stalling every other
    page in flight in the same worker - the difference between concurrent and
    serial extraction at worker_concurrency pages each.
    """
    import asyncio

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.001)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    await asyncio.gather(
        *(extract_page_async(pdf_file, i, text_limit=2048) for i in range(PAGES))
    )
    beat.cancel()

    assert ticks > 0, "the event loop never ran during extraction"


# ---------------------------------------------------------------- lifecycle


def test_page_ref_is_a_reference_not_content(pdf_file: Path) -> None:
    """Fixed-size, so the queue entry does not grow with the document."""
    ref = page_ref("abc123def456", 7)

    assert ref == "pdf:abc123def456#7"
    assert len(ref) < 64


def test_document_is_deleted_when_no_longer_needed(
    pdf_bytes: bytes, tmp_path: Path
) -> None:
    """Uploads are the one resource with no TTL.

    Redis state expires on its own; a file does not. Without an explicit
    delete, disk grows monotonically with every job ever ingested - a slow leak
    that only shows up in production.
    """
    job_id = "deadbeef"
    path = pdf_path(str(tmp_path), job_id)
    path.write_bytes(pdf_bytes)

    assert delete_document(str(tmp_path), job_id) is True
    assert not path.exists()
    assert delete_document(str(tmp_path), job_id) is False, "must be idempotent"


# ------------------------------------------------------- raw body streaming


async def test_raw_body_stream_writes_every_chunk(
    pdf_bytes: bytes, tmp_path: Path
) -> None:
    """The memory-optimal ingestion path.

    FastAPI parses a whole multipart body before the endpoint runs, so
    `UploadFile` means the request is already fully received - spooled to a temp
    file, through the parser's buffers - before the handler sees it. Consuming
    `request.stream()` takes chunks from the transport straight to disk.

    Measured at 50 concurrent 20 MiB uploads: api peak 268.2 MiB for multipart
    against 125.7 MiB for raw body, and 425.1 MiB versus 298.5 MiB across all
    containers.
    """

    async def chunks():
        for start in range(0, len(pdf_bytes), 8192):
            yield pdf_bytes[start : start + 8192]

    dest = tmp_path / "streamed.pdf"
    written = await stream_body_to_disk(chunks(), dest, max_bytes=50 * 1024 * 1024)

    assert written == len(pdf_bytes)
    assert dest.read_bytes() == pdf_bytes
    assert count_pages(dest) == PAGES


async def test_raw_body_stream_refuses_mid_stream(tmp_path: Path) -> None:
    """Same mid-stream bound as the multipart path: a client can lie about
    Content-Length or omit it entirely under chunked encoding."""
    consumed = 0

    async def chunks():
        nonlocal consumed
        for _ in range(500):
            consumed += 64 * 1024
            yield b"x" * (64 * 1024)

    dest = tmp_path / "big.pdf"
    with pytest.raises(UploadTooLarge):
        await stream_body_to_disk(chunks(), dest, max_bytes=256 * 1024)

    assert not dest.exists(), "partial file left behind"
    assert consumed < 500 * 64 * 1024, "kept consuming the body past the limit"


async def test_raw_body_stream_tolerates_empty_chunks(tmp_path: Path) -> None:
    """Transports legitimately deliver zero-length chunks; treating one as
    end-of-body would silently truncate the document."""

    async def chunks():
        yield b"%PDF-1.4\n"
        yield b""
        yield b"trailing"

    dest = tmp_path / "sparse.bin"
    written = await stream_body_to_disk(chunks(), dest, max_bytes=1024)

    assert written == len(b"%PDF-1.4\ntrailing")
