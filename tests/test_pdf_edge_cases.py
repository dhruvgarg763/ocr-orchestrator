"""PDF ingestion edge cases, and the bugs found by working through them.

Separate from test_pdf_splitter.py, which covers the memory bound. Everything
here is a boundary, a malformed input, or a race - and three of them were real
defects rather than hypotheticals:

  * a 0-page PDF was accepted, reported `complete: true` having done nothing,
    and leaked its upload on disk permanently
  * `pdf_path` composed a path from an unvalidated string, so
    `pdf_path("/data", "../../etc/passwd")` escaped the data directory
  * `/Rotate (ninety)` raised and failed an otherwise perfectly readable page
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.pdf.splitter import (
    InvalidPdf,
    count_pages,
    delete_document,
    extract_page,
    extract_page_async,
    orphan_candidates,
    pdf_path,
)


def build_pdf(pages: int, kb_per_page: int = 1) -> bytes:
    import sys

    sys.path.insert(0, ".")
    from bench.make_pdf import build

    return build(pages, kb_per_page)


# ------------------------------------------------------------ page-count edges


def test_a_zero_page_pdf_is_rejected(tmp_path: Path) -> None:
    """REGRESSION: an observed leak, not a hypothetical.

    A 0-page PDF used to be accepted with 202. The job was created with
    total_pages=0, immediately reported `complete: true` having done nothing,
    and enqueued no pages - so no worker ever reached the path that deletes the
    source document, and the upload leaked on disk permanently.

    Rejecting at the count is the fix: a document with no pages is not
    processable, and "instantly complete, zero work done" is not a success.
    """
    doc = tmp_path / "zero.pdf"
    doc.write_bytes(build_pdf(pages=0))

    with pytest.raises(InvalidPdf, match="no pages"):
        count_pages(doc)


def test_single_page_pdf_works(tmp_path: Path) -> None:
    """The other boundary. An off-by-one in page-tree handling shows up here."""
    doc = tmp_path / "one.pdf"
    doc.write_bytes(build_pdf(pages=1, kb_per_page=4))

    assert count_pages(doc) == 1
    assert extract_page(doc, 0, text_limit=0).width == pytest.approx(612.0)
    with pytest.raises(InvalidPdf):
        extract_page(doc, 1, text_limit=0)


def test_an_empty_file_is_a_client_error(tmp_path: Path) -> None:
    """A zero-byte body reaches the parser as a real file. 400, not 500."""
    doc = tmp_path / "empty.pdf"
    doc.write_bytes(b"")

    with pytest.raises(InvalidPdf):
        count_pages(doc)


def test_an_encrypted_pdf_is_refused_once_not_per_page(tmp_path: Path) -> None:
    """Detected at the count, in the API, rather than 100 times in the workers.

    Every page read would fail anyway, so checking at ingest turns a hundred
    confusing worker errors into one clear 400.
    """
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret")
    buf = io.BytesIO()
    writer.write(buf)

    doc = tmp_path / "enc.pdf"
    doc.write_bytes(buf.getvalue())

    with pytest.raises(InvalidPdf, match="encrypted"):
        count_pages(doc)


def test_a_document_above_the_page_limit_still_counts(tmp_path: Path) -> None:
    """Counting and validating are separate concerns.

    The splitter reports what the file contains; the API decides whether that
    is acceptable. Conflating them would leave no way to tell a client HOW
    oversized their document was.
    """
    doc = tmp_path / "150.pdf"
    doc.write_bytes(build_pdf(pages=150))

    assert count_pages(doc) == 150


# --------------------------------------------------------------- page geometry


def test_mediabox_inherited_from_the_pages_node(tmp_path: Path) -> None:
    """Legal, and extremely common: /MediaBox is an inheritable attribute.

    Plenty of real PDFs declare it once on the /Pages node instead of on every
    page. Reading it only from the page object would report wrong geometry - or
    fail outright - on a large share of real documents.
    """
    raw = build_pdf(pages=3)
    inherited = raw.replace(b" /MediaBox [0 0 612 792]", b"")
    inherited = inherited.replace(
        b"<< /Type /Pages /Count", b"<< /MediaBox [0 0 595 842] /Type /Pages /Count"
    )
    doc = tmp_path / "inherit.pdf"
    doc.write_bytes(inherited)

    content = extract_page(doc, 0, text_limit=0)

    assert content.width == pytest.approx(595.0), "A4 width, from the parent node"
    assert content.height == pytest.approx(842.0)


def test_a_malformed_rotation_does_not_cost_the_page(tmp_path: Path) -> None:
    """REGRESSION: `/Rotate (ninety)` used to fail a readable page.

    The spec says /Rotate is an integer multiple of 90; real files carry
    strings and junk. Geometry is what the layout stage actually needs, so a
    cosmetic field has to degrade to a default rather than lose the page - the
    same reasoning as the text-extraction fallback.
    """
    raw = build_pdf(pages=2).replace(
        b"/MediaBox [0 0 612 792]", b"/Rotate (ninety) /MediaBox [0 0 612 792]", 1
    )
    doc = tmp_path / "badrot.pdf"
    doc.write_bytes(raw)

    content = extract_page(doc, 0, text_limit=0)

    assert content.rotation == 0
    assert content.width == pytest.approx(612.0), "geometry must survive"


def test_negative_rotation_is_preserved(tmp_path: Path) -> None:
    """-90 is legal. Clamping it to 0 would silently mis-orient the page."""
    raw = build_pdf(pages=2).replace(
        b"/MediaBox [0 0 612 792]", b"/Rotate -90 /MediaBox [0 0 612 792]", 1
    )
    doc = tmp_path / "negrot.pdf"
    doc.write_bytes(raw)

    assert extract_page(doc, 0, text_limit=0).rotation == -90


def test_a_degenerate_mediabox_is_not_an_error(tmp_path: Path) -> None:
    """A zero-area page is odd but structurally valid.

    Rejecting it would fail a whole document over one strange page, and the
    layout endpoint is perfectly capable of returning nothing for it.
    """
    raw = build_pdf(pages=2).replace(
        b"/MediaBox [0 0 612 792]", b"/MediaBox [0 0 0 0]", 1
    )
    doc = tmp_path / "zerobox.pdf"
    doc.write_bytes(raw)

    assert extract_page(doc, 0, text_limit=0).width == pytest.approx(0.0)


# ------------------------------------------------------------------- races


def test_a_document_removed_mid_flight_is_not_fatal(tmp_path: Path) -> None:
    """The worker races its own cleanup.

    `_resolve_page` checks existence and then extracts, and between those steps
    another worker may have completed the job and deleted the document. The
    extraction has to raise something the caller can treat as "no descriptor",
    not something that fails the page.
    """
    doc = tmp_path / "gone.pdf"
    doc.write_bytes(build_pdf(pages=2))
    doc.unlink()

    with pytest.raises(InvalidPdf):
        extract_page(doc, 0, text_limit=0)


def test_concurrent_extraction_of_one_document_is_consistent(
    tmp_path: Path,
) -> None:
    """Each extraction opens its own handle and its own reader, so there is no
    shared file cursor for concurrent readers to corrupt."""
    doc = tmp_path / "conc.pdf"
    doc.write_bytes(build_pdf(pages=8, kb_per_page=8))

    async def run() -> list:
        return await asyncio.gather(
            *(extract_page_async(doc, i % 8, text_limit=0) for i in range(24))
        )

    results = asyncio.run(run())

    assert len(results) == 24
    assert len({r.width for r in results}) == 1, "inconsistent geometry under load"


# --------------------------------------------------------- traversal safety


@pytest.mark.parametrize(
    "job_id",
    ["../../etc/passwd", "a/b", "..", ".", "", "x" * 100, "a\\b", "/abs", "a b"],
)
def test_unsafe_job_ids_cannot_name_a_path(job_id: str) -> None:
    """REGRESSION. `pdf_path` builds a filesystem path from a string, which
    makes it a traversal sink.

    Job ids come from `uuid4().hex` today, but "the caller happens to pass safe
    input" is not a boundary - `pdf_path("/data", "../../etc/passwd")` returned
    `/data/../../etc/passwd.pdf` before this check existed.
    """
    with pytest.raises(ValueError, match="unsafe job id"):
        pdf_path("/data", job_id)


@pytest.mark.parametrize("job_id", ["0134d956a2cb", "worker-job", "pipe_job", "A1"])
def test_ordinary_job_ids_are_accepted(job_id: str) -> None:
    """The guard was hex-only at first, which was too strict.

    It rejected readable ids, and because `_resolve_page` let the ValueError
    escape, every page of such a job crashed on a path question that has
    nothing to do with whether the page can be processed. Safety comes from the
    character class - no separators, no dots - not from the alphabet.
    """
    assert pdf_path("/data", job_id).name == f"{job_id}.pdf"


def test_delete_document_tolerates_an_unsafe_id() -> None:
    """An id that cannot name a file cannot have one to delete, so this returns
    False rather than raising. Cleanup must never be the thing that fails."""
    assert delete_document("/data", "../nope") is False


# ------------------------------------------------------------- orphan sweep


def test_orphan_sweep_ignores_recent_documents(tmp_path: Path) -> None:
    """Age-gated, so the sweeper cannot race a job that is still ingesting."""
    (tmp_path / "abc123.pdf").write_bytes(b"%PDF-1.4\n")

    assert orphan_candidates(str(tmp_path), min_age_s=3600) == []


def test_orphan_sweep_finds_old_documents(tmp_path: Path) -> None:
    """The backstop for a leak the happy path cannot cover.

    Deleting on completion only fires when a worker acks the LAST page of a
    job. A page stranded in a dead worker's pending list, a kill between the
    final ack and the delete, or Redis state expiring mid-flight all leave the
    upload behind - and a file has no TTL of its own.
    """
    (tmp_path / "abc123.pdf").write_bytes(b"%PDF-1.4\n")

    assert orphan_candidates(str(tmp_path), min_age_s=0) == ["abc123"]


def test_orphan_sweep_only_considers_names_it_recognises(tmp_path: Path) -> None:
    """The sweeper deletes files, so it must never consider a name it does not
    recognise as a job document."""
    (tmp_path / "abc123.pdf").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / "..weird.pdf").write_bytes(b"x")

    assert orphan_candidates(str(tmp_path), min_age_s=0) == ["abc123"]


def test_orphan_sweep_on_a_missing_directory_is_empty(tmp_path: Path) -> None:
    """A worker that starts before the volume is mounted must not crash."""
    assert orphan_candidates(str(tmp_path / "nope"), min_age_s=0) == []
