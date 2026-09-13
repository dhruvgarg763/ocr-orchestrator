"""Generate a large, valid, multi-page PDF for the memory benchmark.

Two properties matter, and the first attempt at this script got both wrong.

Real per-page content, not blank pages: a blank 30-page PDF is 4 KB, which
would make a memory measurement meaningless.

UNCOMPRESSED, and random rather than repetitive: the first version used Flate
with a single repeated line, and zlib crushed a nominal 100 MiB down to 0.3 MiB.
Real scanned documents are dominated by incompressible image data, so an
uncompressed stream of random text is the more faithful stand-in and gives a
predictable target size.

The PDF is assembled by hand so the xref offsets are exact and the size is
controllable; pypdf's writer recompresses, which makes the size hard to target.
"""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path


def build(pages: int, kb_per_page: int, seed: int = 1234) -> bytes:
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + " "

    def text_line() -> bytes:
        body = "".join(rng.choice(alphabet) for _ in range(60))
        return f"({body}) Tj T*\n".encode()

    ops_per_page = max(1, (kb_per_page * 1024) // len(text_line()))

    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # object numbers are 1-based

    objects.append(b"")  # reserve 1 = Catalog
    objects.append(b"")  # reserve 2 = Pages
    font_num = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    kids: list[int] = []
    for _ in range(pages):
        content = (
            b"BT /F1 9 Tf 12 TL 40 750 Td\n"
            + b"".join(text_line() for _ in range(ops_per_page))
            + b"ET"
        )
        stream_num = add(
            b"<< /Length "
            + str(len(content)).encode()
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )
        kids.append(
            add(
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
                b" /Resources << /Font << /F1 "
                + str(font_num).encode()
                + b" 0 R >> >> /Contents "
                + str(stream_num).encode()
                + b" 0 R >>"
            )
        )

    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = (
        b"<< /Type /Pages /Count "
        + str(pages).encode()
        + b" /Kids ["
        + b" ".join(f"{k} 0 R".encode() for k in kids)
        + b"] >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=100)
    ap.add_argument("--kb-per-page", type=int, default=1024)
    ap.add_argument("--out", default="bench/large.pdf")
    args = ap.parse_args()

    data = build(args.pages, args.kb_per_page)
    Path(args.out).write_bytes(data)
    print(f"{args.out}: {len(data) / 1048576:.1f} MiB, {args.pages} pages")
