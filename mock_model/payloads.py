"""Deterministic synthetic model output.

Determinism is a hard requirement, not a nicety:

1. Idempotency proof (Module D). If a page is redelivered after a crash, the
   replayed response must be byte-identical to the original, otherwise you
   cannot tell "safely retried" from "silently produced different output".
2. Evaluation (Module C). CER/WER/IoU/TED need a stable prediction to compare
   against a stable ground truth. A random mock makes every metric noise.

Seeding uses hashlib, NOT Python's built-in hash(). `hash()` on str is salted
per process (PYTHONHASHSEED), so it returns different values after every
restart - which would break determinism in exactly the crash-recovery scenario
we need it for.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

PAGE_WIDTH = 612.0   # US Letter at 72 dpi
PAGE_HEIGHT = 792.0

_WORDS = (
    "invoice total amount due date vendor address quantity unit price tax "
    "subtotal discount payment terms net remittance purchase order item "
    "description reference account currency balance credit debit"
).split()


def _rng(job_id: str, page_index: int, salt: str) -> random.Random:
    digest = hashlib.sha256(f"{job_id}:{page_index}:{salt}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _sentence(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(words))


def layout_payload(job_id: str, page_index: int) -> dict[str, Any]:
    """Fast-Layout-Model: spatial blocks + reading order, no text content.

    This is the fallback output too, so it must be independently useful: when the
    VLM is exhausted or its breaker is open, a page degrades to *this* plus a low
    confidence flag rather than failing.
    """
    rng = _rng(job_id, page_index, "layout")
    n_blocks = rng.randint(4, 9)

    boxes: list[dict[str, Any]] = []
    y = 40.0
    for i in range(n_blocks):
        height = rng.uniform(30, 90)
        if y + height > PAGE_HEIGHT - 40:
            break
        boxes.append(
            {
                "id": i,
                "type": rng.choice(["title", "paragraph", "paragraph", "table", "figure"]),
                # x/y/w/h as the assignment specifies, not x1/y1/x2/y2.
                "x": round(rng.uniform(40, 80), 2),
                "y": round(y, 2),
                "w": round(rng.uniform(400, 500), 2),
                "h": round(height, 2),
                "confidence": round(rng.uniform(0.82, 0.99), 3),
            }
        )
        y += height + rng.uniform(8, 20)

    return {
        "model": "fast-layout",
        "page_index": page_index,
        "boxes": boxes,
        # Reading order is a permutation of box ids, not their spatial order:
        # multi-column documents genuinely read out of spatial sequence, and
        # Module C's tree comparison has to respect that.
        "reading_order": [b["id"] for b in boxes],
        "page_size": {"w": PAGE_WIDTH, "h": PAGE_HEIGHT},
    }


def vlm_payload(job_id: str, page_index: int) -> dict[str, Any]:
    """Heavy-VLM-Model: text, markdown tables, key-values, and a layout TREE.

    The `tree` is the structural output Module C's Tree Edit Distance consumes.
    Nested tables (table -> row -> cell) are what give it real depth; a flat
    list of blocks would make TED degenerate into string edit distance.
    """
    rng = _rng(job_id, page_index, "vlm")
    layout = layout_payload(job_id, page_index)

    children: list[dict[str, Any]] = []
    text_parts: list[str] = []
    tables: list[str] = []

    for box in layout["boxes"]:
        bbox = [box["x"], box["y"], box["w"], box["h"]]
        kind = box["type"]

        if kind == "table":
            n_rows, n_cols = rng.randint(2, 4), rng.randint(2, 4)
            rows: list[dict[str, Any]] = []
            md_rows: list[str] = []
            for r in range(n_rows):
                cells = [_sentence(rng, 1) for _ in range(n_cols)]
                rows.append(
                    {
                        "type": "row",
                        "children": [{"type": "cell", "text": c} for c in cells],
                    }
                )
                md_rows.append("| " + " | ".join(cells) + " |")
            # markdown separator after the header row
            md_rows.insert(1, "|" + "|".join(["---"] * n_cols) + "|")
            tables.append("\n".join(md_rows))
            children.append({"type": "table", "bbox": bbox, "children": rows})

        elif kind == "figure":
            children.append({"type": "figure", "bbox": bbox, "text": ""})

        else:
            text = _sentence(rng, rng.randint(6, 14))
            text_parts.append(text)
            children.append({"type": kind, "bbox": bbox, "text": text})

    n_kv = rng.randint(1, 3)
    key_values = {
        rng.choice(_WORDS): _sentence(rng, 2) for _ in range(n_kv)
    }

    return {
        "model": "heavy-vlm",
        "page_index": page_index,
        "text": "\n".join(text_parts),
        "tables": tables,
        "key_values": key_values,
        "tree": {
            "type": "page",
            "bbox": [0.0, 0.0, PAGE_WIDTH, PAGE_HEIGHT],
            "children": children,
        },
        "confidence": round(rng.uniform(0.88, 0.99), 3),
    }
