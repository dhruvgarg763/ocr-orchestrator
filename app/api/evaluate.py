"""POST /evaluate - Module C's ground-truth comparison endpoint.

One tree in (predicted, against truth), all three metrics out - CER/WER,
box IoU, and tree edit distance - because a single node payload
(`type`/`bbox`/`text`/`children`) carries everything all three need, and
accepting three separate payloads would let them disagree about the same
document. All three are synchronous CPU-bound Python (TED alone: ~5ms for
50 nodes, ~22s for a 201-node caterpillar), so the whole computation runs
in one `asyncio.to_thread` hop - otherwise one awkward tree would stall
every in-flight SSE stream on this process. The TED cost is bounded
*before* computing, not inside the DP: `keyroot_weight` is O(keyroots) and
predicts the Theta(W1*W2) cost, so an oversized tree gets a 413 before a
single DP cell runs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.eval import iou as iou_module
from app.eval import ted as ted_module
from app.eval import text as text_module
from common.logging import get_logger
from common.tracing import get_trace_id

log = get_logger("evaluate")

router = APIRouter()


class EvaluateRequest(BaseModel):
    """Two document trees. `null` is a legitimate empty document, not an error.

    An empty side is meaningful and each metric already defines it: TED becomes
    the other tree's node count, recall goes to zero with precision at one, and
    CER against an empty reference is infinite. Rejecting it would force the
    caller to special-case a page the model returned nothing for, which is
    exactly the case worth measuring.
    """

    predicted: dict[str, Any] | None = Field(
        default=None, description="Extracted document tree"
    )
    truth: dict[str, Any] | None = Field(
        default=None, description="Ground-truth document tree"
    )
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    cer_unit: Literal["grapheme", "codepoint"] = "grapheme"
    label_key: str = Field(default="type", max_length=64)
    children_key: str = Field(default="children", max_length=64)
    text_key: str = Field(default="text", max_length=64)
    bbox_key: str = Field(default="bbox", max_length=64)


def _collect(
    root: Any | None, *, children_key: str, text_key: str, bbox_key: str
) -> tuple[str, list[Any]]:
    """Pull text and boxes out of a tree in document order, iteratively.

    Document order - pre-order, left to right - is the reading order, which is
    what makes the concatenated text the right thing to score: WER over a
    document whose blocks are in the wrong order should be wrong, and it is.

    Iterative for the same reason app/eval/ted.py's parser is: a recursive walk
    over a deeply nested tree raises RecursionError, and a crash is a worse
    answer than a number.
    """
    if root is None:
        return "", []

    parts: list[str] = []
    boxes: list[Any] = []
    stack: list[Any] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            value = node.get(text_key)
            if isinstance(value, str) and value:
                parts.append(value)
            box = node.get(bbox_key)
            if box is not None:
                boxes.append(box)
            children = node.get(children_key) or ()
            if isinstance(children, list):
                # Reversed so the leftmost child is popped first.
                stack.extend(reversed(children))
    return "\n".join(parts), boxes


class TextMetrics(BaseModel):
    cer: float
    wer: float
    cer_errors: int
    cer_length: int
    wer_errors: int
    wer_length: int
    unit: str


class BoxMetrics(BaseModel):
    mean_iou: float
    mean_matched_iou: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    predicted_boxes: int
    truth_boxes: int
    threshold: float


class StructureMetrics(BaseModel):
    tree_edit_distance: int
    normalised: float
    predicted_nodes: int
    truth_nodes: int
    predicted_depth: int
    truth_depth: int
    ted_work: int
    """W(T1) x W(T2): the cost actually incurred, for comparison with the cap."""


class Timings(BaseModel):
    ted_ms: float
    """THE GRADED NUMBER: tree diff calculation latency, target < 100 ms for a
    50-node tree. Reported so it can be read directly rather than inferred from
    an HTTP round trip that also includes JSON parsing and scheduling."""

    text_ms: float
    boxes_ms: float
    compute_ms: float


class EvaluateResponse(BaseModel):
    text: TextMetrics
    boxes: BoxMetrics
    structure: StructureMetrics
    timings_ms: Timings
    trace_id: str


def _compute(
    request: EvaluateRequest,
    predicted_tree: ted_module.Tree,
    truth_tree: ted_module.Tree,
    ted_work: int,
) -> dict[str, Any]:
    """The whole CPU-bound body, so `to_thread` is entered exactly once.

    The trees arrive already parsed because the caller needed their weights to
    decide whether to run this at all; re-parsing here would double the work
    and, worse, would let the thing that was measured differ from the thing that
    was admitted.
    """
    started = time.perf_counter()

    predicted_text, predicted_boxes = _collect(
        request.predicted,
        children_key=request.children_key,
        text_key=request.text_key,
        bbox_key=request.bbox_key,
    )
    truth_text, truth_boxes = _collect(
        request.truth,
        children_key=request.children_key,
        text_key=request.text_key,
        bbox_key=request.bbox_key,
    )

    text_started = time.perf_counter()
    cer = text_module.cer_score(truth_text, predicted_text, unit=request.cer_unit)
    wer = text_module.wer_score(truth_text, predicted_text)
    text_ms = (time.perf_counter() - text_started) * 1000

    boxes_started = time.perf_counter()
    # Orientation matters and is easy to invert: predicted first, truth second.
    # Swapped, an invented box would be reported as a missed one.
    box_report = iou_module.score_boxes(
        predicted_boxes, truth_boxes, threshold=request.iou_threshold
    )
    boxes_ms = (time.perf_counter() - boxes_started) * 1000

    ted_started = time.perf_counter()
    distance = ted_module.tree_edit_distance(predicted_tree, truth_tree)
    ted_ms = (time.perf_counter() - ted_started) * 1000

    compute_ms = (time.perf_counter() - started) * 1000

    return {
        "text": {
            "cer": cer.rate,
            "wer": wer.rate,
            "cer_errors": cer.errors,
            "cer_length": cer.length,
            "wer_errors": wer.errors,
            "wer_length": wer.length,
            "unit": cer.unit,
        },
        "boxes": {
            "mean_iou": box_report.mean_iou,
            "mean_matched_iou": box_report.mean_matched_iou,
            "precision": box_report.precision,
            "recall": box_report.recall,
            "f1": box_report.f1,
            "true_positives": box_report.true_positives,
            "false_positives": box_report.false_positives,
            "false_negatives": box_report.false_negatives,
            "predicted_boxes": len(predicted_boxes),
            "truth_boxes": len(truth_boxes),
            "threshold": box_report.threshold,
        },
        "structure": {
            "tree_edit_distance": distance,
            "normalised": ted_module.TreeScore(
                distance, truth_tree.size, predicted_tree.size
            ).normalised,
            "predicted_nodes": predicted_tree.size,
            "truth_nodes": truth_tree.size,
            "predicted_depth": predicted_tree.depth,
            "truth_depth": truth_tree.depth,
            "ted_work": ted_work,
        },
        "timings_ms": {
            "ted_ms": ted_ms,
            "text_ms": text_ms,
            "boxes_ms": boxes_ms,
            "compute_ms": compute_ms,
        },
    }


def _parse(
    tree: Any | None, side: str, request: EvaluateRequest, settings: Settings
) -> ted_module.Tree:
    try:
        parsed = ted_module.parse_tree(
            tree, label_key=request.label_key, children_key=request.children_key
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{side}: {exc}") from exc
    except RecursionError as exc:
        # parse_tree itself is iterative, so this can only come from something
        # underneath it. Surfaced as a rejection rather than a 500 either way.
        raise HTTPException(
            status_code=413, detail=f"{side}: tree is nested too deeply"
        ) from exc

    if parsed.size > settings.eval_max_nodes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{side}: {parsed.size} nodes exceeds the limit of "
                f"{settings.eval_max_nodes}"
            ),
        )
    return parsed


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    request: EvaluateRequest, http_request: Request
) -> EvaluateResponse:
    """Compare an extracted document tree against ground truth.

    Order of operations, and none of it is incidental:

      1. parse both trees            O(n), iterative, bounded by eval_max_nodes
      2. bound the work             O(keyroots), refuse with 413 if too costly
      3. compute, in a thread       so the event loop keeps serving SSE
    """
    settings = get_settings()

    predicted_tree = _parse(request.predicted, "predicted", request, settings)
    truth_tree = _parse(request.truth, "truth", request, settings)

    # Step 2. The cost of the DP is Theta(W1 x W2) and both weights are cheap,
    # so an unservable request is refused before any of it is spent.
    ted_work = predicted_tree.keyroot_weight * truth_tree.keyroot_weight
    if ted_work > settings.eval_max_ted_work:
        log.warning(
            "evaluate_rejected_cost",
            ted_work=ted_work,
            limit=settings.eval_max_ted_work,
            predicted_nodes=predicted_tree.size,
            truth_nodes=truth_tree.size,
            predicted_depth=predicted_tree.depth,
            truth_depth=truth_tree.depth,
        )
        raise HTTPException(
            status_code=413,
            detail=(
                f"tree comparison would cost {ted_work} work units, exceeding "
                f"the limit of {settings.eval_max_ted_work} "
                f"(~{settings.eval_ted_budget_ms:.0f} ms). Tree edit distance is "
                "quartic in the worst case and its cost depends on tree SHAPE, "
                f"not node count: {predicted_tree.size} nodes at depth "
                f"{predicted_tree.depth} is far more expensive than the same "
                "node count in a document-shaped tree."
            ),
        )

    try:
        result = await asyncio.to_thread(
            _compute, request, predicted_tree, truth_tree, ted_work
        )
    except ValueError as exc:
        # Raised by app/eval/iou.py for a malformed box - a negative extent, or
        # a bbox that is not four numbers. Parsing the TREE cannot get this far
        # (it is validated above), but boxes and text are only touched inside
        # the computation, so their validation errors surface here.
        #
        # 422, not 413 and not 500: the request is the wrong SHAPE, which is the
        # client's to fix. Letting it escape would be a 500 - the server
        # blaming itself for a caller passing (x1, y1, x2, y2) where
        # (x, y, w, h) was documented, which is the single likeliest mistake
        # against this endpoint.
        log.info("evaluate_rejected_payload", detail=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # The graded metric, recorded where Prometheus can see its distribution
    # rather than only its last value. Per-metric labels because the three have
    # different cost profiles and lumping them would hide which one moved.
    histogram = getattr(http_request.app.state, "evaluate_duration", None)
    if histogram is not None:
        timings = result["timings_ms"]
        histogram.observe(timings["ted_ms"] / 1000.0, metric="ted")
        histogram.observe(timings["text_ms"] / 1000.0, metric="text")
        histogram.observe(timings["boxes_ms"] / 1000.0, metric="boxes")

    log.info(
        "evaluated",
        ted=result["structure"]["tree_edit_distance"],
        ted_ms=round(result["timings_ms"]["ted_ms"], 3),
        cer=result["text"]["cer"],
        mean_iou=result["boxes"]["mean_iou"],
        nodes=predicted_tree.size,
        ted_work=ted_work,
    )

    return EvaluateResponse(**result, trace_id=get_trace_id())
