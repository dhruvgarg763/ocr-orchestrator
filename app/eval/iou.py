"""Bounding-box IoU and box matching.

Per-axis overlap with `max(0, ...)` clamping on each axis before
multiplying - unclamped, two diagonally-disjoint boxes report a phantom
IoU of 1.0 (negative x negative), pinned in tests/test_iou.py. Matching N
predicted against M ground-truth boxes is greedy on descending IoU
(O(NM log NM)); optimality vs. Hungarian (O(n^3)) is measured against a
brute-force oracle, not assumed. `mean_iou` divides by ground-truth count,
not match count, so a missed box counts as zero rather than being invisible
to the average - `mean_matched_iou` is exposed too but is gameable the same
way a per-page mean is. Coordinates are (x, y, w, h) with no "+1" (COCO
convention, not PASCAL VOC's inclusive-pixel one).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

__all__ = [
    "Box",
    "BoxReport",
    "Match",
    "Matching",
    "aggregate",
    "as_box",
    "as_boxes",
    "iou",
    "match_boxes",
    "score_boxes",
]

DEFAULT_THRESHOLD = 0.5


class Box(NamedTuple):
    """An axis-aligned box in (x, y, w, h), the assignment's convention."""

    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> Box:
        """Build from corner coordinates, normalising inverted corners.

        Corners arrive from plenty of real sources (PDF rectangles are often
        bottom-left origin with y2 < y1), so inversion is a convention
        mismatch rather than corrupt data and is silently fixed here. A
        negative w or h passed directly to `Box` is a different matter - see
        `as_box`.
        """
        return cls(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h


def as_box(obj: Any) -> Box:
    """Coerce a box from the shapes this codebase actually produces.

    Accepts a `Box`, a mapping with x/y/w/h (what `mock_model.payloads`
    emits, extra keys such as `id`, `type` and `confidence` ignored), or any
    4-element sequence.

    A negative width or height is rejected rather than clamped. Unlike an
    empty reference string in app/eval/text.py - which is legitimate data
    meaning "nothing to compare against" - a negative extent is structurally
    impossible for a rectangle, so it is a bug in whatever produced it. The
    likeliest cause is corner coordinates passed as (x, y, w, h); silently
    clamping to zero would turn that into an IoU of 0.0, which reads as "the
    model found nothing" rather than "the caller used the wrong convention".
    """
    if isinstance(obj, Box):
        box = obj
    elif isinstance(obj, Mapping):
        try:
            box = Box(
                float(obj["x"]), float(obj["y"]), float(obj["w"]), float(obj["h"])
            )
        except KeyError as exc:
            raise ValueError(f"box mapping is missing {exc.args[0]!r}: {obj!r}") from exc
    elif isinstance(obj, Sequence) and not isinstance(obj, str):
        if len(obj) != 4:
            raise ValueError(f"box sequence must have 4 elements, got {len(obj)}")
        box = Box(*(float(value) for value in obj))
    else:
        raise TypeError(f"cannot interpret {type(obj).__name__} as a box: {obj!r}")

    if box.w < 0 or box.h < 0:
        raise ValueError(
            f"box has negative extent {box!r}; (x, y, w, h) was expected - "
            "for corner coordinates use Box.from_xyxy"
        )
    return box


def as_boxes(objs: Iterable[Any]) -> list[Box]:
    return [as_box(obj) for obj in objs]


# ------------------------------------------------------------------ geometry


def iou(a: Any, b: Any) -> float:
    """Intersection over union of two boxes, in [0, 1].

    Zero-area boxes are permitted - a zero-height text line is a degenerate
    but structurally valid output - and score 0 against anything, since they
    can never contribute intersection. Two zero-area boxes give 0/0, which is
    reported as 0.0 rather than 1.0: they are not evidence of agreement, and
    returning 1.0 would let a model that emits empty boxes score perfectly.
    """
    box_a, box_b = as_box(a), as_box(b)

    # The clamp is what stops two diagonally disjoint boxes producing
    # negative x negative = a positive phantom intersection.
    overlap_w = max(0.0, min(box_a.x2, box_b.x2) - max(box_a.x, box_b.x))
    overlap_h = max(0.0, min(box_a.y2, box_b.y2) - max(box_a.y, box_b.y))
    intersection = overlap_w * overlap_h

    union = box_a.area + box_b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


# ------------------------------------------------------------------ matching


class Match(NamedTuple):
    predicted_index: int
    truth_index: int
    iou: float


@dataclass(frozen=True, slots=True)
class Matching:
    """A one-to-one correspondence, plus what it could not account for.

    The unmatched lists are carried rather than recomputed because they are
    the whole basis of precision and recall: an unmatched prediction is a false
    positive, an unmatched ground truth a false negative. Dropping them would
    leave only the flattering half of the picture.
    """

    matches: tuple[Match, ...]
    unmatched_predicted: tuple[int, ...]
    unmatched_truth: tuple[int, ...]

    @property
    def iou_sum(self) -> float:
        return sum(match.iou for match in self.matches)


def match_boxes(predicted: Iterable[Any], truth: Iterable[Any]) -> Matching:
    """Greedy one-to-one matching on descending IoU.

    Pairs with IoU of exactly 0 are never matched: consuming two boxes to
    record that they do not overlap would deny both of them a partner they
    might genuinely have had, and would inflate the true-positive denominator
    with pairs carrying no evidence.

    Ties are broken by (predicted index, truth index) so the result is
    deterministic. With equal IoUs the choice is arbitrary by definition, but
    an arbitrary-and-stable answer is comparable across runs while an
    arbitrary-and-unstable one silently makes two identical inputs disagree.
    """
    predicted_boxes = as_boxes(predicted)
    truth_boxes = as_boxes(truth)

    candidates = []
    for i, predicted_box in enumerate(predicted_boxes):
        for j, truth_box in enumerate(truth_boxes):
            score = iou(predicted_box, truth_box)
            if score > 0:
                candidates.append((score, i, j))
    # Descending IoU; ascending indices as the deterministic tie-break.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    matches: list[Match] = []
    claimed_predicted: set[int] = set()
    claimed_truth: set[int] = set()
    for score, i, j in candidates:
        if i in claimed_predicted or j in claimed_truth:
            continue
        claimed_predicted.add(i)
        claimed_truth.add(j)
        matches.append(Match(i, j, score))

    return Matching(
        matches=tuple(matches),
        unmatched_predicted=tuple(
            i for i in range(len(predicted_boxes)) if i not in claimed_predicted
        ),
        unmatched_truth=tuple(
            j for j in range(len(truth_boxes)) if j not in claimed_truth
        ),
    )


# ------------------------------------------------------------------- scoring


@dataclass(frozen=True, slots=True)
class BoxReport:
    """Counts and sums, never pre-divided rates - see `aggregate`."""

    true_positives: int
    false_positives: int
    false_negatives: int
    iou_sum: float
    matched_count: int
    truth_count: int
    threshold: float

    @property
    def precision(self) -> float:
        """Of the boxes predicted, how many landed on a ground truth.

        No predictions at all is 1.0, not 0.0: precision asks how much of what
        was emitted is correct, and nothing incorrect was emitted. Recall is
        what catches that model, and it will read 0.0.
        """
        predicted = self.true_positives + self.false_positives
        if predicted == 0:
            return 1.0
        return self.true_positives / predicted

    @property
    def recall(self) -> float:
        """Of the ground-truth boxes, how many were found.

        An empty ground truth is 1.0 - there was nothing to miss. A page with
        no ground-truth boxes and several predictions scores recall 1.0 and
        precision 0.0, which is the honest description.
        """
        relevant = self.true_positives + self.false_negatives
        if relevant == 0:
            return 1.0
        return self.true_positives / relevant

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @property
    def mean_iou(self) -> float:
        """Spatial quality charged against every ground-truth box.

        Missed ground truths count as zero, which is what makes this the
        number worth publishing.
        """
        if self.truth_count == 0:
            return 0.0 if self.iou_sum == 0 else float("inf")
        return self.iou_sum / self.truth_count

    @property
    def mean_matched_iou(self) -> float:
        """Mean over matches only. Exposed to be compared against, not trusted.

        Ignores every miss, so one perfect box out of a hundred scores 1.0.
        """
        if self.matched_count == 0:
            return 0.0
        return self.iou_sum / self.matched_count


def score_boxes(
    predicted: Iterable[Any],
    truth: Iterable[Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> BoxReport:
    """Match, then classify at `threshold`.

    A match below the threshold is a false negative for its ground truth AND a
    false positive for its prediction - the box was emitted (so it is
    unearned output) and the ground truth was not found (so it is a miss).
    Counting it once would make precision and recall disagree about how many
    boxes exist.
    """
    matching = match_boxes(predicted, truth)

    true_positives = sum(1 for match in matching.matches if match.iou >= threshold)
    weak = len(matching.matches) - true_positives

    return BoxReport(
        true_positives=true_positives,
        false_positives=len(matching.unmatched_predicted) + weak,
        false_negatives=len(matching.unmatched_truth) + weak,
        iou_sum=matching.iou_sum,
        matched_count=len(matching.matches),
        truth_count=len(matching.matches) + len(matching.unmatched_truth),
        threshold=threshold,
    )


def aggregate(reports: Iterable[BoxReport]) -> BoxReport:
    """Micro-average across pages: sum the counts, sum the IoUs.

    The same argument as `app.eval.text.aggregate`. Averaging per-page mean
    IoUs weights a page holding two boxes the same as one holding fifty, so
    the corpus number drifts toward whatever the sparse pages happened to do.

    Mixing thresholds is refused: true-positive counts computed at 0.5 and at
    0.75 are answers to different questions and their sum is not an answer to
    either.
    """
    reports = list(reports)
    if not reports:
        return BoxReport(0, 0, 0, 0.0, 0, 0, DEFAULT_THRESHOLD)

    thresholds = {report.threshold for report in reports}
    if len(thresholds) > 1:
        raise ValueError(f"cannot aggregate mixed thresholds: {sorted(thresholds)}")

    return BoxReport(
        true_positives=sum(report.true_positives for report in reports),
        false_positives=sum(report.false_positives for report in reports),
        false_negatives=sum(report.false_negatives for report in reports),
        iou_sum=sum(report.iou_sum for report in reports),
        matched_count=sum(report.matched_count for report in reports),
        truth_count=sum(report.truth_count for report in reports),
        threshold=thresholds.pop(),
    )
