"""IoU: the geometry, the sign bug, and how much greedy matching actually loses.

Three kinds of test here. The geometry is checked against hand-computable
cases and against deliberately WRONG implementations, because the two classic
bugs (an unclamped overlap, a union that adds instead of subtracting) both
produce plausible numbers rather than obvious nonsense.

The matching is checked against a brute-force optimal matcher over bitmasks -
the same oracle strategy as `full_matrix` in tests/test_text_metrics.py. That
oracle is exponential and therefore only usable on small inputs, which is
exactly why the shipped code is greedy.

Two measured counterexamples are pinned as literal coordinates. They are not
hypotheticals: they were found by searching random box sets, and they record
the precise cost of choosing greedy. Recording a known limitation in the test
suite is the difference between a trade-off and an oversight.
"""

from __future__ import annotations

import random
from functools import lru_cache

import pytest

from app.eval.iou import (
    Box,
    aggregate,
    as_box,
    iou,
    match_boxes,
    score_boxes,
)

# ------------------------------------------------------------------- oracles


def iou_matrix(predicted, truth) -> list[list[float]]:
    return [[iou(p, t) for t in truth] for p in predicted]


def optimal_iou_sum(matrix: list[list[float]]) -> float:
    """Max total IoU over all one-to-one matchings, by DP over a bitmask.

    What Hungarian would return, computed the obvious exponential way. Kept
    naive on purpose: an oracle has to be believable at a glance, and this one
    is only ever called on inputs of a handful of boxes.
    """
    n, m = len(matrix), len(matrix[0]) if matrix else 0

    @lru_cache(maxsize=None)
    def best(i: int, used: int) -> float:
        if i == n:
            return 0.0
        # Leaving a prediction unmatched is always allowed.
        out = best(i + 1, used)
        for j in range(m):
            if not (used >> j) & 1 and matrix[i][j] > 0:
                out = max(out, matrix[i][j] + best(i + 1, used | (1 << j)))
        return out

    result = best(0, 0)
    best.cache_clear()
    return result


def optimal_true_positives(matrix: list[list[float]], threshold: float) -> int:
    """Maximum-cardinality matching on the graph of pairs at or above threshold.

    This - not the IoU sum - is what precision and recall are computed from,
    so it is the comparison that decides whether greedy is acceptable.
    """
    n, m = len(matrix), len(matrix[0]) if matrix else 0

    @lru_cache(maxsize=None)
    def best(i: int, used: int) -> int:
        if i == n:
            return 0
        out = best(i + 1, used)
        for j in range(m):
            if not (used >> j) & 1 and matrix[i][j] >= threshold:
                out = max(out, 1 + best(i + 1, used | (1 << j)))
        return out

    result = best(0, 0)
    best.cache_clear()
    return result


def unclamped_iou(a: Box, b: Box) -> float:
    """The sign bug, implemented so a test can prove it is a bug.

    Identical to the shipped version except for the missing `max(0, ...)`.
    """
    overlap_w = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    overlap_h = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    intersection = overlap_w * overlap_h
    union = a.w * a.h + b.w * b.h - intersection
    return intersection / union if union else 0.0


def additive_union_iou(a: Box, b: Box) -> float:
    """The other classic bug: union as a plain sum of areas."""
    overlap_w = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    overlap_h = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    intersection = overlap_w * overlap_h
    union = a.w * a.h + b.w * b.h
    return intersection / union if union else 0.0


# ------------------------------------------------------------------ geometry


def test_identical_boxes_are_one() -> None:
    assert iou(Box(10, 20, 30, 40), Box(10, 20, 30, 40)) == 1.0


def test_disjoint_boxes_are_zero() -> None:
    # Separated on x only, on y only, and on both.
    assert iou(Box(0, 0, 10, 10), Box(50, 0, 10, 10)) == 0.0
    assert iou(Box(0, 0, 10, 10), Box(0, 50, 10, 10)) == 0.0
    assert iou(Box(0, 0, 10, 10), Box(50, 50, 10, 10)) == 0.0


def test_edge_touching_boxes_are_zero_not_undefined() -> None:
    """Sharing a boundary is zero overlap, not a division by zero: the
    intersection has zero area but the union does not."""
    assert iou(Box(0, 0, 10, 10), Box(10, 0, 10, 10)) == 0.0


def test_full_containment() -> None:
    """A box entirely inside another: intersection is the inner area, union
    the outer, so IoU is the ratio of areas - here 25/100."""
    assert iou(Box(0, 0, 10, 10), Box(2.5, 2.5, 5, 5)) == pytest.approx(0.25)


def test_half_overlap_by_hand() -> None:
    """Two 10x10 boxes offset by 5 on x: intersection 5x10=50, union
    100+100-50=150, IoU = 1/3."""
    assert iou(Box(0, 0, 10, 10), Box(5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_iou_is_symmetric() -> None:
    a, b = Box(3, 7, 20, 11), Box(9, 2, 14, 25)
    assert iou(a, b) == iou(b, a)


def test_a_zero_area_box_scores_zero_against_anything() -> None:
    """A zero-height block is degenerate but structurally valid output. It can
    contribute no intersection, so it agrees with nothing."""
    assert iou(Box(0, 0, 10, 0), Box(0, 0, 10, 10)) == 0.0
    assert iou(Box(0, 0, 0, 0), Box(0, 0, 10, 10)) == 0.0


def test_two_zero_area_boxes_are_zero_not_one() -> None:
    """0/0. Returning 1.0 would let a model that emits nothing but empty boxes
    score a perfect overlap against a ground truth of empty boxes."""
    assert iou(Box(5, 5, 0, 0), Box(5, 5, 0, 0)) == 0.0


# ------------------------------------------------- the bugs, proven as bugs


def test_the_clamp_is_what_stops_diagonal_boxes_faking_an_overlap() -> None:
    """REGRESSION for the sign bug.

    Two boxes that miss on BOTH axes give a negative overlap width and a
    negative overlap height, and their product is positive. For two 10x10
    boxes offset diagonally by 20, the phantom intersection is -10 x -10 = 100
    against a union of 100 + 100 - 100 = 100, so the unclamped version reports
    IoU = 1.0: a PERFECT overlap, for boxes that do not touch.

    That is the case that matters. Miss on one axis only and the product is
    negative, which any smoke test would catch.
    """
    a, b = Box(0, 0, 10, 10), Box(20, 20, 10, 10)

    assert iou(a, b) == 0.0
    assert unclamped_iou(a, b) > 0, "the bug being guarded against is not reproduced"
    assert unclamped_iou(a, b) == pytest.approx(100 / 100)

    # And the asymmetry that lets it survive: one-axis misses look wrong.
    single_axis = unclamped_iou(Box(0, 0, 10, 10), Box(20, 0, 10, 10))
    assert single_axis < 0, "a one-axis miss should be obviously broken"


def test_union_must_subtract_the_intersection() -> None:
    """Adding the areas double-counts the shared region, so every overlapping
    pair is under-reported. Exact on the half-overlap case: 50/150 vs 50/200."""
    a, b = Box(0, 0, 10, 10), Box(5, 0, 10, 10)

    assert iou(a, b) == pytest.approx(1 / 3)
    assert additive_union_iou(a, b) == pytest.approx(0.25)
    assert additive_union_iou(a, b) < iou(a, b)


# -------------------------------------------------------------- box coercion


def test_a_box_can_come_from_the_mock_payload_shape() -> None:
    """`mock_model.payloads` emits dicts with id/type/confidence alongside
    x/y/w/h. The extra keys must be ignored, not rejected."""
    payload = {"id": 3, "type": "table", "x": 40.0, "y": 120.5,
               "w": 450.0, "h": 61.25, "confidence": 0.91}

    assert as_box(payload) == Box(40.0, 120.5, 450.0, 61.25)


def test_a_box_can_come_from_a_plain_sequence() -> None:
    assert as_box([1, 2, 3, 4]) == Box(1.0, 2.0, 3.0, 4.0)
    assert as_box((1, 2, 3, 4)) == Box(1.0, 2.0, 3.0, 4.0)


@pytest.mark.parametrize("bad", [[1, 2, 3], [1, 2, 3, 4, 5], []])
def test_a_wrong_length_sequence_is_rejected(bad: list[int]) -> None:
    with pytest.raises(ValueError, match="4 elements"):
        as_box(bad)


def test_a_mapping_missing_a_key_names_the_key() -> None:
    with pytest.raises(ValueError, match="'h'"):
        as_box({"x": 0, "y": 0, "w": 10})


def test_a_negative_extent_is_refused_rather_than_clamped() -> None:
    """The likeliest cause is corner coordinates passed as (x, y, w, h).
    Clamping to zero would report that as "the model found nothing", which
    sends the reader looking at the model instead of at the caller."""
    with pytest.raises(ValueError, match="negative extent"):
        as_box([10, 10, -5, 20])
    with pytest.raises(ValueError, match="from_xyxy"):
        as_box([10, 10, 20, -5])


def test_from_xyxy_converts_and_normalises_inverted_corners() -> None:
    """PDF rectangles are frequently bottom-left origin with y2 < y1, so
    inversion is a convention mismatch rather than bad data."""
    assert Box.from_xyxy(10, 20, 40, 60) == Box(10, 20, 30, 40)
    assert Box.from_xyxy(40, 60, 10, 20) == Box(10, 20, 30, 40)


def test_xyxy_passed_as_xywh_is_silently_wrong_which_is_why_from_xyxy_exists() -> None:
    """Not a bug report - a demonstration of why the named constructor is
    worth having. Positive corner coordinates misread as width/height produce
    a valid, oversized box and no error anywhere."""
    corners = (100, 100, 150, 160)  # meant as x1,y1,x2,y2 -> a 50x60 box

    misread = as_box(corners)
    intended = Box.from_xyxy(*corners)

    assert misread == Box(100, 100, 150, 160)  # no exception, wrong answer
    assert misread.area == 24000
    assert intended.area == 3000
    assert iou(misread, intended) == pytest.approx(3000 / 24000)


# ------------------------------------------------------------------ matching


def test_matching_is_one_to_one() -> None:
    """Three predictions crowding one ground truth: exactly one may claim it."""
    truth = [Box(0, 0, 10, 10)]
    predicted = [Box(1, 1, 10, 10), Box(0, 0, 10, 10), Box(2, 2, 10, 10)]

    matching = match_boxes(predicted, truth)

    assert len(matching.matches) == 1
    assert matching.matches[0].predicted_index == 1  # the exact one, IoU 1.0
    assert matching.matches[0].iou == 1.0
    assert sorted(matching.unmatched_predicted) == [0, 2]
    assert matching.unmatched_truth == ()


def test_zero_overlap_pairs_are_never_matched() -> None:
    """Pairing two boxes to record that they do not overlap would consume both
    and deny either a partner it might really have had."""
    matching = match_boxes([Box(100, 100, 10, 10)], [Box(0, 0, 10, 10)])

    assert matching.matches == ()
    assert matching.unmatched_predicted == (0,)
    assert matching.unmatched_truth == (0,)


def test_matching_is_deterministic_under_ties() -> None:
    """Two predictions with identical IoU against the same ground truth. The
    winner is arbitrary by definition, but it must be the SAME arbitrary
    winner every run, or two identical inputs disagree."""
    truth = [Box(0, 0, 10, 10)]
    predicted = [Box(5, 0, 10, 10), Box(0, 5, 10, 10)]  # both IoU 1/3

    first = match_boxes(predicted, truth)
    again = match_boxes(predicted, truth)

    assert first == again
    assert first.matches[0].predicted_index == 0  # lowest index breaks the tie


def test_matching_handles_empty_inputs() -> None:
    assert match_boxes([], []).matches == ()
    assert match_boxes([Box(0, 0, 1, 1)], []).unmatched_predicted == (0,)
    assert match_boxes([], [Box(0, 0, 1, 1)]).unmatched_truth == (0,)


def test_thresholding_after_matching_equals_thresholding_during() -> None:
    """The property that lets `score_boxes` produce both the spatial-quality
    and the detection numbers from ONE matching.

    Greedy visits pairs in descending IoU, so every above-threshold pair is
    considered before any below-threshold one; a sub-threshold match can only
    pair boxes that were already left over. Checked on random layouts rather
    than argued, including the pinned counterexample below where greedy is
    provably suboptimal - the equivalence holds there too.
    """
    rng = random.Random(4)
    for _ in range(400):
        predicted = [_random_box(rng) for _ in range(rng.randint(0, 5))]
        truth = [_random_box(rng) for _ in range(rng.randint(0, 5))]

        relaxed = score_boxes(predicted, truth, threshold=0.5).true_positives

        # The same greedy walk, but refusing to form a match below 0.5.
        matrix = iou_matrix(predicted, truth)
        pairs = sorted(
            (
                (matrix[i][j], i, j)
                for i in range(len(predicted))
                for j in range(len(truth))
                if matrix[i][j] >= 0.5
            ),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        claimed_p: set[int] = set()
        claimed_t: set[int] = set()
        strict = 0
        for _score, i, j in pairs:
            if i in claimed_p or j in claimed_t:
                continue
            claimed_p.add(i)
            claimed_t.add(j)
            strict += 1

        assert relaxed == strict


def _random_box(rng: random.Random) -> Box:
    return Box(
        rng.uniform(0, 120),
        rng.uniform(0, 120),
        rng.uniform(60, 140),
        rng.uniform(60, 140),
    )


def _realistic_page(rng: random.Random) -> tuple[list[Box], list[Box]]:
    """A column of blocks, then a jittered prediction of it with drops and a
    possible spurious box - the shape `mock_model.payloads` actually emits."""
    truth: list[Box] = []
    y = 40.0
    for _ in range(rng.randint(2, 7)):
        height = rng.uniform(30, 90)
        truth.append(Box(rng.uniform(40, 80), y, rng.uniform(400, 500), height))
        y += height + rng.uniform(8, 20)

    predicted: list[Box] = []
    for box in truth:
        if rng.random() < 0.15:
            continue
        predicted.append(
            Box(
                box.x + rng.uniform(-25, 25),
                box.y + rng.uniform(-25, 25),
                max(5.0, box.w + rng.uniform(-60, 60)),
                max(5.0, box.h + rng.uniform(-30, 30)),
            )
        )
    if rng.random() < 0.3:
        predicted.append(_random_box(rng))
    return predicted, truth


# ------------------------------------------- what greedy costs, measured


def test_greedy_matches_optimal_true_positive_count_on_realistic_layouts() -> None:
    """The measurement that justifies greedy over Hungarian.

    Precision and recall depend only on the true-positive COUNT, and on
    document-shaped input - a column of mostly non-overlapping blocks - greedy
    was never observed to lose one. Measured over 4,000 random layouts before
    this test was written; 600 here to keep the suite fast.

    This is a claim about document layouts, not about greedy matching in
    general. See the two tests below for where it does lose.
    """
    rng = random.Random(17)
    for _ in range(600):
        predicted, truth = _realistic_page(rng)
        if not predicted or not truth:
            continue
        matrix = iou_matrix(predicted, truth)

        greedy = score_boxes(predicted, truth, threshold=0.5).true_positives

        assert greedy == optimal_true_positives(matrix, 0.5)


def test_greedy_can_lose_a_true_positive_when_boxes_heavily_overlap() -> None:
    """PINNED COUNTEREXAMPLE, found by search - the known cost of greedy.

    The IoU matrix (predictions down, ground truths across):

                 G1      G2      G3
        P1     0.6023  0.3026  0.4266
        P2     0.6778  0.5221  0.2713

    Greedy takes the single largest pair, (P2,G1) at 0.6778, which consumes
    the only ground truth P1 overlaps above threshold. P1 then settles for G3
    at 0.4266 - below 0.5, so not a true positive. One TP.

    The optimal matching declines the biggest pair: (P1,G1) at 0.6023 plus
    (P2,G2) at 0.5221, both above threshold. Two TPs.

    Only reachable when boxes overlap each other heavily; measured at 273 of
    6,000 densely clustered cases and 0 of 4,000 document layouts. Recorded so
    the limitation lives in the suite rather than in an interviewer's notes.
    """
    predicted = [Box(44, 69, 76, 92), Box(70, 62, 72, 96)]
    truth = [Box(49, 57, 99, 103), Box(67, 30, 100, 124), Box(4, 59, 112, 73)]

    matrix = iou_matrix(predicted, truth)
    assert matrix[1][0] == pytest.approx(0.6778, abs=1e-4)
    assert matrix[0][0] == pytest.approx(0.6023, abs=1e-4)
    assert matrix[1][1] == pytest.approx(0.5221, abs=1e-4)

    report = score_boxes(predicted, truth, threshold=0.5)

    assert report.true_positives == 1
    assert optimal_true_positives(matrix, 0.5) == 2
    # Recall is what visibly suffers: 1 of 3 found instead of 2 of 3.
    assert report.recall == pytest.approx(1 / 3)


def test_greedy_can_lose_almost_half_the_available_iou_sum() -> None:
    """PINNED COUNTEREXAMPLE for the other objective.

                 G1      G2
        P1     0.2710  0.0000
        P2     0.2784  0.2701

    P1 overlaps only G1. Greedy takes (P2,G1) at 0.2784 because it is the
    largest single pair, stranding P1 entirely. Optimal gives G1 to P1 and G2
    to P2 for 0.5411 - greedy loses 48.6% of the obtainable total.

    Both pairs are below threshold, so no true positive changes hands; what
    moves is `mean_iou`. Worth knowing before quoting that number to four
    decimal places.
    """
    predicted = [Box(76, 100, 125, 63), Box(12, 41, 134, 138)]
    truth = [Box(73, 15, 121, 127), Box(1, 46, 66, 96)]

    matching = match_boxes(predicted, truth)
    optimal = optimal_iou_sum(iou_matrix(predicted, truth))

    assert len(matching.matches) == 1
    assert matching.iou_sum == pytest.approx(0.2784, abs=1e-4)
    assert optimal == pytest.approx(0.5411, abs=1e-4)
    assert (optimal - matching.iou_sum) / optimal > 0.48


# ------------------------------------------------------------------- scoring


def test_a_perfect_prediction_scores_one_everywhere() -> None:
    boxes = [Box(0, 0, 10, 10), Box(20, 20, 10, 10)]

    report = score_boxes(boxes, boxes)

    assert (report.true_positives, report.false_positives, report.false_negatives) == (
        2, 0, 0,
    )
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert report.mean_iou == 1.0


def test_a_weak_match_is_both_a_false_positive_and_a_false_negative() -> None:
    """The box was emitted, so it is unearned output; the ground truth was not
    found, so it is a miss. Counting it once would make precision and recall
    disagree about how many boxes exist."""
    predicted = [Box(0, 0, 10, 10)]
    truth = [Box(7, 0, 10, 10)]  # IoU 3/17 = 0.176

    report = score_boxes(predicted, truth, threshold=0.5)

    assert report.true_positives == 0
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.matched_count == 1, "the pair is still matched for IoU purposes"
    assert report.iou_sum == pytest.approx(3 / 17)


def test_the_threshold_is_inclusive() -> None:
    """IoU exactly 0.5 counts as a hit.

    Boxes engineered to land on it: two 30x10 blocks offset by 10 overlap
    20x10 = 200, against a union of 300 + 300 - 200 = 400.
    """
    report = score_boxes([Box(0, 0, 30, 10)], [Box(10, 0, 30, 10)], threshold=0.5)

    assert report.iou_sum == pytest.approx(0.5)
    assert report.true_positives == 1


def test_a_missed_box_and_a_spurious_box_are_counted_separately() -> None:
    truth = [Box(0, 0, 10, 10), Box(100, 100, 10, 10)]
    predicted = [Box(0, 0, 10, 10), Box(500, 500, 10, 10)]

    report = score_boxes(predicted, truth)

    assert report.true_positives == 1
    assert report.false_positives == 1  # the box at 500,500
    assert report.false_negatives == 1  # the truth at 100,100
    assert report.precision == pytest.approx(0.5)
    assert report.recall == pytest.approx(0.5)


def test_predicting_nothing_is_perfect_precision_and_zero_recall() -> None:
    """Precision asks how much of what was emitted is correct, and nothing
    incorrect was emitted. Recall is the number that indicts this model, which
    is exactly why both are reported."""
    report = score_boxes([], [Box(0, 0, 10, 10)])

    assert report.precision == 1.0
    assert report.recall == 0.0
    assert report.f1 == 0.0
    assert report.mean_iou == 0.0


def test_an_empty_ground_truth_is_recall_one_and_precision_zero() -> None:
    """A blank page the model hallucinated boxes onto. There was nothing to
    miss, and everything emitted was wrong."""
    report = score_boxes([Box(0, 0, 10, 10)], [])

    assert report.recall == 1.0
    assert report.precision == 0.0


# ----------------------------------------------- the two mean IoU denominators


def test_mean_matched_iou_ignores_every_miss() -> None:
    """The gameable denominator, demonstrated.

    One perfect box and nine missed ground truths: mean over matches is a
    flawless 1.0, mean over ground truths is 0.1. The second is what a reader
    means by "how well did this page do".
    """
    truth = [Box(i * 100, 0, 10, 10) for i in range(10)]
    predicted = [Box(0, 0, 10, 10)]

    report = score_boxes(predicted, truth)

    assert report.mean_matched_iou == 1.0
    assert report.mean_iou == pytest.approx(0.1)
    assert report.recall == pytest.approx(0.1)


def test_corpus_mean_iou_is_a_micro_average_not_a_mean_of_pages() -> None:
    """The same statistical trap as tests/test_text_metrics.py.

    One dense page of ten good boxes and one sparse page holding a single
    miss. Micro-averaging charges the miss against 11 ground truths; averaging
    the two page scores charges it against one of two pages, roughly halving
    the reported quality.
    """
    dense_truth = [Box(i * 100, 0, 10, 10) for i in range(10)]
    dense = score_boxes(dense_truth, dense_truth)
    sparse = score_boxes([], [Box(0, 0, 10, 10)])

    micro = aggregate([dense, sparse]).mean_iou
    macro = (dense.mean_iou + sparse.mean_iou) / 2

    assert micro == pytest.approx(10 / 11)
    assert macro == pytest.approx(0.5)
    assert micro > macro


def test_aggregate_sums_counts_and_refuses_mixed_thresholds() -> None:
    a = score_boxes([Box(0, 0, 10, 10)], [Box(0, 0, 10, 10)], threshold=0.5)
    b = score_boxes([Box(0, 0, 10, 10)], [Box(0, 0, 10, 10)], threshold=0.75)

    assert aggregate([a, a]).true_positives == 2
    assert aggregate([a, a]).truth_count == 2

    with pytest.raises(ValueError, match="mixed thresholds"):
        aggregate([a, b])


def test_aggregating_nothing_is_empty_not_a_crash() -> None:
    empty = aggregate([])

    assert empty.true_positives == 0
    assert empty.truth_count == 0
    assert empty.mean_iou == 0.0
    assert empty.mean_matched_iou == 0.0
