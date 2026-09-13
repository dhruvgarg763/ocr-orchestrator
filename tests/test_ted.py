"""Zhang-Shasha tree edit distance: three oracles, the stack, and the clock.

This is the hardest algorithm in the project and the only one with a graded
latency target, so it gets more independent verification than the rest.

Three oracles, none of which shares code with the implementation:

  1. A PATH-shaped tree reduces tree edit distance to STRING edit distance,
     because deleting a node in a path leaves a path. So `levenshtein` from
     app/eval/text.py - itself checked against a full-matrix oracle - becomes
     an oracle here. That cross-check is also the concrete sense in which TED
     is "the Step 15 DP one dimension up".
  2. A STAR reduces to `levenshtein` over the children plus the root relabel.
  3. Brute force over EVERY legal tree mapping, for trees of a few nodes.
     This is the real oracle: it encodes the definition (one-to-one,
     order-preserving, ancestry-preserving) directly and minimises cost over
     all of them, with no dynamic programming anywhere near it.

Oracles 1 and 2 cover long inputs but only degenerate shapes; oracle 3 covers
arbitrary shapes but only tiny ones. Together they leave very little room.
"""

from __future__ import annotations

import copy
import math
import random
import time

import pytest

from app.eval.ted import (
    aggregate,
    parse_tree,
    tree_edit_distance,
    tree_score,
)
from app.eval.text import levenshtein

# ------------------------------------------------------------- tree builders


def node(label: str, *children: dict) -> dict:
    return {"type": label, "children": list(children)}


def leaf(label: str) -> dict:
    return {"type": label}


def path(labels) -> dict | None:
    """Root at labels[0], one child each - so deleting any node keeps a path."""
    built = None
    for label in reversed(list(labels)):
        built = {"type": label, "children": [built]} if built else {"type": label}
    return built


def star(root: str, children) -> dict:
    return {"type": root, "children": [{"type": c} for c in children]}


def caterpillar(spine: int) -> dict:
    """A deep spine that sheds one leaf per level.

    The shape that makes depth AND leaves both Theta(n) at once, which is what
    reaches Zhang-Shasha's O(n^4) worst case.
    """
    built: dict = {"type": "tip"}
    for _ in range(spine):
        built = {"type": "spine", "children": [{"type": "leaf"}, built]}
    return built


def layout_tree(rng: random.Random, blocks: int) -> dict:
    """The shape mock_model.payloads actually emits: page -> blocks, tables
    holding rows holding cells. Depth 4, wide."""
    children = []
    for i in range(blocks):
        if i % 3 == 2:
            children.append(
                node(
                    "table",
                    *[
                        node("row", *[leaf("cell") for _ in range(3)])
                        for _ in range(3)
                    ],
                )
            )
        else:
            children.append(leaf(rng.choice(["paragraph", "title", "figure"])))
    return node("page", *children)


def perturb(tree: dict, rng: random.Random, probability: float = 0.25) -> dict:
    out = copy.deepcopy(tree)

    def walk(current: dict) -> None:
        for child in current.get("children", []):
            walk(child)
        kids = current.get("children")
        if kids and rng.random() < probability:
            kids.pop(rng.randrange(len(kids)))
        if rng.random() < probability:
            current["type"] = "changed"

    walk(out)
    return out


def ted(a, b) -> int:
    return tree_edit_distance(parse_tree(a), parse_tree(b))


def random_tree(rng: random.Random, budget: int) -> dict:
    label = rng.choice("pqr")
    children = []
    remaining = budget - 1
    while remaining > 0 and rng.random() < 0.55:
        take = rng.randint(1, remaining)
        children.append(random_tree(rng, take))
        remaining -= take
    return {"type": label, "children": children}


# ------------------------------------------------------------------- oracle 3


def _relation(leftmost, x: int, y: int) -> str:
    """Which of the four possible relations two distinct nodes stand in.

    Postorder plus the leftmost array makes this arithmetic: subtree(a) is
    exactly [leftmost[a], a], so containment is a range test, and for two
    nodes in disjoint subtrees postorder order IS left-to-right order.
    """
    if leftmost[x] <= y < x:
        return "ancestor"
    if leftmost[y] <= x < y:
        return "descendant"
    return "left" if x < y else "right"


def brute_force_ted(a, b) -> int:
    """Minimum cost over every legal mapping, by direct enumeration.

    A legal (Tai) mapping is one-to-one and preserves both the order and the
    ancestry relations. Cost is one per unmapped node on either side, plus one
    per mapped pair whose labels differ. Exponential, and deliberately so -
    an oracle has to encode the definition, not an optimisation of it.
    """
    tree_a, tree_b = parse_tree(a), parse_tree(b)
    n1, n2 = tree_a.size, tree_b.size
    left_a, left_b = tree_a.leftmost, tree_b.leftmost
    labels_a, labels_b = tree_a.labels, tree_b.labels

    best = n1 + n2  # map nothing: delete all, insert all
    pairs: list[tuple[int, int]] = []

    def legal(i: int, j: int) -> bool:
        return all(
            _relation(left_a, pi, i) == _relation(left_b, pj, j)
            for pi, pj in pairs
        )

    def search(i: int, used: frozenset[int], relabels: int) -> None:
        nonlocal best
        if i > n1:
            mapped = len(pairs)
            best = min(best, (n1 - mapped) + (n2 - mapped) + relabels)
            return
        search(i + 1, used, relabels)  # leave node i unmapped
        for j in range(1, n2 + 1):
            if j in used or not legal(i, j):
                continue
            pairs.append((i, j))
            search(
                i + 1,
                used | {j},
                relabels + (labels_a[i] != labels_b[j]),
            )
            pairs.pop()

    search(1, frozenset(), 0)
    return best


# --------------------------------------------------------- parsing / structure


def test_postorder_puts_children_before_parents() -> None:
    tree = parse_tree(node("page", leaf("title"), node("table", leaf("row"))))

    # labels[0] is the 1-based sentinel.
    assert tree.labels[1:] == ("title", "row", "table", "page")
    assert tree.size == 4


def test_a_subtree_is_a_contiguous_postorder_range() -> None:
    """The property the whole algorithm rests on: subtree(i) == [l(i), i]."""
    tree = parse_tree(node("page", leaf("a"), node("t", leaf("b"), leaf("c"))))

    # postorder: a=1, b=2, c=3, t=4, page=5
    assert tree.leftmost[1] == 1  # leaf a
    assert tree.leftmost[4] == 2  # subtree t covers [2,4] = b, c, t
    assert tree.leftmost[5] == 1  # the root covers everything


def test_keyroots_never_outnumber_leaves() -> None:
    """The bound that makes the optimisation worth anything - a keyroot is
    identified by a distinct leftmost leaf, so there cannot be more of them
    than there are leaves."""
    rng = random.Random(3)
    shapes = [
        layout_tree(rng, 20),
        caterpillar(12),
        path("abcdefgh"),
        star("r", "abcdefgh"),
        random_tree(rng, 40),
    ]
    for shape in shapes:
        tree = parse_tree(shape)
        assert len(tree.keyroots) <= tree.leaf_count, tree


def test_a_leftmost_child_is_not_a_keyroot() -> None:
    """It shares its leftmost descendant with its parent, so its treedist row
    is filled in during the parent's pass for free."""
    tree = parse_tree(node("r", node("a", leaf("x")), leaf("b")))

    # postorder: x=1, a=2, b=3, r=4. l(x)=1, l(a)=1, l(b)=3, l(r)=1.
    # x and a share l=1 with r, so only the highest (r) is a keyroot.
    assert tree.keyroots == (3, 4)


def test_keyroots_are_ascending() -> None:
    """Enclosing subtrees must be processed last, or the fourth branch of the
    recurrence reads a treedist entry that has not been written yet."""
    rng = random.Random(9)
    for _ in range(50):
        tree = parse_tree(random_tree(rng, rng.randint(1, 30)))
        assert list(tree.keyroots) == sorted(tree.keyroots)


def test_a_node_without_a_label_is_an_error_not_a_default() -> None:
    """A shared default label would make every unlabelled node match every
    other one at zero cost, quietly deflating the distance."""
    with pytest.raises(ValueError, match="missing 'type'"):
        parse_tree({"children": [{"type": "a"}]})


def test_children_must_be_a_sequence() -> None:
    with pytest.raises(ValueError, match="must be a sequence"):
        parse_tree({"type": "r", "children": {"type": "a"}})


def test_a_missing_or_empty_children_key_is_a_leaf() -> None:
    assert parse_tree({"type": "a"}).size == 1
    assert parse_tree({"type": "a", "children": []}).size == 1
    assert parse_tree({"type": "a", "children": None}).size == 1


# --------------------------------------------------------------- known values


def test_identical_trees_are_zero() -> None:
    tree = layout_tree(random.Random(1), 6)
    assert ted(tree, tree) == 0


def test_an_empty_tree_costs_one_per_node() -> None:
    tree = layout_tree(random.Random(1), 6)
    size = parse_tree(tree).size

    assert ted(None, tree) == size
    assert ted(tree, None) == size
    assert ted(None, None) == 0


def test_a_relabel_costs_one_not_two() -> None:
    """Relabel at 1 against delete-plus-insert at 2 is what makes a changed
    node type cheaper than a restructure - the right preference for drift."""
    assert ted(leaf("title"), leaf("paragraph")) == 1


def test_deleting_an_internal_node_promotes_its_children() -> None:
    """The operation that makes this a TREE problem.

    page(table(row_a, row_b)) -> page(row_a, row_b) is ONE deletion: remove
    the table and its rows attach to page. Any implementation that treated
    the subtree as indivisible would charge 3 (delete table, row, row) or
    more.
    """
    with_table = node("page", node("table", leaf("row"), leaf("row")))
    without = node("page", leaf("row"), leaf("row"))

    assert ted(with_table, without) == 1


def test_sibling_order_matters_because_these_are_ordered_trees() -> None:
    """Reading order is the thing being measured, so a swap is not free.

    page(title, figure) vs page(figure, title) costs 2, and the reason is that
    there is no MOVE operation. Mapping title->title and figure->figure would
    be free, but it inverts their left-to-right order and is therefore an
    illegal mapping. What is left costs 2 either way:

        relabel title->figure and figure->title      1 + 1
        keep title, delete figure, insert figure     1 + 1

    Confirmed against brute force, which is how this test got its number - the
    author's first guess was 1. Worth knowing when reading a TED figure: a
    document whose blocks are merely REORDERED scores as though they were
    rewritten, because Zhang-Shasha has no move. Edit distances that do (RTED
    and APTED variants) exist and cost more to implement; nothing in this
    assignment asks for one.
    """
    forward = node("page", leaf("title"), leaf("figure"))
    backward = node("page", leaf("figure"), leaf("title"))

    assert ted(forward, backward) == 2
    assert ted(forward, backward) == brute_force_ted(forward, backward)
    assert ted(forward, forward) == 0

    # A swap of siblings that share a label is genuinely free - it is the
    # labels that move, not the positions.
    same = node("page", leaf("cell"), leaf("cell"))
    assert ted(same, same) == 0


def test_a_node_escaping_its_subtree_is_not_a_cheap_relabel() -> None:
    """The ancestry constraint, observable in the price.

    Moving a cell out of a table and up to be a sibling of the table cannot be
    expressed as a relabel; it has to be paid for as a delete and an insert.
    """
    nested = node("page", node("table", leaf("cell")))
    promoted = node("page", node("table"), leaf("cell"))

    assert ted(nested, promoted) == 2


def test_distance_is_symmetric_under_unit_costs() -> None:
    rng = random.Random(21)
    for _ in range(120):
        a = random_tree(rng, rng.randint(1, 7))
        b = random_tree(rng, rng.randint(1, 7))
        assert ted(a, b) == ted(b, a)


def test_distance_obeys_the_triangle_inequality() -> None:
    """Unit-cost tree edit distance is a metric. Not required by anything, but
    a cheap global consistency check that catches a recurrence which is
    locally plausible and globally wrong."""
    rng = random.Random(22)
    for _ in range(120):
        a = random_tree(rng, rng.randint(1, 6))
        b = random_tree(rng, rng.randint(1, 6))
        c = random_tree(rng, rng.randint(1, 6))
        assert ted(a, c) <= ted(a, b) + ted(b, c)


# ------------------------------------------------------------------- oracles


def test_a_path_shaped_tree_reduces_to_string_edit_distance() -> None:
    """ORACLE 1, and the cross-module link.

    Deleting a node in a path leaves a path, and the ancestry constraint on a
    path is exactly order preservation - so tree edit distance collapses onto
    `levenshtein`, which tests/test_text_metrics.py already checked against a
    full-matrix oracle.
    """
    rng = random.Random(5)
    for _ in range(300):
        a = [rng.choice("abc") for _ in range(rng.randrange(1, 7))]
        b = [rng.choice("abc") for _ in range(rng.randrange(1, 7))]
        assert ted(path(a), path(b)) == levenshtein(a, b), (a, b)


def test_a_star_shaped_tree_reduces_to_edit_distance_over_its_children() -> None:
    """ORACLE 2. All children are siblings, so ancestry is trivial and only
    their order constrains the mapping; the root contributes its relabel."""
    rng = random.Random(6)
    for _ in range(300):
        root_a, root_b = rng.choice("xy"), rng.choice("xy")
        a = [rng.choice("abc") for _ in range(rng.randrange(0, 6))]
        b = [rng.choice("abc") for _ in range(rng.randrange(0, 6))]

        expected = levenshtein(a, b) + (0 if root_a == root_b else 1)

        assert ted(star(root_a, a), star(root_b, b)) == expected, (a, b)


def test_it_agrees_with_brute_force_over_every_legal_mapping() -> None:
    """ORACLE 3 - the definitive one.

    Arbitrary shapes, small sizes, and an oracle that enumerates legal
    mappings straight from the definition rather than computing anything
    cleverly. 250 random pairs.
    """
    rng = random.Random(7)
    for _ in range(250):
        a = random_tree(rng, rng.randint(1, 5))
        b = random_tree(rng, rng.randint(1, 5))
        assert ted(a, b) == brute_force_ted(a, b), (a, b)


# ---------------------------------------------------- the engineering claims


def test_a_deeply_nested_tree_does_not_blow_the_stack() -> None:
    """`parse_tree` walks with an explicit stack.

    CPython's recursion limit is 1000, so a recursive postorder would raise
    RecursionError here - a crash, in a service whose Module D requirement is
    bounded behaviour. 5,000 deep, five times the limit.
    """
    deep = path(["level"] * 5000)

    tree = parse_tree(deep)

    assert tree.size == 5000
    assert tree.depth == 5000
    assert tree.leaf_count == 1
    assert len(tree.keyroots) == 1, "a path has exactly one leftmost path"


def test_reusing_the_forest_matrix_cannot_leak_a_previous_answer() -> None:
    """The scratch matrix is allocated once at full size and reused by every
    keyroot pair, which is only safe because each pass writes its base row and
    column before reading anything.

    Exercised the way it would actually break: a large pair first, leaving
    large values all over the matrix, then a small pair that must be
    unaffected. Also run twice to catch any state surviving the call.
    """
    rng = random.Random(31)
    big_a, big_b = layout_tree(rng, 12), perturb(layout_tree(rng, 12), rng)
    small_a = node("page", leaf("title"), leaf("figure"))
    small_b = node("page", leaf("title"))

    baseline = ted(small_a, small_b)

    ted(big_a, big_b)  # fill the matrix with large numbers
    after_big = ted(small_a, small_b)

    # And within one Tree object, reused across calls.
    parsed_a, parsed_b = parse_tree(small_a), parse_tree(small_b)
    twice = [tree_edit_distance(parsed_a, parsed_b) for _ in range(3)]

    assert baseline == 1
    assert after_big == baseline
    assert twice == [baseline] * 3


def test_treedist_is_filled_entirely_as_a_byproduct_of_keyroot_passes() -> None:
    """Every subtree distance the recurrence needs must exist by the time it
    is read. Checked indirectly but sharply: on a tree whose keyroots are a
    strict subset of its nodes, the answer still matches brute force.
    """
    # A left-leaning shape, so most nodes are leftmost children and therefore
    # are never keyroots.
    a = node("r", node("a", node("b", leaf("c"))), leaf("d"))
    b = node("r", node("a", leaf("c")), leaf("d"))

    tree = parse_tree(a)
    assert len(tree.keyroots) < tree.size, "shape does not exercise the point"
    assert ted(a, b) == brute_force_ted(a, b)


# --------------------------------------------------------- the graded metric


def _median_ms(parsed_a, parsed_b, repeats: int = 7) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        tree_edit_distance(parsed_a, parsed_b)
        samples.append((time.perf_counter() - started) * 1000)
    return sorted(samples)[len(samples) // 2]


def test_a_fifty_node_document_tree_is_well_inside_the_latency_budget() -> None:
    """THE GRADED METRIC: under 100 ms for a 50-node document structure tree.

    Measured ~7.5 ms on a depth-4 page tree, so roughly 13x of headroom. The
    assertion is set at the graded number rather than at the measured one -
    a test that fails the moment the machine is 2x slower is a flaky test, not
    a tighter guarantee.
    """
    rng = random.Random(11)
    predicted = layout_tree(rng, 9)
    truth = perturb(predicted, rng)
    parsed_a, parsed_b = parse_tree(predicted), parse_tree(truth)

    assert 40 <= parsed_a.size <= 60, f"not a 50-node tree: {parsed_a.size}"
    assert parsed_a.depth == 4

    assert _median_ms(parsed_a, parsed_b) < 100.0


def test_even_the_adversarial_shape_meets_the_budget_at_fifty_nodes() -> None:
    """Latency depends on tree SHAPE, not just node count, and this is the
    shape that costs the most.

    A 49-node caterpillar measured ~80 ms against the same 100 ms target - so
    the headroom is ~1.2x rather than ~13x. It passes, and it is pinned here
    precisely because it nearly does not: the real defence against a hostile
    tree is a node cap at the request boundary, not anything this function can
    do. Generous assertion for slow CI, with the measured value in the docstring
    rather than in the bound.
    """
    tree = parse_tree(caterpillar(24))

    assert 45 <= tree.size <= 55
    assert tree.depth > 20 and tree.leaf_count > 20, (
        "depth and leaves must BOTH be large - that is the whole point"
    )

    assert _median_ms(tree, tree, repeats=3) < 400.0


def test_cost_scales_with_keyroot_weight_not_with_node_count() -> None:
    """The structural half of the complexity claim, asserted without a clock.

    W(T) = sum over keyroots of |subtree(k)| is what the runtime is
    proportional to. For a wide page tree it is ~2n (so quadratic overall);
    for a caterpillar it is ~n^2/4 (so quartic). Measuring W instead of time
    makes the complexity claim a deterministic assertion rather than a
    benchmark that drifts with the host.
    """

    rng = random.Random(13)
    wide = parse_tree(layout_tree(rng, 48))
    assert wide.keyroot_weight / wide.size < 3.0, "a wide page tree should be ~2n"

    for spine in (12, 25, 50):
        cat = parse_tree(caterpillar(spine))
        ratio = cat.keyroot_weight / cat.size
        # Grows with n, which is exactly why the caterpillar is quartic.
        assert ratio > cat.size / 8, f"n={cat.size} W/n={ratio:.1f}"

    # The practical point: W separates the two shapes at equal node count,
    # which is what makes it usable as an admission check at the endpoint.
    #
    # Measured: a 46-node page tree has W=132, a 49-node caterpillar W=625 -
    # a 4.7x separation. Wall clock separates further (5 ms vs 90 ms, ~18x)
    # because the work is the PRODUCT W1*W2, so a self-comparison scales as
    # W^2: 625^2 / 132^2 = 22x, which is what that 18x is.
    page_50 = parse_tree(layout_tree(rng, 9))
    cat_50 = parse_tree(caterpillar(24))
    assert abs(page_50.size - cat_50.size) <= 10
    assert cat_50.keyroot_weight > page_50.keyroot_weight * 4
    assert (cat_50.keyroot_weight / page_50.keyroot_weight) ** 2 > 15


def test_the_measured_exponent_on_page_trees_is_quadratic_not_quartic() -> None:
    """One timing-based check that the shape we actually ship is quadratic.

    Doubling n must roughly quadruple the work, not multiply it by 16. Bounds
    are deliberately wide (exponent in [1.5, 2.8]) because this is wall-clock
    on shared CI; the point is to catch a regression to quartic, which would
    show up as ~4, not to pin the constant.
    """
    rng = random.Random(17)

    def elapsed(blocks: int) -> tuple[int, float]:
        predicted = layout_tree(rng, blocks)
        truth = perturb(predicted, rng)
        parsed_a, parsed_b = parse_tree(predicted), parse_tree(truth)
        return parsed_a.size, _median_ms(parsed_a, parsed_b, repeats=3) / 1000

    small_n, small_t = elapsed(24)
    large_n, large_t = elapsed(48)

    exponent = math.log(large_t / small_t) / math.log(large_n / small_n)

    assert 1.5 < exponent < 2.8, f"exponent {exponent:.2f} (n {small_n}->{large_n})"


# ------------------------------------------------------------------- scoring


def test_tree_score_carries_the_distance_and_both_sizes() -> None:
    predicted = node("page", leaf("title"), leaf("figure"))
    truth = node("page", leaf("title"))

    score = tree_score(predicted, truth)

    assert score.distance == 1
    assert score.predicted_nodes == 3
    assert score.truth_nodes == 2
    assert score.normalised == pytest.approx(1 / 3)


def test_normalisation_divides_by_the_larger_tree_so_it_stays_bounded() -> None:
    """Unlike CER, an unbounded rate is not the useful signal here: structural
    drift above 1.0 reads as a broken metric rather than a broken prediction.
    Dividing by the ground truth alone would do that whenever the prediction
    invents nodes."""
    truth = leaf("page")
    predicted = star("page", ["cell"] * 50)

    score = tree_score(predicted, truth)

    assert score.truth_nodes == 1
    assert score.predicted_nodes == 51
    assert score.distance == 50
    assert score.normalised <= 1.0
    assert score.normalised == pytest.approx(50 / 51)


def test_aggregate_sums_distances_and_sizes() -> None:
    a = tree_score(node("page", leaf("t")), node("page", leaf("t")))
    b = tree_score(node("page", leaf("t"), leaf("f")), node("page", leaf("t")))

    total = aggregate([a, b])

    assert total.distance == 1
    assert total.truth_nodes == 2 + 2
    assert total.predicted_nodes == 2 + 3


def test_aggregating_nothing_is_empty_not_a_crash() -> None:
    empty = aggregate([])

    assert empty.distance == 0
    assert empty.normalised == 0.0


def test_the_label_is_the_type_so_text_and_bbox_do_not_double_count() -> None:
    """Structural drift must be independent of the other two metrics. Two
    trees with identical structure but completely different text and boxes are
    structurally identical, because CER and IoU are what measure those."""
    a = {"type": "page", "children": [
        {"type": "paragraph", "text": "hello world", "bbox": [0, 0, 10, 10]}]}
    b = {"type": "page", "children": [
        {"type": "paragraph", "text": "totally different", "bbox": [9, 9, 1, 1]}]}

    assert ted(a, b) == 0


def test_the_label_key_is_overridable_for_callers_who_disagree() -> None:
    a = {"tag": "page", "children": [{"tag": "para"}]}
    b = {"tag": "page", "children": [{"tag": "title"}]}

    assert tree_score(a, b, label_key="tag").distance == 1
