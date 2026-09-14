"""Tree edit distance (Zhang-Shasha), iterative and memory-bounded.

Quantifies structural drift between a predicted and ground-truth document
tree via node inserts/deletes/relabels (no move op - a sibling swap costs
2, not 1; see tests/test_ted.py). Graded target: <100ms for a 50-node tree.

Nodes are numbered in postorder so every subtree is a contiguous integer
range, which is what lets the DP index by two ints instead of node sets.
Cost is Theta(W(T1).W(T2)), not the looser textbook O(n.m.min(depth,leaves))
bound - see `Tree.keyroot_weight`. Both DP tables are O(|T1|.|T2|);
`forestdist` is allocated once and reused across keyroot pairs, `treedist`
cannot be (the recurrence reads arbitrary past entries). Traversal is
iterative, not recursive, so a degenerate deep tree cannot raise
RecursionError.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Tree",
    "TreeScore",
    "aggregate",
    "parse_tree",
    "tree_edit_distance",
    "tree_score",
]


@dataclass(frozen=True, slots=True)
class Tree:
    """A tree flattened into the arrays Zhang-Shasha actually needs.

    All three sequences are 1-INDEXED with a sentinel at position 0. That is a
    deliberate concession to reviewability: the published recurrence is written
    over 1-based postorder indices, and an implementation that silently shifts
    everything by one is far harder to check against the paper than one with a
    wasted slot.

    labels    postorder node labels
    leftmost  leftmost[i] = postorder index of the leftmost leaf under node i,
              so subtree(i) is exactly the range [leftmost[i], i]
    keyroots  ascending; the only nodes needing their own forest-DP pass
    """

    labels: tuple[Any, ...]
    leftmost: tuple[int, ...]
    keyroots: tuple[int, ...]
    depth: int

    @property
    def size(self) -> int:
        return len(self.labels) - 1  # sentinel

    @property
    def leaf_count(self) -> int:
        """A leaf is a node that is its own leftmost descendant."""
        return sum(1 for i in range(1, self.size + 1) if self.leftmost[i] == i)

    @property
    def keyroot_weight(self) -> int:
        """W(T) = sum over keyroots of |subtree(k)| - the real cost driver.

        `tree_edit_distance(a, b)` does Theta(a.keyroot_weight *
        b.keyroot_weight) cell updates, which is a much better predictor of
        runtime than node count: a 49-node page tree and a 49-node caterpillar
        differ by ~18x in wall clock (5 ms vs 90 ms) at the same size.

        Exposed because it is O(keyroots) to compute and therefore lets a
        caller bound the work BEFORE doing it. A `/evaluate` endpoint handed a
        hostile tree cannot be defended by anything inside the DP; it has to
        refuse the input, and this is the number to refuse on. Nothing calls it
        yet - the endpoint does not exist until after Step 17 - so it is a hook
        and a measurement, not a guard.
        """
        return sum(k - self.leftmost[k] + 1 for k in self.keyroots)


def parse_tree(
    root: Any | None,
    *,
    label_key: str = "type",
    children_key: str = "children",
) -> Tree:
    """Flatten a nested mapping into postorder arrays, without recursion.

    Accepts the shape `mock_model.payloads` emits - nested dicts with `type`
    and `children`, other keys ignored - and `None` for an empty tree. A node
    with no `children` key is a leaf; a missing `label_key` is an error rather
    than a default, because a silently-shared default label would make every
    unlabelled node match every other one for free.
    """
    labels: list[Any] = [None]  # 1-based sentinel
    leftmost: list[int] = [0]

    if root is None:
        return Tree(tuple(labels), tuple(leftmost), (), 0)

    max_depth = 0
    # (node, already_expanded, subtree_start_index, depth)
    stack: list[tuple[Any, bool, int, int]] = [(root, False, 0, 1)]
    while stack:
        node, expanded, start, depth = stack.pop()

        if expanded:
            labels.append(_label_of(node, label_key))
            # The subtree occupies [start, here], so its leftmost leaf is at
            # `start` - which is why the start index is captured on the way
            # down rather than reconstructed from subtree sizes on the way up.
            leftmost.append(start)
            continue

        max_depth = max(max_depth, depth)
        # Everything strictly left of this subtree has been emitted and
        # nothing inside it has, so the next index is where it begins.
        here = len(labels)
        stack.append((node, True, here, depth))
        children = _children_of(node, children_key)
        # Reversed, so the leftmost child is popped first and postorder comes
        # out left to right.
        for child in reversed(children):
            stack.append((child, False, 0, depth + 1))

    # k is a keyroot unless some later node shares its leftmost descendant.
    # Scanning from the end, the first node seen for a given leftmost value is
    # the highest one holding it.
    seen: set[int] = set()
    keyroots: list[int] = []
    for k in range(len(labels) - 1, 0, -1):
        if leftmost[k] not in seen:
            seen.add(leftmost[k])
            keyroots.append(k)
    keyroots.reverse()  # ascending: enclosing subtrees must come last

    return Tree(tuple(labels), tuple(leftmost), tuple(keyroots), max_depth)


def _label_of(node: Any, label_key: str) -> Any:
    if isinstance(node, Mapping):
        try:
            return node[label_key]
        except KeyError as exc:
            raise ValueError(
                f"tree node is missing {label_key!r}: {node!r}"
            ) from exc
    return node


def _children_of(node: Any, children_key: str) -> Sequence[Any]:
    if isinstance(node, Mapping):
        children = node.get(children_key) or ()
        if not isinstance(children, Sequence) or isinstance(children, str):
            raise ValueError(
                f"{children_key!r} must be a sequence, got {type(children).__name__}"
            )
        return children
    return ()


def tree_edit_distance(a: Tree, b: Tree) -> int:
    """Minimum number of node inserts, deletes and relabels taking `a` to `b`.

    Unit costs: insert 1, delete 1, relabel 1 when the labels differ and 0
    when they agree. Relabel at 1 against delete-plus-insert at 2 is what makes
    a changed node type cheaper than a restructure, which is the right
    preference for layout drift. Non-unit costs would be a change to the three
    constants below and nothing else; they are inlined rather than passed as
    callables because this function is on a graded latency budget and a call
    per DP cell is not free.
    """
    n1, n2 = a.size, b.size
    if n1 == 0:
        return n2  # insert everything
    if n2 == 0:
        return n1  # delete everything

    labels_a, labels_b = a.labels, b.labels
    left_a, left_b = a.leftmost, b.leftmost

    treedist = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    # Allocated once, reused by every keyroot pair. See the module docstring
    # for why no pass can observe another's leftovers.
    forestdist = [[0] * (n2 + 1) for _ in range(n1 + 1)]

    for root_a in a.keyroots:
        spine_a = left_a[root_a]
        offset_a = spine_a - 1
        rows = root_a - offset_a

        for root_b in b.keyroots:
            spine_b = left_b[root_b]
            offset_b = spine_b - 1
            cols = root_b - offset_b

            # Base cases: emptying one forest into the other.
            forestdist[0][0] = 0
            for i in range(1, rows + 1):
                forestdist[i][0] = forestdist[i - 1][0] + 1
            base = forestdist[0]
            for j in range(1, cols + 1):
                base[j] = base[j - 1] + 1

            for i in range(1, rows + 1):
                node_a = i + offset_a
                left_of_a = left_a[node_a]
                label_a = labels_a[node_a]
                on_spine_a = left_of_a == spine_a
                current, previous = forestdist[i], forestdist[i - 1]
                treedist_row = treedist[node_a]

                for j in range(1, cols + 1):
                    node_b = j + offset_b
                    delete = previous[j] + 1
                    insert = current[j - 1] + 1

                    if on_spine_a and left_b[node_b] == spine_b:
                        # Both forests are whole trees, so this cell is the
                        # subtree distance and is worth keeping.
                        relabel = previous[j - 1] + (
                            0 if label_a == labels_b[node_b] else 1
                        )
                        value = delete if delete < insert else insert
                        if relabel < value:
                            value = relabel
                        current[j] = value
                        treedist_row[node_b] = value
                    else:
                        # Match subtree(node_a) against subtree(node_b) whole,
                        # then align what sits to the left of both.
                        detached = (
                            forestdist[left_of_a - 1 - offset_a][
                                left_b[node_b] - 1 - offset_b
                            ]
                            + treedist_row[node_b]
                        )
                        value = delete if delete < insert else insert
                        if detached < value:
                            value = detached
                        current[j] = value

    return treedist[n1][n2]


# ------------------------------------------------------------------- scoring


@dataclass(frozen=True, slots=True)
class TreeScore:
    """Distance kept alongside its scale, so pages can be combined.

    Same discipline as `app.eval.text.TextScore` and `app.eval.iou.BoxReport`:
    a normalised rate cannot be averaged correctly, so the numerator and a
    denominator are carried and `aggregate` sums both.
    """

    distance: int
    truth_nodes: int
    predicted_nodes: int

    @property
    def normalised(self) -> float:
        """Distance per node of the larger tree, in [0, 1].

        The denominator is max(|T1|, |T2|) rather than the ground truth's size,
        because the distance is bounded by that maximum (delete everything,
        insert everything is |T1| + |T2|, but any node can be relabelled
        instead of both deleted and inserted). Dividing by the ground truth
        alone would let a prediction that invents 500 nodes report a drift
        above 1.0, which reads as a broken metric rather than a broken
        prediction - the opposite of the CER case, where an unbounded rate is
        the useful signal because insertion is the failure being measured.
        """
        scale = max(self.truth_nodes, self.predicted_nodes)
        if scale == 0:
            return 0.0
        return self.distance / scale


def tree_score(
    predicted: Any | None,
    truth: Any | None,
    *,
    label_key: str = "type",
    children_key: str = "children",
) -> TreeScore:
    """Parse both trees and score them. The convenience entry point."""
    tree_predicted = parse_tree(
        predicted, label_key=label_key, children_key=children_key
    )
    tree_truth = parse_tree(truth, label_key=label_key, children_key=children_key)
    return TreeScore(
        distance=tree_edit_distance(tree_predicted, tree_truth),
        truth_nodes=tree_truth.size,
        predicted_nodes=tree_predicted.size,
    )


def aggregate(scores: Iterable[TreeScore]) -> TreeScore:
    """Micro-average across pages: sum distances, sum sizes."""
    scores = list(scores)
    if not scores:
        return TreeScore(0, 0, 0)
    return TreeScore(
        distance=sum(score.distance for score in scores),
        truth_nodes=sum(score.truth_nodes for score in scores),
        predicted_nodes=sum(score.predicted_nodes for score in scores),
    )
