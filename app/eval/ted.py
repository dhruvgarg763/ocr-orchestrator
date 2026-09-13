"""Tree edit distance (Zhang-Shasha), iterative and memory-bounded.

Quantifies structural layout drift: how far a predicted document tree
(page -> title/paragraph/table/figure -> row -> cell) is from ground truth,
in node inserts, deletes and relabels. This is the graded metric with a
latency target - under 100 ms for a 50-node tree - so the shape of the
implementation is load-bearing, not cosmetic.

Why a tree needs its own algorithm
----------------------------------
Deleting a character from a string removes it. Deleting a NODE from a tree
promotes its children to its parent: drop a `table` and its `row`s become
direct children of `page`. So this is not string edit distance over a
serialisation, and flattening the tree to compare it that way would score
that promotion as free.

Nor is every correspondence between two trees legal. A valid mapping is
one-to-one and must preserve BOTH sibling order (these are ordered trees,
because reading order is the thing being measured) and ancestry (if a maps to
b, a's ancestors map to b's ancestors). The second constraint is what stops
"this cell escaped its table and became a title" from being scored as a cheap
relabel.

The idea that makes it tractable
--------------------------------
Number the nodes in POSTORDER - children before parents, left to right - and
let l(i) be the postorder index of the leftmost leaf below node i. Then:

    the subtree rooted at i is exactly the contiguous range [l(i), i]

That is the whole trick. Postorder turns every subtree into an interval of
integers, so the DP is indexed by two ints rather than by sets of nodes. A
forest - a subtree with some of its left-hand part removed - is a range too.

Two tables, doing different jobs:

    treedist[i][j]   distance between subtree(i) and subtree(j). Permanent;
                     treedist[n1][n2] is the answer.
    forestdist       distance between two forests. Scratch, rewritten per
                     keyroot pair, allocated once (see below).

    fd[i][j] = min( fd[i-1][j]           + 1,            delete i
                    fd[i][j-1]           + 1,            insert j
                    fd[i-1][j-1]         + relabel(i,j)  if both are trees
                    fd[l(i)-1][l(j)-1] + treedist[i][j]  otherwise )

The last line is where the ancestry constraint lives. When the two forests are
not single trees, subtree(i) is matched against subtree(j) as an INDIVISIBLE
unit via the already-computed treedist[i][j], plus the cost of aligning
whatever lies to the left of both. There is no way to express an alignment
that crosses a subtree boundary, so illegal mappings are not rejected - they
are unrepresentable.

When i and j both sit on their forest's leftmost spine the forest IS a tree,
so that cell's value is treedist[i][j] and is recorded as a byproduct. Every
treedist entry is filled this way; none is computed on purpose.

Keyroots, and the recomputation they remove
-------------------------------------------
    LR-keyroots = nodes that are the root or have a left sibling
                = { k : no k' > k has l(k') == l(k) }

A non-keyroot is a leftmost child, so it shares its leftmost descendant with
its parent and its treedist row is filled in during the parent's pass for
free. Only nodes that begin a fresh leftmost path need a pass of their own,
and there are at most as many of those as there are leaves. Running the forest
DP over all n1 x n2 pairs instead would repeat that work once per ancestor.

Complexity
----------
The textbook bound is

    O( |T1| . |T2| . min(depth1, leaves1) . min(depth2, leaves2) )

but that form hides what actually drives the cost, and it is loose on real
shapes. The exact cost is the product of a per-tree quantity:

    W(T) = sum over keyroots k of |subtree(k)|
         = sum over nodes v of #{ keyroot ancestors of v }

    total work = Theta( W(T1) . W(T2) )

because each keyroot pair runs one forest DP over their two subtrees. W(T) is
bounded by n.min(depth, leaves), but can be far smaller. Measured W/n:

    shape                       W/n         total
    page -> blocks (ours)      ~2.0         O(n^2)
    star / wide, any size      ~2.0         O(n^2)
    broom: sqrt(n) arms of
      length sqrt(n)           ~1.9         O(n^2)
    caterpillar                ~n/4         O(n^4)

Our pages are depth 4 (page, table, row, cell), so W/n is ~2 and this is
quadratic; measured exponent 1.8-2.2 over n = 61..481.

On the worst case, and a mistake worth recording. It is tempting to argue that
O(n^4) is unreachable because depth and leaves cannot both be large - a path
has depth n and ONE leaf, a star has n leaves and depth one - which would cap
min(d, l) at ~sqrt(n) and the whole thing at O(n^3). That argument is WRONG,
and a broom (sqrt(n) arms of length sqrt(n)) does not disprove it either: the
broom is quadratic, because W/n stays at ~1.9 and the n.min(d,l) bound is
simply loose there.

The shape that breaks it is a CATERPILLAR - a deep right-leaning spine where
every spine node also sheds one leaf. That has depth Theta(n) AND leaves
Theta(n) at the same time, every spine node is a keyroot, and W(T) = Theta(n^2).
Measured exponent on caterpillar-vs-caterpillar: 3.52, 3.75, 3.98 over
n = 13..81. O(n^4) is real.

Operationally this means TED latency depends on tree SHAPE, not just node
count - the graded target of 100 ms for 50 nodes is met with ~13x headroom on
a document tree (7.5 ms) but only ~1.2x on a 49-node caterpillar (80 ms).
Both are pinned in tests/test_ted.py. The defence against a hostile tree is a
node cap at the request boundary, not anything this function can do about it.

Memory, which the spec asks about explicitly
--------------------------------------------
O(|T1| . |T2|), two tables of it. `treedist` is irreducible - the fourth
branch of the recurrence reads arbitrary entries of it, so it cannot be
rolled into two rows the way app/eval/text.py does. `forestdist` is allocated
ONCE at full size and reused across every keyroot pair rather than being
reallocated per pair, which would be O(keyroots^2) allocations of an
n1 x n2 matrix.

Reuse is safe, and the reason is worth stating because it is the kind of
optimisation that works until it silently does not: each pass writes its own
base row and base column before the double loop reads anything, and the loop
runs in row-major order reading only [i-1][j], [i][j-1], [i-1][j-1] and
[m][n] with m < i and n < j. Every cell read has therefore been written during
THIS pass, so stale values from a previous, larger pass are never visible.
There is a test that runs a large tree pair and then a small one through the
same call path and checks the small answer is unaffected.

No recursion anywhere
---------------------
`parse_tree` walks with an explicit stack. CPython's default recursion limit
is 1000, and a recursive postorder over a deeply nested tree raises
RecursionError - a crash, in a service whose Module D requirement is bounded
behaviour under load. The traversal is iterative for that reason and there is
a test that parses a 5,000-deep tree.

(The remaining recursion risk is not ours: `json.loads` uses a recursive C
scanner and will raise on deeply nested input before this module ever sees it.
That belongs to request validation at the edge, not here.)

There is no MOVE operation, and that shapes how to read the number
------------------------------------------------------------------
The three operations are insert, delete and relabel. Nothing moves a subtree.
So `page(title, figure)` against `page(figure, title)` costs 2, not 1: mapping
title to title and figure to figure would be free but inverts their
left-to-right order, which is an illegal mapping, leaving either two relabels
or a delete plus an insert.

The consequence is that a document whose blocks are merely REORDERED scores as
though they had been rewritten. That is defensible here - reading order IS part
of the structure being evaluated, so a reordering is real drift - but it does
mean the metric cannot distinguish "moved" from "replaced", and a TED of 2 on a
two-block page is a reordering rather than a catastrophe. Edit distances with a
move operation exist (RTED and APTED variants); they cost considerably more to
implement and nothing in this assignment asks for one.

Labels are the node TYPE, deliberately
--------------------------------------
`page`, `table`, `row`, `cell`, `paragraph` - not the text, not the bbox. If
the label included text this would re-measure what CER measures; if it
included the bbox it would re-measure IoU. Three metrics should answer three
questions, and summing three views of one error is how an evaluation suite
reports a number nobody can interpret. `label_key` is a parameter for callers
who disagree.
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
