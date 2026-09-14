"""CER and WER: normalised edit distance over grapheme clusters and words.

Two-row DP (O(min(m,n)) space, not the full matrix - measured at ~900 MB
for a full matrix vs ~0.36 MB for two rows at n=5000; see
tests/test_text_metrics.py). Characters are Unicode grapheme clusters by
default, not code points, so multi-codepoint scripts like Devanagari score
correctly; `unit="codepoint"` is kept for comparison against code-point
baselines (jiwer, torchmetrics). Rates aggregate as
sum(errors)/sum(lengths), never mean(per-page rate) - see `aggregate()`.
CER is intentionally not bounded to 1.0: it is errors over reference
length, and clamping would hide runaway insertion.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "TextScore",
    "aggregate",
    "cer",
    "cer_score",
    "graphemes",
    "levenshtein",
    "wer",
    "wer_score",
    "words",
]

Unit = Literal["grapheme", "codepoint"]


# --------------------------------------------------------------- segmentation

_MARK_CATEGORIES = frozenset({"Mn", "Mc", "Me"})
_ZWJ = "\u200d"  # ZERO WIDTH JOINER
_ZWNJ = "\u200c"  # ZERO WIDTH NON-JOINER
_VIRAMA_COMBINING_CLASS = 9


def _joins_following(ch: str) -> bool:
    """Does this code point force the NEXT one into the same cluster?

    Two cases: an explicit zero-width joiner, and an Indic virama (canonical
    combining class 9), which is what turns two consonants into a conjunct.
    A zero-width NON-joiner is deliberately excluded - it attaches to the
    cluster behind it but its whole purpose is to stop the conjunct forming.
    """
    return ch == _ZWJ or unicodedata.combining(ch) == _VIRAMA_COMBINING_CLASS


def graphemes(text: str) -> list[str]:
    """Split into user-perceived characters.

    An approximation of UAX #29, covering the cases an Indic-language OCR
    system actually meets: combining marks (Mn/Mc/Me, which includes vowel
    signs, nuktas, viramas and variation selectors), Indic conjuncts formed
    with a virama, and ZWJ/ZWNJ sequences.

    Known gaps, stated rather than hidden: regional-indicator pairs (flag
    emoji) split into two clusters, and emoji modifiers such as skin tone
    (category Sk) do not attach. Neither appears in document OCR, and closing
    them properly means a full UAX #29 state machine or a dependency on
    `regex`. If either ever matters, that is the fix - not more special cases
    here.
    """
    clusters: list[str] = []
    join_next = False
    for ch in text:
        extend = clusters and (
            join_next
            or ch in (_ZWJ, _ZWNJ)
            or unicodedata.category(ch) in _MARK_CATEGORIES
        )
        if extend:
            clusters[-1] += ch
        else:
            clusters.append(ch)
        join_next = _joins_following(ch)
    return clusters


def words(text: str) -> list[str]:
    """Whitespace tokenisation, and nothing else.

    No case folding, no punctuation stripping, no Unicode normalisation. Those
    are evaluation POLICY and they belong to the caller, because each one
    silently lowers the reported error rate: fold case and a model that cannot
    do capitalisation scores as though it can. An OCR benchmark with hidden
    normalisation baked into the metric reports a better number than it earned,
    and the reader has no way to tell.
    """
    return text.split()


def _units(text: str, unit: Unit) -> Sequence[Hashable]:
    if unit == "grapheme":
        return graphemes(text)
    if unit == "codepoint":
        return text
    raise ValueError(f"unit must be 'grapheme' or 'codepoint', got {unit!r}")


# ------------------------------------------------------------------- the DP


def levenshtein(reference: Sequence[Hashable], hypothesis: Sequence[Hashable]) -> int:
    """Edit distance with unit costs, in O(mn) time and O(min(m,n)) space.

    Works on any sequence of hashables - characters, grapheme clusters, words,
    layout-node labels - because the recurrence only ever asks `!=`. Step 17's
    tree edit distance reuses that shape one dimension up.
    """
    # Swap so the inner loop (and therefore the retained row) is the shorter
    # side. Safe here precisely because distance is symmetric.
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    if not hypothesis:
        return len(reference)

    # Row 0: turning an empty reference prefix into hyp[:j] costs j insertions.
    previous = list(range(len(hypothesis) + 1))
    for i, ref_unit in enumerate(reference, start=1):
        # Column 0: turning ref[:i] into an empty hypothesis costs i deletions.
        current = [i]
        for j, hyp_unit in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,  # delete ref_unit
                    current[j - 1] + 1,  # insert hyp_unit
                    previous[j - 1] + (ref_unit != hyp_unit),  # substitute
                )
            )
        previous = current
    return previous[-1]


# ----------------------------------------------------------------- scoring


@dataclass(frozen=True, slots=True)
class TextScore:
    """A rate kept as a fraction so it can be combined.

    `errors` is the edit distance, `length` the reference length in whatever
    unit was scored. Both are carried because a rate on its own is not
    aggregatable - see `aggregate`.
    """

    errors: int
    length: int
    unit: str

    @property
    def rate(self) -> float:
        """Errors per reference unit. Unbounded above; see the module docstring.

        An empty reference with a non-empty hypothesis is `inf`, not 1.0 and
        not an exception: there is genuinely no reference to be wrong about, so
        any finite rate would be a fabrication, while raising would let one bad
        page abort a whole corpus. `inf` is loud, propagates, and is still
        aggregatable - `aggregate` sums the integers, so a corpus containing
        such a page gets a finite, honest number as long as some other page has
        a reference.
        """
        if self.length == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return self.errors / self.length


def cer_score(reference: str, hypothesis: str, *, unit: Unit = "grapheme") -> TextScore:
    """Character error rate for one pair, as an aggregatable fraction."""
    ref_units = _units(reference, unit)
    hyp_units = _units(hypothesis, unit)
    distance = levenshtein(ref_units, hyp_units)
    return TextScore(distance, len(ref_units), unit)


def wer_score(reference: str, hypothesis: str) -> TextScore:
    """Word error rate for one pair, as an aggregatable fraction."""
    ref_words = words(reference)
    hyp_words = words(hypothesis)
    distance = levenshtein(ref_words, hyp_words)
    return TextScore(distance, len(ref_words), "word")


def cer(reference: str, hypothesis: str, *, unit: Unit = "grapheme") -> float:
    """Convenience for a single pair. Use `cer_score` + `aggregate` for a set."""
    return cer_score(reference, hypothesis, unit=unit).rate


def wer(reference: str, hypothesis: str) -> float:
    """Convenience for a single pair. Use `wer_score` + `aggregate` for a set."""
    return wer_score(reference, hypothesis).rate


def aggregate(scores: Iterable[TextScore]) -> TextScore:
    """Micro-average: sum the numerators, sum the denominators.

    This is the corpus-level definition of CER/WER, and the reason `rate` is a
    property rather than the stored value. Averaging per-page rates instead
    (a macro-average) gives every page equal weight regardless of length, so a
    single 3-character page scoring 4.0 outweighs fifty clean 2,000-character
    pages. Both numbers have uses, but only one is what "CER" means, and the
    difference is easy to ship by accident.

    Mixing units is rejected: a summed denominator of graphemes and words is
    not a quantity.
    """
    scores = list(scores)
    if not scores:
        return TextScore(0, 0, "empty")

    units = {score.unit for score in scores}
    if len(units) > 1:
        raise ValueError(f"cannot aggregate mixed units: {sorted(units)}")

    return TextScore(
        errors=sum(score.errors for score in scores),
        length=sum(score.length for score in scores),
        unit=units.pop(),
    )
