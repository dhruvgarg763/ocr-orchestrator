"""CER/WER: the DP's correctness, and the three ways the metric gets reported wrong.

The algorithm half of this file is cheap to test - a naive full-matrix
Levenshtein is six lines, so the shipped two-row version can be checked against
it on thousands of random inputs rather than on a handful of hand-picked pairs.
The interesting half is everything around the algorithm: what counts as a
character, which direction an insertion points, and how per-page rates combine.
Those are where a correct DP still produces a wrong published number.
"""

from __future__ import annotations

import random
import string
import sys
import tracemalloc

import pytest

from app.eval.text import (
    aggregate,
    cer,
    cer_score,
    graphemes,
    levenshtein,
    wer,
    wer_score,
    words,
)

# Real Devanagari, spelled out by code point so the test does not depend on
# this file's own encoding surviving a round trip through an editor.
KA = "क"  # क  consonant
SHA = "ष"  # ष  consonant
HA = "ह"  # ह  consonant
NA = "न"  # न  consonant
DA = "द"  # द  consonant
VIRAMA = "्"  # ्  suppresses the inherent vowel, forming a conjunct
SIGN_I = "ि"  # ि  vowel sign (Mc, spacing)
SIGN_II = "ी"  # ी  vowel sign (Mc, spacing)

KSHI = KA + VIRAMA + SHA + SIGN_I  # क्षि  4 code points, 1 written character
HINDI = HA + SIGN_I + NA + VIRAMA + DA + SIGN_II  # हिन्दी 6 code points, 2 clusters


def full_matrix(a, b) -> int:
    """The textbook O(mn) space version, kept deliberately naive.

    This is the oracle, so it must be obviously right rather than clever. It is
    also exactly what the shipped implementation refuses to allocate.
    """
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )
    return d[m][n]


# ------------------------------------------------------- the DP is the DP


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
        ("kitten", "sitting", 3),  # the canonical example
        ("flaw", "lawn", 2),
        ("saturday", "sunday", 3),
        ("a", "b", 1),
    ],
)
def test_known_distances(a: str, b: str, expected: int) -> None:
    assert levenshtein(a, b) == expected


def test_the_two_row_dp_agrees_with_a_full_matrix_on_random_inputs() -> None:
    """4,000 random pairs over a 3-letter alphabet.

    A small alphabet is the point, not a shortcut: it forces frequent
    coincidental matches, which is what exercises the diagonal branch and the
    tie-breaking between the three predecessors. Random pairs over 26 letters
    are almost all substitutions and would leave those paths mostly untouched.
    """
    rng = random.Random(11)
    for _ in range(4000):
        a = "".join(rng.choice("abc") for _ in range(rng.randrange(0, 9)))
        b = "".join(rng.choice("abc") for _ in range(rng.randrange(0, 9)))
        oracle = full_matrix(a, b)
        assert levenshtein(a, b) == oracle, (a, b)


def test_the_shorter_side_swap_does_not_change_the_answer() -> None:
    """`levenshtein` swaps its arguments so the retained row is the shorter
    side - safe only because distance is symmetric. This is the property that
    makes the swap safe, checked directly rather than assumed."""
    rng = random.Random(13)
    for _ in range(500):
        a = "".join(rng.choice("abcd") for _ in range(rng.randrange(0, 12)))
        b = "".join(rng.choice("abcd") for _ in range(rng.randrange(0, 12)))
        assert levenshtein(a, b) == levenshtein(b, a)


def test_the_dp_works_on_any_sequence_of_hashables() -> None:
    """Nothing in the recurrence is string-specific - it only ever asks `!=`.
    That is what lets WER reuse it over word lists, and Step 17's tree edit
    distance reuse the same shape over node labels."""
    assert levenshtein(["the", "cat", "sat"], ["the", "cat", "stood"]) == 1
    assert levenshtein((1, 2, 3), (1, 3)) == 1


def matrix_container_bytes(n: int) -> int:
    """The list-slot cost of an (n+1)x(n+1) matrix, ints excluded.

    Measured with `getsizeof` on the structure rather than `tracemalloc` around
    the DP, for a boring reason: tracing instruments every individual
    allocation, and the DP performs O(n^2) of them, so measuring a 1,600-unit
    pair that way took 48 seconds - longer than the rest of the suite combined.
    The container cost alone is enough to demonstrate the growth rate, and it
    is a strict UNDER-estimate of the real thing.
    """
    matrix = [[0] * (n + 1) for _ in range(n + 1)]
    return sys.getsizeof(matrix) + sum(sys.getsizeof(row) for row in matrix)


def test_a_full_matrix_grows_quadratically() -> None:
    """Half the claim: what the shipped code declines to allocate.

    Doubling n must roughly quadruple the footprint. Measured 3.95x, and 200 MB
    of bare list slots at n=5000 - before a single int object. With the ints it
    is ~900 MB, against a 500 MB budget for the whole service.
    """
    small = matrix_container_bytes(400)
    large = matrix_container_bytes(800)

    assert 3.5 < large / small < 4.5, f"growth was {large / small:.2f}x, not quadratic"
    assert matrix_container_bytes(5000) > 150 * 1024 * 1024


def test_two_rows_grow_linearly_not_quadratically() -> None:
    """The other half, and the one that is actually a claim about our code.

    Doubling n must roughly DOUBLE the peak, not quadruple it. Measured 1.97x.

    Two constraints on how this is measured. `tracemalloc` rather than RSS,
    because 65 KB of list is served from arenas the allocator already holds and
    RSS would show nothing - the same instrument choice, for the same reason, as
    tests/test_pdf_splitter.py. And n >= 400 in both samples, because Python
    interns ints up to 256: below that every cell reuses a cached object, only
    the list slot is charged, and the measured ratio comes out at 8.9x instead
    of 2x. A memory measurement taken under the small-int cache is measuring
    the cache.
    """
    rng = random.Random(3)

    def peak_bytes(n: int) -> int:
        a = "".join(rng.choice(string.ascii_letters) for _ in range(n))
        b = "".join(rng.choice(string.ascii_letters) for _ in range(n))
        tracemalloc.start()
        levenshtein(a, b)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak

    small = peak_bytes(400)
    large = peak_bytes(800)

    assert 1.6 < large / small < 2.6, f"growth was {large / small:.2f}x, not linear"
    assert large < 1024 * 1024, f"two rows used {large / 1024 / 1024:.2f} MiB"
    # And the comparison that matters: at n=800 the matrix needs 79x more for
    # its list slots alone, ints excluded.
    assert matrix_container_bytes(800) > large * 20


# --------------------------------------------------- what a character is


def test_a_devanagari_conjunct_is_one_character_not_four() -> None:
    """क्षि is a base consonant, a virama, a second consonant and a vowel sign.
    A reader sees one character. `len()` says 4."""
    assert len(KSHI) == 4
    assert graphemes(KSHI) == [KSHI]


def test_vowel_signs_and_conjuncts_cluster_together() -> None:
    """हिन्दी is ह+ि then न+्+द+ी: the virama pulls the following consonant in,
    and the trailing vowel sign attaches to that. Two written characters from
    six code points."""
    assert len(HINDI) == 6
    assert graphemes(HINDI) == [HA + SIGN_I, NA + VIRAMA + DA + SIGN_II]


def test_a_zero_width_non_joiner_attaches_but_does_not_conjoin() -> None:
    """ZWNJ exists precisely to stop the conjunct forming, so it must join the
    cluster behind it while leaving the next consonant standing alone. Treating
    it like ZWJ would merge exactly the pairs the author asked to keep apart."""
    zwnj = "\u200c"
    assert graphemes(KA + VIRAMA + zwnj + SHA) == [KA + VIRAMA + zwnj, SHA]
    assert graphemes(KA + VIRAMA + SHA) == [KA + VIRAMA + SHA]


def test_ascii_is_unaffected_by_grapheme_segmentation() -> None:
    """The safety property: making CER correct for Indic scripts must not move
    any Latin number, or every existing baseline comparison breaks."""
    assert graphemes("hello world") == list("hello world")
    assert cer("hello", "h3llo") == cer("hello", "h3llo", unit="codepoint")


def test_a_leading_combining_mark_does_not_crash_or_vanish() -> None:
    """Malformed input - a vowel sign with no base consonant - is real in OCR
    output. It must start its own cluster rather than being dropped or
    indexing off the front of an empty list."""
    assert graphemes(SIGN_I + KA) == [SIGN_I, KA]


def test_the_unit_changes_the_reported_cer_on_indic_text() -> None:
    """The measurement that justifies `unit="grapheme"` as the default.

    One misread conjunct in a two-character word is half the word wrong.
    Counting code points spreads that same single error over a denominator of
    six and reports 0.33 - a model that mangles Devanagari conjuncts looks
    better than one that drops Latin letters at the same visible rate.
    """
    hypothesis = HA + SIGN_I + DA + SIGN_II  # the conjunct's न् is gone

    as_graphemes = cer(HINDI, hypothesis)
    as_codepoints = cer(HINDI, hypothesis, unit="codepoint")

    assert as_graphemes == pytest.approx(0.5), "1 of 2 written characters wrong"
    assert as_codepoints == pytest.approx(2 / 6), "2 of 6 code points differ"
    assert as_graphemes > as_codepoints


def test_an_unknown_unit_is_rejected_rather_than_defaulted() -> None:
    """A typo'd unit must not silently fall back to one of them: the two
    produce different published numbers, so a quiet default is a wrong result
    rather than an inconvenience."""
    with pytest.raises(ValueError, match="unit must be"):
        cer("a", "b", unit="characters")  # type: ignore[arg-type]


# ------------------------------------------------------------ tokenisation


def test_wer_is_cer_over_words() -> None:
    assert wer("the cat sat on the mat", "the cat sat on a mat") == pytest.approx(1 / 6)
    assert wer_score("a b c", "a b c").errors == 0


def test_word_tokenisation_collapses_arbitrary_whitespace() -> None:
    """`split()` with no argument, not `split(" ")`. OCR output is full of
    double spaces, newlines and tabs at line breaks; `split(" ")` yields empty
    strings for each run of them, and those empties are counted as words -
    inflating the denominator and fabricating insertions."""
    assert words("a  b\n\tc  ") == ["a", "b", "c"]
    assert wer("a b c", "a  b\n c") == 0.0


def test_tokenisation_does_not_normalise_case_or_punctuation() -> None:
    """Deliberate, and the reason it is asserted: every hidden normalisation
    lowers the published number. Folding case makes a model that cannot
    capitalise score as though it can. Callers who want that must ask."""
    assert wer("The Cat.", "the cat") == pytest.approx(1.0)
    assert cer("Hello!", "hello") > 0


# --------------------------------------------------- how rates are reported


def test_cer_is_not_bounded_by_one() -> None:
    """The denominator is the reference, so runaway insertion is unbounded -
    and must stay that way. Clamping to 1.0 would make a model that
    hallucinated 150 characters indistinguishable from one that got three
    characters wrong, which is the single most useful thing this number says."""
    assert cer("abc", "x" * 150) == pytest.approx(50.0)
    assert cer("abc", "abc" + "x" * 297) == pytest.approx(99.0)


def test_an_empty_reference_with_output_is_infinite_not_one() -> None:
    """There is no reference to be wrong about, so any finite rate is invented.
    `inf` is loud and still aggregatable; raising would let one blank page
    abort a whole corpus report."""
    assert cer("", "") == 0.0
    assert wer("", "") == 0.0
    assert cer("", "spurious") == float("inf")
    # ...and the integers behind it are still exact, which is what saves the
    # corpus number below.
    assert cer_score("", "spurious").errors == len("spurious")


def test_corpus_cer_is_a_micro_average_not_a_mean_of_rates() -> None:
    """REGRESSION for the statistics, not the algorithm.

    Fifty clean 11-character pages and one 2-character page scored 3.5. The
    corpus is 552 reference characters with 7 errors: 0.0127. Averaging the
    per-page rates gives 0.0686 - 5.4x worse - because the one 2-character page
    is weighted the same as fifty long correct ones. Both are computable from
    the same data; only one is what "CER" means.
    """
    pairs = [("hello world", "hello world")] * 50 + [("ab", "xyzabcdef")]
    scores = [cer_score(ref, hyp) for ref, hyp in pairs]

    micro = aggregate(scores).rate
    macro = sum(score.rate for score in scores) / len(scores)

    assert micro == pytest.approx(7 / 552, rel=1e-3)
    assert macro == pytest.approx(0.0686, abs=1e-3)
    assert macro > micro * 5


def test_an_infinite_page_does_not_poison_the_corpus_rate() -> None:
    """The payoff for keeping numerator and denominator instead of the float.
    A blank-reference page contributes its errors to the numerator and nothing
    to the denominator, so the corpus number stays finite. Had `cer()` returned
    `inf` into a mean, the whole report would read `inf`."""
    scores = [cer_score("hello", "hello"), cer_score("", "xx")]

    assert scores[1].rate == float("inf")
    assert aggregate(scores).rate == pytest.approx(2 / 5)


def test_aggregate_sums_errors_and_lengths_independently() -> None:
    scores = [cer_score("cat", "cart"), cer_score("cart", "cat"), cer_score("a", "b")]

    total = aggregate(scores)

    assert total.errors == 3  # 1 + 1 + 1
    assert total.length == 3 + 4 + 1


def test_aggregating_mixed_units_is_refused() -> None:
    """A denominator of graphemes plus words is not a quantity. Silently
    summing them yields a plausible-looking float with no meaning, which is
    worse than an error."""
    with pytest.raises(ValueError, match="mixed units"):
        aggregate([cer_score("a b", "a b"), wer_score("a b", "a b")])


def test_aggregating_nothing_is_zero_not_a_crash() -> None:
    """A job where every page failed upstream should report an empty result,
    not divide by zero in the reporting layer."""
    empty = aggregate([])
    assert empty.errors == 0 and empty.length == 0
    assert empty.rate == 0.0
