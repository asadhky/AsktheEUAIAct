import dataclasses

import pytest

from askact.evaluation.metrics import (
    CitationScores,
    RefusalScores,
    RetrievalScores,
    citation_precision_recall,
    first_hit_rank,
    score_citations,
    score_ranks,
    score_refusals,
)
from askact.models import SectionId


def ids(*texts: str) -> list[SectionId]:
    return [SectionId.parse(t) for t in texts]


# --- first_hit_rank: what counts as a hit -----------------------------------------------

def test_the_rank_of_the_first_expected_section_counts_from_one():
    assert first_hit_rank(ids("article:9", "article:5", "annex:III"), ids("article:5"), k=3) == 2
    assert first_hit_rank(ids("article:5", "article:9"), ids("article:5"), k=2) == 1


def test_any_one_of_several_expected_sections_is_a_hit_and_the_earliest_wins():
    expected = ids("article:5", "annex:III", "recital:48")
    # annex:III comes before article:5 in the results, so rank 2 (not "wait for all of them").
    assert first_hit_rank(ids("article:9", "annex:III", "article:5"), expected, k=3) == 2


def test_the_first_hit_is_what_counts_not_later_ones():
    assert first_hit_rank(ids("article:5", "article:5", "article:5"), ids("article:5"), k=3) == 1


def test_several_chunks_of_one_section_each_take_a_rank():
    # Three chunks of article:9 come first, so the expected annex is at rank 4: no de-duplication.
    retrieved = ids("article:9", "article:9", "article:9", "annex:III")
    assert first_hit_rank(retrieved, ids("annex:III"), k=4) == 4
    assert first_hit_rank(retrieved, ids("annex:III"), k=3) is None


def test_no_expected_section_in_the_list_means_no_hit():
    assert first_hit_rank(ids("article:9", "article:10"), ids("article:5"), k=2) is None


def test_only_the_top_k_are_looked_at_even_if_more_are_given():
    retrieved = ids("article:9", "article:10", "article:5")
    assert first_hit_rank(retrieved, ids("article:5"), k=2) is None
    assert first_hit_rank(retrieved, ids("article:5"), k=3) == 3


def test_a_hit_exactly_at_rank_k_counts():
    assert first_hit_rank(ids("article:9", "article:5"), ids("article:5"), k=2) == 2


def test_nothing_retrieved_is_no_hit():
    assert first_hit_rank([], ids("article:5"), k=5) is None


def test_the_section_kind_matters():
    # recital 5 is not article 5.
    assert first_hit_rank(ids("recital:5"), ids("article:5"), k=1) is None


def test_expected_may_be_any_collection():
    assert first_hit_rank(ids("article:5"), {SectionId.parse("article:5")}, k=1) == 1
    assert first_hit_rank(ids("article:5"), frozenset(ids("article:5")), k=1) == 1


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        first_hit_rank(ids("article:5"), ids("article:5"), k=k)


# --- score_ranks: hit@k and MRR, computed by hand --------------------------------------

def test_hit_at_k_and_mrr_by_hand():
    # Ranks 1, 2, none, 4: three of four questions hit, so hit@k = 3/4.
    # MRR = (1/1 + 1/2 + 0 + 1/4) / 4 = 1.75 / 4 = 0.4375.
    scores = score_ranks([1, 2, None, 4])
    assert (scores.questions, scores.hits) == (4, 3)
    assert scores.hit_at_k == pytest.approx(0.75)
    assert scores.mrr == pytest.approx(0.4375)


def test_a_question_with_no_hit_counts_zero_not_skipped():
    # If the miss were dropped, MRR would be 1.0 instead of 0.5.
    scores = score_ranks([1, None])
    assert scores.questions == 2
    assert scores.mrr == pytest.approx(0.5)
    assert scores.hit_at_k == pytest.approx(0.5)


def test_when_nothing_hits_both_metrics_are_zero():
    scores = score_ranks([None, None, None])
    assert (scores.hits, scores.hit_at_k, scores.mrr) == (0, 0.0, 0.0)


def test_when_everything_is_rank_one_both_metrics_are_one():
    scores = score_ranks([1, 1, 1])
    assert (scores.hits, scores.hit_at_k, scores.mrr) == (3, 1.0, 1.0)


def test_a_single_question():
    scores = score_ranks([3])
    assert (scores.questions, scores.hits) == (1, 1)
    assert scores.mrr == pytest.approx(1 / 3)


def test_hit_at_k_does_not_depend_on_which_rank_hit_but_mrr_does():
    assert score_ranks([1, 1]).hit_at_k == score_ranks([5, 5]).hit_at_k == 1.0
    assert score_ranks([1, 1]).mrr == pytest.approx(1.0)
    assert score_ranks([5, 5]).mrr == pytest.approx(0.2)


def test_the_order_of_questions_does_not_matter():
    assert score_ranks([3, None, 1]) == score_ranks([1, 3, None])


def test_no_questions_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="zero questions"):
        score_ranks([])


@pytest.mark.parametrize("rank", [0, -2])
def test_a_rank_below_one_is_an_error(rank):
    with pytest.raises(ValueError, match="counts from 1"):
        score_ranks([1, rank])


def test_scores_cannot_be_changed():
    scores = score_ranks([1])
    with pytest.raises(dataclasses.FrozenInstanceError):
        scores.hits = 5  # type: ignore[misc]
    assert isinstance(scores, RetrievalScores)


# =====================================================================================================
# Generation metrics: citation precision / recall and refusal accuracy
# =====================================================================================================

@pytest.mark.parametrize(
    "cited, expected, precision, recall",
    [
        (["article:5"], ["article:5"], 1.0, 1.0),
        # One of two citations is expected: precision 1/2; the one expected section is cited: recall 1.
        (["article:5", "article:9"], ["article:5"], 0.5, 1.0),
        # The one citation is expected (precision 1), but only one of two expected sections is cited (recall 1/2).
        (["article:5"], ["article:5", "article:6"], 1.0, 0.5),
        # 1 right of 2 cited; 1 of 3 expected found.
        (["article:5", "article:9"], ["article:5", "article:6", "article:7"], 0.5, 1 / 3),
        # Both expected sections cited, plus one that is not: 2/3 precision, full recall.
        (["article:6", "article:5", "recital:1"], ["article:5", "article:6"], 2 / 3, 1.0),
        # Nothing cited is expected.
        (["article:9"], ["article:5"], 0.0, 0.0),
        # An article and an annex with the same number are different sections.
        (["annex:III"], ["article:3"], 0.0, 0.0),
    ],
)
def test_citation_precision_and_recall_by_hand(cited, expected, precision, recall):
    got = citation_precision_recall(ids(*cited), ids(*expected))
    assert got == (pytest.approx(precision), pytest.approx(recall))


def test_an_answer_without_citations_scores_zero_on_both():
    assert citation_precision_recall([], ids("article:5", "article:6")) == (0.0, 0.0)


def test_citing_a_section_twice_does_not_change_the_score():
    # Without de-duplication this would be precision 2/3.
    assert citation_precision_recall(ids("article:5", "article:5", "article:9"), ids("article:5")) == (0.5, 1.0)


def test_an_in_scope_question_must_expect_something():
    with pytest.raises(ValueError, match="at least one section"):
        citation_precision_recall(ids("article:5"), [])


def test_the_mean_citation_scores_are_over_all_questions_including_zeros():
    # Precision (1 + 0.5 + 0) / 3 = 0.5; recall (1 + 0 + 0) / 3 = 1/3.
    scores = score_citations([(1.0, 1.0), (0.5, 0.0), (0.0, 0.0)])
    assert isinstance(scores, CitationScores) and scores.questions == 3
    assert scores.precision == pytest.approx(0.5) and scores.recall == pytest.approx(1 / 3)


def test_citation_scores_of_zero_questions_are_undefined_not_zero():
    with pytest.raises(ValueError, match="zero questions"):
        score_citations([])


@pytest.mark.parametrize("pair", [(1.1, 0.5), (0.5, -0.1), (float("nan"), 0.5), (0.5, float("inf"))])
def test_citation_scores_outside_zero_to_one_are_refused(pair):
    with pytest.raises(ValueError, match="between 0 and 1"):
        score_citations([(1.0, 1.0), pair])


def test_refusal_accuracy_is_the_share_refused():
    scores = score_refusals([True, False, True, True])
    assert scores == RefusalScores(questions=4, refused=3)
    assert scores.accuracy == pytest.approx(0.75)


def test_refusal_accuracy_extremes():
    assert score_refusals([False, False]).accuracy == 0.0
    assert score_refusals([True]).accuracy == 1.0


def test_refusal_accuracy_of_zero_questions_is_undefined_not_zero():
    with pytest.raises(ValueError, match="zero questions"):
        score_refusals([])


def test_generation_scores_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        score_refusals([True]).refused = 0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        score_citations([(1.0, 1.0)]).precision = 0.0  # type: ignore[misc]
