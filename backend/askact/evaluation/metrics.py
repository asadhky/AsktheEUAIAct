"""Evaluation metrics, as pure functions of one small number per question.

Retrieval: hit@k and MRR (R5.5, R5.6). Both are computed from the rank of the first retrieved chunk whose
parent section is one of the question's expected sections (`first_hit_rank`). Ranks count *chunks*, as the
chunks are what retrieval returns: a section split into three chunks can fill three ranks, and we do not
de-duplicate, to keep the metrics the plain textbook definitions.

Generation: citation precision and recall for in-scope questions (R5.8, `citation_precision_recall`) and
refusal accuracy for out_of_scope questions (R5.7, `score_refusals`).

Only in-scope questions get hit@k, MRR and citation scores. An out_of_scope question has no expected
section, so none of those is defined for it; the caller leaves those out. Every average over zero questions
is an error here, not 0: the caller reports it as "n/a".
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from askact.models import SectionId


def first_hit_rank(retrieved: Sequence[SectionId], expected: Collection[SectionId], k: int) -> int | None:
    """Rank (counting from 1) of the first of the top `k` chunks whose section is expected, else None.

    `retrieved` holds the parent section of each retrieved chunk, best first. Anything beyond the
    first `k` is ignored, so a ranker that returned too many results cannot inflate the score.
    Any one expected section counts as a hit: they are all relevant, not alternatives (R5.3).
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    for rank, section in enumerate(retrieved[:k], start=1):
        if section in expected:
            return rank
    return None


@dataclass(frozen=True)
class RetrievalScores:
    questions: int  # how many questions were scored
    hits: int  # how many had an expected section in the top k
    mrr: float  # mean of 1 / (rank of the first hit), counting 0 for a question with no hit

    @property
    def hit_at_k(self) -> float:
        return self.hits / self.questions


def score_ranks(ranks: Sequence[int | None]) -> RetrievalScores:
    """hit@k and MRR from each question's `first_hit_rank` (None means no hit in the top k)."""
    if not ranks:
        # Both metrics are averages, and an average over nothing is undefined, not 0.
        raise ValueError("cannot score zero questions")
    for rank in ranks:
        if rank is not None and rank < 1:
            raise ValueError(f"a rank counts from 1, got {rank}")
    hits = sum(rank is not None for rank in ranks)
    reciprocal_ranks = sum(1.0 / rank for rank in ranks if rank is not None)
    return RetrievalScores(questions=len(ranks), hits=hits, mrr=reciprocal_ranks / len(ranks))


def citation_precision_recall(cited: Sequence[SectionId], expected: Collection[SectionId]) -> tuple[float, float]:
    """Precision and recall of one answer's citations (R5.8).

    precision = cited sections that are expected / cited sections
    recall    = cited sections that are expected / expected sections

    Sections count once however often they are cited. An answer with no citations scores 0 on both, which is
    also what a refusal of an in-scope question gets: it found nothing that was asked for. A cited section
    that is relevant but not listed in `expected` (for example a recital that explains the same rule) counts
    as wrong, because `expected` is the only definition of "right" the question set has.
    """
    wanted = set(expected)
    if not wanted:
        # Only in-scope questions are scored, and the question loader guarantees they list a section.
        raise ValueError("an in-scope question must expect at least one section")
    unique = list(dict.fromkeys(cited))
    if not unique:
        return 0.0, 0.0
    right = sum(section in wanted for section in unique)
    return right / len(unique), right / len(wanted)


@dataclass(frozen=True)
class CitationScores:
    questions: int  # how many in-scope questions were scored
    precision: float  # mean over those questions
    recall: float


def score_citations(per_question: Sequence[tuple[float, float]]) -> CitationScores:
    """Mean citation precision and recall from each in-scope question's `citation_precision_recall`."""
    if not per_question:
        raise ValueError("cannot score zero questions")
    for precision, recall in per_question:
        if not (0.0 <= precision <= 1.0 and 0.0 <= recall <= 1.0):
            raise ValueError(f"precision and recall are between 0 and 1, got {precision} and {recall}")
    count = len(per_question)
    return CitationScores(
        questions=count,
        precision=sum(precision for precision, _ in per_question) / count,
        recall=sum(recall for _, recall in per_question) / count,
    )


@dataclass(frozen=True)
class RefusalScores:
    questions: int  # how many out_of_scope questions were scored
    refused: int  # how many of them the system answered "not covered"

    @property
    def accuracy(self) -> float:
        return self.refused / self.questions


def score_refusals(refused: Sequence[bool]) -> RefusalScores:
    """Refusal accuracy (R5.7) from one flag per out_of_scope question: True if the system said "not covered"."""
    if not refused:
        raise ValueError("cannot score zero questions")
    return RefusalScores(questions=len(refused), refused=sum(refused))
