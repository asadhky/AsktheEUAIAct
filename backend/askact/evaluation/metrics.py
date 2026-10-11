"""Retrieval metrics: hit@k and MRR (R5.5, R5.6).

Both are computed from one number per question: the rank of the first retrieved chunk whose parent
section is one of the question's expected sections (`first_hit_rank`). Ranks count *chunks*, as the
chunks are what retrieval returns: a section split into three chunks can fill three ranks, and we do not
de-duplicate, to keep the metrics the plain textbook definitions.

Only in-scope questions are scored. An out_of_scope question has no expected section, so a hit is not
defined for it; the caller leaves those out.
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
