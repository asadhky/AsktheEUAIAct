"""Rerankers: re-score a short list of candidate passages against the question (R2.4).

A reranker reads the question and a passage *together*, which is slower but sharper than comparing two
separately computed embeddings, so it is used only on the few candidates that hybrid retrieval returns.

The real one (`CrossEncoderReranker`) runs a cross-encoder on this machine. The stub (`OverlapReranker`)
needs no model download and no torch, so tests and CI can exercise everything that depends on a reranker
offline (R6.4). Pick one with `RERANKER=stub|real`. Whether reranking is used at all is a separate switch,
`RERANKER_ENABLED`; see `askact.retrieval`.
"""

import math
from collections.abc import Sequence
from typing import Any, Protocol

from askact.config import Settings
from askact.models import tokenize


class RerankerError(Exception):
    """The reranker cannot be used or misbehaved. The message says why and what to do."""


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """One relevance score per passage, in the same order; higher means more relevant.

        Scores are only comparable within one implementation: the cross-encoder's are unbounded logits
        (positive for a good match, negative for a poor one), the stub's are in [0, 1].
        """
        ...


class OverlapReranker:
    """Jaccard overlap of the question's words and the passage's words: shared / combined distinct words.

    Not a good reranker, but deterministic, in [0, 1], and it prefers the passage that shares the most
    of the question's words, which is enough for tests to assert that reranking reorders results.
    """

    name = "stub-overlap"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        query_words = set(tokenize(query))
        scores = []
        for passage in passages:
            passage_words = set(tokenize(passage))
            union = query_words | passage_words
            scores.append(len(query_words & passage_words) / len(union) if union else 0.0)
        return scores


class CrossEncoderReranker:
    """A sentence-transformers cross-encoder run locally on the CPU. Needs the `models` extra.

    Scores are the model's raw logits, which `predict()` returns unchanged in every sentence-transformers
    version tested (3.4.1 to 6.1.0). A slow test pins that, because the relevance threshold for reranked
    results (`RELEVANCE_THRESHOLD_RERANK`) is tuned on this scale and would silently stop meaning the same
    thing if a library update applied a sigmoid.
    """

    def __init__(self, model_name: str, *, batch_size: int = 16):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "RERANKER=real needs the sentence-transformers package, which is the optional `models` "
                "extra: install it with `pip install -e 'backend[models]'` (install torch's CPU wheel first: "
                "`pip install torch --index-url https://download.pytorch.org/whl/cpu`), "
                "or set RERANKER=stub to use the test stub."
            ) from exc
        try:
            # CPU on purpose: the model is small and only about twenty passages are scored per question.
            self._model: Any = CrossEncoder(model_name, device="cpu")
        except Exception as exc:  # network failure, unknown model name, corrupt cache, ...
            raise RerankerError(
                f"could not load the reranker model {model_name!r} ({type(exc).__name__}: {exc}). "
                "Check RERANKER_MODEL and the network, or the model cache if running offline."
            ) from exc
        self.name = model_name
        self._batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        # The model reads query and passage together and silently truncates beyond its token limit
        # (512), so the end of a passage can be cut off when the two are long.
        scores = self._model.predict(
            [(query, passage) for passage in passages], batch_size=self._batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]


def get_reranker(settings: Settings) -> Reranker:
    """The reranker chosen by RERANKER: the stub, or the real model named by RERANKER_MODEL."""
    if settings.reranker == "stub":
        return OverlapReranker()
    return CrossEncoderReranker(settings.reranker_model)


def checked_scores(reranker: Reranker, query: str, passages: Sequence[str]) -> list[float]:
    """The reranker's scores, verified: one finite number per passage.

    A reranker that returns the wrong number of scores, or NaN, would silently reorder or drop results,
    and a NaN relevance slips past a `< threshold` check, so a bad answer is an error, not a result.
    """
    scores = reranker.score(query, passages)
    if len(scores) != len(passages):
        raise RerankerError(f"{reranker.name} returned {len(scores)} scores for {len(passages)} passages")
    if not all(math.isfinite(s) for s in scores):
        raise RerankerError(f"{reranker.name} returned a score that is not a finite number: {scores}")
    return scores
