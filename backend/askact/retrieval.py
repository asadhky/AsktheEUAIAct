"""Retrieval: find the chunks most relevant to a question (R2.1 to R2.5).

Four ways of ranking the index, called "configs" because the evaluation compares them:

  bm25           lexical: Okapi BM25 over the words of each chunk's embedded string
  dense          semantic: cosine similarity between the query and chunk embeddings
  hybrid         both, combined by Reciprocal Rank Fusion
  hybrid+rerank  hybrid, then a cross-encoder re-scores the best candidates

Every result also carries a *relevance* score for the "not covered" gate (R3.8):
  * for `hybrid+rerank`, the reranker's score for the top-ranked chunk (`relevance_kind="rerank"`);
  * for every other config, the dense cosine between the query and the top-ranked chunk, computed the
    same way whichever config ranked it (`relevance_kind="cosine"`).
The two are on different scales (a cosine is at most 1, a cross-encoder score is an unbounded logit),
which is why the gate has two thresholds.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args

import numpy as np
from rank_bm25 import BM25Okapi

from askact.config import Settings
from askact.embedders import Embedder, Vectors, get_embedder
from askact.index import INDEX_DIR, Index, load_index
from askact.models import RetrievalResult, ScoredChunk, tokenize
from askact.rerankers import Reranker, checked_scores, get_reranker

Config = Literal["bm25", "dense", "hybrid", "hybrid+rerank"]
CONFIGS: tuple[str, ...] = get_args(Config)

# Reciprocal Rank Fusion: each ranker gives a chunk 1 / (RRF_K + rank). 60 is the constant from the
# original paper (Cormack, Clarke and Buettcher, 2009) and is not worth tuning on a small question set.
RRF_K = 60
# How many chunks each ranker contributes to the fusion.
HYBRID_CANDIDATES = 30
# How many of the best fused chunks the reranker re-scores. A cross-encoder reads the question and the
# passage together, so it is run on a short list only.
RERANK_CANDIDATES = 20


def app_config(settings: Settings) -> Config:
    """The config the running app uses: reranked hybrid if RERANKER_ENABLED, otherwise plain hybrid (R2.4).

    The evaluation harness does not use this: it compares all four configs whatever the flag says.
    """
    return "hybrid+rerank" if settings.reranker_enabled else "hybrid"


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Fuse ranked lists of chunk indexes into one list of (index, score), best first.

    A chunk's score is the sum over the lists it appears in of 1 / (k + rank), rank counting from 1.
    Only ranks are used, so BM25 scores and cosine similarities never have to be put on one scale.
    A chunk missing from a list gets nothing from it. Equal scores keep the order in which the chunks
    first appear, so earlier lists win ties and the result is deterministic.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])  # sorted() is stable: ties keep insertion order


def _best_first(scores: np.ndarray, n: int) -> list[int]:
    """Indexes of the n highest scores; equal scores stay in document order."""
    return np.argsort(-scores, kind="stable")[:n].tolist()


class Retriever:
    """Searches one loaded index. Build it once and reuse it: it keeps the BM25 index in memory.

    `reranker` is optional: without one, the `hybrid+rerank` config is unavailable and no reranking
    model is ever loaded or called.
    """

    def __init__(
        self,
        index: Index,
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        candidates: int = HYBRID_CANDIDATES,
        rerank_candidates: int = RERANK_CANDIDATES,
    ):
        manifest = index.manifest
        if embedder.name != manifest.embedding_model or embedder.dimension != manifest.dimension:
            # Query vectors from another model are meaningless against these embeddings.
            raise ValueError(
                f"the index was built with {manifest.embedding_model!r} ({manifest.dimension} dimensions) but the "
                f"embedder is {embedder.name!r} ({embedder.dimension} dimensions): run `python -m askact.ingest`"
            )
        if candidates < 1 or rerank_candidates < 1:
            raise ValueError(f"candidates must be at least 1, got {candidates} and {rerank_candidates}")
        self._index = index
        self._embedder = embedder
        self._reranker = reranker
        self._candidates = candidates
        self._rerank_candidates = rerank_candidates
        self._bm25 = BM25Okapi([tokenize(chunk.embed_text) for chunk in index.chunks])

    @classmethod
    def from_settings(cls, settings: Settings, directory: Path = INDEX_DIR) -> "Retriever":
        """Load the index, and the reranker only if RERANKER_ENABLED (R2.4).

        Raises IndexUnavailableError, with instructions, if the index is missing or stale. With the flag off
        the reranking model is not even constructed, so it is neither downloaded nor held in memory.
        """
        reranker = get_reranker(settings) if settings.reranker_enabled else None
        return cls(load_index(settings, directory), get_embedder(settings), reranker)

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)

    def search(self, query: str, config: Config, k: int) -> RetrievalResult:
        """The top `k` chunks for `query` under `config`, best first, with scores and metadata.

        `score` is the config's own ranking score (BM25, cosine, the fused RRF score, or the reranker's
        score), so it is only comparable within one result. `relevance` is described in the module docstring.
        """
        if config not in CONFIGS:
            raise ValueError(f"unknown retrieval config {config!r}; expected one of {', '.join(CONFIGS)}")
        if config == "hybrid+rerank" and self._reranker is None:
            raise ValueError(
                "the config 'hybrid+rerank' needs a reranker, but this retriever has none because reranking is "
                "disabled: set RERANKER_ENABLED=true (the running app picks its config with app_config())"
            )
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not query.strip():
            raise ValueError("the query is empty")

        # Every config embeds the query: the cosine relevance needs the top chunk's cosine even when BM25
        # did the ranking. The whole matrix product is a few hundred rows, so there is no reason to
        # compute only the one row bm25 needs.
        query_vector = self._embedder.embed_query(query)
        cosines = self._index.embeddings @ query_vector

        if config == "hybrid+rerank":
            ranked = self._rerank(query, cosines, k)
            relevance, kind = ranked[0][1], "rerank"
        else:
            if config == "dense":
                ranked = [(i, float(cosines[i])) for i in _best_first(cosines, k)]
            elif config == "bm25":
                bm25 = self._bm25_scores(query)
                ranked = [(i, float(bm25[i])) for i in _best_first(bm25, k)]
            else:
                ranked = self._hybrid(query, cosines, k)
            relevance, kind = float(cosines[ranked[0][0]]), "cosine"

        return RetrievalResult(
            chunks=[ScoredChunk(chunk=self._index.chunks[i], score=score) for i, score in ranked],
            relevance=relevance,
            relevance_kind=kind,
        )

    def _hybrid(self, query: str, cosines: Vectors, k: int) -> list[tuple[int, float]]:
        pool = max(self._candidates, k)  # enough candidates to be able to return k results
        dense_ranking = _best_first(cosines, pool)
        bm25 = self._bm25_scores(query)
        # A chunk with BM25 score 0 shares no word with the query: that is no lexical evidence, and
        # letting the arbitrary first `pool` such chunks into the fusion would reward them for nothing.
        bm25_ranking = [i for i in _best_first(bm25, pool) if bm25[i] > 0]
        return rrf([dense_ranking, bm25_ranking])[:k]

    def _rerank(self, query: str, cosines: Vectors, k: int) -> list[tuple[int, float]]:
        """Re-score the best fused chunks with the reranker and return the top k by its score."""
        assert self._reranker is not None  # search() has checked
        candidates = self._hybrid(query, cosines, max(self._rerank_candidates, k))
        passages = [self._index.chunks[i].embed_text for i, _ in candidates]  # header included, as for embedding
        scores = checked_scores(self._reranker, query, passages)
        # Stable sort: chunks the reranker scores equally keep their fused order.
        reranked = sorted(zip((i for i, _ in candidates), scores), key=lambda item: -item[1])
        return reranked[:k]
