import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from askact.config import Settings
from askact.embedders import HashEmbedder
from askact.index import (
    SCHEMA_VERSION,
    Index,
    IndexCounts,
    IndexUnavailableError,
    Manifest,
    SourceRecord,
    compute_build_hash,
    write_index,
)
from askact.ingest.chunk import chunk_act, chunking_params
from askact.ingest.parse import parse_act
from askact.models import Chunk, SectionId
from askact.rerankers import RerankerError
from askact.retrieval import (
    CONFIGS,
    HYBRID_CANDIDATES,
    RERANK_CANDIDATES,
    RRF_K,
    Config,
    Retriever,
    app_config,
    rrf,
    tokenize,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_act.html"
REAL = Path(__file__).resolve().parents[2] / "data" / "raw" / "ai-act-oj-2024-1689.html"
SHA = "c" * 64
NON_RERANK: list[Config] = ["bm25", "dense", "hybrid"]


# --- helpers ----------------------------------------------------------------------------------------

def chunk(sid: str, text: str, title: str | None = None, ordinal: int = 1) -> Chunk:
    return Chunk(section_id=SectionId.parse(sid), ordinal=ordinal, title=title, text=text)


def make_index(chunks: list[Chunk], embedder) -> Index:
    settings = Settings(_env_file=None, embedder="stub")
    counts = {k: len({c.section_id for c in chunks if c.kind == k}) for k in ("recital", "article", "annex")}
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        source=SourceRecord(regulation="Regulation (EU) 2024/1689", celex="32024R1689", oj_reference="OJ L, 2024/1689, 12.7.2024",
                            url="https://example.org", retrieved_at=date(2026, 10, 9), sha256=SHA, origin="fetched"),
        counts=IndexCounts(recitals=counts["recital"], articles=counts["article"], annexes=counts["annex"], chunks=len(chunks)),
        chunking=chunking_params(settings),
        embedding_model=embedder.name,
        dimension=embedder.dimension,
        build_hash=compute_build_hash(SHA, chunking_params(settings), embedder.name),
    )
    return Index(manifest, tuple(chunks), embedder.embed_documents([c.embed_text for c in chunks]))


def ids(result) -> list[str]:
    return [s.chunk.chunk_id for s in result.chunks]


def sections(result) -> list[str]:
    return [str(s.chunk.section_id) for s in result.chunks]


# A small corpus with distinct vocabulary, searched with a stub embedder wide enough not to collide.
CORPUS = [
    chunk("recital:12", "The notion of an AI system should be clearly defined."),
    chunk("article:5", "The use of real-time remote biometric identification systems in publicly accessible spaces.", "Prohibited AI practices"),
    chunk("article:99", "Member States shall lay down the rules on penalties applicable to infringements.", "Penalties"),
    chunk("article:3", "For the purposes of this Regulation the following definitions apply.", "Definitions"),
    chunk("annex:III", "Biometrics, in so far as their use is permitted: remote biometric identification systems.", "High-risk AI systems"),
]


@pytest.fixture
def stub() -> HashEmbedder:
    return HashEmbedder(dimension=1024)  # wide enough that the few words here do not collide


@pytest.fixture
def retriever(stub) -> Retriever:
    return Retriever(make_index(CORPUS, stub), stub)


# --- tokenizing ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, tokens", [
    ("Article 5 — Prohibited AI practices", ["article", "5", "prohibited", "ai", "practices"]),
    ("point (a) of Article 5(1)", ["point", "a", "of", "article", "5", "1"]),
    ("‘general-purpose AI model’", ["general", "purpose", "ai", "model"]),
    ("PENALTIES, penalties.", ["penalties", "penalties"]),
    ("", []),
    ("— … !!!", []),
])
def test_tokenize_lowercases_and_splits_on_anything_that_is_not_a_word_character(text, tokens):
    assert tokenize(text) == tokens


# --- BM25 (R2.1) ---------------------------------------------------------------------------------------

def test_bm25_finds_the_passage_that_contains_a_rare_exact_term(retriever):
    result = retriever.search("penalties", "bm25", k=3)
    assert sections(result)[0] == "article:99"
    assert result.chunks[0].score > result.chunks[1].score


def test_bm25_prefers_a_rare_term_to_a_common_one(retriever):
    """'ai' is in four of the five chunks, 'biometric' in two: the rarer word carries more weight."""
    result = retriever.search("ai biometric", "bm25", k=5)
    assert sections(result)[0] in {"article:5", "annex:III"}


def test_bm25_matches_the_words_and_numbers_of_the_header(retriever):
    """The embedded string starts with 'Article 99 — Penalties', so a citation-style query finds it."""
    assert sections(retriever.search("Article 99", "bm25", k=1)) == ["article:99"]
    assert sections(retriever.search("prohibited practices", "bm25", k=1)) == ["article:5"]


def test_bm25_ignores_case_and_punctuation(retriever):
    plain = retriever.search("penalties", "bm25", k=5)
    assert ids(retriever.search("PENALTIES!?", "bm25", k=5)) == ids(plain)
    assert [s.score for s in retriever.search("PENALTIES!?", "bm25", k=5).chunks] == [s.score for s in plain.chunks]


def test_bm25_scores_are_in_descending_order_and_finite(retriever):
    scores = [s.score for s in retriever.search("remote biometric identification systems", "bm25", k=5).chunks]
    assert scores == sorted(scores, reverse=True)
    assert all(math.isfinite(s) for s in scores)


def test_a_query_sharing_no_word_with_the_index_still_returns_k_chunks_in_document_order(retriever):
    """BM25 has no signal; ties (all zero) fall back to document order. The relevance gate, not the
    ranking, is what decides such a question is not covered."""
    result = retriever.search("zebra giraffe", "bm25", k=3)
    assert [s.score for s in result.chunks] == [0.0, 0.0, 0.0]
    assert ids(result) == [c.chunk_id for c in CORPUS[:3]]


# --- dense (R2.2) --------------------------------------------------------------------------------------------

def test_dense_scores_are_the_cosine_between_query_and_chunk(retriever, stub):
    query = "penalties applicable to infringements"
    expected = stub.embed_documents([c.embed_text for c in CORPUS]) @ stub.embed_query(query)
    result = retriever.search(query, "dense", k=5)
    for scored in result.chunks:
        assert scored.score == pytest.approx(float(expected[CORPUS.index(scored.chunk)]), abs=1e-6)


def test_dense_ranks_the_chunk_sharing_most_words_first_and_in_descending_order(retriever):
    result = retriever.search("remote biometric identification in publicly accessible spaces", "dense", k=5)
    assert sections(result)[0] == "article:5"
    scores = [s.score for s in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_dense_equal_scores_keep_document_order():
    """Two chunks with identical embeddings tie exactly; the earlier one in the document comes first,
    whichever order they happen to be stored in memory."""
    chunks = [chunk(f"recital:{n}", f"tie{n}") for n in (1, 2, 3)]
    same = [1.0, 0.0, 0.0, 0.0]
    embedder = ScriptedEmbedder({c.embed_text: same for c in chunks}, same)
    result = Retriever(make_index(chunks, embedder), embedder).search("anything", "dense", k=3)
    assert [s.score for s in result.chunks] == [pytest.approx(1.0)] * 3
    assert ids(result) == ["recital:1#1", "recital:2#1", "recital:3#1"]


# --- Reciprocal Rank Fusion (R2.3) -----------------------------------------------------------------------------

def test_rrf_arithmetic_on_a_hand_computed_example():
    """Two rankings of chunks 0, 1, 2:  [0, 1, 2]  and  [1, 2, 0].  With k = 60:
         chunk 0:  1/61 + 1/63        = 0.016393 + 0.015873 = 0.032266
         chunk 1:  1/62 + 1/61        = 0.016129 + 0.016393 = 0.032522
         chunk 2:  1/63 + 1/62        = 0.015873 + 0.016129 = 0.032002
       so the order is 1, 0, 2."""
    fused = rrf([[0, 1, 2], [1, 2, 0]])
    assert [index for index, _ in fused] == [1, 0, 2]
    assert dict(fused)[0] == pytest.approx(1 / 61 + 1 / 63)
    assert dict(fused)[1] == pytest.approx(1 / 62 + 1 / 61)
    assert dict(fused)[2] == pytest.approx(1 / 63 + 1 / 62)
    assert dict(fused)[1] == pytest.approx(0.032522, abs=1e-6)


def test_rrf_uses_ranks_counted_from_one_and_the_constant_sixty():
    assert RRF_K == 60
    assert rrf([[7]]) == [(7, pytest.approx(1 / 61))]
    assert rrf([[7]], k=0) == [(7, pytest.approx(1.0))]  # rank 1 with k = 0 is 1/1: pins "ranks start at 1"


def test_rrf_includes_a_chunk_that_appears_in_only_one_ranking():
    fused = dict(rrf([[0, 1], [2]]))
    assert set(fused) == {0, 1, 2}
    assert fused[2] == pytest.approx(1 / 61)


def test_rrf_ties_keep_the_order_in_which_chunks_first_appear():
    assert [i for i, _ in rrf([[5], [3]])] == [5, 3]   # equal scores: the first ranking wins
    assert [i for i, _ in rrf([[3], [5]])] == [3, 5]


def test_hybrid_hands_the_dense_ranking_to_the_fusion_first_so_dense_wins_ties(monkeypatch):
    """Fusion ties go to the earlier list, so which list comes first is a documented behaviour. The retriever
    cannot be made to produce an exact tie from the outside (a chunk missing from one ranker's candidates
    reappears when k widens the pool), so check what it passes to rrf(): the dense list, then the BM25 list."""
    from askact import retrieval

    seen = []
    real = retrieval.rrf
    monkeypatch.setattr(retrieval, "rrf", lambda rankings, k=RRF_K: (seen.append([list(r) for r in rankings]), real(rankings, k))[1])
    retriever, chunks = scripted_retriever()
    retriever.search("zeta", "hybrid", k=2)

    dense_list, bm25_list = seen[0]
    assert dense_list[:4] == [0, 1, 2, 3]   # d0 d1 d2 d3 by cosine: the first list is the dense one
    assert bm25_list == [2, 1]              # d2 d1: the lexical matches only, in BM25 order


def test_rrf_gives_an_exact_tie_to_the_first_list():
    """The other half of the above: with equal scores, the first list's chunk comes first."""
    fused = rrf([[0], [1]])
    assert fused[0][1] == pytest.approx(fused[1][1])   # truly tied
    assert [i for i, _ in fused] == [0, 1]
    assert [i for i, _ in rrf([[1], [0]])] == [1, 0]


def test_equal_scores_stay_in_document_order_when_there_are_several_groups_of_ties():
    """numpy's default sort keeps an all-equal input in order but scrambles one with several groups of ties
    (about 180 of 200 positions move), which is the realistic case: some chunks share one cosine, some another.
    So the ties here come in two groups, interleaved through the document."""
    count = 200
    chunks = [chunk(f"recital:{n}", f"tie{n}") for n in range(1, count + 1)]
    high, low = [1.0, 0.0, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0]
    vectors = {c.embed_text: (high if n % 3 == 0 else low) for n, c in enumerate(chunks)}   # every third chunk is "high"
    embedder = ScriptedEmbedder(vectors, [1.0, 0.0, 0.0, 0.0])
    result = Retriever(make_index(chunks, embedder), embedder).search("anything", "dense", k=count)
    expected = [c.chunk_id for n, c in enumerate(chunks) if n % 3 == 0] + [c.chunk_id for n, c in enumerate(chunks) if n % 3 != 0]
    assert ids(result) == expected   # all the "high" chunks first, each group in document order


def test_bm25_zero_scores_stay_in_document_order_across_a_large_corpus():
    """With no word in common every BM25 score is 0, an all-equal input; it must come back in document order."""
    chunks = [chunk(f"recital:{n}", f"tie{n}") for n in range(1, 201)]
    embedder = ScriptedEmbedder({c.embed_text: [1.0, 0.0, 0.0, 0.0] for c in chunks}, [1.0, 0.0, 0.0, 0.0])
    result = Retriever(make_index(chunks, embedder), embedder).search("nothingmatches", "bm25", k=200)
    assert ids(result) == [c.chunk_id for c in chunks]


def test_rrf_of_nothing_is_nothing():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_rrf_rewards_agreement_over_a_single_first_place():
    """A chunk ranked 2nd by both rankers beats chunks that one ranker puts first and the other omits."""
    fused = [i for i, _ in rrf([[10, 20], [11, 20]])]
    assert fused[0] == 20


# --- hybrid, with rankings the test controls ----------------------------------------------------------------------

class ScriptedEmbedder:
    """Returns the vectors it is given, so the dense ranking in a test is exactly what the test says."""

    name = "scripted"
    dimension = 4

    def __init__(self, documents: dict[str, list[float]], query: list[float]):
        self.documents, self.query = documents, np.array(query, dtype=np.float32)

    def embed_documents(self, texts):
        return np.array([self.documents[t] for t in texts], dtype=np.float32)

    def embed_query(self, text):
        return self.query


def scripted_retriever(candidates: int = HYBRID_CANDIDATES) -> tuple[Retriever, list[Chunk]]:
    """Four chunks d0..d3 plus six fillers. Query vector (1,0,0,0) gives the DENSE ranking d0, d1, d2, d3
    (cosines 1, .8, .6, 0, with the fillers last at cosine 0). The query word 'zeta' gives the BM25 ranking
    d2, d1 (d2 repeats it; d0 and d3 do not contain it).

    The fillers are needed: Okapi BM25's idf is ln((N - n + .5) / (n + .5)), which is exactly 0 for a word
    that appears in half of the documents, so with only four chunks 'zeta' would score nothing at all."""
    texts = ["alpha", "zeta alpha", "zeta zeta zeta", "beta"] + [f"filler{n} padding" for n in range(1, 7)]
    chunks = [chunk(f"recital:{n}", text) for n, text in enumerate(texts, start=1)]
    unit_vectors = [[1.0, 0.0, 0.0, 0.0], [0.8, 0.6, 0.0, 0.0], [0.6, 0.0, 0.8, 0.0], [0.0, 1.0, 0.0, 0.0]]
    vectors = {c.embed_text: (unit_vectors[n] if n < 4 else [0.0, 0.0, 0.0, 1.0]) for n, c in enumerate(chunks)}
    embedder = ScriptedEmbedder(vectors, [1.0, 0.0, 0.0, 0.0])
    return Retriever(make_index(chunks, embedder), embedder, candidates=candidates), chunks


def test_the_scripted_rankings_are_what_the_tests_below_assume():
    retriever, chunks = scripted_retriever()
    assert ids(retriever.search("zeta", "dense", k=4)) == [c.chunk_id for c in chunks[:4]]            # d0 d1 d2 d3
    assert ids(retriever.search("zeta", "bm25", k=2)) == [chunks[2].chunk_id, chunks[1].chunk_id]   # d2 d1


def test_hybrid_fuses_the_two_rankings_by_hand_computed_scores():
    """dense: d0 d1 d2 d3 (ranks 1-4)   bm25: d2 d1 (ranks 1-2; d0 and d3 score 0 and are left out)
         d0: 1/61                = 0.016393
         d1: 1/62 + 1/62         = 0.032258
         d2: 1/63 + 1/61         = 0.032266   <- wins by a hair, so a wrong constant or rank offset changes the order
         d3: 1/64                = 0.015625
       Neither ranker alone puts d2 first overall: dense says d0, bm25 says d2 then d1."""
    retriever, chunks = scripted_retriever()
    result = retriever.search("zeta", "hybrid", k=4)
    assert ids(result) == [chunks[2].chunk_id, chunks[1].chunk_id, chunks[0].chunk_id, chunks[3].chunk_id]
    assert [s.score for s in result.chunks] == [
        pytest.approx(1 / 63 + 1 / 61), pytest.approx(1 / 62 + 1 / 62), pytest.approx(1 / 61), pytest.approx(1 / 64)
    ]
    assert result.chunks[0].score == pytest.approx(0.032266, abs=1e-6)
    assert result.chunks[1].score == pytest.approx(0.032258, abs=1e-6)


def test_hybrid_is_not_just_one_of_its_rankers():
    """Dense puts d0 first (it has no 'zeta' at all); hybrid puts d2 first, like BM25."""
    retriever, chunks = scripted_retriever()
    hybrid = retriever.search("zeta", "hybrid", k=4)
    dense = retriever.search("zeta", "dense", k=4)
    bm25 = retriever.search("zeta", "bm25", k=4)
    assert hybrid.chunks[0].chunk == chunks[2] != dense.chunks[0].chunk
    assert ids(hybrid) != ids(dense)
    # In this corpus the *order* of hybrid's top four equals BM25's (BM25 pads with zero-score chunks in
    # document order), so what shows hybrid is more than BM25 is its scores: the tail gets its points from
    # the dense ranking (d0 at dense rank 1, d3 at rank 4) whereas BM25 scores them exactly 0.
    assert [s.score for s in bm25.chunks][2:] == [0.0, 0.0]
    assert [s.score for s in hybrid.chunks][2:] == [pytest.approx(1 / 61), pytest.approx(1 / 64)]


def test_hybrid_leaves_out_chunks_with_no_lexical_match_instead_of_rewarding_them():
    """A query sharing no word with the index has no BM25 evidence: only the dense ranking counts, so the
    fused scores are exactly 1/(60+rank) of the dense ranks and nothing else."""
    retriever, chunks = scripted_retriever()
    result = retriever.search("omega", "hybrid", k=4)
    assert ids(result) == [c.chunk_id for c in chunks[:4]]
    assert [s.score for s in result.chunks] == [pytest.approx(1 / (60 + r)) for r in (1, 2, 3, 4)]


def test_only_the_top_candidates_of_each_ranker_enter_the_fusion():
    """With 2 candidates the dense list is d0 d1 and the bm25 list is d2 d1, so d2 gets only its bm25 point
    and d1 (in both lists) wins; with the default 30, d2 also collects its dense point and wins instead."""
    narrow, chunks = scripted_retriever(candidates=2)
    wide, _ = scripted_retriever()
    assert ids(narrow.search("zeta", "hybrid", k=1)) == [chunks[1].chunk_id]
    assert ids(wide.search("zeta", "hybrid", k=1)) == [chunks[2].chunk_id]


def test_enough_candidates_are_used_to_return_k_results_even_if_candidates_is_small():
    retriever, _ = scripted_retriever(candidates=1)
    assert len(retriever.search("zeta", "hybrid", k=3).chunks) == 3


def test_hybrid_never_returns_a_chunk_twice(retriever):
    result = retriever.search("remote biometric identification systems", "hybrid", k=5)
    assert len(set(ids(result))) == len(result.chunks)


def test_hybrid_scores_are_descending(retriever):
    scores = [s.score for s in retriever.search("biometric identification penalties", "hybrid", k=5).chunks]
    assert scores == sorted(scores, reverse=True)


# --- reranking (R2.4) ------------------------------------------------------------------------------------------------------------

class SpyReranker:
    """Scores passages by a lookup, and records every call so tests can see when it is (not) used."""

    name = "spy"

    def __init__(self, by_word: dict[str, float] | None = None, default: float = 0.0):
        self.by_word, self.default = by_word or {}, default
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        # The longest matching key wins, so "zeta alpha" is not shadowed by its substring "alpha".
        return [
            self.by_word[max((w for w in self.by_word if w in p), key=len)] if any(w in p for w in self.by_word) else self.default
            for p in passages
        ]


def reranked_retriever(reranker, **kwargs) -> tuple[Retriever, list[Chunk]]:
    """The scripted corpus (hybrid order for 'zeta': d2, d1, d0, d3, then fillers) with a reranker attached."""
    base, chunks = scripted_retriever()
    return Retriever(base._index, base._embedder, reranker, **kwargs), chunks


def test_the_reranker_decides_the_final_order_not_the_fusion():
    """Hybrid order is d2 d1 d0 d3. This reranker prefers d3, then d0, then d1, then d2: the reverse."""
    reranker = SpyReranker({"beta": 9.0, "alpha": 5.0, "zeta alpha": 3.0, "zeta zeta zeta": 1.0})
    retriever, chunks = reranked_retriever(reranker)
    fused = ids(retriever.search("zeta", "hybrid", k=4))
    result = retriever.search("zeta", "hybrid+rerank", k=4)
    assert fused == [chunks[2].chunk_id, chunks[1].chunk_id, chunks[0].chunk_id, chunks[3].chunk_id]
    assert ids(result) == [chunks[3].chunk_id, chunks[0].chunk_id, chunks[1].chunk_id, chunks[2].chunk_id]


def test_the_scores_of_a_reranked_result_are_the_rerankers_own():
    retriever, chunks = reranked_retriever(SpyReranker({"beta": 9.0, "alpha": 5.0, "zeta alpha": 3.0, "zeta zeta zeta": 1.0}))
    result = retriever.search("zeta", "hybrid+rerank", k=4)
    assert [s.score for s in result.chunks] == [9.0, 5.0, 3.0, 1.0]


def test_the_relevance_of_a_reranked_result_is_the_top_rerank_score_and_says_so():
    retriever, _ = reranked_retriever(SpyReranker({"beta": 9.0, "alpha": 5.0}))
    result = retriever.search("zeta", "hybrid+rerank", k=3)
    assert result.relevance == 9.0 and result.relevance_kind == "rerank"
    assert result.relevance == result.chunks[0].score


def test_a_rerank_relevance_can_be_negative_like_a_cross_encoder_logit():
    retriever, _ = reranked_retriever(SpyReranker(default=-11.0))
    result = retriever.search("zeta", "hybrid+rerank", k=2)
    assert result.relevance == -11.0 and result.relevance_kind == "rerank"


@pytest.mark.parametrize("config", NON_RERANK)
def test_the_other_configs_still_use_the_cosine_even_when_a_reranker_is_attached(config):
    retriever, _ = reranked_retriever(SpyReranker(default=7.0))
    result = retriever.search("zeta", config, k=2)
    assert result.relevance_kind == "cosine" and result.relevance != 7.0


@pytest.mark.parametrize("config", NON_RERANK)
def test_the_reranker_is_never_called_for_the_other_configs(config):
    """R2.4: when not reranking, the reranker is skipped entirely."""
    spy = SpyReranker()
    retriever, _ = reranked_retriever(spy)
    retriever.search("zeta", config, k=3)
    assert spy.calls == []


def test_the_reranker_is_called_once_per_search_with_the_question_and_the_candidates():
    spy = SpyReranker()
    retriever, chunks = reranked_retriever(spy)
    retriever.search("zeta", "hybrid+rerank", k=3)
    assert len(spy.calls) == 1
    query, passages = spy.calls[0]
    assert query == "zeta"
    assert passages[0] == chunks[2].embed_text            # the best fused candidate comes first
    assert passages[:3] == [chunks[i].embed_text for i in (2, 1, 0)]


def test_the_reranker_reads_the_header_too_not_just_the_body():
    spy = SpyReranker()
    chunks = [chunk("article:99", "Member States shall lay down rules.", "Penalties")] + [
        chunk(f"recital:{n}", f"filler{n} padding") for n in range(1, 8)
    ]
    embedder = HashEmbedder(dimension=1024)
    Retriever(make_index(chunks, embedder), embedder, spy).search("penalties", "hybrid+rerank", k=1)
    assert any(p.startswith("Article 99 — Penalties\n") for p in spy.calls[0][1])


def test_only_the_top_rerank_candidates_are_reranked():
    """With 3 candidates the reranker sees just the 3 best fused chunks, even though there are ten chunks."""
    spy = SpyReranker()
    retriever, chunks = reranked_retriever(spy, rerank_candidates=3)
    retriever.search("zeta", "hybrid+rerank", k=2)
    assert [p for p in spy.calls[0][1]] == [chunks[i].embed_text for i in (2, 1, 0)]


def test_a_chunk_outside_the_rerank_candidates_cannot_win_however_the_reranker_would_score_it():
    retriever, chunks = reranked_retriever(SpyReranker({"beta": 99.0}), rerank_candidates=3)   # d3 would win, but is 4th
    assert chunks[3].chunk_id not in ids(retriever.search("zeta", "hybrid+rerank", k=3))


def test_enough_candidates_are_reranked_to_return_k_even_if_rerank_candidates_is_small():
    spy = SpyReranker()
    retriever, _ = reranked_retriever(spy, rerank_candidates=2)
    assert len(retriever.search("zeta", "hybrid+rerank", k=5).chunks) == 5
    assert len(spy.calls[0][1]) == 5


def test_the_default_number_of_rerank_candidates_is_twenty():
    spy = SpyReranker()
    retriever, _ = reranked_retriever(spy)      # the scripted index has only ten chunks
    retriever.search("zeta", "hybrid+rerank", k=1)
    assert len(spy.calls[0][1]) == 10 and RERANK_CANDIDATES == 20
    chunks = [chunk(f"recital:{n}", f"word{n} common") for n in range(1, 41)]
    embedder = HashEmbedder(dimension=1024)
    spy2 = SpyReranker()
    Retriever(make_index(chunks, embedder), embedder, spy2).search("common", "hybrid+rerank", k=1)
    assert len(spy2.calls[0][1]) == 20


def test_chunks_the_reranker_scores_equally_keep_their_fused_order():
    retriever, chunks = reranked_retriever(SpyReranker(default=1.0))
    result = retriever.search("zeta", "hybrid+rerank", k=4)
    assert ids(result) == ids(retriever.search("zeta", "hybrid", k=4))


def test_a_reranked_result_has_distinct_chunks_and_k_of_them():
    retriever, _ = reranked_retriever(SpyReranker({"zeta": 1.0}))
    result = retriever.search("zeta", "hybrid+rerank", k=5)
    assert len(result.chunks) == 5 and len(set(ids(result))) == 5


def test_a_reranker_that_returns_the_wrong_number_of_scores_is_an_error():
    class Short(SpyReranker):
        def score(self, query, passages):
            return [1.0]

    retriever, _ = reranked_retriever(Short())
    with pytest.raises(RerankerError, match="returned 1 scores for"):
        retriever.search("zeta", "hybrid+rerank", k=3)


def test_a_reranker_that_returns_nan_is_an_error_and_not_a_result():
    class Nan(SpyReranker):
        def score(self, query, passages):
            return [float("nan")] * len(passages)

    retriever, _ = reranked_retriever(Nan())
    with pytest.raises(RerankerError, match="not a finite number"):
        retriever.search("zeta", "hybrid+rerank", k=3)


def test_a_reranked_search_validates_its_input_before_calling_the_reranker():
    spy = SpyReranker()
    retriever, _ = reranked_retriever(spy)
    for query, k in (("", 3), ("   ", 3), ("zeta", 0)):
        with pytest.raises(ValueError):
            retriever.search(query, "hybrid+rerank", k=k)
    assert spy.calls == []


def test_rerank_candidates_must_be_positive(stub):
    with pytest.raises(ValueError, match="candidates must be at least 1"):
        Retriever(make_index(CORPUS, stub), stub, SpyReranker(), rerank_candidates=0)


# --- the flag: RERANKER_ENABLED (R2.4) -----------------------------------------------------------------------------------------------

def test_the_app_uses_plain_hybrid_unless_reranking_is_enabled():
    assert app_config(Settings(_env_file=None, reranker_enabled=False)) == "hybrid"
    assert app_config(Settings(_env_file=None, reranker_enabled=True)) == "hybrid+rerank"
    assert app_config(Settings(_env_file=None)) == "hybrid"          # off by default


def test_with_the_flag_off_the_reranker_is_never_constructed(tmp_path, monkeypatch):
    """Not loaded, not downloaded, not held in memory: get_reranker must not even be called."""
    from askact import retrieval

    def explode(settings):
        raise AssertionError("the reranker must not be constructed when RERANKER_ENABLED is false")

    monkeypatch.setattr(retrieval, "get_reranker", explode)
    settings = write_stub_index(tmp_path / "index", CORPUS)
    retriever = Retriever.from_settings(settings, tmp_path / "index")
    assert sections(retriever.search("penalties", "hybrid", k=2))[0] == "article:99"
    with pytest.raises(ValueError, match="needs a reranker"):
        retriever.search("penalties", "hybrid+rerank", k=2)


def test_with_the_flag_on_the_reranker_is_constructed_from_the_settings(tmp_path, monkeypatch):
    from askact import retrieval

    built = []
    spy = SpyReranker({"penalties": 5.0})

    def make(settings):
        built.append(settings.reranker_model)
        return spy

    monkeypatch.setattr(retrieval, "get_reranker", make)
    base = write_stub_index(tmp_path / "index", CORPUS)
    settings = base.model_copy(update={"reranker_enabled": True, "reranker_model": "some/cross-encoder"})
    retriever = Retriever.from_settings(settings, tmp_path / "index")
    result = retriever.search("penalties", "hybrid+rerank", k=2)
    assert built == ["some/cross-encoder"] and spy.calls and result.relevance_kind == "rerank"


def test_the_stub_reranker_works_end_to_end_through_settings(tmp_path):
    base = write_stub_index(tmp_path / "index", CORPUS)
    settings = base.model_copy(update={"reranker_enabled": True, "reranker": "stub"})
    result = Retriever.from_settings(settings, tmp_path / "index").search("penalties applicable to infringements", app_config(settings), k=3)
    assert sections(result)[0] == "article:99" and result.relevance_kind == "rerank"
    assert 0.0 < result.relevance <= 1.0     # the stub's Jaccard scale


def test_a_missing_index_still_fails_clearly_before_any_reranker_is_built(tmp_path, monkeypatch):
    from askact import retrieval

    monkeypatch.setattr(retrieval, "get_reranker", lambda settings: SpyReranker())
    base = Settings(_env_file=None, embedder="stub", reranker_enabled=True)
    with pytest.raises(IndexUnavailableError, match="python -m askact.ingest"):
        Retriever.from_settings(base, tmp_path / "nowhere")


# --- how many results (R2.5) --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("config", NON_RERANK)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_exactly_k_chunks_are_returned(retriever, config, k):
    assert len(retriever.search("remote biometric identification systems", config, k=k).chunks) == k


@pytest.mark.parametrize("config", NON_RERANK)
def test_asking_for_more_than_the_index_holds_returns_everything_once(retriever, config):
    result = retriever.search("remote biometric identification systems", config, k=50)
    assert len(result.chunks) <= len(CORPUS)
    assert len(set(ids(result))) == len(result.chunks)
    if config != "hybrid":
        assert len(result.chunks) == len(CORPUS)


# --- result shape and metadata (R2.5) -------------------------------------------------------------------------------------

@pytest.mark.parametrize("config", NON_RERANK)
def test_each_result_carries_full_citation_metadata_and_a_finite_score(retriever, config):
    result = retriever.search("penalties", config, k=3)
    for scored in result.chunks:
        c = scored.chunk
        assert isinstance(c.section_id, SectionId)                       # the parent section, used for citations
        assert c.chunk_id == f"{c.section_id}#{c.ordinal}"
        assert c.kind == c.section_id.kind and c.number == c.section_id.number
        assert c.text and (c.title is None) == (c.kind == "recital")
        assert math.isfinite(scored.score)


def test_chunks_of_one_long_section_share_the_parent_section_id(stub):
    pieces = [chunk("article:3", f"definition number {n} says something", "Definitions", ordinal=n) for n in (1, 2, 3)]
    result = Retriever(make_index(pieces, stub), stub).search("definition says", "bm25", k=3)
    assert {str(s.chunk.section_id) for s in result.chunks} == {"article:3"}
    assert sorted(s.chunk.chunk_id for s in result.chunks) == ["article:3#1", "article:3#2", "article:3#3"]


def test_the_title_and_text_are_those_of_the_chunk_not_the_embedded_string(retriever):
    top = retriever.search("penalties", "bm25", k=1).chunks[0].chunk
    assert top.title == "Penalties" and top.text.startswith("Member States") and "Article 99" not in top.text


# --- relevance (the input to the not-covered gate) ---------------------------------------------------------------------------

@pytest.mark.parametrize("config", NON_RERANK)
def test_relevance_is_the_dense_cosine_of_the_top_chunk_for_every_config(config):
    retriever, chunks = scripted_retriever()
    result = retriever.search("zeta", config, k=2)
    top = chunks.index(result.chunks[0].chunk)
    cosine = float(retriever._index.embeddings[top] @ np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert result.relevance == pytest.approx(cosine)
    assert result.relevance_kind == "cosine"


def test_bm25_relevance_comes_from_the_embedder_not_from_the_bm25_score():
    retriever, chunks = scripted_retriever()
    result = retriever.search("zeta", "bm25", k=2)
    assert result.chunks[0].chunk == chunks[2]
    assert result.relevance == pytest.approx(0.6)       # cosine of d2 with the query
    assert result.relevance != pytest.approx(result.chunks[0].score)


def test_relevance_follows_the_top_ranked_chunk_of_each_config():
    """Hybrid ranks d2 first (cosine .6), dense ranks d0 first (cosine 1.0): the same question gets
    different relevance under different configs, because each uses its own top chunk."""
    retriever, _ = scripted_retriever()
    assert retriever.search("zeta", "dense", k=2).relevance == pytest.approx(1.0)
    assert retriever.search("zeta", "hybrid", k=2).relevance == pytest.approx(0.6)
    assert retriever.search("zeta", "bm25", k=2).relevance == pytest.approx(0.6)


# --- invalid use ------------------------------------------------------------------------------------------------------------------

def test_an_unknown_config_is_refused_and_the_valid_ones_are_listed(retriever):
    with pytest.raises(ValueError, match=r"unknown retrieval config 'fuzzy'; expected one of bm25, dense, hybrid, hybrid\+rerank"):
        retriever.search("penalties", "fuzzy", k=3)


def test_the_reranked_config_without_a_reranker_says_how_to_enable_it(retriever):
    with pytest.raises(ValueError, match=r"'hybrid\+rerank' needs a reranker.*RERANKER_ENABLED=true"):
        retriever.search("penalties", "hybrid+rerank", k=3)


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(retriever, k):
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever.search("penalties", "bm25", k=k)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_a_blank_query_is_refused(retriever, query):
    with pytest.raises(ValueError, match="query is empty"):
        retriever.search(query, "bm25", k=3)


def test_the_configs_are_the_four_documented_ones():
    assert CONFIGS == ("bm25", "dense", "hybrid", "hybrid+rerank")


def test_the_documented_defaults():
    assert (RRF_K, HYBRID_CANDIDATES, RERANK_CANDIDATES) == (60, 30, 20)


def test_the_default_number_of_candidates_is_used_when_none_is_given(stub):
    """With the default of 30, a chunk at dense rank 25 still enters the fusion; with a tiny default it would not."""
    chunks = [chunk(f"recital:{n}", f"word{n} common") for n in range(1, 41)]
    vectors = {c.embed_text: [1.0 - 0.02 * n, 0.0, 0.0, 0.0] for n, c in enumerate(chunks)}
    embedder = ScriptedEmbedder(vectors, [1.0, 0.0, 0.0, 0.0])
    result = Retriever(make_index(chunks, embedder), embedder).search("common", "hybrid", k=40)
    assert len(result.chunks) >= 30


def test_a_dimension_mismatch_alone_is_refused_even_if_the_name_matches(stub):
    index = make_index(CORPUS, stub)

    class SameNameWrongSize(HashEmbedder):
        def __init__(self):
            super().__init__(dimension=64)
            self.name = stub.name  # same name as the index's embedder, different width

    with pytest.raises(ValueError, match=r"\(1024 dimensions\) but the embedder is 'stub-hash-1024' \(64 dimensions\)"):
        Retriever(index, SameNameWrongSize())


def test_an_embedder_from_another_model_than_the_index_is_refused(stub):
    index = make_index(CORPUS, stub)
    with pytest.raises(ValueError, match=r"built with 'stub-hash-1024' \(1024 dimensions\) but the embedder is 'stub-hash-64'"):
        Retriever(index, HashEmbedder(dimension=64))


def test_candidates_must_be_positive(stub):
    with pytest.raises(ValueError, match="candidates must be at least 1"):
        Retriever(make_index(CORPUS, stub), stub, candidates=0)


# --- loading from settings (R2.8) --------------------------------------------------------------------------------------------------

def write_stub_index(directory: Path, chunks: list[Chunk]) -> Settings:
    settings = Settings(_env_file=None, embedder="stub")
    index = make_index(chunks, HashEmbedder())
    write_index(directory, manifest=index.manifest, chunks=list(index.chunks), embeddings=index.embeddings)
    return settings


def test_from_settings_loads_the_index_and_searches_it(tmp_path):
    settings = write_stub_index(tmp_path / "index", CORPUS)
    result = Retriever.from_settings(settings, tmp_path / "index").search("penalties", "hybrid", k=2)
    assert sections(result)[0] == "article:99"


def test_a_missing_index_is_an_error_that_says_to_run_ingestion(tmp_path):
    with pytest.raises(IndexUnavailableError) as excinfo:
        Retriever.from_settings(Settings(_env_file=None, embedder="stub"), tmp_path / "nowhere")
    assert excinfo.value.kind == "missing" and "python -m askact.ingest" in str(excinfo.value)


def test_a_stale_index_is_an_error_that_says_to_run_ingestion(tmp_path):
    write_stub_index(tmp_path / "index", CORPUS)
    with pytest.raises(IndexUnavailableError) as excinfo:
        Retriever.from_settings(Settings(_env_file=None, embedder="stub", chunk_max_chars=1500), tmp_path / "index")
    assert excinfo.value.kind == "stale"


# --- the fixture (a trimmed copy of the real page) ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_retriever() -> Retriever:
    embedder = HashEmbedder(dimension=1024)
    return Retriever(make_index(chunk_act(parse_act(FIXTURE.read_bytes()), 1800), embedder), embedder)


@pytest.mark.parametrize("config", ["bm25", "hybrid"])
def test_the_exact_title_words_find_the_article(fixture_retriever, config):
    top3 = sections(fixture_retriever.search("Prohibited AI practices", config, k=3))
    assert top3[0] == "article:5"


def test_dense_with_the_bag_of_words_stub_does_not_weight_rare_words_so_it_may_miss_the_title(fixture_retriever):
    """The stub has no idf: a short chunk repeating 'AI' can beat the chunk that has the title words.
    That is why lexical search is part of the design, and why this test only asserts that the dense
    ranking is sensible (descending) and that the article's chunks are among the results at all."""
    result = fixture_retriever.search("Prohibited AI practices", "dense", k=8)
    scores = [s.score for s in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert "article:5" in sections(result)


def test_all_chunks_of_the_long_fixture_article_are_reachable(fixture_retriever):
    result = fixture_retriever.search("Prohibited AI practices", "bm25", k=20)
    assert len([s for s in result.chunks if str(s.chunk.section_id) == "article:5"]) >= 6


# --- the real document (runs only when the file is present) -------------------------------------------------------------------------------

real_source = pytest.mark.skipif(not REAL.exists(), reason=f"the full source is not present at {REAL}")


@pytest.fixture(scope="module")
def real_retriever() -> Retriever:
    embedder = HashEmbedder(dimension=1024)
    return Retriever(make_index(chunk_act(parse_act(REAL.read_bytes()), 1800), embedder), embedder)


@real_source
def test_real_source_has_the_expected_number_of_chunks_to_search(real_retriever):
    assert len(real_retriever._index.chunks) == 495


@real_source
def test_real_source_bm25_finds_an_article_by_its_number_and_title(real_retriever):
    """A sanity check of the plumbing on the real text, not a measure of retrieval quality (that is the
    evaluation's job): the header words 'Article 99' and 'Penalties' are in the indexed string."""
    assert {str(s.chunk.section_id) for s in real_retriever.search("Article 99 Penalties", "bm25", k=4).chunks} >= {"article:99"}
    assert sections(real_retriever.search("Article 99 Penalties", "bm25", k=1)) == ["article:99"]


@real_source
@pytest.mark.parametrize("config", NON_RERANK)
def test_real_source_every_config_returns_k_distinct_chunks_with_finite_scores(real_retriever, config):
    result = real_retriever.search("What are the obligations of providers of high-risk AI systems?", config, k=10)
    assert len(result.chunks) == 10 and len(set(ids(result))) == 10
    assert all(math.isfinite(s.score) for s in result.chunks) and math.isfinite(result.relevance)


# --- the real embedder (slow, local only) --------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_the_real_model_ranks_the_matching_fixture_article_first_in_every_config():
    pytest.importorskip("sentence_transformers", reason="the `models` extra is not installed")
    from askact.embedders import get_embedder

    settings = Settings(_env_file=None, embedder="real")
    embedder = get_embedder(settings)
    retriever = Retriever(make_index(chunk_act(parse_act(FIXTURE.read_bytes()), 1800), embedder), embedder)
    for config in NON_RERANK:
        result = retriever.search("Which AI practices are prohibited?", config, k=3)
        assert str(result.chunks[0].chunk.section_id) in {"article:5", "recital:8"}, config
        assert 0.0 < result.relevance <= 1.0001 and result.relevance_kind == "cosine"


@pytest.mark.slow
def test_the_real_cross_encoder_reranks_the_fixture_and_separates_on_topic_from_off_topic():
    pytest.importorskip("sentence_transformers", reason="the `models` extra is not installed")
    from askact.embedders import get_embedder
    from askact.rerankers import get_reranker

    settings = Settings(_env_file=None, embedder="real", reranker="real", reranker_enabled=True)
    embedder = get_embedder(settings)
    retriever = Retriever(
        make_index(chunk_act(parse_act(FIXTURE.read_bytes()), 1800), embedder), embedder, get_reranker(settings)
    )
    assert app_config(settings) == "hybrid+rerank"

    on_topic = retriever.search("Which AI practices are prohibited?", "hybrid+rerank", k=3)
    off_topic = retriever.search("How do I bake sourdough bread?", "hybrid+rerank", k=3)

    assert str(on_topic.chunks[0].chunk.section_id) == "article:5"
    assert on_topic.relevance_kind == "rerank" and off_topic.relevance_kind == "rerank"
    scores = [s.score for s in on_topic.chunks]
    assert scores == sorted(scores, reverse=True)          # reranked order is by the reranker's score
    assert on_topic.relevance == scores[0]
    assert on_topic.relevance > 0 > off_topic.relevance    # a logit scale: relevant is positive, irrelevant negative
