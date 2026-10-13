"""Command line: `python -m askact.evaluation.run [--no-generation] [--k K] [--split dev|test|both] [--configs ...]`.

Scores each retrieval config on the questions in `eval/questions.yaml`.

Retrieval (no LLM, free): hit@k and MRR on the in-scope questions, plus the range of the relevance score (the
input to the "not covered" gate) for in-scope and out_of_scope questions, which is what the thresholds are tuned
from (R5.10). `--no-generation` stops here.

Generation (the default; calls the LLM, bounded and cached, see `evaluation.cache`): every question goes through
the same `answer()` pipeline as the API, and the report adds refusal accuracy on the out_of_scope questions and
citation precision and recall on the in-scope ones (R5.7, R5.8).

The report is printed and written to `eval/results.md`; nothing is written, and no table is printed, if anything
fails or the call budget runs out (R5.4, R5.9). Exit code 0 on success, 1 on any failure.
"""

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from askact.config import Settings
from askact.embedders import Embedder, EmbedderError, get_embedder
from askact.evaluation.cache import CACHE_PATH, BudgetExceededError, CacheError, CachingLLM
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
from askact.evaluation.schema import QUESTIONS_PATH, EvalSetError, Question, Split, load_questions
from askact.generation.answer import AnswerResult, GateNotConfiguredError, answer, relevance_threshold
from askact.generation.llm import LLM, LLMBadOutputError, LLMUnavailableError, get_llm
from askact.index import INDEX_DIR, IndexUnavailableError, load_index
from askact.models import RetrievalResult
from askact.rerankers import Reranker, RerankerError, get_reranker
from askact.retrieval import CONFIGS, Config, Retriever, relevance_kind

RESULTS_PATH = Path("eval/results.md")
_EXPECTED_FAILURES = (
    EvalSetError, IndexUnavailableError, EmbedderError, RerankerError, GateNotConfiguredError, CacheError, OSError,
)
# These two leave replies behind in the cache, so their message says that running again continues from there.
_PAID_FOR_FAILURES = (BudgetExceededError, LLMUnavailableError)

# Turns a question and its retrieval result into the system's answer. In production it is `answer()` over a
# caching LLM; tests script it.
Answerer = Callable[[str, RetrievalResult], AnswerResult]


class Searcher(Protocol):
    """What the evaluation needs from retrieval. `Retriever` provides it; tests script it with known answers."""

    def search(self, query: str, config: Config, k: int) -> RetrievalResult: ...


@dataclass(frozen=True)
class Spread:
    """The lowest and highest relevance score seen, and how many questions that covers."""

    low: float
    high: float
    count: int


@dataclass(frozen=True)
class GenerationResult:
    refusal: RefusalScores | None  # None if the split has no out_of_scope question
    citations: CitationScores | None  # None if the split has no in-scope question
    # Replies the model gave that could not be read (`LLMBadOutputError`). They are scored as "no answer": not a
    # refusal, and no citations. Shown, so that a bad run of replies cannot hide inside the other numbers.
    unusable_replies: int


@dataclass(frozen=True)
class ConfigResult:
    config: Config
    scores: RetrievalScores | None  # None if the split has no in-scope question: the averages are undefined
    relevance_scale: str  # "cosine" or "rerank": the two are not comparable (R3.8)
    in_scope_relevance: Spread | None  # None if there is no question of that kind
    out_of_scope_relevance: Spread | None
    generation: GenerationResult | None = None  # None for a retrieval-only run


@dataclass(frozen=True)
class SplitResult:
    split: Split
    in_scope: int
    out_of_scope: int
    configs: list[ConfigResult]


@dataclass(frozen=True)
class GenerationInfo:
    """What a generation run states about itself, on top of `RunInfo`."""

    provider: str
    model: str
    max_output_tokens: int
    cosine_threshold: float | None
    rerank_threshold: float | None
    max_calls: int  # EVAL_MAX_LLM_CALLS
    uncached_calls: int  # calls made to the provider in this run
    cache_hits: int  # replies reused from the cache (from earlier runs, or an identical request earlier in this one)


@dataclass(frozen=True)
class RunInfo:
    """What the report states about the run, so that its numbers can be traced to what produced them."""

    day: str
    k: int
    embedding_model: str
    reranker: str | None  # None if no selected config uses one
    build_hash: str
    chunks: int
    uses_stubs: bool
    generation: GenerationInfo | None = None  # None for a retrieval-only run


def _spread(values: Sequence[float]) -> Spread | None:
    return Spread(min(values), max(values), len(values)) if values else None


def evaluate_split(
    questions: Sequence[Question],
    retriever: Searcher,
    configs: Sequence[Config],
    k: int,
    answerer: Answerer | None = None,
) -> list[ConfigResult]:
    """Run every question through every config. `questions` are all from one split.

    With an `answerer`, each question's retrieval result is also turned into an answer and scored. Without one
    (`--no-generation`) nothing but retrieval runs, and the results carry no generation scores.
    """
    results = []
    for config in configs:
        ranks: list[int | None] = []
        in_scope_relevance: list[float] = []
        out_of_scope_relevance: list[float] = []
        refusals: list[bool] = []  # out_of_scope questions: did the system say "not covered"?
        citations: list[tuple[float, float]] = []  # in-scope questions: (precision, recall)
        unusable_replies = 0
        scale = ""
        for question in questions:
            found = retriever.search(question.question, config, k)
            scale = found.relevance_kind
            out_of_scope = question.category == "out_of_scope"
            if out_of_scope:
                out_of_scope_relevance.append(found.relevance)  # no expected section, so no hit to look for (R5.6)
            else:
                in_scope_relevance.append(found.relevance)
                retrieved = [scored.chunk.section_id for scored in found.chunks]
                ranks.append(first_hit_rank(retrieved, question.expected_ids, k))
            if answerer is None:
                continue
            try:
                given = answerer(question.question, found)
            except LLMBadOutputError:
                # The model answered, but not in the required form. That is a result about the system, not a
                # reason to abandon a paid run (and a cached bad reply would fail the same way on every re-run).
                # It counts as no answer: no citations, and not a refusal, since "not covered" was not said.
                unusable_replies += 1
                refused, cited = False, []
            else:
                refused, cited = given.status == "not_covered", given.citations
            if out_of_scope:
                refusals.append(refused)
            else:
                citations.append(citation_precision_recall(cited, question.expected_ids))
        results.append(
            ConfigResult(
                config=config,
                scores=score_ranks(ranks) if ranks else None,
                relevance_scale=scale,
                in_scope_relevance=_spread(in_scope_relevance),
                out_of_scope_relevance=_spread(out_of_scope_relevance),
                generation=None if answerer is None else GenerationResult(
                    refusal=score_refusals(refusals) if refusals else None,
                    citations=score_citations(citations) if citations else None,
                    unusable_replies=unusable_replies,
                ),
            )
        )
    return results


# --- report ---------------------------------------------------------------------------------------------

def _scores_cells(scores: RetrievalScores | None) -> str:
    if scores is None:
        return "n/a | n/a"
    return f"{scores.hit_at_k:.3f} ({scores.hits}/{scores.questions}) | {scores.mrr:.3f}"


def _generation_cells(generation: GenerationResult | None) -> str:
    """Refusal accuracy, citation precision, citation recall: three cells, each starting with ' | '."""
    if generation is None:
        return " | n/a | n/a | n/a"
    refusal, citations = generation.refusal, generation.citations
    return (
        f" | {'n/a' if refusal is None else f'{refusal.accuracy:.3f} ({refusal.refused}/{refusal.questions})'}"
        f" | {'n/a' if citations is None else f'{citations.precision:.3f}'}"
        f" | {'n/a' if citations is None else f'{citations.recall:.3f}'}"
    )


def _threshold_text(value: float | None) -> str:
    return "not set" if value is None else str(value)


def _spread_cell(spread: Spread | None) -> str:
    if spread is None:
        return "no questions"
    return f"{spread.low:.3f} to {spread.high:.3f} (n={spread.count})"


def render_report(info: RunInfo, splits: Sequence[SplitResult]) -> str:
    generation = info.generation
    lines = ["# Retrieval evaluation" if generation is None else "# Evaluation: retrieval and generation", ""]
    if info.uses_stubs:
        # The LLM is only named when one is part of the run; a retrieval-only report keeps its original wording.
        names, what = (
            ("EMBEDDER=stub or RERANKER=stub", "retrieval quality")
            if generation is None
            else ("EMBEDDER=stub, RERANKER=stub or LLM_PROVIDER=fake", "quality")
        )
        lines += [
            f"> **WARNING: stub models were used ({names}).** These numbers only exercise the code; "
            f"they say nothing about {what}. Do not publish them.",
            "",
        ]
    lines += [
        f"- Date: {info.day} (UTC)",
        f"- k: {info.k} (chunks; hit@{info.k} and MRR look at the top {info.k} chunks of each question)",
        f"- Embedding model: {info.embedding_model}",
        f"- Reranker: {info.reranker or 'not used'}",
        f"- Index: {info.chunks} chunks, build hash {info.build_hash}",
    ]
    lines += [f"- Questions in {s.split}: {s.in_scope} in-scope, {s.out_of_scope} out_of_scope" for s in splits]
    if generation is None:
        lines += ["- Generation: not run (`--no-generation`); retrieval metrics only", ""]
    else:
        lines += [
            f"- Generation: LLM {generation.provider or 'not set'} / {generation.model or 'not set'}, "
            f"at most {generation.max_output_tokens} output tokens per answer",
            f"- Relevance thresholds (the gate): cosine {_threshold_text(generation.cosine_threshold)}, "
            f"rerank {_threshold_text(generation.rerank_threshold)}",
            f"- LLM calls: {generation.uncached_calls} made (limit {generation.max_calls}), "
            f"{generation.cache_hits} answered from the cache. Each request is answered once and its reply is kept "
            "(`eval/.cache/`), so running again reproduces these numbers without new calls.",
            "",
        ]

    for result in splits:
        if result.split == "test":
            lines += [
                "## test split: FINAL",
                "",
                "The final result. It must not be used to choose parameters or thresholds (R5.10).",
            ]
        else:
            lines += ["## dev split", "", "The split that parameters and thresholds are tuned on."]
        lines += [
            "",
            f"hit@{info.k} and MRR are over the {result.in_scope} in-scope "
            f"question{'' if result.in_scope == 1 else 's'} only. MRR is the mean of "
            "1 / (rank of the first chunk from an expected section), counting 0 when there is none in the top "
            f"{info.k}.",
        ]
        if generation is not None:
            lines += [
                "",
                f"Refusal accuracy is over the {result.out_of_scope} out_of_scope "
                f"question{'' if result.out_of_scope == 1 else 's'}: the share the system answered \"not covered\" "
                "(the relevance gate refused, or the model said so, or none of its citations named a retrieved "
                f"section). Citation precision and recall are means over the {result.in_scope} in-scope "
                f"question{'' if result.in_scope == 1 else 's'} of: cited sections that are expected / cited "
                "sections, and cited sections that are expected / expected sections. An answer with no "
                "citations, including a refusal, scores 0 on both.",
            ]
        header = f"| Config | hit@{info.k} | MRR |" + (
            " Refusal accuracy | Citation precision | Citation recall |" if generation else ""
        )
        lines += ["", header, "|---|---|---|" + ("---|---|---|" if generation else "")]
        lines += [
            f"| {c.config} | {_scores_cells(c.scores)}{_generation_cells(c.generation) if generation else ''} |"
            for c in result.configs
        ]
        if generation is not None:
            counts = ", ".join(f"{c.config} {c.generation.unusable_replies if c.generation else 0}" for c in result.configs)
            lines += [
                "",
                "Replies the model gave that could not be read (scored as no answer, not as a refusal): " + counts,
            ]
        lines += [
            "",
            "Relevance score of the top-ranked chunk, lowest to highest (the gate's input; "
            "cosine and rerank scores are on different scales):",
            "",
            "| Config | Scale | in-scope questions | out_of_scope questions |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {c.config} | {c.relevance_scale} | {_spread_cell(c.in_scope_relevance)} | "
            f"{_spread_cell(c.out_of_scope_relevance)} |"
            for c in result.configs
        ]
        lines.append("")
    return "\n".join(lines)


def _write_atomically(path: Path, text: str) -> None:
    """Replace the file in one step, so an interrupted run cannot leave half a report behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


class _Unmetered:
    """The `SpendGuard` of an evaluation run: it reserves nothing and refuses nothing.

    The evaluation is an operator-run tool, not the public endpoint, so the daily cap does not apply to it and
    the ledger is not touched. What bounds its cost is the cap on uncached LLM calls (EVAL_MAX_LLM_CALLS) in
    `CachingLLM`. It is one object because the reservation it hands back does nothing either.
    """

    def reserve(self, system: str, user: str, max_output_tokens: int) -> "_Unmetered":
        return self

    def reconcile(self, input_tokens: int | None, output_tokens: int | None) -> None:
        pass


def _answerer(settings: Settings, llm: LLM) -> Answerer:
    """The real `answer()` pipeline over `llm`, with no spend ledger."""
    unmetered = _Unmetered()

    def answer_one(text: str, found: RetrievalResult) -> AnswerResult:
        return answer(text, found, settings, llm, unmetered)

    return answer_one


# --- command line ----------------------------------------------------------------------------------------

def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    index_dir: Path = INDEX_DIR,
    questions_path: Path = QUESTIONS_PATH,
    results_path: Path = RESULTS_PATH,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    llm: LLM | None = None,
    cache_path: Path = CACHE_PATH,
    now: datetime | None = None,
) -> int:
    """Run the evaluation; returns the process exit code. The keyword arguments exist for tests."""
    parser = argparse.ArgumentParser(
        prog="python -m askact.evaluation.run",
        description="Score each retrieval config on eval/questions.yaml: hit@k and MRR, and (unless --no-generation) "
        "refusal accuracy and citation precision and recall from real answers.",
    )
    parser.add_argument(
        "--no-generation", action="store_true",
        help="retrieval metrics only: no LLM is called, nothing is spent",
    )
    parser.add_argument("--k", type=_positive_int, help="chunks per question (default: TOP_K)")
    parser.add_argument(
        "--split", choices=["dev", "test", "both"], default="dev",
        help="which questions to score; the default, dev, is the split to tune on (test is the final result)",
    )
    parser.add_argument(
        "--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS), metavar="CONFIG",
        help=f"retrieval configs to compare, from: {', '.join(CONFIGS)} (default: all four)",
    )
    args = parser.parse_args(argv)

    try:
        settings = settings or Settings()
    except ValidationError as exc:  # bad environment variables
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 1
    k: int = args.k or settings.top_k
    configs: list[Config] = [c for c in CONFIGS if c in args.configs]  # always in the same order, no repeats
    split_names: list[Split] = ["dev", "test"] if args.split == "both" else [args.split]
    generate: bool = not args.no_generation
    provider = (settings.llm_provider or "").strip().lower()  # as get_llm reads it
    model = settings.llm_model or ""
    caching: CachingLLM | None = None

    try:
        questions = load_questions(questions_path)
        by_split: dict[Split, list[Question]] = {name: [q for q in questions if q.split == name] for name in split_names}
        for name, chosen in by_split.items():
            if not chosen:
                raise EvalSetError(f"{questions_path}: no questions in the {name} split, so there is nothing to score")

        if generate:
            # Before the first paid call, not when the first question reaches the gate: a threshold that is not
            # set for one of the chosen configs would otherwise stop the run after the other configs were paid for.
            for config in configs:
                relevance_threshold(settings, relevance_kind(config))
            if llm is None and provider == "fake":
                # `fake` is built with no script (it is for tests, which replace it). Run as it is, the first call
                # would end in a traceback from the fake and an evaluation of nothing.
                raise LLMUnavailableError(
                    "LLM_PROVIDER=fake has no scripted replies, so it cannot be evaluated: set a real provider, "
                    "or pass --no-generation"
                )
            caching = CachingLLM(llm or get_llm(settings), cache_path, provider=provider, model=model,
                                 max_calls=settings.eval_max_llm_calls)
        index = load_index(settings, index_dir)
        embedder = embedder or get_embedder(settings)
        # The harness compares all four configs whatever RERANKER_ENABLED says (that flag is for the app), so
        # it builds the reranker itself, and only if a selected config needs it.
        uses_reranker = "hybrid+rerank" in configs
        if uses_reranker and reranker is None:
            reranker = get_reranker(settings)
        try:
            retriever = Retriever(index, embedder, reranker)
        except ValueError as exc:  # the index and the embedder do not match
            print(f"error: {exc}", file=sys.stderr)
            return 1

        answerer = _answerer(settings, caching) if caching is not None else None
        results = [
            SplitResult(
                split=name,
                in_scope=sum(q.category != "out_of_scope" for q in chosen),
                out_of_scope=sum(q.category == "out_of_scope" for q in chosen),
                configs=evaluate_split(chosen, retriever, configs, k, answerer),
            )
            for name, chosen in by_split.items()
        ]
        info = RunInfo(
            day=(now or datetime.now(UTC)).strftime("%Y-%m-%d"),
            k=k,
            embedding_model=embedder.name,
            reranker=reranker.name if uses_reranker and reranker else None,
            build_hash=index.manifest.build_hash,
            chunks=len(index.chunks),
            uses_stubs=settings.embedder == "stub" or (uses_reranker and settings.reranker == "stub")
            or (generate and provider == "fake"),
            generation=None if caching is None else GenerationInfo(
                provider=provider, model=model, max_output_tokens=settings.max_output_tokens,
                cosine_threshold=settings.relevance_threshold_cosine, rerank_threshold=settings.relevance_threshold_rerank,
                max_calls=settings.eval_max_llm_calls, uncached_calls=caching.uncached_calls, cache_hits=caching.cache_hits,
            ),
        )
        report = render_report(info, results)
        _write_atomically(results_path, report)
    except _EXPECTED_FAILURES as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except _PAID_FOR_FAILURES as exc:
        print(f"error: {exc}", file=sys.stderr)
        if caching is not None:  # False if the provider could not even be built: nothing was asked, nothing saved
            print(f"The replies received so far are saved in {cache_path}. Fix the problem (or raise "
                  "EVAL_MAX_LLM_CALLS) and run again to continue from there. No report was written.", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
