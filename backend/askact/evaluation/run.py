"""Command line: `python -m askact.evaluation.run --no-generation [--k K] [--split dev|test|both] [--configs ...]`.

Scores each retrieval config on the hand-written questions in `eval/questions.yaml`: hit@k and MRR on
the in-scope questions, plus the range of the relevance score (the input to the "not covered" gate)
for in-scope and out_of_scope questions, which is what the thresholds are tuned from (R5.10). No LLM
is involved. The report is printed and written to `eval/results.md`; nothing is written, and no table
is printed, if anything fails (R5.4). Exit code 0 on success, 1 on any failure.

Generation metrics (refusal accuracy, citation precision and recall) arrive with Task 19, so for now
`--no-generation` is required: a table that silently lacked those columns would look complete.
"""

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from askact.config import Settings
from askact.embedders import Embedder, EmbedderError, get_embedder
from askact.evaluation.metrics import RetrievalScores, first_hit_rank, score_ranks
from askact.evaluation.schema import QUESTIONS_PATH, EvalSetError, Question, Split, load_questions
from askact.index import INDEX_DIR, IndexUnavailableError, load_index
from askact.models import RetrievalResult
from askact.rerankers import Reranker, RerankerError, get_reranker
from askact.retrieval import CONFIGS, Config, Retriever

RESULTS_PATH = Path("eval/results.md")
_EXPECTED_FAILURES = (EvalSetError, IndexUnavailableError, EmbedderError, RerankerError, OSError)


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
class ConfigResult:
    config: Config
    scores: RetrievalScores | None  # None if the split has no in-scope question: the averages are undefined
    relevance_scale: str  # "cosine" or "rerank": the two are not comparable (R3.8)
    in_scope_relevance: Spread | None  # None if there is no question of that kind
    out_of_scope_relevance: Spread | None


@dataclass(frozen=True)
class SplitResult:
    split: Split
    in_scope: int
    out_of_scope: int
    configs: list[ConfigResult]


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


def _spread(values: Sequence[float]) -> Spread | None:
    return Spread(min(values), max(values), len(values)) if values else None


def evaluate_split(
    questions: Sequence[Question], retriever: Searcher, configs: Sequence[Config], k: int
) -> list[ConfigResult]:
    """Run every question through every config. `questions` are all from one split."""
    results = []
    for config in configs:
        ranks: list[int | None] = []
        in_scope_relevance: list[float] = []
        out_of_scope_relevance: list[float] = []
        scale = ""
        for question in questions:
            found = retriever.search(question.question, config, k)
            scale = found.relevance_kind
            if question.category == "out_of_scope":
                out_of_scope_relevance.append(found.relevance)
                continue  # no expected section, so no hit to look for (R5.6)
            in_scope_relevance.append(found.relevance)
            retrieved = [scored.chunk.section_id for scored in found.chunks]
            ranks.append(first_hit_rank(retrieved, question.expected_ids, k))
        results.append(
            ConfigResult(
                config=config,
                scores=score_ranks(ranks) if ranks else None,
                relevance_scale=scale,
                in_scope_relevance=_spread(in_scope_relevance),
                out_of_scope_relevance=_spread(out_of_scope_relevance),
            )
        )
    return results


# --- report ---------------------------------------------------------------------------------------------

def _scores_cells(scores: RetrievalScores | None) -> str:
    if scores is None:
        return "n/a | n/a"
    return f"{scores.hit_at_k:.3f} ({scores.hits}/{scores.questions}) | {scores.mrr:.3f}"


def _spread_cell(spread: Spread | None) -> str:
    if spread is None:
        return "no questions"
    return f"{spread.low:.3f} to {spread.high:.3f} (n={spread.count})"


def render_report(info: RunInfo, splits: Sequence[SplitResult]) -> str:
    lines = ["# Retrieval evaluation", ""]
    if info.uses_stubs:
        lines += [
            "> **WARNING: stub models were used (EMBEDDER=stub or RERANKER=stub).** These numbers only exercise the "
            "code; they say nothing about retrieval quality. Do not publish them.",
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
    lines += ["- Generation: not run (`--no-generation`); retrieval metrics only", ""]

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
            "",
            f"| Config | hit@{info.k} | MRR |",
            "|---|---|---|",
        ]
        lines += [f"| {c.config} | {_scores_cells(c.scores)} |" for c in result.configs]
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
    now: datetime | None = None,
) -> int:
    """Run the evaluation; returns the process exit code. The keyword arguments exist for tests."""
    parser = argparse.ArgumentParser(
        prog="python -m askact.evaluation.run",
        description="Score each retrieval config (hit@k, MRR) on eval/questions.yaml. No LLM is called.",
    )
    parser.add_argument("--no-generation", action="store_true", help="retrieval metrics only (required for now)")
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
    if not args.no_generation:
        parser.error("generation metrics are not implemented yet (Task 19); pass --no-generation")

    try:
        settings = settings or Settings()
    except ValidationError as exc:  # bad environment variables
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 1
    k: int = args.k or settings.top_k
    configs: list[Config] = [c for c in CONFIGS if c in args.configs]  # always in the same order, no repeats
    split_names: list[Split] = ["dev", "test"] if args.split == "both" else [args.split]

    try:
        questions = load_questions(questions_path)
        by_split: dict[Split, list[Question]] = {name: [q for q in questions if q.split == name] for name in split_names}
        for name, chosen in by_split.items():
            if not chosen:
                raise EvalSetError(f"{questions_path}: no questions in the {name} split, so there is nothing to score")

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

        results = [
            SplitResult(
                split=name,
                in_scope=sum(q.category != "out_of_scope" for q in chosen),
                out_of_scope=sum(q.category == "out_of_scope" for q in chosen),
                configs=evaluate_split(chosen, retriever, configs, k),
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
            uses_stubs=settings.embedder == "stub" or (uses_reranker and settings.reranker == "stub"),
        )
        report = render_report(info, results)
        _write_atomically(results_path, report)
    except _EXPECTED_FAILURES as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
