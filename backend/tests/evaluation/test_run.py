import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from askact.config import Settings
from askact.embedders import EmbedderError, HashEmbedder
from askact.evaluation import run as run_module
from askact.evaluation.run import (
    ConfigResult,
    RunInfo,
    SplitResult,
    Spread,
    evaluate_split,
    main,
    render_report,
)
from askact.evaluation.metrics import RetrievalScores
from askact.evaluation.schema import Question, Split
from askact.index import load_index
from askact.ingest.build import build_index
from askact.ingest.fetch import CACHE_FILENAME
from askact.ingest.parse import Counts
from askact.models import Chunk, RetrievalResult, ScoredChunk, SectionId
from askact.rerankers import OverlapReranker, RerankerError
from askact.retrieval import CONFIGS, Config, Retriever

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_act.html"
NOW = datetime(2026, 10, 11, 9, 30, tzinfo=UTC)


# =====================================================================================================
# evaluate_split: the scoring loop, against a retriever whose answers are known
# =====================================================================================================

def question(qid, category, expected, text=None, split: Split = "dev") -> Question:
    return Question(id=qid, question=text or f"text of {qid}", split=split, category=category, expected_ids=expected)


class Scripted:
    """A retriever with fixed answers: for each (question text, config), the sections to return, best first."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def search(self, query, config, k):
        self.calls.append((query, config, k))
        sections, relevance, kind = self.answers[(query, config)]
        chunks = [
            ScoredChunk(chunk=Chunk(section_id=SectionId.parse(s), ordinal=i, text="t"), score=1.0 / i)
            for i, s in enumerate(sections, start=1)
        ]
        return RetrievalResult(chunks=chunks, relevance=relevance, relevance_kind=kind)


# Five questions with hand-computed ranks at k=3 under "bm25":
#   q1 expects article:5, which comes second          -> rank 2
#   q2 expects article:6, which comes first           -> rank 1
#   q3 expects article:9 or annex:III; annex:III is third -> rank 3 (any one of several is a hit)
#   q4 expects recital:1, which is not retrieved      -> no hit
#   q5 is out_of_scope: it has no hit to find and must not be scored
QUESTIONS = [
    question("q1", "direct", ["article:5"]),
    question("q2", "paraphrase", ["article:6"]),
    question("q3", "multi", ["article:9", "annex:III"]),
    question("q4", "direct", ["recital:1"]),
    question("q5", "out_of_scope", []),
]
BM25_ANSWERS = {
    ("text of q1", "bm25"): (["article:9", "article:5", "article:7"], 0.70, "cosine"),
    ("text of q2", "bm25"): (["article:6", "article:7", "article:8"], 0.85, "cosine"),
    ("text of q3", "bm25"): (["article:1", "article:2", "annex:III"], 0.60, "cosine"),
    ("text of q4", "bm25"): (["article:1", "article:2", "article:3"], 0.55, "cosine"),
    ("text of q5", "bm25"): (["article:5", "article:6", "article:9"], 0.40, "cosine"),
}


def test_hit_at_k_and_mrr_by_hand():
    [result] = evaluate_split(QUESTIONS, Scripted(BM25_ANSWERS), ["bm25"], k=3)
    scores = result.scores
    assert scores is not None
    # Four in-scope questions; hits at ranks 2, 1, 3 and none.
    assert (scores.questions, scores.hits) == (4, 3)
    assert scores.hit_at_k == pytest.approx(3 / 4)
    assert scores.mrr == pytest.approx((1 / 2 + 1 + 1 / 3 + 0) / 4)


def test_a_smaller_k_turns_a_late_hit_into_a_miss():
    [result] = evaluate_split(QUESTIONS, Scripted(BM25_ANSWERS), ["bm25"], k=2)
    scores = result.scores
    assert scores is not None
    assert (scores.questions, scores.hits) == (4, 2)  # q3's hit was at rank 3
    assert scores.mrr == pytest.approx((1 / 2 + 1 + 0 + 0) / 4)


def test_out_of_scope_questions_are_not_scored_for_hits():
    with_out_of_scope = evaluate_split(QUESTIONS, Scripted(BM25_ANSWERS), ["bm25"], k=3)[0]
    without = evaluate_split(QUESTIONS[:4], Scripted(BM25_ANSWERS), ["bm25"], k=3)[0]
    assert with_out_of_scope.scores == without.scores
    assert with_out_of_scope.scores is not None and with_out_of_scope.scores.questions == 4


def test_relevance_is_collected_separately_for_in_scope_and_out_of_scope_questions():
    [result] = evaluate_split(QUESTIONS, Scripted(BM25_ANSWERS), ["bm25"], k=3)
    assert result.in_scope_relevance == Spread(low=0.55, high=0.85, count=4)
    assert result.out_of_scope_relevance == Spread(low=0.40, high=0.40, count=1)
    assert result.relevance_scale == "cosine"


def test_every_question_is_searched_once_per_config_with_its_text_and_k():
    answers = {**BM25_ANSWERS, **{(text, "dense"): value for (text, _), value in BM25_ANSWERS.items()}}
    searcher = Scripted(answers)
    evaluate_split(QUESTIONS, searcher, ["bm25", "dense"], k=4)
    assert sorted(searcher.calls) == sorted(
        (f"text of q{n}", config, 4) for n in range(1, 6) for config in ("bm25", "dense")
    )


def test_each_config_is_scored_on_its_own_results_and_keeps_its_own_scale():
    answers = dict(BM25_ANSWERS)
    # Under the other config every in-scope question hits at rank 1, with relevance on the rerank scale.
    for n, expected in enumerate(["article:5", "article:6", "annex:III", "recital:1", "article:1"], start=1):
        answers[(f"text of q{n}", "hybrid+rerank")] = ([expected, "article:99"], -3.0 + n, "rerank")
    bm25, rerank = evaluate_split(QUESTIONS, Scripted(answers), ["bm25", "hybrid+rerank"], k=3)
    assert (bm25.config, rerank.config) == ("bm25", "hybrid+rerank")
    assert bm25.scores is not None and bm25.scores.hits == 3
    assert rerank.scores is not None
    assert (rerank.scores.hits, rerank.scores.mrr) == (4, pytest.approx(1.0))
    assert (bm25.relevance_scale, rerank.relevance_scale) == ("cosine", "rerank")
    assert rerank.in_scope_relevance == Spread(low=-2.0, high=1.0, count=4)  # q1..q4 give -2, -1, 0, 1
    assert rerank.out_of_scope_relevance == Spread(low=2.0, high=2.0, count=1)


def test_configs_are_reported_in_the_order_given():
    answers = {(text, config): value for (text, _), value in BM25_ANSWERS.items() for config in CONFIGS}
    results = evaluate_split(QUESTIONS, Scripted(answers), ["hybrid", "bm25", "dense"], k=3)
    assert [r.config for r in results] == ["hybrid", "bm25", "dense"]


def test_a_split_with_only_out_of_scope_questions_has_no_scores_but_has_relevance():
    only_out = [question("q5", "out_of_scope", [])]
    [result] = evaluate_split(only_out, Scripted(BM25_ANSWERS), ["bm25"], k=3)
    assert result.scores is None  # an average over nothing is not zero
    assert result.in_scope_relevance is None
    assert result.out_of_scope_relevance == Spread(low=0.40, high=0.40, count=1)


def test_a_split_with_only_in_scope_questions_has_no_out_of_scope_relevance():
    [result] = evaluate_split(QUESTIONS[:2], Scripted(BM25_ANSWERS), ["bm25"], k=3)
    assert result.out_of_scope_relevance is None
    assert result.scores is not None and result.scores.questions == 2


def test_a_failing_retriever_stops_the_scoring():
    class Failing:
        def search(self, query, config, k):
            raise RerankerError("model exploded")

    with pytest.raises(RerankerError):
        evaluate_split(QUESTIONS, Failing(), ["bm25"], k=3)


# =====================================================================================================
# render_report
# =====================================================================================================

def info(reranker: str | None = None, uses_stubs: bool = False) -> RunInfo:
    return RunInfo(day="2026-10-11", k=3, embedding_model="BAAI/bge-small-en-v1.5", reranker=reranker,
                   build_hash="abc123", chunks=495, uses_stubs=uses_stubs)


def config_result(config: Config = "bm25", scores: RetrievalScores | None = RetrievalScores(questions=4, hits=3, mrr=0.4583333),
                  scale="cosine", in_scope: Spread | None = Spread(0.55, 0.85, 4),
                  out_of_scope: Spread | None = Spread(0.40, 0.42, 2)) -> ConfigResult:
    return ConfigResult(config, scores, scale, in_scope, out_of_scope)


def split_result(split: Split = "dev", in_scope=4, out_of_scope=2, configs=None) -> SplitResult:
    return SplitResult(split, in_scope, out_of_scope, configs or [config_result()])


def test_the_report_states_what_produced_it():
    report = render_report(info(reranker="cross-encoder/x"), [split_result()])
    assert "- Date: 2026-10-11 (UTC)" in report
    assert "- k: 3 (chunks; hit@3 and MRR look at the top 3 chunks of each question)" in report
    assert "- Embedding model: BAAI/bge-small-en-v1.5" in report
    assert "- Reranker: cross-encoder/x" in report
    assert "- Index: 495 chunks, build hash abc123" in report
    assert "- Questions in dev: 4 in-scope, 2 out_of_scope" in report
    assert "- Generation: not run (`--no-generation`); retrieval metrics only" in report


def test_an_unused_reranker_is_reported_as_not_used():
    assert "- Reranker: not used" in render_report(info(reranker=None), [split_result()])


def test_the_scope_of_the_averages_is_stated_with_the_right_plural():
    assert "over the 4 in-scope questions only." in render_report(info(), [split_result(in_scope=4)])
    assert "over the 1 in-scope question only." in render_report(info(), [split_result(in_scope=1)])


def test_the_metrics_table_shows_hits_over_questions_and_mrr():
    report = render_report(info(), [split_result(configs=[config_result("bm25"), config_result("hybrid")])]).splitlines()
    assert "| Config | hit@3 | MRR |" in report
    assert "| bm25 | 0.750 (3/4) | 0.458 |" in report
    assert "| hybrid | 0.750 (3/4) | 0.458 |" in report


def test_the_relevance_table_shows_the_scale_and_both_ranges_with_their_sizes():
    rerank = config_result("hybrid+rerank", scale="rerank", in_scope=Spread(-2.0, 7.5, 4), out_of_scope=Spread(-11.2, -9.0, 2))
    report = render_report(info(), [split_result(configs=[config_result("bm25"), rerank])]).splitlines()
    assert "| Config | Scale | in-scope questions | out_of_scope questions |" in report
    assert "| bm25 | cosine | 0.550 to 0.850 (n=4) | 0.400 to 0.420 (n=2) |" in report
    assert "| hybrid+rerank | rerank | -2.000 to 7.500 (n=4) | -11.200 to -9.000 (n=2) |" in report


def test_an_undefined_average_is_n_a_not_zero():
    report = render_report(info(), [split_result(in_scope=0, configs=[config_result(scores=None, in_scope=None)])])
    assert "| bm25 | n/a | n/a |" in report.splitlines()
    assert "| bm25 | cosine | no questions | 0.400 to 0.420 (n=2) |" in report.splitlines()


def test_a_missing_kind_of_question_says_so():
    report = render_report(info(), [split_result(out_of_scope=0, configs=[config_result(out_of_scope=None)])])
    assert "| bm25 | cosine | 0.550 to 0.850 (n=4) | no questions |" in report.splitlines()


def test_the_test_split_is_labelled_final_and_dev_is_not():
    dev_only = render_report(info(), [split_result("dev")])
    test_only = render_report(info(), [split_result("test")])
    assert "FINAL" not in dev_only and "## dev split" in dev_only
    assert "## test split: FINAL" in test_only and "## dev split" not in test_only
    assert "must not be used to choose parameters" in test_only


def test_both_splits_get_their_own_table_in_the_order_given():
    report = render_report(info(), [split_result("dev", configs=[config_result("bm25")]),
                                    split_result("test", configs=[config_result("dense")])])
    dev, test = report.index("## dev split"), report.index("## test split: FINAL")
    assert dev < report.index("| bm25 |") < test < report.index("| dense |")
    assert "- Questions in dev:" in report and "- Questions in test:" in report


def test_stub_runs_carry_a_loud_warning_and_real_runs_do_not():
    assert "WARNING: stub models were used" in render_report(info(uses_stubs=True), [split_result()])
    assert "WARNING" not in render_report(info(uses_stubs=False), [split_result()])


def test_the_report_ends_with_exactly_one_newline():
    report = render_report(info(), [split_result("dev"), split_result("test")])
    assert report.endswith("\n") and not report.endswith("\n\n")


# =====================================================================================================
# main: the command line, end to end on the fixture index
# =====================================================================================================

DEV = [
    # The wording is taken from the passage, so the expected section is clearly the one to find.
    ("d1", "direct", "Which AI practices shall be prohibited?", ["article:5"]),
    ("d2", "paraphrase", "Union institutions acting as provider or deployer", ["recital:23"]),
    ("d3", "multi", "What is the purpose of this Regulation?", ["recital:1", "article:1"]),
    ("d4", "direct", "harmonised transparency rules", ["annex:VI"]),  # deliberately not found in the top 3
    ("d5", "out_of_scope", "What is the capital of France?", []),
]
TEST = [
    ("t1", "direct", "Which AI practices shall be prohibited?", ["article:5"]),
    ("t2", "out_of_scope", "What is the capital of France?", []),
]


def write_questions(path: Path, dev=DEV, test=TEST) -> Path:
    entries = [
        {"id": qid, "question": text, "split": split, "category": category, "expected_ids": expected}
        for split, rows in (("dev", dev), ("test", test)) for qid, category, text, expected in rows
    ]
    path.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")
    return path


class Project:
    """A scratch project: a real index built from the fixture document, and a question file."""

    def __init__(self, tmp_path: Path, settings: Settings | None = None, embedder=None):
        self.settings = settings or Settings(_env_file=None, embedder="stub", reranker="stub")
        raw = tmp_path / "data" / "raw"
        raw.mkdir(parents=True)
        (raw / CACHE_FILENAME).write_bytes(FIXTURE.read_bytes())
        self.index = tmp_path / "data" / "index"
        build_index(self.settings, raw_dir=raw, index_dir=self.index, expected_counts=Counts(3, 2, 1), embedder=embedder)
        (tmp_path / "eval").mkdir()
        self.questions = write_questions(tmp_path / "eval" / "questions.yaml")
        self.results = tmp_path / "eval" / "results.md"

    def run(self, *argv: str, settings: Settings | None = None, **kwargs) -> int:
        kwargs.setdefault("embedder", None)
        return main(["--no-generation", *argv], settings=settings or self.settings, index_dir=self.index,
                    questions_path=self.questions, results_path=self.results, now=NOW, **kwargs)


@pytest.fixture
def project(tmp_path) -> Project:
    return Project(tmp_path)


def table_rows(report: str) -> dict[str, str]:
    """The metrics table of the first split: {config: 'hit cell | mrr cell'}."""
    rows = {}
    lines = report.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Config | hit@"))
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows[cells[0]] = " | ".join(cells[1:])
    return rows


def test_the_default_run_scores_the_dev_split_with_all_four_configs(project, capsys):
    assert project.run("--k", "3") == 0
    report = capsys.readouterr().out
    # Hand-checked from the retrieval results: d1, d2 hit at rank 1 everywhere; d3 (recital:1 or article:1) at
    # rank 2 for bm25 and hybrid+rerank, rank 1 for dense and hybrid; d4 is not in the top 3. d5 is not scored.
    assert table_rows(report) == {
        "bm25": "0.750 (3/4) | 0.625",
        "dense": "0.750 (3/4) | 0.750",
        "hybrid": "0.750 (3/4) | 0.750",
        "hybrid+rerank": "0.750 (3/4) | 0.625",
    }
    assert "## dev split" in report and "FINAL" not in report
    assert "- Questions in dev: 4 in-scope, 1 out_of_scope" in report
    assert "Questions in test" not in report


def test_the_report_is_printed_and_written_to_the_results_file(project, capsys):
    assert project.run("--k", "3") == 0
    printed = capsys.readouterr().out
    assert project.results.read_text(encoding="utf-8") + "\n" == printed  # print() adds the final newline
    assert not project.results.with_name("results.md.tmp").exists()


def test_the_results_directory_is_created_if_needed(project):
    nested = project.results.parent / "deeper" / "results.md"
    assert main(["--no-generation", "--k", "3"], settings=project.settings, index_dir=project.index,
                questions_path=project.questions, results_path=nested, now=NOW) == 0
    assert nested.read_text(encoding="utf-8").startswith("# Retrieval evaluation")


def test_the_metadata_names_the_models_the_index_and_the_date(project, capsys):
    project.run("--k", "3")
    report = capsys.readouterr().out
    build_hash = load_index(project.settings, project.index).manifest.build_hash
    assert "- Date: 2026-10-11 (UTC)" in report
    assert f"build hash {build_hash}" in report
    assert "- Embedding model: stub-hash-256" in report
    assert "- Reranker: stub-overlap" in report
    assert "- k: 3 " in report


def test_k_defaults_to_top_k_and_can_be_overridden(project, capsys):
    project.run()
    assert "| Config | hit@5 | MRR |" in capsys.readouterr().out  # Settings.top_k is 5
    project.run("--k", "2")
    out = capsys.readouterr().out
    assert "| Config | hit@2 | MRR |" in out and "- k: 2 " in out


def test_a_different_top_k_setting_is_used_when_k_is_not_given(project, capsys):
    settings = project.settings.model_copy(update={"top_k": 4})
    project.run(settings=settings)
    assert "| Config | hit@4 | MRR |" in capsys.readouterr().out


def test_the_test_split_is_scored_and_labelled_final(project, capsys):
    assert project.run("--split", "test", "--k", "3") == 0
    report = capsys.readouterr().out
    assert "## test split: FINAL" in report and "## dev split" not in report
    assert "- Questions in test: 1 in-scope, 1 out_of_scope" in report
    assert table_rows(report)["bm25"] == "1.000 (1/1) | 1.000"


def test_both_splits_are_reported_separately(project, capsys):
    assert project.run("--split", "both", "--k", "3") == 0
    report = capsys.readouterr().out
    assert report.index("## dev split") < report.index("## test split: FINAL")
    assert "- Questions in dev: 4 in-scope, 1 out_of_scope" in report
    assert "- Questions in test: 1 in-scope, 1 out_of_scope" in report
    assert report.count("| bm25 | ") == 4  # a metrics row and a relevance row, for each of the two splits


def test_configs_can_be_chosen_and_come_out_in_a_fixed_order_without_repeats(project, capsys):
    assert project.run("--k", "3", "--configs", "hybrid", "bm25", "bm25") == 0
    assert list(table_rows(capsys.readouterr().out)) == ["bm25", "hybrid"]


def test_the_relevance_of_the_top_chunk_is_reported_per_config_with_its_scale(project, capsys):
    project.run("--k", "3")
    report = capsys.readouterr().out
    index = load_index(project.settings, project.index)
    retriever = Retriever(index, HashEmbedder(), OverlapReranker())
    france = retriever.search("What is the capital of France?", "dense", 3).relevance
    rerank_france = retriever.search("What is the capital of France?", "hybrid+rerank", 3).relevance
    # The out_of_scope question is the only one of its kind, so its range is a single number (n=1).
    assert f"| dense | cosine | " in report and f"| {france:.3f} to {france:.3f} (n=1) |" in report
    assert f"| hybrid+rerank | rerank | " in report and f"| {rerank_france:.3f} to {rerank_france:.3f} (n=1) |" in report


def test_the_reranker_runs_for_hybrid_rerank_even_though_the_app_flag_is_off(project, capsys):
    assert project.settings.reranker_enabled is False
    seen = []

    class Spy(OverlapReranker):
        def score(self, query, passages):
            seen.append(query)
            return super().score(query, passages)

    assert project.run("--k", "3", "--configs", "hybrid+rerank", reranker=Spy()) == 0
    assert len(seen) == 5  # all five dev questions, in-scope or not: the relevance range needs them
    assert "- Reranker: stub-overlap" in capsys.readouterr().out


def test_the_reranker_is_neither_built_nor_called_when_no_selected_config_uses_it(project, capsys, monkeypatch):
    def forbidden(settings):
        raise AssertionError("the reranker must not be built")

    monkeypatch.setattr(run_module, "get_reranker", forbidden)

    class Spy(OverlapReranker):
        def score(self, query, passages):
            raise AssertionError("the reranker must not be called")

    assert project.run("--k", "3", "--configs", "bm25", "dense", "hybrid", reranker=Spy()) == 0
    assert "- Reranker: not used" in capsys.readouterr().out


def test_stub_models_trigger_the_warning(project, capsys):
    project.run("--k", "3")
    assert "WARNING: stub models were used" in capsys.readouterr().out


def test_a_stub_embedder_alone_triggers_the_warning(project, capsys):
    # Only bm25, so the (stub) reranker is not part of the run: the embedder is what gives it away.
    assert project.run("--k", "3", "--configs", "bm25") == 0
    assert "WARNING: stub models were used" in capsys.readouterr().out


class NamedLikeTheRealModel(HashEmbedder):
    """Pretends to be the default real model so that a settings object with EMBEDDER=real accepts the index."""

    def __init__(self):
        super().__init__(dimension=384)
        self.name = Settings(_env_file=None).embedding_model


def test_no_warning_when_neither_the_embedder_nor_a_used_reranker_is_a_stub(tmp_path, capsys):
    settings = Settings(_env_file=None, embedder="real", reranker="real")
    project = Project(tmp_path, settings, embedder=NamedLikeTheRealModel())
    assert project.run("--k", "3", "--configs", "bm25", embedder=NamedLikeTheRealModel()) == 0
    assert "WARNING" not in capsys.readouterr().out


def test_a_stub_reranker_alone_triggers_the_warning_only_if_it_is_used(tmp_path, capsys):
    settings = Settings(_env_file=None, embedder="real", reranker="stub")
    project = Project(tmp_path, settings, embedder=NamedLikeTheRealModel())
    assert project.run("--k", "3", "--configs", "bm25", embedder=NamedLikeTheRealModel()) == 0
    assert "WARNING" not in capsys.readouterr().out  # the stub reranker is not part of this run
    assert project.run("--k", "3", "--configs", "hybrid+rerank", embedder=NamedLikeTheRealModel()) == 0
    assert "WARNING: stub models were used" in capsys.readouterr().out


# --- refusals: nothing is printed or written -----------------------------------------------------------

def assert_failed_cleanly(project, code, capsys, expected_in_stderr):
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""  # no table
    assert "error:" in captured.err and expected_in_stderr in captured.err
    assert not project.results.exists()  # and no file


def test_a_problem_in_the_test_split_stops_a_dev_run_too(project, capsys):
    # The whole file is checked, whichever split is being scored: a typo must not hide until the final run.
    project.questions.write_text(project.questions.read_text().replace("split: test", "split: train", 1), encoding="utf-8")
    assert_failed_cleanly(project, project.run("--k", "3"), capsys, "id 't1'")


def test_the_error_for_a_bad_entry_names_it(project, capsys):
    text = project.questions.read_text().replace("category: direct", "category: easy", 1)
    project.questions.write_text(text, encoding="utf-8")
    assert_failed_cleanly(project, project.run("--k", "3"), capsys, "entry 1 (line 1, id 'd1')")


def test_a_missing_question_file_is_an_error(project, capsys):
    project.questions.unlink()
    assert_failed_cleanly(project, project.run("--k", "3"), capsys, "not found")


def test_an_existing_results_file_is_left_alone_when_the_run_fails(project, capsys):
    project.results.write_text("OLD RESULTS", encoding="utf-8")
    project.questions.unlink()
    assert project.run("--k", "3") == 1
    assert project.results.read_text(encoding="utf-8") == "OLD RESULTS"


def test_a_split_without_questions_is_an_error(project, capsys):
    write_questions(project.questions, test=[])
    assert_failed_cleanly(project, project.run("--split", "both", "--k", "3"), capsys, "no questions in the test split")


def test_a_missing_index_says_to_run_ingestion(project, capsys):
    for path in project.index.iterdir():
        path.unlink()
    assert_failed_cleanly(project, project.run("--k", "3"), capsys, "python -m askact.ingest")


def test_a_stale_index_says_to_run_ingestion(project, capsys):
    stale = project.settings.model_copy(update={"chunk_max_chars": 1500})
    assert_failed_cleanly(project, project.run("--k", "3", settings=stale), capsys, "out of date")


def test_an_embedder_that_does_not_match_the_index_is_an_error(project, capsys):
    code = project.run("--k", "3", embedder=HashEmbedder(dimension=64))
    assert_failed_cleanly(project, code, capsys, "run `python -m askact.ingest`")


def test_a_failing_embedder_is_an_error_not_a_traceback(project, capsys, monkeypatch):
    def broken(settings):
        raise EmbedderError("model could not be loaded")

    monkeypatch.setattr(run_module, "get_embedder", broken)
    assert_failed_cleanly(project, project.run("--k", "3"), capsys, "model could not be loaded")


def test_a_failing_reranker_is_an_error_and_writes_nothing(project, capsys):
    class Broken(OverlapReranker):
        def score(self, query, passages):
            raise RerankerError("reranker exploded")

    assert_failed_cleanly(project, project.run("--k", "3", reranker=Broken()), capsys, "reranker exploded")


def test_the_results_file_is_replaced_in_one_step_so_a_failure_keeps_the_old_one(project, capsys, monkeypatch):
    project.results.write_text("OLD RESULTS", encoding="utf-8")

    def fail(source, target):
        raise OSError("disk went away")

    monkeypatch.setattr(run_module.os, "replace", fail)
    code = project.run("--k", "3")
    captured = capsys.readouterr()
    assert code == 1 and captured.out == "" and "disk went away" in captured.err
    assert project.results.read_text(encoding="utf-8") == "OLD RESULTS"  # never half-overwritten
    assert not project.results.with_name("results.md.tmp").exists()


def test_a_results_file_that_cannot_be_written_is_an_error_and_prints_no_table(project, capsys):
    project.results.mkdir()  # a directory where the file should go
    code = project.run("--k", "3")
    captured = capsys.readouterr()
    assert code == 1 and captured.out == "" and "error:" in captured.err
    assert not project.results.with_name("results.md.tmp").exists()  # the half-written file is removed


def test_invalid_environment_settings_are_an_error(project, capsys, monkeypatch):
    monkeypatch.setenv("TOP_K", "0")
    code = main(["--no-generation"], index_dir=project.index, questions_path=project.questions,
                results_path=project.results, now=NOW)
    assert code == 1 and "invalid configuration" in capsys.readouterr().err


# --- the command line itself ----------------------------------------------------------------------------

def test_generation_metrics_do_not_exist_yet_so_the_flag_is_required(project, capsys):
    with pytest.raises(SystemExit) as raised:
        main([], settings=project.settings, index_dir=project.index, questions_path=project.questions,
             results_path=project.results)
    assert raised.value.code == 2
    assert "--no-generation" in capsys.readouterr().err
    assert not project.results.exists()


@pytest.mark.parametrize("argv", [
    ["--k", "0"], ["--k", "-1"], ["--k", "three"], ["--split", "train"], ["--configs", "magic"], ["--configs"],
])
def test_bad_arguments_are_refused_before_anything_runs(project, capsys, argv):
    with pytest.raises(SystemExit) as raised:
        project.run(*argv)
    assert raised.value.code == 2
    assert not project.results.exists()


def test_python_dash_m_exits_one_with_a_message_when_the_questions_are_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "askact.evaluation.run", "--no-generation"], cwd=tmp_path,
        env={**os.environ, "EMBEDDER": "stub", "RERANKER": "stub"}, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "error:" in result.stderr and "questions.yaml" in result.stderr and "not found" in result.stderr
    assert result.stdout == ""
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_python_dash_m_runs_end_to_end_in_a_project_directory(tmp_path):
    project = Project(tmp_path)  # builds data/index and eval/questions.yaml where the defaults look
    result = subprocess.run(
        [sys.executable, "-m", "askact.evaluation.run", "--no-generation", "--k", "3"], cwd=tmp_path,
        env={**os.environ, "EMBEDDER": "stub", "RERANKER": "stub"}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "| bm25 | 0.750 (3/4) | 0.625 |" in result.stdout
    assert project.results.read_text(encoding="utf-8") + "\n" == result.stdout
