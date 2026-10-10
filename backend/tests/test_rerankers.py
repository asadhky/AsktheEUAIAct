import builtins
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from askact import rerankers
from askact.config import Settings
from askact.rerankers import (
    CrossEncoderReranker,
    OverlapReranker,
    RerankerError,
    checked_scores,
    get_reranker,
)

BACKEND = Path(__file__).resolve().parents[1]


# --- the stub ---------------------------------------------------------------------------------------

def test_identical_text_scores_one():
    assert OverlapReranker().score("penalties for infringements", ["penalties for infringements"]) == [1.0]


def test_text_sharing_no_word_scores_zero():
    assert OverlapReranker().score("penalties", ["biometric identification systems"]) == [0.0]


def test_the_score_is_shared_words_over_combined_distinct_words():
    # query {a, b}; passage {b, c, d}; shared {b} = 1; combined {a, b, c, d} = 4
    assert OverlapReranker().score("a b", ["b c d"]) == [pytest.approx(0.25)]


def test_the_score_is_in_the_unit_interval_and_never_nan():
    scores = OverlapReranker().score("the provider shall ensure", ["", "x", "the provider", "the provider shall ensure compliance"])
    assert all(0.0 <= s <= 1.0 and math.isfinite(s) for s in scores)


def test_the_stub_ignores_case_and_punctuation_like_the_rest_of_the_system():
    assert OverlapReranker().score("PENALTIES!", ["penalties."]) == [1.0]


def test_a_repeated_word_does_not_count_twice():
    """Jaccard is over distinct words, so repeating one in a passage cannot inflate the score."""
    assert OverlapReranker().score("fine", ["fine fine fine fine"]) == OverlapReranker().score("fine", ["fine"])


def test_the_stub_prefers_the_passage_that_shares_more_of_the_question():
    scores = OverlapReranker().score("remote biometric identification", ["remote biometric identification systems", "biometric data"])
    assert scores[0] > scores[1] > 0


def test_the_stub_returns_one_score_per_passage_in_order():
    passages = ["alpha", "beta alpha", "gamma"]
    scores = OverlapReranker().score("alpha", passages)
    assert len(scores) == 3 and scores[1] < scores[0] and scores[2] == 0.0


def test_the_stub_handles_no_passages_and_an_empty_question():
    assert OverlapReranker().score("anything", []) == []
    assert OverlapReranker().score("", ["some passage"]) == [0.0]
    assert OverlapReranker().score("", [""]) == [0.0]


def test_the_stub_is_deterministic_and_named():
    assert OverlapReranker().score("a b", ["a c"]) == OverlapReranker().score("a b", ["a c"])
    assert OverlapReranker().name == "stub-overlap"


def test_the_stub_gives_the_same_scores_in_every_process():
    code = (
        "from askact.rerankers import OverlapReranker;"
        "print(OverlapReranker().score('the provider shall ensure compliance', ['the provider and the deployer', 'penalties']))"
    )
    outputs = set()
    for seed in ("1", "2", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.add(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=BACKEND, check=True).stdout)
    assert len(outputs) == 1


# --- choosing a reranker ---------------------------------------------------------------------------------

def test_the_stub_is_selected_by_the_setting():
    assert isinstance(get_reranker(Settings(_env_file=None, reranker="stub")), OverlapReranker)


def test_the_real_reranker_is_the_default_choice(monkeypatch):
    monkeypatch.delenv("RERANKER", raising=False)  # CI sets RERANKER=stub; this test is about the default
    seen = {}

    class Recorder:
        def __init__(self, model_name):
            seen["model"] = model_name

    monkeypatch.setattr(rerankers, "CrossEncoderReranker", Recorder)
    assert Settings(_env_file=None).reranker == "real"
    get_reranker(Settings(_env_file=None, reranker_model="some/cross-encoder"))
    assert seen["model"] == "some/cross-encoder"


def test_using_the_stub_never_imports_torch_or_sentence_transformers():
    code = (
        "import sys; from askact.config import Settings; from askact.rerankers import get_reranker;"
        "r = get_reranker(Settings(_env_file=None, reranker='stub')); r.score('a b', ['a c']);"
        "bad = [m for m in ('torch', 'sentence_transformers', 'transformers') if m in sys.modules];"
        "print('IMPORTED:' + ','.join(bad) if bad else 'clean')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=BACKEND, check=True)
    assert out.stdout.strip() == "clean", out.stdout


# --- the real reranker's failure modes and wrapper logic, without needing the model -----------------------------

def install_fake_sentence_transformers(monkeypatch, cross_encoder_class) -> None:
    """A stand-in `sentence_transformers`, so these tests neither need the real package (CI has none) nor
    spend seconds importing torch."""
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = cross_encoder_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_a_missing_models_extra_gives_an_actionable_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RerankerError) as excinfo:
        CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    message = str(excinfo.value)
    assert "models" in message and "pip install" in message and "RERANKER=stub" in message


def test_a_model_that_cannot_be_loaded_gives_an_actionable_error(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no such model on the hub")

    install_fake_sentence_transformers(monkeypatch, boom)
    with pytest.raises(RerankerError, match=r"could not load the reranker model 'not/a-model'.*RERANKER_MODEL"):
        CrossEncoderReranker("not/a-model")


class FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, **kwargs):
        assert kwargs.get("show_progress_bar") is False
        self.calls.append(list(pairs))
        return [float(len(passage)) - 10.0 for _, passage in pairs]  # arbitrary, signed, like logits


@pytest.fixture
def wrapped(monkeypatch):
    install_fake_sentence_transformers(monkeypatch, FakeCrossEncoder)
    return CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_each_passage_is_paired_with_the_question(wrapped):
    wrapped.score("what is x?", ["first passage", "second"])
    assert wrapped._model.calls == [[("what is x?", "first passage"), ("what is x?", "second")]]


def test_scores_come_back_as_plain_floats_in_order(wrapped):
    scores = wrapped.score("q", ["a", "abcdef", ""])
    assert scores == [-9.0, -4.0, -10.0] and all(type(s) is float for s in scores)


def test_negative_scores_are_kept_as_they_are(wrapped):
    """Cross-encoder scores are logits: a poor match is negative and must not be clipped or rescaled."""
    assert wrapped.score("q", ["x"]) == [-9.0]


def test_no_passages_does_not_call_the_model(wrapped):
    assert wrapped.score("q", []) == []
    assert wrapped._model.calls == []


def test_the_wrapper_names_itself_after_the_model(wrapped):
    assert wrapped.name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --- verifying what a reranker returns -------------------------------------------------------------------------------

class Scripted:
    name = "scripted"

    def __init__(self, scores):
        self._scores = scores

    def score(self, query, passages):
        return self._scores


def test_good_scores_pass_through_unchanged():
    assert checked_scores(Scripted([1.5, -2.0]), "q", ["a", "b"]) == [1.5, -2.0]


@pytest.mark.parametrize("scores", [[1.0], [1.0, 2.0, 3.0], []])
def test_the_wrong_number_of_scores_is_an_error_not_a_silent_misalignment(scores):
    with pytest.raises(RerankerError, match=r"scripted returned \d+ scores for 2 passages"):
        checked_scores(Scripted(scores), "q", ["a", "b"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_score_that_is_not_finite_is_an_error(bad):
    """NaN is the dangerous one: `nan < threshold` is False, so it would pass the not-covered gate."""
    with pytest.raises(RerankerError, match="not a finite number"):
        checked_scores(Scripted([0.5, bad]), "q", ["a", "b"])


# --- the real model (slow, local only) ---------------------------------------------------------------------------------

@pytest.mark.slow
def test_the_real_cross_encoder_scores_a_matching_passage_far_above_an_unrelated_one():
    pytest.importorskip("sentence_transformers", reason="the `models` extra is not installed")
    reranker = get_reranker(Settings(_env_file=None, reranker="real"))
    scores = reranker.score(
        "Which AI practices are prohibited?",
        [
            "Article 5 — Prohibited AI practices\n1. The following AI practices shall be prohibited: the placing on the market of an AI system that deploys subliminal techniques",
            "Article 99 — Penalties\nMember States shall lay down the rules on penalties and other enforcement measures",
            "How to bake sourdough bread with a wild yeast starter.",
        ],
    )
    # The on-topic passage wins clearly; the penalties article (also about AI law) is not the answer; the
    # unrelated one is far below. Only the on-topic-vs-unrelated gap is asserted as a sign change, since the
    # exact scores are the model's business.
    assert scores[0] > scores[1] and scores[0] > scores[2]
    assert scores[0] > 0 > scores[2]


@pytest.mark.slow
def test_the_real_cross_encoder_scores_are_raw_logits_not_probabilities():
    """The threshold for reranked results is tuned on this scale. If a library update started returning
    probabilities (a sigmoid), every score would land in [0, 1] and the tuned threshold would silently
    mean something else. Verified identical on sentence-transformers 3.4.1, 4.1.0, 5.1.0 and 6.1.0."""
    pytest.importorskip("sentence_transformers", reason="the `models` extra is not installed")
    reranker = get_reranker(Settings(_env_file=None, reranker="real"))
    scores = reranker.score("Which AI practices are prohibited?", ["Article 5 — Prohibited AI practices", "sourdough bread recipe"])
    # A sigmoid would keep both inside (0, 1). Logits have a negative value for the poor match and a
    # value above 1 for the good one.
    assert min(scores) < 0 and max(scores) > 1, f"expected unbounded logits, got {scores}"
    assert all(math.isfinite(s) for s in scores)


@pytest.mark.slow
def test_the_real_cross_encoder_handles_a_passage_longer_than_its_token_limit():
    pytest.importorskip("sentence_transformers", reason="the `models` extra is not installed")
    reranker = get_reranker(Settings(_env_file=None, reranker="real"))
    scores = reranker.score("Which AI practices are prohibited?", ["word " * 3000, "prohibited AI practices"])
    assert len(scores) == 2 and all(math.isfinite(s) for s in scores)
