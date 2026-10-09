import builtins
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from askact import embedders
from askact.config import Settings
from askact.embedders import (
    BGE_QUERY_INSTRUCTION,
    EmbedderError,
    HashEmbedder,
    SentenceTransformerEmbedder,
    default_query_prefix,
    embedder_name,
    get_embedder,
)
from askact.ingest.chunk import chunk_act
from askact.ingest.parse import parse_act

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_act.html"
BACKEND = Path(__file__).resolve().parents[1]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)  # unit-length vectors, so the dot product is the cosine


# --- the stub: the properties tests rely on ---------------------------------------------------------

def test_stub_vectors_are_unit_length_float32():
    vectors = HashEmbedder().embed_documents(["The provider shall ensure", "Prohibited AI practices", "x"])
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_stub_has_a_fixed_dimension():
    embedder = HashEmbedder(dimension=64)
    assert embedder.dimension == 64
    assert embedder.embed_documents(["short", "a much longer text " * 50]).shape == (2, 64)
    assert embedder.embed_query("anything").shape == (64,)


def test_stub_default_dimension_matches_what_it_reports():
    embedder = HashEmbedder()
    assert embedder.embed_query("text").shape == (embedder.dimension,)


def test_stub_is_deterministic_within_a_process():
    embedder = HashEmbedder()
    text = "High-risk AI systems referred to in Article 6(2)"
    assert np.array_equal(embedder.embed_query(text), embedder.embed_query(text))
    assert np.array_equal(embedder.embed_documents([text, text])[0], embedder.embed_documents([text])[0])


def test_stub_gives_the_same_vectors_in_every_process():
    """Python randomises str hashes per process; the stub must not depend on them, or tests and
    index build hashes would change from run to run."""
    code = (
        "from askact.embedders import HashEmbedder; import hashlib;"
        "v = HashEmbedder().embed_query('the provider shall ensure compliance');"
        "print(hashlib.sha256(v.tobytes()).hexdigest())"
    )
    digests = set()
    for seed in ("1", "2", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=BACKEND, check=True)
        digests.add(out.stdout.strip())
    assert len(digests) == 1


def test_stub_ignores_case_and_punctuation_but_not_words():
    embedder = HashEmbedder()
    assert np.array_equal(embedder.embed_query("Prohibited AI practices!"), embedder.embed_query("prohibited ai PRACTICES"))
    assert not np.array_equal(embedder.embed_query("prohibited practices"), embedder.embed_query("permitted practices"))


def test_identical_texts_have_cosine_one():
    embedder = HashEmbedder()
    text = "biometric identification in publicly accessible spaces"
    assert cosine(embedder.embed_query(text), embedder.embed_documents([text])[0]) == pytest.approx(1.0, abs=1e-6)


def test_cosine_grows_with_the_words_two_texts_share():
    embedder = HashEmbedder(dimension=1024)  # wide enough that these few words do not collide
    query = embedder.embed_query("biometric identification systems")
    shares_all = embedder.embed_documents(["biometric identification systems are regulated"])[0]
    shares_some = embedder.embed_documents(["biometric data are regulated here"])[0]
    shares_none = embedder.embed_documents(["penalties for infringements apply"])[0]
    assert cosine(query, shares_all) > cosine(query, shares_some) > cosine(query, shares_none)
    assert cosine(query, shares_none) == pytest.approx(0.0, abs=1e-6)


def test_the_right_chunk_ranks_first_for_a_matching_query():
    texts = [
        "Article 5 — Prohibited AI practices\nThe use of real-time remote biometric identification systems",
        "Article 99 — Penalties\nMember States shall lay down the rules on penalties and other enforcement measures",
        "Recital 12\nThe notion of AI system should be clearly defined",
    ]
    embedder = HashEmbedder(dimension=1024)
    scores = embedder.embed_documents(texts) @ embedder.embed_query("what penalties do Member States lay down")
    assert int(np.argmax(scores)) == 1


def test_text_without_any_word_gives_the_zero_vector_and_cosine_zero():
    embedder = HashEmbedder()
    for text in ("", "   ", "!!! --- ???"):
        vector = embedder.embed_query(text)
        assert not vector.any()
        assert cosine(vector, embedder.embed_query("a real sentence")) == 0.0


def test_stub_has_no_query_instruction():
    embedder = HashEmbedder()
    text = "some question about penalties"
    assert np.array_equal(embedder.embed_query(text), embedder.embed_documents([text])[0])


def test_a_repeated_word_weighs_more_than_a_single_mention():
    """The stub counts words. Both texts have the same three distinct words, so the only difference
    is repetition: with presence/absence instead of counts they would score identically."""
    embedder = HashEmbedder(dimension=1024)
    query = embedder.embed_query("penalties")
    repeated = embedder.embed_documents(["penalties penalties penalties other word"])[0]
    once = embedder.embed_documents(["penalties other other other word"])[0]
    assert cosine(query, repeated) > cosine(query, once) > 0


def test_every_word_of_a_long_text_counts_not_just_the_first_few():
    embedder = HashEmbedder(dimension=1024)
    filler = "lorem ipsum dolor sit amet " * 10
    ends_with_target = embedder.embed_documents([filler + "biometric"])[0]
    without_target = embedder.embed_documents([filler + "penalties"])[0]
    query = embedder.embed_query("biometric")
    assert cosine(query, ends_with_target) > 0 == cosine(query, without_target)


def test_stub_batches_keep_their_order_and_handle_empty_input():
    embedder = HashEmbedder()
    texts = ["zulu three", "alpha one", "mike two"]  # deliberately not alphabetical
    batch = embedder.embed_documents(texts)
    for i, text in enumerate(texts):
        assert np.array_equal(batch[i], embedder.embed_query(text))
    assert embedder.embed_documents([]).shape == (0, embedder.dimension)


def test_stub_name_changes_with_its_dimension():
    """The name goes into the index build hash, so a different stub must have a different name."""
    assert HashEmbedder(64).name != HashEmbedder(128).name
    assert HashEmbedder().name.startswith("stub")


def test_stub_rejects_a_nonsensical_dimension():
    with pytest.raises(ValueError, match="at least 1"):
        HashEmbedder(0)


def test_stub_embeds_the_real_chunks_of_the_fixture():
    chunks = chunk_act(parse_act(FIXTURE.read_bytes()), 1800)
    vectors = HashEmbedder().embed_documents([c.embed_text for c in chunks])
    assert vectors.shape == (len(chunks), HashEmbedder().dimension)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


# --- the stub needs nothing downloaded and does not even import torch -----------------------------------

def test_using_the_stub_never_imports_torch_or_sentence_transformers():
    code = (
        "import sys; from askact.config import Settings; from askact.embedders import get_embedder;"
        "e = get_embedder(Settings(_env_file=None, embedder='stub')); e.embed_documents(['a b c']); e.embed_query('a');"
        "bad = [m for m in ('torch', 'sentence_transformers', 'transformers') if m in sys.modules];"
        "print('IMPORTED:' + ','.join(bad) if bad else 'clean')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=BACKEND, check=True)
    assert out.stdout.strip() == "clean", out.stdout


# --- choosing an embedder ---------------------------------------------------------------------------------

def test_stub_is_selected_by_the_setting():
    assert isinstance(get_embedder(Settings(_env_file=None, embedder="stub")), HashEmbedder)


def test_real_is_the_default_choice(monkeypatch):
    monkeypatch.delenv("EMBEDDER", raising=False)  # CI sets EMBEDDER=stub; this test is about the default
    seen = {}

    class Recorder:
        def __init__(self, model_name):
            seen["model"] = model_name

    monkeypatch.setattr(embedders, "SentenceTransformerEmbedder", Recorder)
    assert Settings(_env_file=None).embedder == "real"
    get_embedder(Settings(_env_file=None, embedding_model="some/model"))
    assert seen["model"] == "some/model"


def test_the_embedder_name_is_known_without_loading_a_model(monkeypatch):
    """Index staleness is decided from the name alone; loading the real model takes seconds."""
    def explode(*args, **kwargs):
        raise AssertionError("no model may be loaded just to learn a name")

    monkeypatch.setattr(embedders, "SentenceTransformerEmbedder", explode)
    assert embedder_name(Settings(_env_file=None, embedder="real", embedding_model="some/model")) == "some/model"
    assert embedder_name(Settings(_env_file=None, embedder="stub")) == HashEmbedder().name


def test_the_embedder_name_matches_the_name_of_the_embedder_it_describes():
    settings = Settings(_env_file=None, embedder="stub")
    assert embedder_name(settings) == get_embedder(settings).name


# --- the real embedder's failure modes, without needing the model ----------------------------------------

def test_a_missing_models_extra_gives_an_actionable_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(EmbedderError) as excinfo:
        SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    message = str(excinfo.value)
    assert "models" in message and "pip install" in message and "EMBEDDER=stub" in message


def install_fake_sentence_transformers(monkeypatch, model_class) -> None:
    """Put a stand-in `sentence_transformers` module in place, so these tests neither need the real
    package (CI does not install it) nor spend seconds importing torch."""
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = model_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_a_model_that_cannot_be_loaded_gives_an_actionable_error(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no such model on the hub")

    install_fake_sentence_transformers(monkeypatch, boom)
    with pytest.raises(EmbedderError, match=r"could not load the embedding model 'not/a-model'.*EMBEDDING_MODEL"):
        SentenceTransformerEmbedder("not/a-model")


# --- the query instruction -----------------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"])
def test_bge_v15_models_get_the_documented_instruction(model):
    assert default_query_prefix(model) == BGE_QUERY_INSTRUCTION == "Represent this sentence for searching relevant passages: "


@pytest.mark.parametrize("model", [
    "BAAI/bge-small-en",                     # v1, whose card has the same text but was not verified here
    "BAAI/bge-small-zh-v1.5",                # Chinese: a different instruction
    "intfloat/e5-small-v2",                  # uses "query: " instead
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5-extra",
])
def test_other_models_do_not_silently_get_a_bge_instruction(model, caplog):
    with caplog.at_level(logging.WARNING, logger="askact.embedders"):
        assert default_query_prefix(model) == ""
    assert model in caplog.text and "without one" in caplog.text


class FakeModel:
    """Stands in for a SentenceTransformer so the wrapper's own logic can be tested offline."""

    def __init__(self, *args, **kwargs):
        self.encoded: list[list[str]] = []

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True and kwargs["convert_to_numpy"] is True
        self.encoded.append(list(texts))
        return np.array([[1.0, 0.0, 0.0, 0.0]] * len(texts), dtype=np.float64)


@pytest.fixture
def wrapped(monkeypatch):
    install_fake_sentence_transformers(monkeypatch, FakeModel)
    return SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")


def test_the_instruction_is_added_to_queries_only(wrapped):
    wrapped.embed_query("what is prohibited?")
    wrapped.embed_documents(["Article 5 — Prohibited AI practices\ntext", "second"])
    query_call, document_call = wrapped._model.encoded
    assert query_call == [BGE_QUERY_INSTRUCTION + "what is prohibited?"]
    assert document_call == ["Article 5 — Prohibited AI practices\ntext", "second"]  # untouched


def test_the_instruction_can_be_switched_off_explicitly(monkeypatch):
    install_fake_sentence_transformers(monkeypatch, FakeModel)
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", query_prefix="")
    embedder.embed_query("plain question")
    assert embedder._model.encoded == [["plain question"]]


def test_the_wrapper_returns_float32_in_the_documented_shapes(wrapped):
    documents = wrapped.embed_documents(["a", "b", "c"])
    query = wrapped.embed_query("q")
    assert documents.shape == (3, 4) and documents.dtype == np.float32
    assert query.shape == (4,) and query.dtype == np.float32
    assert wrapped.dimension == 4 and wrapped.name == "BAAI/bge-small-en-v1.5"


def test_an_empty_batch_does_not_call_the_model(wrapped):
    assert wrapped.embed_documents([]).shape == (0, 4)
    assert wrapped._model.encoded == []


def test_the_dimension_works_with_the_old_and_the_new_method_name(monkeypatch):
    class NewName(FakeModel):
        def get_embedding_dimension(self):
            return 7

        def get_sentence_embedding_dimension(self):
            raise AssertionError("the deprecated name must not be used when the new one exists")

    install_fake_sentence_transformers(monkeypatch, NewName)
    assert SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5").dimension == 7
    install_fake_sentence_transformers(monkeypatch, FakeModel)  # old name only
    assert SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5").dimension == 4


# --- the real model (slow, local only) ----------------------------------------------------------------------------

@pytest.mark.slow
def test_the_real_model_embeds_the_fixture_with_the_right_shape_and_unit_length():
    pytest.importorskip("sentence_transformers", reason="the `models` extra is not installed")
    embedder = get_embedder(Settings(_env_file=None, embedder="real"))
    chunks = chunk_act(parse_act(FIXTURE.read_bytes()), 1800)

    vectors = embedder.embed_documents([c.embed_text for c in chunks])

    assert embedder.dimension == 384  # BAAI/bge-small-en-v1.5
    assert vectors.shape == (len(chunks), embedder.dimension)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert embedder.embed_query("Which AI practices are prohibited?").shape == (embedder.dimension,)


@pytest.mark.slow
def test_the_real_model_applies_the_instruction_to_queries_and_not_to_documents():
    pytest.importorskip("sentence_transformers", reason="the `models` extra is not installed")
    with_instruction = get_embedder(Settings(_env_file=None, embedder="real"))
    assert isinstance(with_instruction, SentenceTransformerEmbedder)
    without_instruction = SentenceTransformerEmbedder(with_instruction.name, query_prefix="")
    text = "Which AI practices are prohibited?"

    # Queries differ with and without the instruction; the instruction is really being applied...
    query_a, query_b = with_instruction.embed_query(text), without_instruction.embed_query(text)
    assert not np.allclose(query_a, query_b, atol=1e-4)
    assert cosine(query_a, query_b) > 0.9  # ...but it only nudges the vector
    # ...and equals encoding the prefixed string by hand, while documents are encoded untouched.
    by_hand = without_instruction.embed_documents([BGE_QUERY_INSTRUCTION + text])[0]
    assert np.allclose(query_a, by_hand, atol=1e-5)
    assert np.allclose(with_instruction.embed_documents([text]), without_instruction.embed_documents([text]), atol=1e-6)


@pytest.mark.slow
def test_the_real_model_ranks_the_matching_fixture_chunk_first():
    pytest.importorskip("sentence_transformers", reason="the `models` extra is not installed")
    embedder = get_embedder(Settings(_env_file=None, embedder="real"))
    chunks = chunk_act(parse_act(FIXTURE.read_bytes()), 1800)
    scores = embedder.embed_documents([c.embed_text for c in chunks]) @ embedder.embed_query(
        "Which AI practices are prohibited?"
    )
    assert chunks[int(np.argmax(scores))].section_id.kind == "article"
    assert str(chunks[int(np.argmax(scores))].section_id) == "article:5"
