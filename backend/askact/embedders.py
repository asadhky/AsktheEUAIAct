"""Embedders: turn text into vectors, with a real local model and a deterministic stub.

The real one (`SentenceTransformerEmbedder`) runs a model on this machine: no API calls, no cost
(R2.2). The stub (`HashEmbedder`) needs no model download and no torch, so tests and CI can
exercise everything that depends on an embedder offline (R6.4). Pick one with `EMBEDDER=stub|real`.

Both return unit-length float32 vectors, so cosine similarity is a plain dot product.
"""

import hashlib
import logging
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from askact.config import Settings

log = logging.getLogger(__name__)

Vectors = NDArray[np.float32]

# The instruction the BGE v1.5 models recommend in front of a *query* when searching passages
# (from the model card). Documents never get it. The card calls it optional for v1.5 ("only a slight
# degradation" without it) but recommended for short queries against long passages, which is this app.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_BGE_V15_EN = re.compile(r"BAAI/bge-(small|base|large)-en-v1\.5")


class EmbedderError(Exception):
    """The embedding model cannot be used. The message says why and what to do."""


class Embedder(Protocol):
    name: str  # identifies the model; part of the index build hash, so a different model rebuilds
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """One unit-length row per text, shape (len(texts), dimension). No query instruction."""
        ...

    def embed_query(self, text: str) -> Vectors:
        """A single unit-length vector, shape (dimension,)."""
        ...


# --- the stub ----------------------------------------------------------------------------

class HashEmbedder:
    """A bag of words hashed into a fixed number of buckets, counted and length-normalised.

    It is not a good embedder, but it is deterministic, needs nothing downloaded, and cosine
    similarity still grows with the words two texts share, which is enough for tests to assert a
    sensible ranking. Text with no word characters has no direction, so it maps to the zero vector
    (a real model never does that; the dot product with it is simply 0).
    """

    def __init__(self, dimension: int = 256):
        if dimension < 1:
            raise ValueError(f"dimension must be at least 1, got {dimension}")
        self.dimension = dimension
        self.name = f"stub-hash-{dimension}"

    def _bucket(self, token: str) -> int:
        # hashlib, not hash(): Python randomises str hashes per process, which would make vectors
        # (and so tests, and index build hashes) differ from run to run.
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimension

    def _embed(self, text: str) -> Vectors:
        vector = np.zeros(self.dimension, dtype=np.float64)
        for token in re.findall(r"\w+", text.lower()):
            vector[self._bucket(token)] += 1.0
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else vector).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._embed(text) for text in texts])

    def embed_query(self, text: str) -> Vectors:
        return self._embed(text)  # the stub has no query instruction


# --- the real model -----------------------------------------------------------------------

def default_query_prefix(model_name: str) -> str:
    """The query instruction to use for a model, or '' when none is known.

    Only the models whose card was checked get one. Giving another model the BGE instruction would
    quietly hurt its results, so an unknown model is embedded as-is and the log says so.
    """
    if _BGE_V15_EN.fullmatch(model_name):
        return BGE_QUERY_INSTRUCTION
    log.warning("No query instruction is known for %s; queries are embedded without one.", model_name)
    return ""


class SentenceTransformerEmbedder:
    """A sentence-transformers model run locally on the CPU.

    Needs the `models` extra. `query_prefix=None` picks the default for the model (see
    `default_query_prefix`); pass '' to disable the instruction explicitly.
    """

    def __init__(self, model_name: str, *, query_prefix: str | None = None, batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderError(
                "EMBEDDER=real needs the sentence-transformers package, which is the optional `models` "
                "extra: install it with `pip install -e 'backend[models]'` (install torch's CPU wheel first: "
                "`pip install torch --index-url https://download.pytorch.org/whl/cpu`), "
                "or set EMBEDDER=stub to use the test stub."
            ) from exc
        try:
            # CPU on purpose: the model is small and the corpus is a few hundred chunks.
            self._model: Any = SentenceTransformer(model_name, device="cpu")
        except Exception as exc:  # network failure, unknown model name, corrupt cache, ...
            raise EmbedderError(
                f"could not load the embedding model {model_name!r} ({type(exc).__name__}: {exc}). "
                "Check EMBEDDING_MODEL and the network, or the model cache if running offline."
            ) from exc
        self.name = model_name
        # sentence-transformers 5.x renamed this method and warns about the old name; 3.x only has the old one.
        get_dimension = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dimension = int(get_dimension())
        self.query_prefix = default_query_prefix(model_name) if query_prefix is None else query_prefix
        self._batch_size = batch_size

    def _encode(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,  # unit length even for a model whose pipeline does not normalise
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> Vectors:
        return self._encode([self.query_prefix + text])[0]


def get_embedder(settings: Settings) -> Embedder:
    """The embedder chosen by EMBEDDER: the stub, or the real local model named by EMBEDDING_MODEL."""
    if settings.embedder == "stub":
        return HashEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)
