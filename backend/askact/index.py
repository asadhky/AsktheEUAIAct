"""The on-disk index: format, build hash, atomic writing, and loading with a freshness check.

Layout of `data/index/` (written by `python -m askact.ingest`, read by the retriever and the API):

  chunks.jsonl     one chunk per line (`Chunk` as JSON: section_id, ordinal, title, text and the
                   derived chunk_id, kind, number), UTF-8, in document order
  embeddings.npy   float32 array of shape (number of chunks, dimension), unit-length rows; row i
                   is the embedding of line i of chunks.jsonl (`Chunk.embed_text`)
  manifest.json    what the index was built from (see `Manifest`), including `build_hash`

The BM25 index is not stored: it is rebuilt from chunks.jsonl at startup, so there is nothing
extra to go stale.
"""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from askact.config import Settings
from askact.embedders import Vectors, embedder_name
from askact.ingest.chunk import chunking_params
from askact.models import Chunk

INDEX_DIR = Path("data/index")
CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
MANIFEST_FILE = "manifest.json"
# Bump when the files above change shape. An index with another version is reported as stale.
SCHEMA_VERSION = 1

_RUN_INGESTION = "Run `python -m askact.ingest` to build it."


class IndexUnavailableError(Exception):
    """The index cannot be used. `kind` says why; the message says what to do (R2.8)."""

    def __init__(self, kind: Literal["missing", "stale", "corrupt"], message: str):
        super().__init__(message)
        self.kind = kind


# --- the manifest ----------------------------------------------------------------------------

class _Frozen(BaseModel):
    # forbid: a manifest with fields this code does not know was written by something else.
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceRecord(_Frozen):
    """The exact source the index was built from (R1.5), so results are reproducible against it."""

    regulation: str
    celex: str
    oj_reference: str
    url: str
    # The day the file was downloaded. For a file already on disk (`origin` cache or manual) this is
    # the file's modification date, the best record of when it was saved.
    retrieved_at: date
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin: Literal["fetched", "cache", "manual"]


class IndexCounts(_Frozen):
    recitals: int = Field(ge=0)
    articles: int = Field(ge=0)
    annexes: int = Field(ge=0)
    chunks: int = Field(ge=1)


class Manifest(_Frozen):
    schema_version: int
    source: SourceRecord
    counts: IndexCounts
    chunking: dict[str, int]
    embedding_model: str
    dimension: int = Field(ge=1)
    build_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def compute_build_hash(source_sha256: str, chunking: dict[str, int], embedding_model: str) -> str:
    """Identifies what an index was built from: the source file, the chunking, and the embedding model.

    An index is stale exactly when this differs from the hash of the current settings (R2.6). The
    three parts are hashed as canonical JSON, not concatenated, so no choice of values can make two
    different triples produce the same input string.
    """
    canonical = json.dumps(
        {"source_sha256": source_sha256, "chunking": chunking, "embedding_model": embedding_model},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- writing ---------------------------------------------------------------------------------

def _check_consistent(manifest: Manifest, chunks: Sequence[Chunk], embeddings: Vectors) -> None:
    """Refuse to write an index that load_index would reject."""
    problems = []
    if embeddings.dtype != np.float32 or embeddings.ndim != 2:
        problems.append(f"embeddings must be a 2-D float32 array, got {embeddings.dtype} with {embeddings.ndim} dimensions")
    elif len(chunks) != embeddings.shape[0]:
        problems.append(f"{len(chunks)} chunks but {embeddings.shape[0]} embeddings")
    elif embeddings.shape[1] != manifest.dimension:
        problems.append(f"embeddings have dimension {embeddings.shape[1]} but the manifest says {manifest.dimension}")
    if len(chunks) != manifest.counts.chunks:
        problems.append(f"{len(chunks)} chunks but the manifest says {manifest.counts.chunks}")
    if manifest.schema_version != SCHEMA_VERSION:
        problems.append(f"schema_version {manifest.schema_version}, expected {SCHEMA_VERSION}")
    if problems:
        raise ValueError("inconsistent index: " + "; ".join(problems))


def write_index(directory: Path, *, manifest: Manifest, chunks: Sequence[Chunk], embeddings: Vectors) -> None:
    """Write the three files as `directory`, replacing any index already there, all or nothing.

    The files go into a temporary directory next to `directory` and are swapped in by renaming, so
    a failure or a crash at any point leaves either the previous index untouched or, if there was
    none, no index at all, never a half-written one. (It does not claim to survive power loss.)
    """
    _check_consistent(manifest, chunks, embeddings)
    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-new-", dir=parent))
    previous: Path | None = None
    try:
        (staging / CHUNKS_FILE).write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
        with open(staging / EMBEDDINGS_FILE, "wb") as handle:
            np.save(handle, embeddings, allow_pickle=False)
        (staging / MANIFEST_FILE).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.chmod(staging, 0o755)  # mkdtemp makes it private (0700), which would lock out another user

        if directory.exists():
            previous = Path(tempfile.mkdtemp(prefix=f".{directory.name}-old-", dir=parent))
            previous.rmdir()  # only wanted a unique free name
            os.rename(directory, previous)
        os.rename(staging, directory)
    except BaseException:
        if previous is not None and previous.exists() and not directory.exists():
            os.rename(previous, directory)  # put the old index back
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if previous is not None:
        shutil.rmtree(previous, ignore_errors=True)


# --- reading ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Index:
    manifest: Manifest
    chunks: tuple[Chunk, ...]
    embeddings: Vectors  # row i belongs to chunks[i]


def _corrupt(directory: Path, what: str) -> IndexUnavailableError:
    return IndexUnavailableError("corrupt", f"The index in {directory} is damaged ({what}). {_RUN_INGESTION}")


def _read_manifest(directory: Path) -> Manifest:
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise IndexUnavailableError("missing", f"There is no index at {directory} ({MANIFEST_FILE} not found). {_RUN_INGESTION}")
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, UnicodeDecodeError, OSError) as exc:
        raise _corrupt(directory, f"{MANIFEST_FILE} is unreadable: {str(exc).splitlines()[0]}") from exc


def _why_stale(manifest: Manifest, settings: Settings) -> str:
    """Say what differs, as far as the manifest lets us tell."""
    reasons = []
    wanted_model = embedder_name(settings)
    if manifest.embedding_model != wanted_model:
        reasons.append(f"the embedding model is now {wanted_model!r} (the index was built with {manifest.embedding_model!r})")
    wanted_chunking = chunking_params(settings)
    if manifest.chunking != wanted_chunking:
        reasons.append(f"the chunking is now {wanted_chunking} (the index was built with {manifest.chunking})")
    if not reasons:
        reasons.append("the source text differs from the one the index was built from")
    return "; ".join(reasons)


def load_index(settings: Settings, directory: Path = INDEX_DIR, *, source_sha256: str | None = None) -> Index:
    """Load the index, checking that it is intact and was built from the current configuration.

    The expected build hash is recomputed from the source hash recorded in the manifest and the
    current settings, so no raw file is needed (`source_sha256` overrides it: ingestion passes the
    hash of the source it has just obtained). Raises `IndexUnavailableError` if the index is
    missing, stale or damaged, and nothing else for a bad index.
    """
    manifest = _read_manifest(directory)
    if manifest.schema_version != SCHEMA_VERSION:
        raise IndexUnavailableError(
            "stale",
            f"The index in {directory} has format version {manifest.schema_version} but this code reads "
            f"version {SCHEMA_VERSION}. {_RUN_INGESTION}",
        )
    expected = compute_build_hash(
        source_sha256 or manifest.source.sha256, chunking_params(settings), embedder_name(settings)
    )
    if manifest.build_hash != expected:
        raise IndexUnavailableError(
            "stale", f"The index in {directory} is out of date: {_why_stale(manifest, settings)}. {_RUN_INGESTION}"
        )
    # The hash matches the current settings; it must also match the values recorded beside it.
    if manifest.build_hash != compute_build_hash(manifest.source.sha256, manifest.chunking, manifest.embedding_model):
        raise _corrupt(directory, "its build_hash does not match the values recorded next to it")

    chunks = _read_chunks(directory)
    embeddings = _read_embeddings(directory)
    _check_loaded(directory, manifest, chunks, embeddings)
    return Index(manifest, tuple(chunks), embeddings)


def _read_chunks(directory: Path) -> list[Chunk]:
    path = directory / CHUNKS_FILE
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise _corrupt(directory, f"{CHUNKS_FILE} is unreadable: {exc}") from exc
    chunks = []
    for number, line in enumerate(lines, start=1):
        try:
            chunks.append(Chunk.model_validate_json(line))
        except ValidationError as exc:
            raise _corrupt(directory, f"{CHUNKS_FILE} line {number} is not a valid chunk") from exc
    return chunks


def _read_embeddings(directory: Path) -> Vectors:
    try:
        return np.load(directory / EMBEDDINGS_FILE, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise _corrupt(directory, f"{EMBEDDINGS_FILE} is unreadable: {exc}") from exc


def _check_loaded(directory: Path, manifest: Manifest, chunks: list[Chunk], embeddings: Vectors) -> None:
    if embeddings.dtype != np.float32 or embeddings.ndim != 2:
        raise _corrupt(directory, f"{EMBEDDINGS_FILE} is not a 2-D float32 array")
    if not (len(chunks) == embeddings.shape[0] == manifest.counts.chunks):
        raise _corrupt(
            directory,
            f"{len(chunks)} chunks, {embeddings.shape[0]} embeddings and {manifest.counts.chunks} in the manifest do not agree",
        )
    if embeddings.shape[1] != manifest.dimension:
        raise _corrupt(directory, f"embeddings have dimension {embeddings.shape[1]}, the manifest says {manifest.dimension}")
    if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3):
        raise _corrupt(directory, f"{EMBEDDINGS_FILE} holds values that are not unit-length vectors")
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise _corrupt(directory, "chunk ids are not unique")
    sections: dict[str, set[object]] = {"recital": set(), "article": set(), "annex": set()}
    for chunk in chunks:
        sections[chunk.kind].add(chunk.section_id)
    found = (len(sections["recital"]), len(sections["article"]), len(sections["annex"]))
    recorded = (manifest.counts.recitals, manifest.counts.articles, manifest.counts.annexes)
    if found != recorded:
        raise _corrupt(directory, f"the chunks cover {found} recitals/articles/annexes but the manifest says {recorded}")
