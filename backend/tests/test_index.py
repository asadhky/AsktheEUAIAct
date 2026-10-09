import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from askact import index as index_module
from askact.config import Settings
from askact.embedders import HashEmbedder, embedder_name
from askact.index import (
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    MANIFEST_FILE,
    SCHEMA_VERSION,
    IndexCounts,
    IndexUnavailableError,
    Manifest,
    SourceRecord,
    compute_build_hash,
    load_index,
    write_index,
)
from askact.ingest.chunk import chunking_params
from askact.models import Chunk, SectionId

SHA = "a" * 64
OTHER_SHA = "b" * 64


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, embedder="stub", **overrides)


def make_chunks() -> list[Chunk]:
    def chunk(sid: str, ordinal: int, title: str | None, text: str) -> Chunk:
        return Chunk(section_id=SectionId.parse(sid), ordinal=ordinal, title=title, text=text)

    return [
        chunk("recital:1", 1, None, "The purpose of this Regulation is to improve the functioning of the internal market."),
        chunk("article:5", 1, "Prohibited AI practices", "1. The following AI practices shall be prohibited:"),
        chunk("article:5", 2, "Prohibited AI practices", "(a) the placing on the market of AI systems that deploy subliminal techniques;"),
        chunk("annex:III", 1, "High-risk AI systems", "1. Biometrics, in so far as their use is permitted."),
    ]


def make_manifest(chunks: list[Chunk], s: Settings | None = None, source_sha: str = SHA, embedder: HashEmbedder | None = None) -> Manifest:
    s = s or settings()
    embedder = embedder or HashEmbedder()
    kinds = {k: len({c.section_id for c in chunks if c.kind == k}) for k in ("recital", "article", "annex")}
    return Manifest(
        schema_version=SCHEMA_VERSION,
        source=SourceRecord(
            regulation="Regulation (EU) 2024/1689", celex="32024R1689", oj_reference="OJ L, 2024/1689, 12.7.2024",
            url="https://example.org/act", retrieved_at=date(2026, 10, 9), sha256=source_sha, origin="fetched",
        ),
        counts=IndexCounts(recitals=kinds["recital"], articles=kinds["article"], annexes=kinds["annex"], chunks=len(chunks)),
        chunking=chunking_params(s),
        embedding_model=embedder.name,
        dimension=embedder.dimension,
        build_hash=compute_build_hash(source_sha, chunking_params(s), embedder.name),
    )


@pytest.fixture
def written(tmp_path):
    """An index on disk, plus what went into it."""
    chunks = make_chunks()
    vectors = HashEmbedder().embed_documents([c.embed_text for c in chunks])
    directory = tmp_path / "index"
    write_index(directory, manifest=make_manifest(chunks), chunks=chunks, embeddings=vectors)
    return directory, chunks, vectors


def listing(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


# --- round trip (R1.10) ----------------------------------------------------------------------

def test_the_index_is_three_files(written):
    directory, _, _ = written
    assert listing(directory) == [CHUNKS_FILE, EMBEDDINGS_FILE, MANIFEST_FILE]


def test_round_trip_gives_back_the_same_chunks_embeddings_and_manifest(written):
    directory, chunks, vectors = written
    loaded = load_index(settings(), directory)
    assert list(loaded.chunks) == chunks
    assert np.array_equal(loaded.embeddings, vectors)
    assert loaded.embeddings.dtype == np.float32
    assert loaded.manifest == make_manifest(chunks)


def test_chunks_file_has_one_json_line_per_chunk_in_order(written):
    directory, chunks, _ = written
    lines = (directory / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(chunks)
    second = json.loads(lines[1])
    assert second["chunk_id"] == "article:5#1" and second["section_id"] == "article:5"
    assert second["title"] == "Prohibited AI practices" and second["kind"] == "article" and second["number"] == "5"


def test_non_ascii_text_survives_the_round_trip(tmp_path):
    chunks = [Chunk(section_id=SectionId.parse("article:1"), ordinal=1, title="Subject matter", text="‘AI system’ means … — 10^25 ≥ ½")]
    vectors = HashEmbedder().embed_documents([c.embed_text for c in chunks])
    write_index(tmp_path / "i", manifest=make_manifest(chunks), chunks=chunks, embeddings=vectors)
    assert load_index(settings(), tmp_path / "i").chunks[0].text == "‘AI system’ means … — 10^25 ≥ ½"


def test_the_manifest_records_the_pinned_source(written):
    directory, _, _ = written
    recorded = json.loads((directory / MANIFEST_FILE).read_text())
    assert recorded["source"] == {
        "regulation": "Regulation (EU) 2024/1689", "celex": "32024R1689", "oj_reference": "OJ L, 2024/1689, 12.7.2024",
        "url": "https://example.org/act", "retrieved_at": "2026-10-09", "sha256": SHA, "origin": "fetched",
    }
    assert recorded["counts"] == {"recitals": 1, "articles": 1, "annexes": 1, "chunks": 4}
    assert recorded["chunking"] == {"max_chars": 1800, "chunker_version": 1}
    assert recorded["embedding_model"] == "stub-hash-256" and recorded["dimension"] == 256
    assert recorded["schema_version"] == 1 and len(recorded["build_hash"]) == 64


# --- the build hash (R2.6, R2.7) ---------------------------------------------------------------

BASE = compute_build_hash(SHA, {"max_chars": 1800, "chunker_version": 1}, "model-a")


def test_the_hash_is_a_stable_sha256():
    assert BASE == compute_build_hash(SHA, {"max_chars": 1800, "chunker_version": 1}, "model-a")
    assert len(BASE) == 64 and set(BASE) <= set("0123456789abcdef")


@pytest.mark.parametrize("changed, args", [
    ("the source file", (OTHER_SHA, {"max_chars": 1800, "chunker_version": 1}, "model-a")),
    ("max_chars", (SHA, {"max_chars": 1799, "chunker_version": 1}, "model-a")),
    ("the chunker version", (SHA, {"max_chars": 1800, "chunker_version": 2}, "model-a")),
    ("an extra chunking parameter", (SHA, {"max_chars": 1800, "chunker_version": 1, "overlap": 0}, "model-a")),
    ("the embedding model name", (SHA, {"max_chars": 1800, "chunker_version": 1}, "model-b")),
])
def test_the_hash_changes_with_each_of_its_inputs(changed, args):
    assert compute_build_hash(*args) != BASE, f"changing {changed} must change the hash"


def test_the_hash_does_not_depend_on_the_order_of_chunking_parameters():
    assert compute_build_hash(SHA, {"a": 1, "b": 2}, "m") == compute_build_hash(SHA, {"b": 2, "a": 1}, "m")


def test_values_cannot_be_shifted_between_inputs_to_collide():
    """Plain concatenation would hash 'ab'+'c' and 'a'+'bc' alike; the canonical JSON does not."""
    assert compute_build_hash("ab", {}, "c") != compute_build_hash("a", {}, "bc")


def test_the_hash_is_recorded_next_to_the_index(written):
    directory, chunks, _ = written
    assert load_index(settings(), directory).manifest.build_hash == make_manifest(chunks).build_hash


def test_changing_the_chunk_size_makes_the_index_stale(written):
    directory, _, _ = written
    with pytest.raises(IndexUnavailableError) as excinfo:
        load_index(settings(chunk_max_chars=1500), directory)
    assert excinfo.value.kind == "stale"
    assert "chunking" in str(excinfo.value) and "1500" in str(excinfo.value) and "python -m askact.ingest" in str(excinfo.value)


def test_changing_the_embedding_model_makes_the_index_stale(written):
    directory, _, _ = written
    with pytest.raises(IndexUnavailableError, match=r"embedding model is now 'BAAI/bge-small-en-v1\.5'") as excinfo:
        load_index(Settings(_env_file=None, embedder="real"), directory)
    assert excinfo.value.kind == "stale"


def test_a_different_source_makes_the_index_stale_at_ingest_time(written):
    directory, _, _ = written
    with pytest.raises(IndexUnavailableError, match="source text differs") as excinfo:
        load_index(settings(), directory, source_sha256=OTHER_SHA)
    assert excinfo.value.kind == "stale"


def test_the_same_source_given_explicitly_is_not_stale(written):
    directory, _, _ = written
    assert load_index(settings(), directory, source_sha256=SHA).manifest.source.sha256 == SHA


def test_loading_needs_no_raw_file_and_no_model(written, monkeypatch):
    """The API checks freshness at startup from the manifest alone."""
    directory, _, _ = written
    monkeypatch.chdir(directory)  # no data/raw anywhere on the path
    assert load_index(settings(), directory).manifest.embedding_model == embedder_name(settings())


# --- missing and damaged indexes (R2.8) ------------------------------------------------------------

def test_a_missing_index_says_to_run_ingestion(tmp_path):
    with pytest.raises(IndexUnavailableError) as excinfo:
        load_index(settings(), tmp_path / "nothing-here")
    assert excinfo.value.kind == "missing"
    assert "no index" in str(excinfo.value).lower() and "python -m askact.ingest" in str(excinfo.value)


def test_an_empty_directory_counts_as_missing(tmp_path):
    (tmp_path / "index").mkdir()
    with pytest.raises(IndexUnavailableError) as excinfo:
        load_index(settings(), tmp_path / "index")
    assert excinfo.value.kind == "missing"


@pytest.mark.parametrize("name", [CHUNKS_FILE, EMBEDDINGS_FILE])
def test_a_missing_data_file_is_reported_as_damage_not_a_crash(written, name):
    directory, _, _ = written
    (directory / name).unlink()
    with pytest.raises(IndexUnavailableError) as excinfo:
        load_index(settings(), directory)
    assert excinfo.value.kind == "corrupt" and "python -m askact.ingest" in str(excinfo.value)


def corrupt(directory: Path, name: str, mutate) -> None:
    path = directory / name
    path.write_bytes(mutate(path.read_bytes()))


@pytest.mark.parametrize("name, mutate, expected", [
    (MANIFEST_FILE, lambda b: b"{not json", "manifest.json is unreadable"),
    (MANIFEST_FILE, lambda b: b"{}", "manifest.json is unreadable"),
    (MANIFEST_FILE, lambda b: b.replace(b'"schema_version": 1,', b'"schema_version": 1, "surprise": true,'), "manifest.json is unreadable"),
    (CHUNKS_FILE, lambda b: b + b"not a chunk\n", "line 5 is not a valid chunk"),
    (CHUNKS_FILE, lambda b: b"\n".join(b.splitlines()[:-1]) + b"\n", "do not agree"),
    (EMBEDDINGS_FILE, lambda b: b[:100], "embeddings.npy is unreadable"),
    (EMBEDDINGS_FILE, lambda b: b"garbage", "embeddings.npy is unreadable"),
])
def test_damaged_files_give_a_clear_error_naming_the_problem(written, name, mutate, expected):
    directory, _, _ = written
    corrupt(directory, name, mutate)
    with pytest.raises(IndexUnavailableError, match=expected) as excinfo:
        load_index(settings(), directory)
    assert excinfo.value.kind == "corrupt"
    assert "python -m askact.ingest" in str(excinfo.value)


def test_embeddings_that_are_not_unit_vectors_are_rejected(written):
    directory, _, vectors = written
    with open(directory / EMBEDDINGS_FILE, "wb") as handle:
        np.save(handle, vectors * 3)
    with pytest.raises(IndexUnavailableError, match="not unit-length"):
        load_index(settings(), directory)


def test_embeddings_with_nan_are_rejected(written):
    directory, _, vectors = written
    bad = vectors.copy()
    bad[0, 0] = np.nan
    with open(directory / EMBEDDINGS_FILE, "wb") as handle:
        np.save(handle, bad)
    with pytest.raises(IndexUnavailableError, match="not unit-length"):
        load_index(settings(), directory)


def test_embeddings_of_the_wrong_dtype_or_shape_are_rejected(written):
    directory, _, vectors = written
    for bad in (vectors.astype(np.float64), vectors[:, 0], vectors[:, :10]):
        with open(directory / EMBEDDINGS_FILE, "wb") as handle:
            np.save(handle, bad)
        with pytest.raises(IndexUnavailableError) as excinfo:
            load_index(settings(), directory)
        assert excinfo.value.kind == "corrupt"


def test_embeddings_of_the_wrong_width_are_rejected_even_if_they_are_unit_vectors(written):
    """Slicing columns off makes vectors non-unit, so another check would catch it first; these are
    properly normalised, so only the dimension check can notice that the width is not the manifest's."""
    directory, _, vectors = written
    narrow = vectors[:, :128] / np.linalg.norm(vectors[:, :128], axis=1, keepdims=True)
    narrow = np.nan_to_num(narrow).astype(np.float32)
    assert np.allclose(np.linalg.norm(narrow, axis=1), 1.0, atol=1e-3)  # really unit-length
    with open(directory / EMBEDDINGS_FILE, "wb") as handle:
        np.save(handle, narrow)
    with pytest.raises(IndexUnavailableError, match="dimension 128, the manifest says 256") as excinfo:
        load_index(settings(), directory)
    assert excinfo.value.kind == "corrupt"


def test_pickled_arrays_are_never_loaded(written):
    """np.load with pickle enabled would run code from a file."""
    directory, _, _ = written
    with open(directory / EMBEDDINGS_FILE, "wb") as handle:
        np.save(handle, np.array([{"x": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(IndexUnavailableError, match="unreadable"):
        load_index(settings(), directory)


def test_duplicate_chunk_ids_are_rejected(written):
    directory, _, _ = written
    lines = (directory / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
    lines[2] = lines[1]
    (directory / CHUNKS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IndexUnavailableError, match="not unique"):
        load_index(settings(), directory)


def test_a_manifest_whose_counts_do_not_match_the_chunks_is_rejected(written):
    directory, _, _ = written
    recorded = json.loads((directory / MANIFEST_FILE).read_text())
    recorded["counts"]["articles"] = 99
    (directory / MANIFEST_FILE).write_text(json.dumps(recorded))
    with pytest.raises(IndexUnavailableError, match="manifest says"):
        load_index(settings(), directory)


def test_a_manifest_edited_to_claim_another_source_is_not_trusted(written):
    directory, _, _ = written
    recorded = json.loads((directory / MANIFEST_FILE).read_text())
    recorded["source"]["sha256"] = OTHER_SHA  # claims another source but keeps the old hash
    (directory / MANIFEST_FILE).write_text(json.dumps(recorded))
    with pytest.raises(IndexUnavailableError) as excinfo:
        load_index(settings(), directory)
    assert excinfo.value.kind in {"stale", "corrupt"}


def test_a_hash_that_does_not_match_the_fields_recorded_beside_it_is_reported_as_damage(written):
    """The hash agrees with the current settings, but the manifest's own record of what it was built
    with says something else: the file was edited or damaged, and it cannot be trusted."""
    directory, _, _ = written
    recorded = json.loads((directory / MANIFEST_FILE).read_text())
    recorded["chunking"] = {"max_chars": 500, "chunker_version": 1}  # claims a different chunking
    (directory / MANIFEST_FILE).write_text(json.dumps(recorded))
    with pytest.raises(IndexUnavailableError, match="build_hash does not match") as excinfo:
        load_index(settings(), directory)  # the current settings still hash to the stored build_hash
    assert excinfo.value.kind == "corrupt"


def test_an_index_with_another_format_version_is_stale(written):
    directory, _, _ = written
    recorded = json.loads((directory / MANIFEST_FILE).read_text())
    recorded["schema_version"] = SCHEMA_VERSION + 1
    (directory / MANIFEST_FILE).write_text(json.dumps(recorded))
    with pytest.raises(IndexUnavailableError, match="format version") as excinfo:
        load_index(settings(), directory)
    assert excinfo.value.kind == "stale"


# --- writing is all or nothing (R1.13) ---------------------------------------------------------------

def test_inconsistent_inputs_are_refused_before_anything_is_written(tmp_path):
    chunks = make_chunks()
    vectors = HashEmbedder().embed_documents([c.embed_text for c in chunks])
    directory = tmp_path / "index"
    for bad_vectors in (vectors[:-1], vectors.astype(np.float64), vectors[:, :5]):
        with pytest.raises(ValueError, match="inconsistent index"):
            # float64 on purpose: the type checker rejects it too, and the runtime check must as well
            write_index(directory, manifest=make_manifest(chunks), chunks=chunks, embeddings=bad_vectors)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="manifest says"):
        write_index(directory, manifest=make_manifest(chunks[:-1]), chunks=chunks, embeddings=vectors)
    assert not directory.exists() and listing(tmp_path) == []


def test_a_failure_while_writing_leaves_no_index_and_no_temporary_files(tmp_path, monkeypatch):
    chunks = make_chunks()
    vectors = HashEmbedder().embed_documents([c.embed_text for c in chunks])

    def disk_full(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(index_module.np, "save", disk_full)
    with pytest.raises(OSError, match="No space left"):
        write_index(tmp_path / "index", manifest=make_manifest(chunks), chunks=chunks, embeddings=vectors)
    assert listing(tmp_path) == []  # not even a half-written directory or a hidden temp dir


def test_a_failure_while_writing_keeps_the_previous_index_intact(written, monkeypatch):
    directory, chunks, vectors = written
    before = {name: (directory / name).read_bytes() for name in listing(directory)}
    monkeypatch.setattr(index_module.np, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        write_index(directory, manifest=make_manifest(chunks, source_sha=OTHER_SHA), chunks=chunks, embeddings=vectors)
    assert {name: (directory / name).read_bytes() for name in listing(directory)} == before
    assert listing(directory.parent) == ["index"]


def test_a_failure_while_swapping_puts_the_previous_index_back(written, monkeypatch):
    directory, chunks, vectors = written
    before = {name: (directory / name).read_bytes() for name in listing(directory)}
    real_rename = os.rename
    calls = []

    def rename(src, dst):
        calls.append((Path(src).name, Path(dst).name))
        if len(calls) == 2:  # the second rename moves the new index into place: fail it
            raise OSError("rename failed")
        return real_rename(src, dst)

    monkeypatch.setattr(index_module.os, "rename", rename)
    with pytest.raises(OSError, match="rename failed"):
        write_index(directory, manifest=make_manifest(chunks, source_sha=OTHER_SHA), chunks=chunks, embeddings=vectors)
    monkeypatch.undo()
    assert {name: (directory / name).read_bytes() for name in listing(directory)} == before
    assert listing(directory.parent) == ["index"]
    assert load_index(settings(), directory).manifest.source.sha256 == SHA


def test_writing_over_an_existing_index_replaces_it_completely(written):
    directory, chunks, vectors = written
    write_index(directory, manifest=make_manifest(chunks, source_sha=OTHER_SHA), chunks=chunks, embeddings=vectors)
    assert load_index(settings(), directory).manifest.source.sha256 == OTHER_SHA
    assert listing(directory) == [CHUNKS_FILE, EMBEDDINGS_FILE, MANIFEST_FILE]
    assert listing(directory.parent) == ["index"]  # old and staging directories are gone


def test_the_index_directory_and_files_are_readable_by_other_users(written):
    directory, _, _ = written
    assert directory.stat().st_mode & 0o777 == 0o755  # mkdtemp's private 0700 would lock out the app's user
    for name in listing(directory):
        assert (directory / name).stat().st_mode & 0o044 == 0o044, f"{name} is not world-readable"


def test_the_parent_directory_is_created_when_missing(tmp_path):
    chunks = make_chunks()
    vectors = HashEmbedder().embed_documents([c.embed_text for c in chunks])
    write_index(tmp_path / "data" / "index", manifest=make_manifest(chunks), chunks=chunks, embeddings=vectors)
    assert (tmp_path / "data" / "index" / MANIFEST_FILE).is_file()


def test_writing_to_an_unwritable_place_raises_and_leaves_nothing(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed")
    chunks = make_chunks()
    vectors = HashEmbedder().embed_documents([c.embed_text for c in chunks])
    with pytest.raises(OSError):
        write_index(blocker / "index", manifest=make_manifest(chunks), chunks=chunks, embeddings=vectors)
    assert listing(tmp_path) == ["blocker"]


def test_manifest_rejects_a_malformed_hash_or_unknown_field():
    good = make_manifest(make_chunks()).model_dump(mode="json")
    for patch in ({"build_hash": "xyz"}, {"source": {**good["source"], "sha256": "short"}}, {"unexpected": 1}, {"dimension": 0}):
        with pytest.raises(ValueError):
            Manifest.model_validate({**good, **patch})
