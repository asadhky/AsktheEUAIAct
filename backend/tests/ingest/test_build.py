import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest

from askact.config import Settings
from askact.embedders import Embedder, EmbedderError, HashEmbedder, Vectors
from askact.index import CHUNKS_FILE, EMBEDDINGS_FILE, MANIFEST_FILE, IndexUnavailableError, load_index
from askact.ingest import build as build_module
from askact.ingest.__main__ import main
from askact.ingest.build import BuildResult, build_index, retrieval_date
from askact.ingest.chunk import CHUNKER_VERSION, ChunkError, chunk_act
from askact.ingest.fetch import CACHE_FILENAME, SourceError, SourceFile
from askact.ingest.parse import Counts, ParseError, parse_act

BACKEND = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_act.html"
REAL = Path(__file__).resolve().parents[3] / "data" / "raw" / "ai-act-oj-2024-1689.html"
FIXTURE_COUNTS = Counts(recitals=3, articles=2, annexes=1)
FIXTURE_BYTES = FIXTURE.read_bytes()
FIXTURE_SHA = hashlib.sha256(FIXTURE_BYTES).hexdigest()
TODAY = date(2026, 10, 9)


class CountingEmbedder:
    """Wraps the stub and counts the texts it is asked to embed."""

    def __init__(self, inner: Embedder | None = None):
        self.inner = inner or HashEmbedder()
        self.name = self.inner.name
        self.dimension = self.inner.dimension
        self.texts_embedded = 0
        self.calls = 0
        self.seen: list[str] = []

    def embed_documents(self, texts):
        self.calls += 1
        self.texts_embedded += len(texts)
        self.seen.extend(texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        return self.inner.embed_query(text)


class Broken(CountingEmbedder):
    def embed_documents(self, texts):
        raise EmbedderError("model exploded")


class Env:
    """A scratch project directory: raw/ for the source, index/ for the output."""

    def __init__(self, tmp_path: Path):
        self.raw, self.index = tmp_path / "data" / "raw", tmp_path / "data" / "index"
        self.settings = Settings(_env_file=None, embedder="stub", source_url="https://example.org/act")
        self.requests: list[httpx.Request] = []
        self.body: bytes = FIXTURE_BYTES

    def client(self) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, content=self.body, headers={"content-type": "application/xhtml+xml"})

        return httpx.Client(transport=httpx.MockTransport(handler))

    def build(self, settings: Settings | None = None, **kwargs) -> BuildResult:
        kwargs.setdefault("expected_counts", FIXTURE_COUNTS)
        kwargs.setdefault("today", TODAY)
        client = kwargs.pop("client", None) or self.client()
        return build_index(settings or self.settings, raw_dir=self.raw, index_dir=self.index, client=client, **kwargs)

    def cli(self, *argv: str, settings: Settings | None = None, embedder=None, expected=FIXTURE_COUNTS) -> int:
        return main(list(argv), settings=settings or self.settings, raw_dir=self.raw, index_dir=self.index,
                    client=self.client(), expected_counts=expected, embedder=embedder)

    def sections_dir_entries(self) -> list[str]:
        data = self.index.parent
        return sorted(p.name for p in data.iterdir()) if data.exists() else []


@pytest.fixture
def env(tmp_path) -> Env:
    return Env(tmp_path)


def set_file_date(path: Path, moment: datetime) -> None:
    os.utime(path, (moment.timestamp(), moment.timestamp()))


# --- building the index end to end (R1.10) -------------------------------------------------------------

def test_a_first_build_writes_the_index_and_reports_what_it_built(env):
    result = env.build()
    assert result.status == "built" and result.index_dir == env.index
    counts = result.manifest.counts
    assert (counts.recitals, counts.articles, counts.annexes) == (3, 2, 1)
    assert sorted(p.name for p in env.index.iterdir()) == [CHUNKS_FILE, EMBEDDINGS_FILE, MANIFEST_FILE]


def test_the_built_index_matches_the_chunks_of_the_source(env):
    env.build()
    loaded = load_index(env.settings, env.index)
    expected = chunk_act(parse_act(FIXTURE_BYTES), env.settings.chunk_max_chars)
    assert list(loaded.chunks) == expected
    assert loaded.manifest.counts.chunks == len(expected)
    assert np.array_equal(loaded.embeddings, HashEmbedder().embed_documents([c.embed_text for c in expected]))


def test_embeddings_are_made_from_the_header_prefixed_string(env):
    embedder = CountingEmbedder()
    env.build(embedder=embedder)
    assert any(text.startswith("Article 5 — Prohibited AI practices\n") for text in embedder.seen)
    assert any(text.startswith("Recital 1\n") for text in embedder.seen)


def test_the_manifest_records_the_pinned_source(env):
    env.build()
    source = json.loads((env.index / MANIFEST_FILE).read_text())["source"]
    assert source == {
        "regulation": "Regulation (EU) 2024/1689",
        "celex": "32024R1689",
        "oj_reference": "OJ L, 2024/1689, 12.7.2024",
        "url": "https://example.org/act",
        "retrieved_at": "2026-10-09",
        "sha256": FIXTURE_SHA,
        "origin": "fetched",
    }


def test_the_manifest_records_the_chunking_model_and_dimension(env):
    manifest = env.build().manifest
    assert manifest.chunking == {"max_chars": 1800, "chunker_version": CHUNKER_VERSION}
    assert manifest.embedding_model == "stub-hash-256" and manifest.dimension == 256


# --- skipping work when nothing changed (R2.6) -----------------------------------------------------------

def test_an_unchanged_index_is_not_rebuilt_or_re_embedded(env):
    first = CountingEmbedder()
    env.build(embedder=first)
    assert first.texts_embedded > 0
    stamps = {p.name: p.stat().st_mtime_ns for p in env.index.iterdir()}

    second = CountingEmbedder()
    result = env.build(embedder=second)

    assert result.status == "up_to_date"
    assert second.calls == 0, "nothing should be embedded when the hash matches"
    assert {p.name: p.stat().st_mtime_ns for p in env.index.iterdir()} == stamps  # files untouched


def test_an_up_to_date_result_describes_the_existing_index(env):
    built = env.build().manifest
    assert env.build().manifest == built


def test_force_downloads_again_but_an_identical_source_still_skips_embedding(env):
    env.build()
    assert len(env.requests) == 1
    again = CountingEmbedder()
    result = env.build(force=True, embedder=again)
    assert len(env.requests) == 2 and result.status == "up_to_date" and again.calls == 0


def test_a_changed_source_rebuilds(env):
    env.build()
    old_hash = load_index(env.settings, env.index).manifest.build_hash
    env.body = FIXTURE_BYTES.replace(b"Prohibited AI practices", b"Prohibited AI practices (edited)")
    embedder = CountingEmbedder()

    result = env.build(force=True, embedder=embedder)

    assert result.status == "built" and embedder.texts_embedded > 0
    assert result.manifest.build_hash != old_hash
    assert result.manifest.source.sha256 == hashlib.sha256(env.body).hexdigest()
    assert any(c.title == "Prohibited AI practices (edited)" for c in load_index(env.settings, env.index).chunks)


def test_a_changed_chunk_size_rebuilds(env):
    env.build()
    embedder = CountingEmbedder()
    result = env.build(env.settings.model_copy(update={"chunk_max_chars": 900}), embedder=embedder)
    assert result.status == "built" and embedder.calls == 1
    assert result.manifest.chunking["max_chars"] == 900


def test_a_changed_chunker_version_rebuilds(env, monkeypatch):
    env.build()
    from askact.ingest import chunk as chunk_module

    monkeypatch.setattr(chunk_module, "CHUNKER_VERSION", CHUNKER_VERSION + 1)
    assert env.build(embedder=CountingEmbedder()).status == "built"


def test_an_embedder_that_is_not_the_one_the_settings_name_is_refused(env):
    """The build hash uses the name from the settings. Recording a different name in the manifest
    would make every later check call the index stale, so it is refused up front."""
    with pytest.raises(EmbedderError, match="the embedder is called 'stub-hash-64' but the settings say 'stub-hash-256'"):
        env.build(embedder=CountingEmbedder(HashEmbedder(dimension=64)))
    assert not env.index.exists()


def test_an_index_built_with_another_embedding_model_is_seen_as_stale(env):
    """Switching EMBEDDER from the stub to the real model changes the model name, hence the hash."""
    env.build()
    real = Settings(_env_file=None, embedder="real", source_url=env.settings.source_url)
    with pytest.raises(IndexUnavailableError, match="embedding model is now 'BAAI/bge-small-en-v1.5'"):
        load_index(real, env.index)
    assert load_index(env.settings, env.index).manifest.embedding_model == "stub-hash-256"


def test_a_damaged_index_is_rebuilt_instead_of_trusted(env):
    env.build()
    (env.index / EMBEDDINGS_FILE).write_bytes(b"garbage")
    embedder = CountingEmbedder()
    assert env.build(embedder=embedder).status == "built" and embedder.calls == 1
    assert load_index(env.settings, env.index).manifest.counts.chunks > 0


# --- failure: nothing partial, previous index kept (R1.13) --------------------------------------------------

def test_a_source_that_cannot_be_obtained_writes_no_index(env):
    env.body = b"<html>not the act</html>"
    with pytest.raises(SourceError):
        env.build()
    assert not env.index.exists()
    assert env.sections_dir_entries() in ([], ["raw"])


def test_a_source_that_does_not_parse_writes_no_index(env):
    env.body = b"<html><head><title>L_202401689EN.000101.fmx.xml</title></head><body><p>no sections</p></body></html>"
    with pytest.raises(ParseError, match="no recitals, articles or annexes"):
        env.build()
    assert not env.index.exists()


def test_the_pinned_counts_are_enforced_by_default(env):
    with pytest.raises(ParseError, match="parsed 3 recitals, 2 articles and 1 annexes, but the pinned text has 180, 113 and 13"):
        env.build(expected_counts=None)  # None means: the real pinned counts and numbering
    assert not env.index.exists()


def test_custom_counts_check_only_the_counts_not_the_numbering(env):
    """The fixture keeps the real, non-contiguous ids (1, 8, 23); supplying counts is how a trimmed
    document is checked. Without them the full check, including numbering, applies."""
    assert env.build(expected_counts=FIXTURE_COUNTS).status == "built"


def test_an_embedder_that_fails_writes_no_index(env):
    with pytest.raises(EmbedderError, match="exploded"):
        env.build(embedder=Broken())
    assert not env.index.exists()
    assert env.sections_dir_entries() == ["raw"]


def test_a_failed_rebuild_keeps_the_previous_index(env):
    env.build()
    before = {p.name: p.read_bytes() for p in env.index.iterdir()}
    with pytest.raises(EmbedderError):
        env.build(env.settings.model_copy(update={"chunk_max_chars": 900}), embedder=Broken())
    assert {p.name: p.read_bytes() for p in env.index.iterdir()} == before
    assert env.sections_dir_entries() == ["index", "raw"]


def test_an_embedder_returning_the_wrong_shape_is_refused_before_writing(env):
    class WrongShape(CountingEmbedder):
        def embed_documents(self, texts) -> Vectors:
            return super().embed_documents(texts)[:-1]

    with pytest.raises(ValueError, match="inconsistent index"):
        env.build(embedder=WrongShape())
    assert not env.index.exists()


def test_a_chunk_size_too_small_for_the_headers_writes_no_index(env):
    with pytest.raises(ChunkError, match="too small"):
        env.build(env.settings.model_copy(update={"chunk_max_chars": 105}))
    assert not env.index.exists()


# --- the retrieval date and origin ---------------------------------------------------------------------------

def test_a_just_downloaded_source_is_dated_today():
    source = SourceFile(Path("unused"), "0" * 64, "fetched")
    assert retrieval_date(source, today=date(2026, 1, 2)) == date(2026, 1, 2)


@pytest.mark.parametrize("origin", ["cache", "manual"])
def test_a_file_already_on_disk_is_dated_by_the_file(tmp_path, origin):
    path = tmp_path / "saved.html"
    path.write_bytes(b"x")
    set_file_date(path, datetime(2025, 12, 24, 23, 30, tzinfo=UTC))
    assert retrieval_date(SourceFile(path, "0" * 64, origin), today=date(2030, 1, 1)) == date(2025, 12, 24)


def test_a_cached_source_is_recorded_as_cached_with_its_file_date(env):
    env.raw.mkdir(parents=True)
    cached = env.raw / CACHE_FILENAME
    cached.write_bytes(FIXTURE_BYTES)
    set_file_date(cached, datetime(2025, 3, 4, 12, 0, tzinfo=UTC))
    source = env.build().manifest.source
    assert env.requests == [] and source.origin == "cache" and source.retrieved_at == date(2025, 3, 4)


def test_a_manual_file_is_recorded_as_manual(env, tmp_path):
    manual = tmp_path / "mine.html"
    manual.write_bytes(FIXTURE_BYTES)
    broken = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    result = env.build(client=broken, source_file=manual)
    assert result.manifest.source.origin == "manual" and result.manifest.source.sha256 == FIXTURE_SHA


# --- the command line (R1.10, R1.13) ------------------------------------------------------------------------------

def test_the_cli_prints_the_counts_and_exits_zero(env, capsys):
    assert env.cli() == 0
    out = capsys.readouterr().out
    assert "Built:" in out
    for line in ("Recitals: 3", "Articles: 2", "Annexes: 1"):
        assert line in out
    assert f"Chunks: {load_index(env.settings, env.index).manifest.counts.chunks}" in out
    assert "Regulation (EU) 2024/1689" in out and "CELEX 32024R1689" in out and FIXTURE_SHA[:16] in out


def test_the_cli_says_when_the_index_was_already_up_to_date(env, capsys):
    env.cli()
    capsys.readouterr()
    assert env.cli() == 0
    out = capsys.readouterr().out
    assert "Already up to date" in out and "Chunks:" in out


def test_the_cli_exits_one_with_a_clear_message_when_nothing_can_be_fetched(env, capsys):
    env.body = b""
    assert env.cli() == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # no summary on failure
    assert captured.err.startswith("error: Could not get the Act")
    assert not env.index.exists()


def test_the_cli_exits_one_when_the_counts_are_wrong(env, capsys):
    assert env.cli(expected=Counts(180, 113, 13)) == 1
    err = capsys.readouterr().err
    assert "error: parsed 3 recitals" in err and "pinned text has 180, 113 and 13" in err
    assert not env.index.exists()


def test_the_cli_exits_one_when_the_embedder_fails(env, capsys):
    assert env.cli(embedder=Broken()) == 1
    assert "error: model exploded" in capsys.readouterr().err
    assert not env.index.exists()


def test_the_cli_exits_one_on_invalid_configuration(env, capsys, monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_CHARS", "0")
    code = main([], raw_dir=env.raw, index_dir=env.index, client=env.client(), expected_counts=FIXTURE_COUNTS)
    err = capsys.readouterr().err
    assert code == 1 and "invalid configuration" in err and "chunk_max_chars" in err


def test_the_cli_exits_one_when_the_index_cannot_be_written(env, capsys, monkeypatch):
    """A full disk or read-only directory, simulated at the write step so it does not depend on
    who runs the tests (file permissions are not enforced for root)."""
    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(build_module, "write_index", disk_full)
    assert env.cli() == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:") and "No space left on device" in captured.err
    assert captured.out == "" and not env.index.exists()


def test_force_and_source_file_options_are_passed_through(env, tmp_path, capsys):
    manual = tmp_path / "mine.html"
    manual.write_bytes(FIXTURE_BYTES)
    env.body = b""  # the fetch fails, so the manual file must be used
    assert env.cli("--force", "--source-file", str(manual)) == 0
    assert "manual" in capsys.readouterr().out


def test_unknown_options_are_a_usage_error(env):
    with pytest.raises(SystemExit) as excinfo:
        env.cli("--nonsense")
    assert excinfo.value.code == 2


def test_running_main_does_not_change_global_logging_state(env):
    """main() once attached a handler to the 'askact' logger on first use. It kept a stale stderr, so a
    later test failed or not depending on test order. Logging is configured only when run as a program."""
    logger = logging.getLogger("askact")
    before = (list(logger.handlers), logger.level, logger.propagate)
    env.cli()
    env.cli()
    assert (list(logger.handlers), logger.level, logger.propagate) == before


# --- a real subprocess run, as the Docker build will do it ------------------------------------------------------------

def test_python_dash_m_refuses_a_document_that_is_not_the_pinned_act(tmp_path):
    """The fixture is not the pinned Act, so the real entry point must exit 1, writing no index."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / CACHE_FILENAME).write_bytes(FIXTURE_BYTES)
    result = subprocess.run(
        [sys.executable, "-m", "askact.ingest"], cwd=tmp_path, env={**os.environ, "EMBEDDER": "stub"},
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "pinned text has 180, 113 and 13" in result.stderr
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["raw"]


def test_python_dash_m_prints_progress_to_stderr_and_the_counts_to_stdout(tmp_path):
    """Run as a program, the app's own progress messages appear (on stderr), library chatter does not."""
    code = (
        "import sys; sys.argv = ['askact.ingest']; import runpy; "
        "from askact.ingest import __main__ as m; m._configure_logging(); "
        "from askact.ingest.parse import Counts; from pathlib import Path; from askact.config import Settings; import httpx; "
        "raise SystemExit(m.main([], settings=Settings(_env_file=None, embedder='stub'), raw_dir=Path('raw'), index_dir=Path('index'), "
        "expected_counts=Counts(3, 2, 1)))"
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / CACHE_FILENAME).write_bytes(FIXTURE_BYTES)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, "EMBEDDER": "stub"})
    assert result.returncode == 0, result.stderr
    assert "Building the index" in result.stderr and "Embedding " in result.stderr
    assert "Recitals: 3" in result.stdout and "Chunks:" in result.stdout
    assert "HTTP Request" not in result.stderr


# --- the real document (runs only when the file is present) ---------------------------------------------------------------

@pytest.mark.skipif(not REAL.exists(), reason=f"the full source is not present at {REAL}")
def test_real_source_builds_the_pinned_index_with_the_stub_embedder(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / CACHE_FILENAME).write_bytes(REAL.read_bytes())
    settings = Settings(_env_file=None, embedder="stub")
    code = main([], settings=settings, raw_dir=raw, index_dir=tmp_path / "data" / "index")
    assert code == 0
    manifest = load_index(settings, tmp_path / "data" / "index").manifest
    assert (manifest.counts.recitals, manifest.counts.articles, manifest.counts.annexes) == (180, 113, 13)
    assert manifest.counts.chunks == 495
