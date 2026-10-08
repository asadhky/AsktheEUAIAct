import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from askact.config import Settings
from askact.ingest import fetch
from askact.ingest.fetch import CACHE_FILENAME, SourceError, get_source

URL = "https://eur-lex.example/act"
# The real page's <title> is "L_202401689EN.000101.fmx.xml".
GOOD = b"<html><head><title>L_202401689EN.000101.fmx.xml</title></head><body>Article 1</body></html>"
OTHER_GOOD = b"<html><head><title>L_202401689EN.000101.fmx.xml</title></head><body>Article 2</body></html>"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


Outcome = httpx.Response | Exception


class Server:
    """A fake EUR-Lex that records requests. `respond` is a Response, an exception to raise, or a callable."""

    def __init__(self, respond: Outcome | Callable[[httpx.Request], Outcome]):
        self.respond = respond
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        result = self.respond if isinstance(self.respond, Outcome) else self.respond(request)
        if isinstance(result, Exception):
            raise result
        return result

    @property
    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def ok(body: bytes = GOOD, status: int = 200, content_type: str = "text/html; charset=UTF-8") -> httpx.Response:
    return httpx.Response(status, content=body, headers={"content-type": content_type})


def settings(manual: Path | None = None, url: str = URL) -> Settings:
    extra = {"source_fallback_path": manual} if manual is not None else {}
    return Settings(_env_file=None, source_url=url, **extra)


@pytest.fixture
def raw(tmp_path) -> Path:
    return tmp_path / "raw"  # deliberately not created: the code must create it only when needed


@pytest.fixture
def manual(tmp_path) -> Path:
    return tmp_path / "manual.html"


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


# --- fetching and caching (R1.1, R1.3) -------------------------------------------------

def test_fetches_and_caches_the_document(raw, manual):
    server = Server(ok())
    source = get_source(settings(manual), raw_dir=raw, client=server.client)

    assert source.origin == "fetched"
    assert source.path == raw / CACHE_FILENAME
    assert source.path.read_bytes() == GOOD
    assert source.sha256 == sha(GOOD)
    assert len(server.requests) == 1
    assert str(server.requests[0].url) == URL
    assert leftovers(raw) == [CACHE_FILENAME]  # no temp file left behind


def test_request_asks_for_english_xhtml(raw, manual):
    """The Publications Office chooses the document variant from these headers (text/html gets a 404)."""
    server = Server(ok())
    get_source(settings(manual), raw_dir=raw, client=server.client)

    headers = server.requests[0].headers
    assert headers["accept"] == "application/xhtml+xml"
    assert headers["accept-language"] == "eng"


def test_the_default_source_is_the_publications_office_resource_for_the_act():
    """eur-lex.europa.eu answers the project's honest User-Agent with a 202 challenge page."""
    assert Settings(_env_file=None).source_url == "https://publications.europa.eu/resource/celex/32024R1689"


def test_request_identifies_the_client_and_sets_timeouts(raw, manual):
    server = Server(ok())
    get_source(settings(manual), raw_dir=raw, client=server.client)

    request = server.requests[0]
    assert "askact" in request.headers["user-agent"]
    timeout = request.extensions["timeout"]
    assert timeout["connect"] == 10.0 and timeout["read"] == 30.0


def test_a_cached_copy_is_reused_without_any_request(raw, manual):
    raw.mkdir()
    (raw / CACHE_FILENAME).write_bytes(GOOD)
    server = Server(ok(OTHER_GOOD))

    source = get_source(settings(manual), raw_dir=raw, client=server.client)

    assert source.origin == "cache"
    assert source.sha256 == sha(GOOD)
    assert server.requests == []


def test_force_fetches_again_and_replaces_the_cache(raw, manual):
    raw.mkdir()
    (raw / CACHE_FILENAME).write_bytes(GOOD)
    server = Server(ok(OTHER_GOOD))

    source = get_source(settings(manual), force=True, raw_dir=raw, client=server.client)

    assert source.origin == "fetched"
    assert (raw / CACHE_FILENAME).read_bytes() == OTHER_GOOD
    assert len(server.requests) == 1


def test_force_with_a_failing_fetch_falls_back_and_leaves_the_old_cache_alone(raw, manual):
    raw.mkdir()
    (raw / CACHE_FILENAME).write_bytes(GOOD)
    manual.write_bytes(OTHER_GOOD)

    source = get_source(settings(manual), force=True, raw_dir=raw, client=Server(ok(b"", 503)).client)

    assert source.origin == "manual"
    assert (raw / CACHE_FILENAME).read_bytes() == GOOD


def test_a_redirect_is_followed(raw, manual):
    def respond(request):
        if request.url.path == "/act":
            return httpx.Response(301, headers={"location": "https://eur-lex.example/final"})
        return ok()

    source = get_source(settings(manual), raw_dir=raw, client=Server(respond).client)
    assert source.origin == "fetched"


def test_default_client_is_created_and_used_when_none_is_given(raw, manual, monkeypatch):
    server = Server(ok())
    mock_client = server.client  # built first: the patch below replaces httpx.Client for the whole process
    monkeypatch.setattr(fetch.httpx, "Client", lambda: mock_client)
    assert get_source(settings(manual), raw_dir=raw).origin == "fetched"
    assert len(server.requests) == 1
    assert mock_client.is_closed, "the client created by get_source must be closed afterwards"


def test_an_unusable_cache_is_ignored_and_fetched_again(raw, manual, caplog):
    """A bad page that got cached once must not poison every later run."""
    raw.mkdir()
    (raw / CACHE_FILENAME).write_bytes(b"<html>Please verify you are human</html>")
    server = Server(ok())

    with caplog.at_level(logging.WARNING, logger="askact.ingest.fetch"):
        source = get_source(settings(manual), raw_dir=raw, client=server.client)

    assert source.origin == "fetched"
    assert (raw / CACHE_FILENAME).read_bytes() == GOOD
    assert "Ignoring the cached file" in caplog.text


# --- falling back to the manual file (R1.4) ---------------------------------------------

@pytest.mark.parametrize("failure", [
    pytest.param(ok(b"", 503), id="http 503"),
    pytest.param(ok(b"", 404), id="http 404"),
    pytest.param(ok(GOOD, 202), id="202 challenge status, even with good-looking body"),
    pytest.param(ok(GOOD, 403), id="http 403"),
    pytest.param(httpx.ConnectError("no route to host"), id="connection error"),
    pytest.param(httpx.ReadTimeout("slow"), id="timeout"),
    pytest.param(ok(b""), id="empty body"),
    pytest.param(ok(b"   \n"), id="blank body"),
    pytest.param(ok(b"<html>Please verify you are human</html>"), id="html without the document id"),
    pytest.param(ok(b'{"error": "rate limited"}', content_type="application/json"), id="json body"),
    pytest.param(ok(b"<html>02024R1689-20260727 consolidated text</html>"), id="consolidated version"),
])
def test_failed_or_invalid_fetch_falls_back_to_the_manual_file(raw, manual, caplog, failure):
    manual.write_bytes(OTHER_GOOD)
    server = Server(failure)

    with caplog.at_level(logging.WARNING, logger="askact.ingest.fetch"):
        source = get_source(settings(manual), raw_dir=raw, client=server.client)

    assert source.origin == "manual"
    assert source.path == manual
    assert source.sha256 == sha(OTHER_GOOD)
    assert "trying the manual file" in caplog.text
    assert not raw.exists(), "a failed fetch must not create or write anything under data/raw"


@pytest.mark.parametrize("url", ["not-a-url", "", "ftp://example.org/act"])
def test_an_unusable_url_falls_back_too(raw, manual, url):
    """Uses a real httpx client on purpose: a mock transport would not validate the URL.
    These URLs are rejected before any connection is attempted, so the test stays offline."""
    manual.write_bytes(GOOD)
    with httpx.Client() as real_client:
        source = get_source(settings(manual, url=url), raw_dir=raw, client=real_client)
    assert source.origin == "manual"


def test_the_default_manual_file_in_data_raw_is_used_when_nothing_is_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    default = Path("data/raw/ai-act-oj-2024-1689.manual.html")
    default.parent.mkdir(parents=True)
    default.write_bytes(GOOD)

    # No raw_dir and no fallback path: both come from the defaults, as the CLI will call it.
    source = get_source(Settings(_env_file=None, source_url=URL), client=Server(ok(b"", 503)).client)

    assert source.origin == "manual"
    assert source.path == default


def test_an_explicit_source_file_wins_over_the_configured_fallback(raw, manual, tmp_path):
    manual.write_bytes(GOOD)
    explicit = tmp_path / "mine.html"
    explicit.write_bytes(OTHER_GOOD)

    source = get_source(settings(manual), source_file=explicit, raw_dir=raw, client=Server(ok(b"", 503)).client)

    assert source.path == explicit
    assert source.sha256 == sha(OTHER_GOOD)


def test_the_manual_file_is_only_a_fallback(raw, manual, tmp_path):
    """Pins the documented behaviour: a working fetch is used even when --source-file is given."""
    explicit = tmp_path / "mine.html"
    explicit.write_bytes(OTHER_GOOD)

    source = get_source(settings(manual), source_file=explicit, raw_dir=raw, client=Server(ok()).client)

    assert source.origin == "fetched"


def test_a_fetch_never_overwrites_the_manual_download(raw):
    """The default manual file lives in data/raw next to the cache, so the names must differ."""
    default_manual = Settings(_env_file=None).source_fallback_path
    assert default_manual.name != CACHE_FILENAME
    assert default_manual.parent == fetch.RAW_DIR

    raw.mkdir()
    saved_by_hand = raw / default_manual.name
    saved_by_hand.write_bytes(OTHER_GOOD)
    get_source(settings(saved_by_hand), raw_dir=raw, client=Server(ok()).client)

    assert saved_by_hand.read_bytes() == OTHER_GOOD
    assert (raw / CACHE_FILENAME).read_bytes() == GOOD


# --- nothing usable: fail clearly, write nothing (R1.13) ---------------------------------

def test_no_fetch_and_no_manual_file_raises_a_helpful_error(raw, manual):
    with pytest.raises(SourceError) as excinfo:
        get_source(settings(manual), raw_dir=raw, client=Server(ok(b"", 503)).client)

    message = str(excinfo.value)
    assert URL in message                  # what we tried to fetch
    assert "HTTP status 503" in message    # why it failed
    assert str(manual) in message          # where a manual copy is expected
    assert "--source-file" in message      # the other way out
    assert not raw.exists()


@pytest.mark.parametrize("content, reason", [
    (b"", "it is empty"),
    (b"<html>not the act</html>", "not the Official Journal text"),
])
def test_an_unusable_manual_file_is_rejected_with_the_reason(raw, manual, content, reason):
    manual.write_bytes(content)
    with pytest.raises(SourceError, match=reason):
        get_source(settings(manual), raw_dir=raw, client=Server(httpx.ConnectError("offline")).client)
    assert not raw.exists()


def test_a_manual_path_that_is_a_directory_is_an_error_not_a_crash(raw, tmp_path):
    folder = tmp_path / "a_folder"
    folder.mkdir()
    with pytest.raises(SourceError, match="no usable manual file"):
        get_source(settings(folder), raw_dir=raw, client=Server(ok(b"", 503)).client)


def test_a_failure_while_saving_the_download_leaves_no_partial_file(raw, manual, monkeypatch):
    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(fetch.os, "replace", broken_replace)

    with pytest.raises(SourceError, match="could not save the download: disk full"):
        get_source(settings(manual), raw_dir=raw, client=Server(ok()).client)

    assert leftovers(raw) == []  # neither the cache nor a .part temp file


def test_a_failure_while_saving_falls_back_to_the_manual_file(raw, manual, monkeypatch):
    manual.write_bytes(OTHER_GOOD)
    monkeypatch.setattr(fetch.os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("read-only")))

    source = get_source(settings(manual), raw_dir=raw, client=Server(ok()).client)

    assert source.origin == "manual"
    assert leftovers(raw) == []
