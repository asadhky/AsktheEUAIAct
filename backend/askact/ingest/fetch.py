"""Ingestion step 1: get the Act's source file onto disk, or fail clearly (R1.1-R1.4, R1.13).

Order of preference:
  1. the cached download in `data/raw/`, unless `force` is set or the cache is unusable;
  2. a fresh download from `SOURCE_URL`, which becomes the cache;
  3. the manually downloaded file (`source_file`, else `SOURCE_FALLBACK_PATH`).
If none of these works, `SourceError` is raised and nothing has been written.
"""

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from askact.config import Settings

log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
# Deliberately different from the default manual file name (`...manual.html`), so a download
# can never overwrite a copy the user saved by hand.
CACHE_FILENAME = "ai-act-oj-2024-1689.html"
# The Official Journal file id. It appears in the original text (the <title> is
# "L_202401689EN.000101.fmx.xml") but not in a consolidated version. Here it is only a cheap
# sanity check that we received the right kind of document; the parser's <title> check is the pin.
OJ_DOCUMENT_ID = "L_202401689EN"
USER_AGENT = "askact/0.1 (EU AI Act RAG demo; +https://github.com/asadhky/AsktheEUAIAct)"
FETCH_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class SourceError(Exception):
    """The Act's source file could not be obtained. The message says what was tried and what to do."""


@dataclass(frozen=True)
class SourceFile:
    path: Path
    sha256: str  # of the bytes on disk; recorded in the index manifest
    origin: Literal["cache", "fetched", "manual"]  # the manifest needs to know where it came from


class _FetchFailed(Exception):
    """Internal: the download did not produce a usable document. Carries a human-readable reason."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _problem_with(data: bytes) -> str | None:
    """Why `data` cannot be the Official Journal text, or None if it looks right."""
    if not data.strip():
        return "it is empty"
    if OJ_DOCUMENT_ID.encode() not in data:
        return (
            f"it does not contain {OJ_DOCUMENT_ID!r}, so it is not the Official Journal text "
            "of Regulation (EU) 2024/1689 (for example an error page or a consolidated version)"
        )
    return None


def _download(url: str, client: httpx.Client) -> bytes:
    # Headers, timeout and redirects are per request, so they also apply to an injected client.
    try:
        response = client.get(
            url,
            # The Publications Office picks the document variant by content negotiation. Asking for
            # XHTML in English returns the Official Journal file; asking for text/html returns 404.
            headers={"User-Agent": USER_AGENT, "Accept": "application/xhtml+xml", "Accept-Language": "eng"},
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
        )
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise _FetchFailed(f"{type(exc).__name__}: {exc}") from exc
    # Exactly 200: bot-protection pages are typically served as 202 or as a 200 without the Act.
    if response.status_code != 200:
        raise _FetchFailed(f"HTTP status {response.status_code}")
    problem = _problem_with(response.content)
    if problem:
        content_type = response.headers.get("content-type", "unknown type")
        raise _FetchFailed(f"unexpected response ({content_type}, {len(response.content)} bytes): {problem}")
    return response.content


def _save(data: bytes, directory: Path) -> Path:
    """Write the cache atomically: a failure part-way never leaves a half-written file behind."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / CACHE_FILENAME
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=CACHE_FILENAME + ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def _how_to_fix(url: str, path: Path) -> str:
    # The curl command is the route that was verified to return the right file; a browser's
    # "save page" produces a different document, which the checks would reject.
    return (
        f"To supply the file yourself, download it on a machine that can reach {url}, for example: "
        f"curl -L -H 'Accept: application/xhtml+xml' -H 'Accept-Language: eng' -o {path} {url} "
        f"(or save it elsewhere and pass --source-file PATH), then run again."
    )


def _use_manual(path: Path, fetch_failure: str, url: str) -> SourceFile:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SourceError(
            f"Could not get the Act. Fetching {url} failed ({fetch_failure}), and there is no usable "
            f"manual file at {path} ({exc.strerror or exc}). {_how_to_fix(url, path)}"
        ) from exc
    problem = _problem_with(data)
    if problem:
        # Never carry on with a wrong file: a bad source would silently become a bad index.
        raise SourceError(
            f"Could not get the Act. Fetching {url} failed ({fetch_failure}), and the manual file "
            f"{path} cannot be used: {problem}. {_how_to_fix(url, path)}"
        )
    return SourceFile(path, _sha256(data), "manual")


def get_source(
    settings: Settings,
    *,
    force: bool = False,
    source_file: Path | None = None,
    raw_dir: Path = RAW_DIR,
    client: httpx.Client | None = None,
) -> SourceFile:
    """Return the Act's source file, fetching it if needed. See the module docstring for the order.

    `source_file` overrides `SOURCE_FALLBACK_PATH` as the manual file. It is still only used when
    the fetch fails: a cached copy is reused first, and a fetch is attempted before a manual file.
    `client` is for tests; normally a short-lived client is created.
    """
    cached = raw_dir / CACHE_FILENAME
    if cached.is_file() and not force:
        cached_data = cached.read_bytes()
        cache_problem = _problem_with(cached_data)
        if cache_problem is None:
            return SourceFile(cached, _sha256(cached_data), "cache")
        # Reusing a bad cache would break every later run until someone thought of --force.
        log.warning("Ignoring the cached file %s because %s; fetching again.", cached, cache_problem)

    try:
        if client is None:
            with httpx.Client() as own_client:
                data = _download(settings.source_url, own_client)
        else:
            data = _download(settings.source_url, client)
        path = _save(data, raw_dir)
        return SourceFile(path, _sha256(data), "fetched")
    except (_FetchFailed, OSError) as exc:  # OSError: could not save the download
        failure = str(exc) if isinstance(exc, _FetchFailed) else f"could not save the download: {exc}"
        log.warning("Fetching the Act from %s failed (%s); trying the manual file.", settings.source_url, failure)

    return _use_manual(source_file or settings.source_fallback_path, failure, settings.source_url)
