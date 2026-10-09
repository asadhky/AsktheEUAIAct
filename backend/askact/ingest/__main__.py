"""Command line: `python -m askact.ingest [--force] [--source-file PATH]`.

Fetches the Act (or uses the manually downloaded file), parses and chunks it, embeds the chunks and
writes the index to data/index/. Prints the section and chunk counts. Exits 0 on success and 1 on
any failure, having written nothing to the index.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx
from pydantic import ValidationError

from askact.config import Settings
from askact.embedders import Embedder, EmbedderError
from askact.index import INDEX_DIR
from askact.ingest.build import BuildResult, build_index
from askact.ingest.chunk import ChunkError
from askact.ingest.fetch import RAW_DIR, SourceError
from askact.ingest.parse import Counts, ParseError

# Everything build_index can raise on purpose. Anything else is a bug and should show a traceback.
_EXPECTED_FAILURES = (SourceError, ParseError, ChunkError, EmbedderError, OSError)


def _summary(result: BuildResult) -> str:
    manifest = result.manifest
    counts = manifest.counts
    source = manifest.source
    state = "Built" if result.status == "built" else "Already up to date"
    return "\n".join(
        [
            f"{state}: {result.index_dir}",
            f"Source: {source.regulation} (CELEX {source.celex}), {source.origin}, "
            f"retrieved {source.retrieved_at}, sha256 {source.sha256[:16]}...",
            f"Embedding model: {manifest.embedding_model} ({manifest.dimension} dimensions)",
            f"Recitals: {counts.recitals}",
            f"Articles: {counts.articles}",
            f"Annexes: {counts.annexes}",
            f"Chunks: {counts.chunks}",
        ]
    )


def _configure_logging() -> None:
    """Show this app's progress messages, not the libraries': at INFO, httpx and huggingface_hub log
    every request, which buries the output (and would flood a Docker build log).

    Called only when run as a program, never from `main()`: it changes global logger state, which
    would leak from one test into the next.
    """
    app = logging.getLogger("askact")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    app.addHandler(handler)
    app.setLevel(logging.INFO)
    app.propagate = False


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    raw_dir: Path = RAW_DIR,
    index_dir: Path = INDEX_DIR,
    client: httpx.Client | None = None,
    expected_counts: Counts | None = None,
    embedder: Embedder | None = None,
) -> int:
    """Run ingestion; returns the process exit code. The keyword arguments exist for tests."""
    parser = argparse.ArgumentParser(
        prog="python -m askact.ingest",
        description="Build the search index for the EU AI Act: fetch the text, parse it, chunk it, embed "
        "the chunks and write data/index/. Does nothing if the index is already up to date.",
    )
    parser.add_argument("--force", action="store_true", help="download the source again even if a copy is cached")
    parser.add_argument("--source-file", type=Path, metavar="PATH", help="use this manually downloaded file if the download fails")
    args = parser.parse_args(argv)

    try:
        result = build_index(
            settings or Settings(),
            force=args.force,
            source_file=args.source_file,
            raw_dir=raw_dir,
            index_dir=index_dir,
            client=client,
            expected_counts=expected_counts,
            embedder=embedder,
        )
    except ValidationError as exc:  # bad environment variables
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 1
    except _EXPECTED_FAILURES as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(_summary(result))
    return 0


if __name__ == "__main__":
    _configure_logging()
    sys.exit(main())
