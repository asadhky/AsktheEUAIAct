"""Does every chunk of the real Act fit the real embedder? Slow, local only.

CHUNK_MAX_CHARS counts characters, but the embedding model reads tokens and silently truncates
anything beyond its limit, which would hide the end of a chunk from search. Characters per token
varies a lot (about 5 for prose, under 3 for lists of directive numbers), so this checks the real
chunks against the real tokenizer instead of trusting the ratio.

Run with:  pytest -m slow tests/ingest/test_chunk_tokens.py -s
Needs the `models` extra (sentence-transformers), the model weights (downloaded on first use),
and the full source file; it is skipped when any of them is absent.
"""

from pathlib import Path

import pytest

from askact.config import Settings
from askact.ingest.chunk import chunk_act
from askact.ingest.parse import parse_act

REAL = Path(__file__).resolve().parents[3] / "data" / "raw" / "ai-act-oj-2024-1689.html"


@pytest.mark.slow
def test_every_real_chunk_fits_the_embedders_token_limit():
    if not REAL.exists():  # cheap checks first: loading the model takes seconds
        pytest.skip(f"the full source is not present at {REAL}")
    sentence_transformers = pytest.importorskip("sentence_transformers", reason="the `models` extra is not installed")

    settings = Settings(_env_file=None)  # the default model and the default CHUNK_MAX_CHARS
    model = sentence_transformers.SentenceTransformer(settings.embedding_model, device="cpu")
    limit = model.max_seq_length  # read from the loaded model, not hard-coded

    chunks = chunk_act(parse_act(REAL.read_bytes()), settings.chunk_max_chars)
    # Exactly the string that gets embedded, with the [CLS] and [SEP] tokens, and no truncation.
    encoded = model.tokenizer([c.embed_text for c in chunks], add_special_tokens=True, truncation=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]

    too_long = sorted(
        ((n, c.chunk_id, len(c.embed_text)) for c, n in zip(chunks, lengths) if n > limit), reverse=True
    )
    longest = max(zip(lengths, (c.chunk_id for c in chunks)))
    print(f"\n{len(chunks)} chunks at CHUNK_MAX_CHARS={settings.chunk_max_chars}; "
          f"longest is {longest[0]} tokens ({longest[1]}); the {settings.embedding_model} limit is {limit}")

    assert not too_long, (
        f"{len(too_long)} chunk(s) exceed the {limit}-token limit of {settings.embedding_model}; "
        f"lower CHUNK_MAX_CHARS (currently {settings.chunk_max_chars}) and rebuild. Offenders "
        f"(tokens, chunk id, characters): {too_long}"
    )
