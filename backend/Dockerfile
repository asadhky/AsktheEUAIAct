# The backend: FastAPI under uvicorn, with the local embedding and reranker models and the search index BUILT INTO
# the image (R6.2). The build needs the network (pip, the Act's text from publications.europa.eu, and the two models
# from Hugging Face); the running container needs none, and is told not to try (HF_HUB_OFFLINE=1).
#
# Build from the repository root:   docker build -f backend/Dockerfile -t askact-backend .
# (docker-compose.yml does this and passes the build arguments below from your .env.)

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The settings that decide what the index and the downloaded models ARE. They are build arguments because the build
# cannot see your .env. They are also kept as environment variables, so the running app asks for the same models and
# chunking the index was built with. If your .env later names a different model, the app finds the index stale and
# says so (/health is 503) instead of searching with the wrong model: rebuild the image.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ARG CHUNK_MAX_CHARS=1800
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    RERANKER_MODEL=${RERANKER_MODEL} \
    CHUNK_MAX_CHARS=${CHUNK_MAX_CHARS} \
    HF_HOME=/opt/hf

WORKDIR /app

# torch first and from the CPU-only index: the default wheel pulls gigabytes of CUDA libraries that a CPU container
# never uses. It is the largest layer and changes least, so it comes before the source and stays cached.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# The package, with the `models` extra (sentence-transformers). Not editable: the code is installed into the image.
COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/askact backend/askact
RUN pip install "./backend[models]"

# The Act's text, if the owner has a copy (data/raw is gitignored, so a clean clone has none). One COPY with a source
# that always exists (pyproject.toml) so that "no HTML file" is not an error; the file is then moved into place.
COPY backend/pyproject.toml data/raw/*.html /tmp/optional/
RUN mkdir -p data/raw \
    && (cp /tmp/optional/*.html data/raw/ 2>/dev/null || true) \
    && rm -rf /tmp/optional

# Build the index (fetches the text unless a copy is present, downloads the embedding model, embeds the chunks), then
# download the reranker through the app's own loader, so what is cached is exactly what the app will ask for.
RUN python -m askact.ingest \
    && python -c "from askact.config import Settings; from askact.rerankers import get_reranker; get_reranker(Settings())"

# From here on the container must not reach for the network: a missing model fails loudly instead of downloading.
ENV HF_HUB_OFFLINE=1

# Not root. The spend ledger lives in a volume mounted at /var/lib/askact; creating and owning the directory here is
# what makes a new named volume writable by this user.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /var/lib/askact \
    && chown app:app /var/lib/askact
USER app

EXPOSE 8000
# One worker: the rate limiter keeps its state in the process. --no-access-log: uvicorn's access log records the client
# address and path, and the app writes its own one-line log instead (request_log). --no-proxy-headers: uvicorn must not
# rewrite the client address from X-Forwarded-For; the app decides that itself, from TRUSTED_PROXY_HEADER.
CMD ["uvicorn", "--factory", "askact.api:build_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--no-proxy-headers"]
