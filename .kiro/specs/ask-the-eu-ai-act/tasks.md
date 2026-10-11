# Tasks — Ask the EU AI Act

Implements `requirements.md` and `design.md`. Section references (§) point to `design.md`; `R5.9` means Requirement 5, criterion 9.

## How we work (from `.kiro/steering/rules.md`)

- **One task at a time. Stop for your review after each task.** I do not start the next one until you say so.
- **Definition of done, for every task:**
  - The task's tests exist and pass with `pytest -m "not slow"` (frontend: `vitest run`). Tests marked `slow` are run locally where the task says so.
  - Every module the task adds has tests.
  - Any new environment variable is added to `.env.example` and to `config.py` in the same task (R6.13).
  - Non-obvious choices get a short comment or a line in `design.md`. No unused abstractions or features.
  - No secrets in the repo.
  - I report what I actually ran and its real output. No metric appears anywhere unless a real run produced it.
- **Task order is fixed by §9:** retrieval and its evaluation come before any generation work.

Checkboxes are ticked only after you approve the task.

---

## A. Foundations and retrieval (no LLM anywhere)

- [x] **Task 1 — Scaffold, config, `.env.example`, minimal CI**
  - *Goal:* a runnable Python 3.12 project with one source of truth for configuration.
  - *Files:* `backend/pyproject.toml` (base deps, `dev` extra, `models` extra), `backend/askact/__init__.py`, `backend/askact/config.py` (pydantic-settings; the only place env vars are read; every variable in §4, with the two relevance thresholds defaulting to `None` until tuned), pytest config with the `slow` marker, `backend/tests/test_config.py`, `.env.example` (every variable in §4 documented, including the proxy-hop assumption and the `LOG_QUESTIONS` warning), `.github/workflows/ci.yml` (backend job only: Python 3.12, `pip install -e ".[dev]"`, `pytest -m "not slow"` with stub/fake env). Extend `.gitignore` only if something new needs ignoring.
  - *Tests:* defaults load; env overrides work; invalid values (negative cap, bad JSON price table) raise clear errors; thresholds are `None` when unset; `.env.example` lists exactly the variables `config.py` defines (a test compares the two).
  - *Covers:* R6.13 (initial), R6.4 (backend half), `rules.md` config/secrets rules.
  - *Review focus:* the variable list and defaults in §4; CI file.

- [x] **Task 2 — Section ids and data models**
  - *Goal:* the shared vocabulary every later module uses.
  - *Files:* `backend/askact/models.py` (`SectionId` parse/format/normalise, `Chunk`, `ScoredChunk`, `RetrievalResult`), `backend/tests/test_models.py`.
  - *Tests:* `article:5`, `recital:12`, `annex:III` round-trip; normalisation of messy forms (`Article 5`, `annex:iii`, whitespace); rejection of invalid ids; chunk-id format `article:5#2`; a chunk exposes its parent section id.
  - *Covers:* R1.8 (metadata shape), supports R3.5 and R5.3.

- [x] **Task 3 — Ingestion: fetch, cache, fallback**
  - *Goal:* get the source file onto disk, or fail clearly.
  - *Files:* `backend/askact/ingest/fetch.py`, `backend/tests/ingest/test_fetch.py`.
  - *Behaviour:* reuse `data/raw/ai-act-oj-2024-1689.html` unless `--force`; otherwise GET `SOURCE_URL` with `httpx` (descriptive User-Agent, timeout); on failure or an invalid response, use the manually downloaded file: `--source-file` if given, otherwise `SOURCE_FALLBACK_PATH`, which defaults to `data/raw/ai-act-oj-2024-1689.manual.html` (a different name from the cache file, so a fetch can never overwrite a manual download); if neither works, raise an error that the CLI turns into a non-zero exit with nothing written. Return the file path and its `sha256`.
  - *Tests (`httpx.MockTransport`, no network):* cache reuse; `--force` refetches; HTTP error falls back; non-HTML/empty response falls back; with no path configured and the fetch failing, the default manual file in `data/raw/` is used; an explicit `--source-file` wins over the default; nothing available → error.
  - *Covers:* R1.1, R1.2 (source URL), R1.3, R1.4, R1.13 (fetch half).

- [x] **Task 4 — Ingestion: parser, fixture, pin check**
  - *Goal:* turn the Official Journal HTML into recitals, articles and annexes.
  - *Files:* `backend/askact/ingest/parse.py`, `backend/tests/fixtures/mini_act.html` (3 recitals, 2 articles one of them long, 1 annex, same markup as the real page), `backend/tests/ingest/test_parse.py`.
  - *Behaviour:* recitals from `#rct_N`, articles from `#art_N` (number from `p.oj-ti-art`, title from `.eli-title p.oj-sti-art`), annexes from `#anx_ROMAN` (§3.1 table); table rows flattened to `"(a) text"` lines; pin check: document `<title>` starts with `L_202401689EN`, and parsed counts equal `EXPECTED_COUNTS`.
  - *Real-file step:* run the parser on the real file (fetched, or the manually downloaded file in `data/raw/` if the fetch fails; Task 3 owns that fallback, and the same pin check applies to both) and **report the actual counts to you**. `EXPECTED_COUNTS` is set from that run only after you confirm the numbers. The raw-marker counts seen in the design probe (180 / 113 / 13) are not assumed to be the answer.
  - *Tests:* fixture counts asserted; article number/title extraction; table flattening; pin check rejects a wrong `<title>` and a wrong count (including when the file came from the manual fallback); the full-source count test runs only if the real file exists, else `pytest.skip` with a reason.
  - *Covers:* R1.5 (pin check), R1.6, R1.7, R1.11, R1.12, R1.13 (parse half).

- [x] **Task 5 — Ingestion: chunker and token-length check**
  - *Goal:* citable, bounded-size chunks that carry their metadata.
  - *Files:* `backend/askact/ingest/chunk.py`, `backend/tests/ingest/test_chunk.py`, a `slow` test in the same folder.
  - *Behaviour:* one algorithm for recitals, articles and annexes (35 of the 180 recitals exceed the limit, so recitals are not always one chunk): blocks packed whole up to `CHUNK_MAX_CHARS` measured on the embedded string (header + text); an oversize block is split at line, then sentence, then word boundaries; no overlap; stable `chunk_id`; every chunk keeps number, title, parent section id; the header-prefixed embed string is built in one place (§2).
  - *Tests:* every chunk ≤ limit; long fixture article yields several chunks, each with the article number and title; ids stable across runs; recital stays whole. **`slow`:** over the real source, tokenize every chunk's embed string (header + text, special tokens included) with the real embedder's tokenizer and assert none exceeds the model's max sequence length (read from the loaded model); failure lists the chunk ids. Run it locally and report the result; if it fails, lower `CHUNK_MAX_CHARS` and rerun.
  - *Covers:* R1.8, R1.9.

- [x] **Task 6 — Embedders (stub and real)**
  - *Goal:* one small interface with a deterministic stub for tests/CI and the real local model.
  - *Files:* `backend/askact/embedders.py` (`HashEmbedder`, `SentenceTransformerEmbedder`, selected by `EMBEDDER=stub|real`), `backend/tests/test_embedders.py`.
  - *Tests:* stub is deterministic, L2-normalised, fixed dimension, and cosine is higher for texts that share words; the stub needs no download; a `slow` test loads the real model, embeds the fixture, and checks shape and normalisation. The query-instruction prefix for BGE is applied to queries only.
  - *Covers:* R2.2 (local, no-cost embeddings), R6.4 (stub embedder).

- [ ] **Task 7 — Index artifacts and the `ingest` CLI**
  - *Goal:* `python -m askact.ingest` builds the index end to end, and the API can later check it.
  - *Files:* `backend/askact/index.py`, `backend/askact/ingest/build.py`, `backend/askact/ingest/__main__.py`, `backend/tests/test_index.py`, `backend/tests/ingest/test_build.py`.
  - *Behaviour:* write `chunks.jsonl`, `embeddings.npy`, `manifest.json` atomically (temp dir then rename); manifest records the pinned source (regulation, CELEX, OJ reference, URL, retrieval date, source `sha256`), counts, chunking params, embedding model, schema version and `build_hash = sha256(source_sha256 + chunking params + embedding model)`; skip re-embedding when the hash matches; `load_index()` raises a clear "run ingestion" error when missing or stale; the CLI prints recital/article/annex/chunk counts and exits non-zero on any failure with no partial output.
  - *Tests (fixture + stub embedder):* round-trip; hash changes with source, each chunking param, and model name; unchanged hash skips embedding; stale and missing index errors; a failure mid-build leaves no partial index; CLI exit codes and printed counts.
  - *Real-file step:* run the CLI on the real source with the real model and report the printed counts and chunk total.
  - *Covers:* R1.5 (recording), R1.10, R1.13, R2.6, R2.7, R2.8.

- [ ] **Task 8 — Retrieval: BM25, dense, RRF hybrid**
  - *Goal:* three of the four retrieval configs.
  - *Files:* `backend/askact/retrieval.py` (`Retriever.search(query, config, k)`), `backend/tests/test_retrieval.py`.
  - *Behaviour:* BM25Okapi over lowercase word tokens of the embed string; dense = `embeddings @ q`; hybrid = RRF (k=60) over the top `CANDIDATES` of each; results carry scores and full citation metadata including the parent section id.
  - *Tests (fixture index + stub embedder):* BM25 finds an exact-term passage; dense ordering with the stub; RRF arithmetic on a hand-computed example; result shape and metadata; `k` respected.
  - *Covers:* R2.1, R2.2, R2.3, R2.5 (three configs).

- [ ] **Task 9 — Retrieval: reranker and relevance score**
  - *Goal:* the fourth config and the score the relevance gate will use.
  - *Files:* `backend/askact/rerankers.py` (`OverlapReranker` stub, cross-encoder real, selected by `RERANKER`), updates to `retrieval.py`, tests.
  - *Behaviour:* `hybrid+rerank` reranks the top `RERANK_CANDIDATES` fused candidates; with `RERANKER_ENABLED=false` the reranker is never loaded or called; every result carries `relevance` and `relevance_kind`: the reranker score of rank 1 for `hybrid+rerank`, otherwise the dense cosine of the rank-1 chunk, including for `bm25` and `hybrid`.
  - *Tests:* reranker order applied and not called when disabled (spy); `relevance_kind` is `cosine` for `bm25`/`dense`/`hybrid` and `rerank` for `hybrid+rerank`; cosine for a `bm25` result is computed from the embedder, not from the BM25 score; a `slow` test runs the real reranker on the fixture.
  - *Covers:* R2.4, R2.5 (all four configs), R3.8 (score definition).

---

## B. Retrieval-only evaluation (still no LLM)

- [ ] **Task 10 — Eval schema and loader**
  - *Goal:* `eval/questions.yaml` is validated strictly, so no score is ever computed from bad input.
  - *Files:* `backend/askact/evaluation/schema.py`, `backend/tests/evaluation/test_schema.py`.
  - *Behaviour:* pydantic entries with `id`, `question`, `split` (`dev|test`, required), `category`, `expected_ids` (section ids; empty exactly for `out_of_scope`); unique ids; errors name the offending entry and the CLI exits non-zero.
  - *Tests:* valid file loads; missing file, malformed YAML, missing `split`, bad `category`, chunk-id style `expected_ids`, non-empty `expected_ids` on `out_of_scope`, empty on in-scope, duplicate ids — each fails with a message naming the entry.
  - *Covers:* R5.1, R5.2, R5.3, R5.4.

- [ ] **Task 11 — Retrieval metrics and report (`--no-generation`)**
  - *Goal:* hit@k and MRR per retrieval config and split, with no LLM involved.
  - *Files:* `backend/askact/evaluation/metrics.py`, `backend/askact/evaluation/run.py` (`--k`, `--split dev|test|both`, `--configs`, `--no-generation`), `backend/tests/evaluation/test_metrics.py`, `test_run.py`.
  - *Behaviour:* hit = a retrieved chunk whose parent section is in `expected_ids`; hit@k and MRR over in-scope entries only; `out_of_scope` excluded; per config and per split; prints a Markdown table plus a metadata block (date, `k`, models, index `build_hash`, question counts per split) and writes `eval/results.md`; also prints, per config and split, the range of the relevance score for in-scope vs `out_of_scope` questions; default `--split dev`, and a `test` run is labelled FINAL; no table is produced on a validation error.
  - *Tests:* hand-computed cases covering a multi-id question (any one is a hit), several chunks of one section, no hit in the list (MRR 0), `out_of_scope` excluded, both splits reported, `k` reported; run on the fixture index with stubs.
  - *Covers:* R5.5, R5.6, R5.11 (retrieval columns), R5.12 (table written to file), R5.13.

> **Checkpoint — your turn.** You write `eval/questions.yaml` (hand-written questions, `dev`/`test` split). I then run the retrieval-only eval on `dev` with the real models and show you the real output. We review those numbers together before any generation work starts.

---

## C. Generation and safety

- [ ] **Task 12 — LLM interface, fake provider, prompt, response parsing**
  - *Goal:* the provider-agnostic seam, testable without any network.
  - *Files:* `backend/askact/generation/llm.py` (`LLMReply`, `complete(system, user, max_output_tokens)`, `FakeLLM` that scripts replies and counts calls, `get_llm(settings)` factory keyed on `LLM_PROVIDER`), `backend/askact/generation/prompt.py` (system prompt, passages labelled by section id), response parser, tests.
  - *Behaviour:* the parser reads `{answer, citations, covered}` JSON, tolerating a code fence; anything else raises `llm_bad_output`; the factory raises `llm_unavailable` for an unknown provider or a missing key/model.
  - *Tests:* prompt labels passages by section id and states the grounding/JSON rules; parser accepts valid and fenced JSON, rejects prose, a missing field and a wrong type; `FakeLLM` scripting and call counting; factory errors.
  - *Covers:* R3.1 (prompt), R3.2, R3.3, R3.9 (selection by env), R3.10 (config errors), R6.12 (max-token parameter in the interface).

- [ ] **Task 13 — Answer pipeline: gate, call, validate**
  - *Goal:* `answer()` implements the grounding rules.
  - *Files:* `backend/askact/generation/answer.py`, tests.
  - *Behaviour (order from §3.3):* choose the threshold by `relevance_kind`; below it → `not_covered` with zero LLM calls; otherwise reserve spend through a small `SpendGuard` protocol (tests use a double; the real ledger arrives in Task 16), call the LLM, parse, drop citations not in the retrieved set, treat `covered=false` or no valid citation as `not_covered`, return sources for cited sections.
  - *Tests (fake LLM):* normal answer with sources; invented citation dropped; `covered=true` with no valid citation → `not_covered`; `covered=false` → `not_covered`; gate refusal makes zero LLM calls and no reservation, for both the cosine and the rerank threshold; a `None` threshold fails with a clear configuration error rather than silently passing.
  - *Covers:* R3.1, R3.4, R3.5, R3.6, R3.7, R3.8, R3.11.

- [ ] **Task 14 — OpenAI adapter**
  - *Goal:* the one real provider (OpenAI, by your choice; no other real adapter is built), plus the proof that the provider is chosen by env var.
  - *Process:* **read OpenAI's current API docs first**; record the doc URL and read date in a comment at the top of the adapter.
  - *Files:* adapter in `generation/llm.py` (or its own module if it grows), tests.
  - *Behaviour:* Chat Completions over `httpx` with the current output-token parameter; `LLM_BASE_URL` override; extract text and usage; auth/limit/model errors → `llm_unavailable`.
  - *Tests:* request building and response/usage parsing via `httpx.MockTransport`; missing usage returns `None` counts; error mapping; **provider switch:** `get_llm(settings)` returns the fake provider for `LLM_PROVIDER=fake` and the OpenAI adapter for `LLM_PROVIDER=openai` with no other change, and rejects any other value (including `anthropic`) with the `llm_unavailable` configuration error. **`slow`/manual live smoke test:** one tiny real call, asserts a non-empty reply and reported usage; skipped without the key. Run once locally if you provide a key, and report the result.
  - *Covers:* R3.9, R3.10, R6.12.

- [ ] **Task 15 — Rate limiter**
  - *Goal:* bound requests per client.
  - *Files:* `backend/askact/guards/ratelimit.py`, tests.
  - *Behaviour:* in-process sliding window per client key; key = socket peer, or the last entry of `TRUSTED_PROXY_HEADER` when set; empty windows pruned; returns `Retry-After`.
  - *Tests (injected clock):* allows up to the limit and refuses the next; window slides; separate keys independent; header parsing takes the last entry and ignores spoofed earlier ones; memory pruned.
  - *Covers:* R6.5.

- [ ] **Task 16 — Spend ledger**
  - *Goal:* the daily cap that fails closed.
  - *Files:* `backend/askact/guards/spend.py` (implements the `SpendGuard` protocol from Task 13), tests.
  - *Behaviour:* SQLite `spend(day, micro_usd)`, UTC day; reserve `(estimated_input_tokens + MAX_OUTPUT_TOKENS) × price` in one `BEGIN IMMEDIATE` transaction and refuse if the total would exceed `DAILY_SPEND_CAP_USD`; reconcile to actual cost on the reservation's day; reservation stands if usage is missing or reconciliation never happens; unreadable/unwritable DB, model missing from `LLM_PRICE_TABLE`, or an unparsable table → refuse; integer micro-dollars, rounded up.
  - *Tests:* reserve and refuse at the cap; reconcile up and down; persistence across a reopened DB; new day starts at zero; call straddling midnight reconciles on the right day; usage missing → reservation stands; unwritable path, missing price entry, bad table → refused; two concurrent reservations cannot both pass when only one fits.
  - *Covers:* R6.6, R6.7, R6.8, R6.9, R6.10.

- [ ] **Task 17 — Request logging**
  - *Goal:* useful operational logs with no question text.
  - *Files:* `backend/askact/request_log.py`, tests.
  - *Behaviour:* one structured line per `/ask` with outcome, retrieval config, latency, and input/output token counts (or null); never the question, answer, passages or client address; exception handlers log the code and exception type only; `LOG_QUESTIONS=true` adds only the question.
  - *Tests (`caplog`):* line contains the expected fields; a distinctive question string never appears in any captured log, including on error paths; the client address never appears; `LOG_QUESTIONS=true` adds only the question.
  - *Covers:* design-level logging decision (§3.4); no requirement.

- [ ] **Task 18 — FastAPI app**
  - *Goal:* wire everything into `POST /ask` and `GET /health`.
  - *Files:* `backend/askact/api.py`, `backend/tests/test_api.py`.
  - *Behaviour:* order of checks from §3.4: length → rate limit → retrieve → gate → spend reservation → LLM → validate → reconcile; response and error shapes from §3.4; startup recomputes the build hash and refuses to serve a missing/stale index (`index_unavailable`); CORS from `CORS_ORIGINS`; uvicorn started with `--no-access-log` and a single worker.
  - *Tests (`TestClient`, all stubs):* answered; `not_covered`; every error code and HTTP status in the §3.4 table (`question_too_long`, `rate_limited` with `Retry-After`, `spend_cap_reached`, `llm_unavailable`, `llm_bad_output`, `index_unavailable`); an over-length question never reaches retrieval or the LLM; a rate-limited request never reaches the LLM; `/health`; CORS headers.
  - *Covers:* R2.8 (API side), R3.10, R3.11, R6.11, and the integration of R6.5–R6.9.

---

## D. Generation evaluation

- [ ] **Task 19 — Eval: generation metrics, cache, call cap**
  - *Goal:* refusal accuracy and citation precision/recall, bounded in cost.
  - *Files:* `backend/askact/evaluation/cache.py`, extensions to `metrics.py` and `run.py`, tests; add `eval/.cache/` to `.gitignore`.
  - *Behaviour:* run the real `answer()` pipeline per config; refusal accuracy over `out_of_scope` (refused = gate fires, or `covered=false`, or all citations dropped); citation precision and recall over in-scope entries (no citations → 0 for both); cache keyed by `sha256(provider, model, system, user, max_output_tokens)` in JSON lines; `EVAL_MAX_LLM_CALLS` counts only uncached calls and aborts before exceeding it, printing no generation metrics from an incomplete run; both splits reported; table extended and written to `eval/results.md`.
  - *Tests (fake LLM):* hand-computed precision/recall including multi-id and empty citations; refusal accuracy for each refusal path; cache hit avoids a call; a changed question, passage or model misses the cache; the cap aborts cleanly with no partial generation numbers; `--no-generation` still works unchanged.
  - *Covers:* R5.7, R5.8, R5.9, R5.11.

- [ ] **Task 20 — Tune thresholds on dev; final test run** *(needs your LLM key and budget approval)*
  - *Goal:* set the two relevance thresholds honestly and produce the final numbers.
  - *Process:* using the score ranges printed by the eval, choose `RELEVANCE_THRESHOLD_COSINE` and `RELEVANCE_THRESHOLD_RERANK` from **`dev` results only**; record the chosen values and the dev evidence; set them as the documented values in `.env.example`/`config.py`; then run the `test` split **once** as the final result and write `eval/results.md`. Before any paid run I tell you the maximum number of calls and an estimated cost for your approval.
  - *Tests:* none new in code; the verification is the real run output, which I show you unedited. If the dev set is too small to separate in-scope from out-of-scope scores, I report that rather than force a threshold.
  - *Covers:* R3.8 (tuned thresholds), R5.10, R5.11 (final), R5.12.

---

## E. UI and delivery

- [ ] **Task 21 — Next.js chat page**
  - *Goal:* the minimal UI.
  - *Files:* `frontend/` (App Router + TypeScript, `app/page.tsx`, `lib/api.ts`, plain CSS), vitest + Testing Library tests.
  - *Behaviour:* question box; Q&A turns; sources panel for the latest answer (section label, title, text); loading state with submit disabled; distinct `not_covered` and error states showing the server's message; always-visible notices: "Not legal advice." and the Official Journal 12 July 2024 / Regulation (EU) 2026/1744 notice; backend URL from `NEXT_PUBLIC_API_URL`. **Answer text, source text and server error messages are rendered as plain text** through ordinary React text nodes (`white-space: pre-wrap` keeps line breaks); never as HTML or Markdown, and `dangerouslySetInnerHTML` is not used anywhere.
  - *Tests (`fetch` mocked):* each state; submit disabled while loading; sources panel contents; both notices present in every state. **Plain-text rendering:** an answer, a source passage and an error message containing `<img src=x onerror=alert(1)>`, `<script>alert(1)</script>` and `<b>bold</b>` appear literally as text and create no `img`, `script` or `b` elements in the DOM; and a source-scan test fails if `dangerouslySetInnerHTML` appears anywhere under `frontend/` (excluding `node_modules` and `.next`).
  - *Covers:* R4.1–R4.6.

- [ ] **Task 22 — Dockerfiles and compose**
  - *Goal:* `docker compose up` starts everything.
  - *Files:* `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, `.dockerignore`, notes in `.env.example`.
  - *Behaviour (§3.7):* backend image installs the `models` extra, copies the source and `data/raw/` if present, runs `python -m askact.ingest` at build time, downloads the reranker model at build time, sets `HF_HUB_OFFLINE=1`; named volume for the spend DB; healthcheck; UI built with `NEXT_PUBLIC_API_URL` as a build arg (documented as build-time and needing the public backend URL, plus `CORS_ORIGINS`, for deployment); `.env` never copied into an image.
  - *Verification:* run `docker compose build` and `up`, hit `/health` and `/ask`, and check that the running backend makes no EUR-Lex request and no model download. If Docker is not available in this environment I say so explicitly and list what remains unverified rather than claim success.
  - *Measurements (reported to you unedited, then used in Task 24):* **backend image size** (`docker image ls`, uncompressed size as Docker reports it) and **idle memory use** (`docker stats --no-stream` on the backend container once it is healthy and has received no requests, after a fixed settling time that I state). I record the exact commands, the date, the Docker version, and the settings that affect memory (notably `RERANKER_ENABLED` and the embedding/reranker models). If Docker is unavailable here, these are marked "not measured" and I do not estimate them.
  - *Covers:* R6.1, R6.2, R6.10 (volume), R6.13.

- [ ] **Task 23 — CI: frontend job**
  - *Goal:* complete the pipeline.
  - *Files:* `.github/workflows/ci.yml` (add `frontend` job: Node LTS, `npm ci`, `vitest run`, `next build`).
  - *Verification:* the workflow runs the backend job with stubs and fake LLM, needs no secrets and makes no test-time network calls to external services; I report the actual CI run result after you push, or state that I could not observe it.
  - *Covers:* R6.4, R6.3.

- [ ] **Task 24 — README**
  - *Goal:* the front door of the project.
  - *Files:* `README.md`.
  - *Content:* architecture diagram (from §1); quick start; configuration pointer to `.env.example`, including the build-time `NEXT_PUBLIC_API_URL` note; the results table pasted from `eval/results.md` (never typed); question counts per split; a statement that the questions were hand-written; a note that small samples mean differences between configs may be noise; the **backend image size and idle memory use** exactly as measured in Task 22, with the measurement method, date and the settings used (or "not measured" if Task 22 could not measure them); the optional manual faithfulness review (N reviewed, how many faithful) only if you actually did one; a limitations section: single pinned document, English Official Journal text only (the HTML rendition, which EUR-Lex says is for information and not the authentic signed PDF), later amendments not reflected and naming Regulation (EU) 2026/1744, retrieval/parsing limits, not legal advice, single worker, non-goals.
  - *Covers:* R7.1–R7.6, R5.12.

---

## Requirement coverage

| Requirement | Tasks |
|---|---|
| R1 Ingestion | 3, 4, 5, 7 (model shapes in 2) |
| R2 Retrieval | 7 (hash, missing index), 8, 9 |
| R3 Generation | 9 (score), 12, 13, 14, 20 (tuned thresholds) |
| R4 UI | 21 |
| R5 Evaluation | 10, 11, 19, 20 |
| R6 Delivery | 1, 15, 16, 18, 22, 23 |
| R7 README | 24 |
| Design-level (logging, adapter docs and smoke test, token-length test, build-time UI URL) | 17, 14, 5, 21–22 |
