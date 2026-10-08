# Requirements — Ask the EU AI Act

## Introduction

"Ask the EU AI Act" is a small, portfolio-grade Retrieval-Augmented
Generation (RAG) application. It answers natural-language questions about
the EU Artificial Intelligence Act (Regulation (EU) 2024/1689) using the
Official Journal text published on EUR-Lex, and it grounds every answer in retrieved passages with explicit
citations to article/recital/annex numbers. The project is optimized for
clarity and quality of a few well-executed features, not for breadth.

The system has seven capability areas: ingestion, retrieval, generation, a
minimal UI, an evaluation harness, delivery (containerization, tests, CI,
and public-demo safety limits), and documentation (README).

### Scope
In scope: a single document (the EU AI Act), hybrid retrieval with a free
local embedding model, grounded generation with a configurable LLM
provider, a minimal chat UI, a reproducible retrieval-evaluation harness,
one-command delivery with safety limits, and a README.

### Non-goals (explicitly out of scope)
- Authentication / user accounts.
- User-provided uploads.
- Multiple or arbitrary documents.
- Model fine-tuning.

### Glossary
- **Section id**: the identifier of a structural unit of the Act, written `<kind>:<number>`, e.g. `article:5`, `recital:12`, `annex:III`. This is what users cite and what `expected_ids` refer to.
- **Chunk**: a retrievable unit of text with metadata (kind, number, title, stable chunk id). A chunk belongs to exactly one section (its *parent section*); one section may produce several chunks.
- **Hit**: a retrieved chunk whose parent section id is among a question's `expected_ids`.
- **hit@k**: fraction of in-scope eval questions with at least one hit in the top-k retrieved chunks.
- **MRR**: Mean Reciprocal Rank, over in-scope eval questions, of the first hit in the ranked chunk list (0 for a question with no hit in the list).
- **Retrieval config**: a named combination of retrieval settings: `bm25`, `dense`, `hybrid`, `hybrid+rerank`.
- **Split**: the `dev` or `test` partition of the eval questions. `dev` is used for tuning; `test` is the final reported result.
- **Citation precision / recall**: the fraction of cited section ids that are in `expected_ids` / the fraction of `expected_ids` that are cited (see R5).

---

## Requirement 1 — Ingestion

**User story:** As a developer running the project, I want to fetch and
parse the EU AI Act into structured, chunked passages with metadata, so
that retrieval has clean, citable units to work with.

#### Acceptance Criteria
1. WHEN the ingestion script is run THEN the system SHALL fetch the Official Journal text of Regulation (EU) 2024/1689 (EUR-Lex CELEX 32024R1689, as published in the Official Journal on 12 July 2024) from a configurable EUR-Lex source URL into `data/raw/`.
2. WHERE the source text is concerned THEN the system SHALL ingest the English-language Official Journal text as published (not a consolidated version), and the configured URL and any fallback file SHALL be that text; later amendments are therefore not reflected.
3. IF the raw document already exists locally THEN the system SHALL reuse the cached copy and SHALL re-fetch only when a `--force` flag is passed.
4. IF automated fetching is unavailable or fails THEN the system SHALL load a manually downloaded source file from a configurable local path as a fallback, so ingestion can run fully offline.
5. WHEN ingestion runs THEN the system SHALL pin and record the exact source it ingested (Regulation (EU) 2024/1689, its CELEX identifier, the Official Journal publication reference, and the retrieval date) into a recorded metadata field in `data/index/`, so results are reproducible against a known version.
6. WHEN parsing the document THEN the system SHALL separate the text into three structural kinds: recitals, articles, and annexes.
7. WHEN an article is parsed THEN the system SHALL capture its article number and title as metadata.
8. WHEN producing chunks THEN the system SHALL attach to each chunk its structural kind, its number (article/recital/annex identifier), its parent section id, its title where one exists, and a stable chunk id.
9. WHEN chunking a long article THEN the system SHALL split it into bounded-size chunks while preserving the article-number and title metadata on every resulting chunk.
10. WHEN ingestion completes THEN the system SHALL write the parsed chunks to `data/index/` in a documented format and SHALL print a summary count of recitals, articles, annexes, and total chunks.
11. WHEN the ingestion module is tested THEN the test suite SHALL assert the parsed recital, article, and annex counts for the small committed fixture (see R6.4), so a parsing regression fails the build.
12. WHEN the real source file is present locally THEN the test suite SHALL additionally assert the full recital, article, and annex counts against the expected values for the pinned Act version; WHEN the real file is absent THEN that test SHALL be skipped with a stated reason rather than failed.
13. IF the source cannot be fetched or parsed (and no fallback file is available) THEN the system SHALL exit non-zero with a clear error message rather than writing a partial index.

---

## Requirement 2 — Retrieval

**User story:** As a user asking a question, I want the system to find the
most relevant passages using both keyword and semantic matching, so that
answers are grounded in the right parts of the Act.

#### Acceptance Criteria
1. WHEN the index is built THEN the system SHALL support lexical retrieval over chunks using BM25.
2. WHEN the index is built THEN the system SHALL support dense retrieval using a local, no-cost embedding model (no paid API calls required to embed or query).
3. WHEN a query is issued in hybrid mode THEN the system SHALL combine BM25 and dense scores into a single ranked list using a documented fusion method.
4. WHEN a reranker flag is enabled THEN the system SHALL re-order the fused top candidates with a reranker model, and WHEN the flag is disabled THEN the system SHALL skip reranking entirely.
5. WHERE a retrieval config is selected (`bm25`, `dense`, `hybrid`, or `hybrid+rerank`) THEN the system SHALL return the top-k chunks with their scores and full citation metadata, including each chunk's parent section id.
6. WHEN embeddings are needed THEN the system SHALL reuse a persisted embedding/index artifact if present and SHALL recompute only when the index is missing or stale, WHERE "stale" is defined as a mismatch between the stored artifact's build hash and a freshly computed hash of (source file content + chunking parameters + embedding model name).
7. WHEN an index artifact is written THEN the system SHALL record its build hash (as defined in criterion 6) alongside it, so staleness can be checked without recomputing embeddings.
8. IF the index artifacts are missing THEN the system SHALL return a clear error instructing the user to run ingestion, rather than crashing.

---

## Requirement 3 — Generation

**User story:** As a user, I want answers that come only from the retrieved
passages and cite their article numbers, so that I can trust and verify
every claim.

#### Acceptance Criteria
1. WHEN generating an answer THEN the system SHALL use only the retrieved passages as evidence and SHALL NOT introduce facts absent from them.
2. WHEN the LLM is called THEN the system SHALL require structured output of the form `{answer, citations, covered}`, where `answer` is text, `citations` is a list of section ids, and `covered` is a boolean stating whether the retrieved passages contain the evidence to answer.
3. IF the LLM output cannot be parsed into that structure THEN the system SHALL return a clear error and SHALL NOT present the raw output as an answer.
4. WHEN an answer is produced THEN the system SHALL cite the section ids (article/recital/annex numbers) of the passages it relied on.
5. WHEN an answer cites a section id THEN every cited id SHALL correspond to a passage present in the retrieved set, and the system SHALL drop or reject any citation that is not in that set.
6. IF the LLM returns `covered: false` THEN the system SHALL respond that the question is "not covered" by the retrieved text rather than guessing.
7. IF the relevance score of the top-ranked candidate is below the relevance threshold that applies to the active retrieval config (criterion 8) THEN the system SHALL return "not covered" WITHOUT calling the LLM.
8. WHERE the relevance gate of criterion 7 is applied THEN the system SHALL use two separately configurable thresholds: `RELEVANCE_THRESHOLD_COSINE` for the `bm25`, `dense`, and `hybrid` configs, compared against the dense cosine similarity between the query and the top-ranked candidate (computed for every one of these configs, including `bm25` and `hybrid`); and `RELEVANCE_THRESHOLD_RERANK` for the `hybrid+rerank` config, compared against the reranker score of the top-ranked candidate. The gate SHALL NOT use a raw BM25 or fused (rank-fusion) score, and both thresholds SHALL be tuned on the `dev` split only (see R5 criterion 10).
9. WHERE the LLM provider is configured via an environment variable THEN the system SHALL route generation to that provider without code changes.
10. IF the configured provider or its API key is missing or invalid THEN the system SHALL return a clear, user-facing error rather than failing silently or fabricating an answer.
11. WHEN returning an answer THEN the system SHALL also return the list of source passages (with metadata) used, so the UI can display them.

---

## Requirement 4 — UI

**User story:** As a visitor, I want a simple chat page that shows the
answer alongside its sources, so that I can read the response and check
where it came from.

#### Acceptance Criteria
1. WHEN a visitor opens the app THEN the system SHALL present a minimal single-page chat interface built with Next.js and TypeScript.
2. WHEN a visitor submits a question THEN the UI SHALL send it to the backend and SHALL display the returned answer.
3. WHEN an answer is displayed THEN the UI SHALL show a sources panel listing the cited passages with their article/recital/annex numbers and titles.
4. WHILE a request is in flight THEN the UI SHALL show a loading indicator and SHALL disable duplicate submission.
5. IF the backend returns an error or a "not covered" result THEN the UI SHALL display that state clearly instead of a blank or broken view.
6. WHEN the chat page is displayed THEN the UI SHALL show a visible "not legal advice" disclaimer AND a visible notice that answers are based on the original Official Journal text of 12 July 2024 and do not reflect later amendments, including Regulation (EU) 2026/1744.

---

## Requirement 5 — Evaluation

**User story:** As the project owner, I want a harness that scores
retrieval and citation quality from a fixed, hand-written question set, so
that I can compare retrieval configurations with reproducible numbers.

#### Acceptance Criteria
1. WHEN the evaluation harness runs THEN the system SHALL read questions and expected references from `eval/questions.yaml` (authored by the project owner).
2. WHERE `eval/questions.yaml` defines each entry THEN each entry SHALL follow this schema: `id` (unique string), `question` (string), `split` (required; one of `dev`, `test`), `expected_ids` (list of section ids), and `category` (one of `direct`, `paraphrase`, `multi`, `out_of_scope`).
3. WHERE `expected_ids` is given THEN its values SHALL be section ids (article/recital/annex ids such as `article:5`), NOT chunk ids, and SHALL be empty exactly for `out_of_scope` entries; the listed sections SHALL all be treated as relevant to the question (not as alternatives), so that a hit requires any one of them to be retrieved and citation recall measures the share of them that the answer cites.
4. IF `eval/questions.yaml` is missing, malformed, or violates the schema THEN the harness SHALL exit non-zero with a clear message naming the offending entry and SHALL NOT emit fabricated scores.
5. WHEN scoring retrieval THEN a hit SHALL be a retrieved chunk whose parent section id matches one of the entry's `expected_ids`, and the harness SHALL compute hit@k and MRR on that basis for each retrieval config (`bm25`, `dense`, `hybrid`, and `hybrid+rerank`).
6. WHEN computing hit@k and MRR THEN the harness SHALL use the in-scope entries (`direct`, `paraphrase`, `multi`) ONLY, excluding `out_of_scope` entries.
7. WHEN evaluating `out_of_scope` entries THEN the harness SHALL report refusal accuracy (the fraction of `out_of_scope` questions the system correctly answers "not covered") as a separate metric.
8. WHEN evaluating in-scope entries THEN the harness SHALL report citation precision (the fraction of the answer's cited section ids that are in the entry's `expected_ids`) and citation recall (the fraction of the entry's `expected_ids` that the answer cites) as separate metrics, distinct from hit@k and MRR, averaged over in-scope entries; an entry whose answer has no citations SHALL contribute 0 to both.
9. WHEN the harness needs LLM answers THEN it SHALL cache them (keyed so that a change to the question, retrieved passages, or model invalidates the entry) and reuse cached answers on re-runs, AND it SHALL enforce a configurable maximum number of uncached LLM calls per run, aborting with a clear message rather than exceeding it and NOT reporting generation metrics from an incomplete run.
10. WHEN retrieval parameters or the relevance thresholds (`RELEVANCE_THRESHOLD_COSINE` and `RELEVANCE_THRESHOLD_RERANK`) are tuned THEN they SHALL be tuned using `dev` split results only; the `test` split SHALL be reported as the final result and SHALL NOT be used to choose parameters.
11. WHEN evaluation completes THEN the harness SHALL report results for both splits separately and SHALL print a per-config results table (hit@k, MRR, refusal accuracy, citation precision, citation recall, per split) in a form that can be pasted into the README.
12. WHEN the README reports metrics THEN those numbers SHALL come only from an actual harness run (never hand-written or assumed).
13. WHERE `k` is configurable THEN the harness SHALL accept the value of `k` and SHALL report which `k` was used.

---

## Requirement 6 — Delivery

**User story:** As someone evaluating the project, I want to start it with
one command, see tests and CI passing, and know the public demo is
protected from runaway cost, so that the project is credible and safe to
host.

#### Acceptance Criteria
1. WHEN `docker compose up` is run THEN the system SHALL start the backend and UI together and SHALL serve the chat app without additional manual steps.
2. WHEN the demo is built THEN the index SHALL be built either at image build time or by an init service, from the pinned source or the fallback file, AND the local embedding and reranker models SHALL be downloaded at build time, so that the running demo needs no live EUR-Lex fetch and no model download.
3. WHEN tests are run with pytest THEN the system SHALL execute unit tests for ingestion, retrieval, generation, and evaluation modules.
4. WHEN code is pushed THEN a GitHub Actions workflow SHALL install dependencies and run the test suite using a small committed text fixture in place of the full Act, a fake LLM provider, a deterministic stub embedder, and a stub reranker (so no model is downloaded), SHALL require no network access to external services and no secrets, and SHALL fail the build on any test failure; tests that use the real embedding or reranker models SHALL be marked `slow` and SHALL be run locally only, not in CI.
5. WHILE the public demo is serving requests THEN the system SHALL rate-limit requests per client, WHERE the client key is defined as the client IP (taken from the configured trusted proxy header when behind a proxy, otherwise the socket peer address).
6. WHILE the public demo is running THEN the system SHALL enforce a daily spend cap, WHERE spend is computed from token usage multiplied by a configurable per-model price table.
7. BEFORE each paid LLM call THEN the system SHALL reserve the worst-case cost of the call, namely (estimated input tokens + the configured maximum output tokens) x the model's price, against the daily total within a single SQLite transaction; AFTER the call THEN the system SHALL reconcile the reservation to the actual cost computed from the reported token usage.
8. IF the reservation would cause the daily total to exceed the cap THEN the system SHALL fail closed: it SHALL NOT make the paid LLM call and SHALL return a clear message.
9. IF the spend database cannot be read or written, OR the configured model has no entry in the price table THEN the system SHALL fail closed in the same way; IF the call completes but actual token usage is not reported, or reconciliation does not happen, THEN the reservation SHALL stand as the amount charged.
10. WHEN the daily spend total is stored THEN it SHALL live in a SQLite database file on a mounted volume, so that the cap survives process and container restarts within the same day, and SHALL reset at the start of each day.
11. WHEN a question is submitted THEN the system SHALL enforce a configurable maximum question length and SHALL reject over-length input with a clear message.
12. WHEN generating an answer THEN the system SHALL enforce a configurable maximum output-token limit on the LLM call.
13. WHEN configuration is needed THEN the repository SHALL provide a `.env.example` documenting every environment variable (LLM provider, model, API key, price table, rate-limit, daily spend cap, spend-counter storage path, trusted proxy header, source URL, fallback source path, reranker flag, `RELEVANCE_THRESHOLD_COSINE`, `RELEVANCE_THRESHOLD_RERANK`, max question length, max output tokens, max eval LLM calls per run) and SHALL keep real secrets out of version control.

---

## Requirement 7 — Documentation (README)

**User story:** As someone evaluating the project, I want a README that
explains the architecture, shows measured results, and is honest about
limitations, so that I can understand and trust the project quickly.

#### Acceptance Criteria
1. WHEN the README is published THEN it SHALL include an architecture diagram showing ingestion, retrieval, generation, UI, and the safety limits, and how they connect.
2. WHEN the README reports results THEN it SHALL include the evaluation results table produced by an actual harness run (per Requirement 5), and SHALL NOT contain hand-written metrics.
3. WHEN the README reports results THEN it SHALL state the number of questions in each split (`dev` and `test`) and SHALL state that the questions were hand-written.
4. WHEN the README reports results THEN it SHALL note that the question sets are small and that differences between configs may therefore be within sampling noise.
5. WHERE the project owner has manually reviewed a sample of generated answers for faithfulness to their retrieved passages THEN the README SHALL report the number of answers reviewed (N), how many were judged faithful, and that the review was manual; those figures SHALL come from the actual review and SHALL NOT be estimated.
6. WHEN the README is published THEN it SHALL include a limitations section stating known constraints (single pinned document, English Official Journal text only, that later amendments are not reflected, naming Regulation (EU) 2026/1744 (the Digital Omnibus on AI) as one such amendment, retrieval/parsing limits, "not legal advice", and the non-goals).

---

## Traceability summary

| Requirement | Capability area | Original brief item |
|-------------|-----------------|---------------------|
| R1 | Ingestion | 1. Ingestion |
| R2 | Retrieval | 2. Retrieval |
| R3 | Generation | 3. Generation |
| R4 | UI | 4. UI |
| R5 | Evaluation | 5. Evaluation |
| R6 | Delivery (compose, pytest, CI, rate limit, spend cap) | 6. Delivery |
| R7 | Documentation (README: diagram, results, limitations) | 5 (results table) and 6 (delivery) |

All non-goals (auth, uploads, multiple documents, fine-tuning) remain out
of scope for every requirement above.
