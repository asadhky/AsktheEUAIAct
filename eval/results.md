# Evaluation: retrieval and generation

- Date: 2026-10-13 (UTC)
- k: 5 (chunks; hit@5 and MRR look at the top 5 chunks of each question)
- Embedding model: BAAI/bge-small-en-v1.5
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- Index: 495 chunks, build hash 6a5aac7af9cc03a7fc5d1755aa4955f68b24f1404c7eed8a9e6b88bf95c0022e
- Questions in dev: 23 in-scope, 6 out_of_scope
- Questions in test: 17 in-scope, 4 out_of_scope
- Generation: LLM openai / gpt-4o-mini, at most 500 output tokens per answer
- Relevance thresholds (the gate): cosine 0.69, rerank 2.5
- LLM calls: 59 made (limit 90), 83 answered from the cache. Each request is answered once and its reply is kept (`eval/.cache/`), so running again reproduces these numbers without new calls.

## dev split

The split that parameters and thresholds are tuned on.

hit@5 and MRR are over the 23 in-scope questions only. MRR is the mean of 1 / (rank of the first chunk from an expected section), counting 0 when there is none in the top 5.

Refusal accuracy is over the 6 out_of_scope questions: the share the system answered "not covered" (the relevance gate refused, or the model said so, or none of its citations named a retrieved section). Citation precision and recall are means over the 23 in-scope questions of: cited sections that are expected / cited sections, and cited sections that are expected / expected sections. An answer with no citations, including a refusal, scores 0 on both.

| Config | hit@5 | MRR | Refusal accuracy | Citation precision | Citation recall |
|---|---|---|---|---|---|
| bm25 | 0.696 (16/23) | 0.415 | 1.000 (6/6) | 0.457 | 0.457 |
| dense | 0.826 (19/23) | 0.563 | 1.000 (6/6) | 0.587 | 0.638 |
| hybrid | 0.783 (18/23) | 0.594 | 1.000 (6/6) | 0.533 | 0.630 |
| hybrid+rerank | 0.913 (21/23) | 0.690 | 1.000 (6/6) | 0.481 | 0.587 |

Replies the model gave that could not be read (scored as no answer, not as a refusal): bm25 0, dense 0, hybrid 0, hybrid+rerank 0

Relevance score of the top-ranked chunk, lowest to highest (the gate's input; cosine and rerank scores are on different scales):

| Config | Scale | in-scope questions | out_of_scope questions |
|---|---|---|---|
| bm25 | cosine | 0.492 to 0.873 (n=23) | 0.464 to 0.653 (n=6) |
| dense | cosine | 0.611 to 0.880 (n=23) | 0.547 to 0.737 (n=6) |
| hybrid | cosine | 0.611 to 0.880 (n=23) | 0.508 to 0.737 (n=6) |
| hybrid+rerank | rerank | -11.107 to 8.975 (n=23) | -10.951 to 3.193 (n=6) |

## test split: FINAL

The final result. It must not be used to choose parameters or thresholds (R5.10).

hit@5 and MRR are over the 17 in-scope questions only. MRR is the mean of 1 / (rank of the first chunk from an expected section), counting 0 when there is none in the top 5.

Refusal accuracy is over the 4 out_of_scope questions: the share the system answered "not covered" (the relevance gate refused, or the model said so, or none of its citations named a retrieved section). Citation precision and recall are means over the 17 in-scope questions of: cited sections that are expected / cited sections, and cited sections that are expected / expected sections. An answer with no citations, including a refusal, scores 0 on both.

| Config | hit@5 | MRR | Refusal accuracy | Citation precision | Citation recall |
|---|---|---|---|---|---|
| bm25 | 0.824 (14/17) | 0.618 | 1.000 (4/4) | 0.382 | 0.529 |
| dense | 0.706 (12/17) | 0.593 | 1.000 (4/4) | 0.525 | 0.647 |
| hybrid | 0.824 (14/17) | 0.608 | 1.000 (4/4) | 0.539 | 0.735 |
| hybrid+rerank | 0.765 (13/17) | 0.632 | 1.000 (4/4) | 0.407 | 0.588 |

Replies the model gave that could not be read (scored as no answer, not as a refusal): bm25 0, dense 0, hybrid 0, hybrid+rerank 0

Relevance score of the top-ranked chunk, lowest to highest (the gate's input; cosine and rerank scores are on different scales):

| Config | Scale | in-scope questions | out_of_scope questions |
|---|---|---|---|
| bm25 | cosine | 0.579 to 0.891 (n=17) | 0.294 to 0.685 (n=4) |
| dense | cosine | 0.648 to 0.891 (n=17) | 0.385 to 0.743 (n=4) |
| hybrid | cosine | 0.622 to 0.891 (n=17) | 0.341 to 0.743 (n=4) |
| hybrid+rerank | rerank | -4.176 to 9.921 (n=17) | -11.052 to 4.398 (n=4) |
