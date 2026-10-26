# rag-eval-pipeline

A small, fully local Retrieval-Augmented Generation (RAG) system with a
real evaluation harness — built to demonstrate the part of RAG that demos
usually skip: **measuring whether retrieval and answering actually work,
and refusing to answer when the evidence isn't there.**

It runs end-to-end with **no API key and no model download**. Retrieval
uses TF-IDF vectors implemented in pure Python, and answers are extracted
verbatim from the retrieved chunks with citations, so an answer cannot
contain a fact that isn't in the corpus. Every number in this README was
produced by running the eval scripts in `eval/` against the labeled
golden set in this repo.

## What it does

- **Ingests** plain-text and Markdown documents and splits them with three
  interchangeable chunking strategies: fixed-size, recursive
  (separator-aware), and sentence-boundary semantic chunking guided by
  local TF similarity.
- **Retrieves** chunks by TF-IDF cosine similarity, with metadata
  filtering applied before ranking.
- **Answers** extractively: the best-matching sentences from the retrieved
  chunks, returned word-for-word with `[chunk-id]` citations. When
  retrieval confidence is below a tuned threshold, it **abstains** with
  "I don't have enough evidence in the indexed documents to answer that."
- **Evaluates** itself on a labeled golden set of 26 queries (22
  answerable, 4 deliberately unanswerable): Recall@k, Precision@k, MRR,
  hit rate, citation precision, answer rate, and abstention accuracy.
- **Serves** the pipeline over FastAPI (`/ingest`, `/query`, `/evaluate`,
  `/health`), with interactive docs at `/docs`.

An optional `LLMGenerator` class in `app/generator.py` marks the seam
where a generative model would plug in (it requires an external LLM
client and API key). It is not used by the default pipeline.

## Architecture

```
documents (.txt / .md)
        │
        ▼
  app/ingest.py ──► chunkers (fixed | recursive | semantic)
        │                        │
        ▼                        ▼
  app/retriever.py ◄── app/embeddings.py (TF-IDF / hashing, sparse vectors)
        │
        ▼  top-k chunks + scores
  app/confidence.py ──► below threshold? ──► abstain
        │
        ▼
  app/generator.py (extractive, cited sentences)
        │
        ▼
  app/pipeline.py ──► app/main.py (FastAPI) ──► eval/ (golden-set scoring)
```

The chunking / embedding / retrieval / evaluation core depends only on
the Python standard library. FastAPI and Pydantic are used solely by the
service layer; pytest and httpx solely by the tests.

## Measured results

Golden set: **26 queries** over the 6 short documents in `data/corpus`
(22 answerable, each labeled with its one relevant document; relevance is
judged at document level). Abstention threshold: 0.15 (tuned on this
golden set — see Limitations).

**Retrieval** (recursive chunker, 30 chunks indexed):

| Metric | Value |
|---|---|
| Recall@1 | 0.9545 |
| Recall@3 | 1.0000 |
| Recall@5 | 1.0000 |
| Precision@5 | 0.6591 |
| MRR | 0.9773 |

**Generation behaviour** (extractive generator):

| Metric | Value |
|---|---|
| Answer rate (answerable queries answered) | 1.0000 (22/22) |
| Citation precision (expected doc among citations) | 1.0000 |
| Abstention accuracy (unanswerable queries refused) | 1.0000 (4/4) |

**Chunker comparison** (same corpus, same queries):

| Chunker | Chunks | Recall@5 | MRR | Precision@5 |
|---|---|---|---|---|
| fixed | 23 | 1.0000 | 1.0000 | 0.6000 |
| recursive | 30 | 1.0000 | 0.9773 | 0.6591 |
| semantic | 50 | 1.0000 | 0.9773 | 0.7091 |

Note on reading this table: on a corpus this small, every chunker finds the
right document within five results, so Recall@5 does not separate them.
The differences show up in MRR, precision, and chunk count — the semantic
chunker buys the best precision with nearly double the chunks. A larger,
harder corpus would be needed to rank the chunkers decisively, and the
harness is set up to do exactly that.

**Tests:** 41 pytest tests, all passing — chunkers, embeddings, retriever,
metrics (exact values), generator/abstention, golden-set regression
guards, and the API via `TestClient`.

Full per-query results are committed in `eval/results/`
(`retrieval_results.json`, `generation_results.json`,
`chunker_comparison.json`) and can be regenerated at any time.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the test suite
python -m pytest

# Run the evaluations (writes JSON into eval/results/)
python -m eval.evaluate_retrieval
python -m eval.evaluate_generation
python -m eval.compare_chunkers

# Serve the API (indexes data/corpus on startup)
uvicorn app.main:app --reload
# -> http://127.0.0.1:8000/docs
```

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does mean reciprocal rank measure?", "k": 5}'
```

## Project structure

```
rag-eval-pipeline/
├── app/
│   ├── main.py            # FastAPI service (/ingest, /query, /evaluate, /health)
│   ├── pipeline.py        # Assembled pipeline: ingest -> index -> retrieve -> answer
│   ├── ingest.py          # .txt/.md loading and chunking orchestration
│   ├── chunkers/
│   │   ├── fixed.py       # Fixed-size windows with overlap
│   │   ├── recursive.py   # Separator-aware recursive splitting
│   │   └── semantic.py    # Sentence grouping by local TF similarity
│   ├── embeddings.py      # TF-IDF and hashing embedders (sparse vectors, stdlib only)
│   ├── retriever.py       # In-memory cosine retrieval + metadata filtering
│   ├── generator.py       # Extractive cited answering; optional LLM seam
│   ├── confidence.py      # Retrieval confidence + abstention threshold
│   ├── memory.py          # Per-session conversation memory
│   └── schemas.py         # Shared dataclasses (Document, Chunk, Answer, ...)
├── data/
│   └── corpus/            # 6 short Markdown documents used by the demo + eval
├── eval/
│   ├── golden_set.json    # 26 labeled queries (22 answerable, 4 unanswerable)
│   ├── metrics.py         # Recall@k, Precision@k, MRR, hit rate
│   ├── evaluate_retrieval.py
│   ├── evaluate_generation.py   # Citation precision, answer rate, abstention accuracy
│   ├── compare_chunkers.py
│   └── results/           # Committed JSON output of the three eval scripts
├── tests/                 # 41 tests: chunkers, retriever, metrics, generator, API, regression
├── conftest.py
├── requirements.txt       # Pinned; core pipeline needs none of it (stdlib only)
└── LICENSE                # MIT
```

## Limitations

- **TF-IDF is not a neural embedding model.** It matches on shared
  vocabulary, so it is weak on paraphrases and synonyms ("car" vs
  "automobile"). The embedder interface in `app/embeddings.py` is the
  plug-in point for a real embedding model.
- **The "semantic" chunker is a lightweight stand-in.** It groups
  sentences by TF cosine similarity between neighbours, not by learned
  embedding similarity.
- **In-memory index.** Retrieval is brute-force cosine over all chunks.
  Right-sized for a corpus of this scale; a vector database would replace
  `app/retriever.py` for a large corpus.
- **No PDF ingestion.** Only `.txt` and `.md` are supported. (An earlier
  version of this README claimed PDF support, ChromaDB, and LLM-as-judge
  evaluation that were never implemented. They have been removed rather
  than left as claims.)
- **The golden set is small and its labels are document-level** — 26
  queries, one relevant document each, over 6 short documents written for
  this project. The metrics demonstrate the harness, not production-grade
  retrieval quality.
- **The abstention threshold was tuned on the same golden set it is
  measured on.** The clean separation (answerable queries score ≥ 0.20
  confidence, unanswerable ≤ 0.09) is encouraging, but a held-out set
  would be needed for an unbiased estimate.
- **Extractive answers are verbatim source sentences** — guaranteed
  grounded, but less fluent and less synthesised than LLM generation.

## License

MIT — see `LICENSE`.
