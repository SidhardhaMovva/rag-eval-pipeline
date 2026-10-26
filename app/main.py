"""FastAPI service for the RAG eval pipeline.

Endpoints:
- ``GET  /health``   — liveness + index size
- ``POST /ingest``   — add one document (text) to the index
- ``POST /query``    — grounded answer with citations, or an abstention
- ``POST /evaluate`` — run the golden-set retrieval evaluation and return
  Recall@k / MRR / hit rate (requires the app to have been created with a
  corpus directory and a golden-set path, as the default app is)

Run: ``uvicorn app.main:app --reload`` — interactive docs at ``/docs``.
The default app indexes ``data/corpus`` on startup if it exists.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.pipeline import RagPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "corpus"
DEFAULT_GOLDEN_SET = REPO_ROOT / "eval" / "golden_set.json"


class IngestRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = None
    filters: dict | None = None


class EvaluateRequest(BaseModel):
    k: int = Field(default=5, ge=1, le=20)


class RetrievedOut(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    abstained: bool
    retrieved: list[RetrievedOut]


def create_app(
    corpus_dir: Path | None = DEFAULT_CORPUS_DIR,
    golden_set_path: Path | None = DEFAULT_GOLDEN_SET,
) -> FastAPI:
    app = FastAPI(title="rag-eval-pipeline", version="1.0.0")
    pipeline = RagPipeline()
    if corpus_dir is not None and Path(corpus_dir).is_dir():
        pipeline.load_corpus(corpus_dir)
    app.state.pipeline = pipeline
    app.state.corpus_dir = corpus_dir
    app.state.golden_set_path = golden_set_path

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **pipeline.stats()}

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict:
        total = pipeline.add_text(req.text, req.doc_id)
        return {"doc_id": req.doc_id, "total_chunks": total, **pipeline.stats()}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        answer = pipeline.query(
            req.query, k=req.k, filters=req.filters, session_id=req.session_id
        )
        return QueryResponse(
            answer=answer.text,
            citations=answer.citations,
            confidence=answer.confidence,
            abstained=answer.abstained,
            retrieved=[
                RetrievedOut(
                    chunk_id=r.chunk.chunk_id,
                    doc_id=r.chunk.doc_id,
                    source=r.chunk.source,
                    score=round(r.score, 6),
                    text=r.chunk.text,
                )
                for r in answer.retrieved
            ],
        )

    @app.post("/evaluate")
    def evaluate(req: EvaluateRequest) -> dict:
        if golden_set_path is None or not Path(golden_set_path).is_file():
            raise HTTPException(status_code=404, detail="golden set not configured")
        if corpus_dir is None or not Path(corpus_dir).is_dir():
            raise HTTPException(status_code=404, detail="corpus not configured")

        # Import here so the app module stays importable without eval/.
        from eval.evaluate_retrieval import evaluate_retrieval

        golden = json.loads(Path(golden_set_path).read_text(encoding="utf-8"))
        fresh = RagPipeline()
        fresh.load_corpus(corpus_dir)
        return evaluate_retrieval(fresh, golden, k=req.k)

    return app


app = create_app()
