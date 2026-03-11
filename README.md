# rag-eval-pipeline

Document Intelligence RAG System with Evaluation Pipeline — PDF ingestion, chunking, embedding, ChromaDB retrieval, LLM generation with citations, and quantitative evaluation (MRR, Recall@5, LLM-as-judge).

## Structure

```
rag-eval-pipeline/
├── app/
│   ├── main.py              # FastAPI app with Swagger
│   ├── ingest.py            # PDF ingestion pipeline
│   ├── chunkers/
│   │   ├── fixed.py         # Fixed-size chunking
│   │   ├── recursive.py    # Recursive chunking
│   │   └── semantic.py     # Semantic chunking
│   ├── embeddings.py        # Embedding generation + model switching
│   ├── retriever.py         # ChromaDB retrieval + metadata filtering
│   ├── generator.py        # LLM answer generation with citations
│   ├── memory.py            # Multi-turn conversation memory
│   ├── confidence.py       # Confidence scoring + refusal logic
│   └── schemas.py           # Pydantic models
├── eval/
│   ├── golden_set.json      # 200+ test queries with ground truth
│   ├── evaluate_retrieval.py
│   ├── evaluate_generation.py
│   ├── compare_chunkers.py
│   └── results/
├── tests/
├── data/
│   └── sample_pdfs/
├── requirements.txt
└── README.md
```

## How to run

*(TODO: docker-compose or local setup instructions.)*

## API documentation

*(TODO: Link to Swagger after implementation.)*
