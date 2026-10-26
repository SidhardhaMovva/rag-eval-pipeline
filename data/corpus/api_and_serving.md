# Serving a RAG Pipeline as an API

A retrieval pipeline becomes useful to other software when it is exposed as an HTTP API. This project serves its pipeline with FastAPI, a Python web framework that generates interactive OpenAPI documentation automatically from the route definitions and request models.

The ingest endpoint accepts a document, chunks it, embeds the chunks, and adds them to the index. Ingestion is where new knowledge enters the system, so it should validate its input and report how many chunks were created.

The query endpoint accepts a question and returns the answer text, the citations, a confidence score, and an abstained flag. Returning the retrieved chunks along with the answer makes the API debuggable: a caller who sees a bad answer can inspect whether retrieval or generation was at fault.

The evaluate endpoint runs the golden-set evaluation on demand and returns recall, MRR, and hit rate. Exposing evaluation as an endpoint lets a deployment pipeline check retrieval quality after each release instead of trusting that nothing regressed.

Request and response bodies are validated with typed models, so a malformed request fails with a clear error instead of corrupting the index. Health endpoints report how many documents and chunks are indexed, which is the first thing to check when answers suddenly degrade after a restart.
