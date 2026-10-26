"""In-memory cosine retrieval over chunk vectors, with metadata filtering.

This replaces the originally-planned ChromaDB dependency: for a corpus of
this size an in-memory index is simpler, has zero infrastructure, and is
exactly reproducible in tests and eval runs. The ``Retriever`` interface
(``index`` / ``query``) is the seam where a vector database would slot in
for a larger corpus.
"""

from __future__ import annotations

from app.embeddings import TfidfEmbedder, cosine_similarity
from app.schemas import Chunk, RetrievedChunk


class Retriever:
    """Indexes chunks with a fitted embedder and answers top-k queries."""

    def __init__(self, embedder: TfidfEmbedder | None = None) -> None:
        self.embedder = embedder or TfidfEmbedder()
        self._chunks: list[Chunk] = []
        self._vectors: list[dict] = []

    def index(self, chunks: list[Chunk]) -> None:
        """(Re)build the index from scratch over ``chunks``."""
        self._chunks = list(chunks)
        texts = [c.text for c in self._chunks]
        self.embedder.fit(texts)
        self._vectors = self.embedder.embed_many(texts)

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks and refit (the corpus here is small; refit is cheap)."""
        self.index(self._chunks + list(chunks))

    @property
    def size(self) -> int:
        return len(self._chunks)

    def query(
        self,
        text: str,
        k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k chunks by cosine similarity.

        ``filters`` is an equality match against chunk metadata (plus the
        synthetic keys ``doc_id`` and ``source``), applied *before* ranking
        so a filtered-out chunk can never occupy a top-k slot. Chunks with a
        zero score are dropped: no shared vocabulary means no evidence.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._chunks:
            return []

        query_vec = self.embedder.embed(text)
        if not query_vec:
            return []

        scored: list[RetrievedChunk] = []
        for chunk, vec in zip(self._chunks, self._vectors):
            if filters and not _matches(chunk, filters):
                continue
            score = cosine_similarity(query_vec, vec)
            if score > 0.0:
                scored.append(RetrievedChunk(chunk=chunk, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]


def _matches(chunk: Chunk, filters: dict) -> bool:
    haystack = {**chunk.metadata, "doc_id": chunk.doc_id, "source": chunk.source}
    return all(haystack.get(key) == value for key, value in filters.items())
