"""The assembled RAG pipeline: ingest -> index -> retrieve -> generate.

One object owns the document store, the chunk list, the retriever, the
generator, and conversation memory, so the API layer and the eval scripts
drive exactly the same code path.
"""

from __future__ import annotations

from pathlib import Path

from app import ingest as ingest_mod
from app.generator import ExtractiveGenerator
from app.memory import ConversationMemory
from app.retriever import Retriever
from app.schemas import Answer, Chunk, Document


class RagPipeline:
    def __init__(
        self,
        chunker_name: str = "recursive",
        abstain_threshold: float | None = None,
    ) -> None:
        self.chunker_name = chunker_name
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.retriever = Retriever()
        kwargs = {} if abstain_threshold is None else {"abstain_threshold": abstain_threshold}
        self.generator = ExtractiveGenerator(**kwargs)
        self.memory = ConversationMemory()

    # -- ingestion ---------------------------------------------------------
    def add_documents(self, documents: list[Document]) -> int:
        """Add documents, re-chunk everything, refit the index. Returns chunk count."""
        self.documents.extend(documents)
        self.chunks = ingest_mod.ingest_documents(self.documents, self.chunker_name)
        self.retriever.index(self.chunks)
        return len(self.chunks)

    def add_text(self, text: str, doc_id: str) -> int:
        doc = Document(doc_id=doc_id, text=text, metadata={"source": doc_id})
        return self.add_documents([doc])

    def load_corpus(self, directory: str | Path) -> int:
        return self.add_documents(ingest_mod.load_directory(directory))

    # -- querying ----------------------------------------------------------
    def query(
        self,
        text: str,
        k: int = 5,
        filters: dict | None = None,
        session_id: str | None = None,
    ) -> Answer:
        retrieved = self.retriever.query(text, k=k, filters=filters)
        answer = self.generator.generate(text, retrieved)
        if session_id:
            self.memory.add_turn(session_id, text, answer.text)
        return answer

    # -- introspection -----------------------------------------------------
    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "chunker": self.chunker_name,
        }
