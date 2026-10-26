"""Shared data types for the RAG pipeline.

Plain dataclasses on purpose: the retrieval / chunking / evaluation core has
no web-framework dependency, so it can be imported and tested without
FastAPI or Pydantic installed. The API layer (app/main.py) converts these
to its own Pydantic response models.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """A source document before chunking."""

    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """One retrievable unit of text, with provenance metadata."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", self.doc_id))


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, with its similarity score."""

    chunk: Chunk
    score: float


@dataclass
class Answer:
    """A generated answer with citations and a confidence signal.

    ``abstained`` is True when the system decided the retrieved evidence was
    too weak to answer from — in that case ``text`` is the refusal message
    and ``citations`` is empty.
    """

    text: str
    citations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False
    retrieved: list[RetrievedChunk] = field(default_factory=list)
