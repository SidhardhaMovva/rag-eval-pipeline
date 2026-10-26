"""Document ingestion: text/markdown in, chunks out.

Supported inputs are plain text (``.txt``) and Markdown (``.md``) files, or
raw strings via ``ingest_text``. PDF ingestion is *not* implemented — the
previous README claimed it, this one does not (see "Limitations" in
README.md).

Document ids come from the file stem (``evaluation_metrics.md`` ->
``evaluation_metrics``) so golden-set labels can refer to them readably.
"""

from __future__ import annotations

from pathlib import Path

from app.chunkers import get_chunker
from app.schemas import Chunk, Document

SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_document(path: str | Path) -> Document:
    """Load one .txt/.md file as a Document. Raises ValueError otherwise."""
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {p.suffix!r}; supported: "
            f"{sorted(SUPPORTED_SUFFIXES)}"
        )
    text = p.read_text(encoding="utf-8")
    return Document(
        doc_id=p.stem,
        text=text,
        metadata={"source": p.name, "path": str(p)},
    )


def load_directory(directory: str | Path) -> list[Document]:
    """Load every supported file in a directory, sorted by filename."""
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"Not a directory: {d}")
    docs = [
        load_document(p)
        for p in sorted(d.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return docs


def chunk_document(document: Document, chunker_name: str = "recursive", **kwargs) -> list[Chunk]:
    """Chunk one document, attaching provenance metadata to each chunk."""
    chunker = get_chunker(chunker_name)
    pieces = chunker(document.text, **kwargs)
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#{i}",
                doc_id=document.doc_id,
                text=piece,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "chunker": chunker_name,
                },
            )
        )
    return chunks


def ingest_documents(
    documents: list[Document], chunker_name: str = "recursive", **kwargs
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, chunker_name, **kwargs))
    return chunks


def ingest_directory(
    directory: str | Path, chunker_name: str = "recursive", **kwargs
) -> list[Chunk]:
    return ingest_documents(load_directory(directory), chunker_name, **kwargs)


def ingest_text(
    text: str, doc_id: str = "inline", chunker_name: str = "recursive", **kwargs
) -> list[Chunk]:
    doc = Document(doc_id=doc_id, text=text, metadata={"source": doc_id})
    return chunk_document(doc, chunker_name, **kwargs)
