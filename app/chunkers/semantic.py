"""Sentence-boundary chunking guided by local similarity.

Sentences are the atomic unit. Consecutive sentences are grouped into one
chunk while they stay topically similar; a new chunk starts when the
similarity between the previous sentence and the next one drops below
``similarity_threshold`` (a topic shift), or when the chunk reaches
``max_chunk_chars``.

Similarity is computed with sparse TF vectors and cosine similarity from
``app.embeddings`` — no external embedding model or API key is involved.
That keeps the chunker deterministic and testable. It is a lightweight
stand-in for embedding-based semantic chunking, not the same thing, and
the README says so.
"""

from __future__ import annotations

import re

from app.embeddings import cosine_similarity, sparse_tf_vector

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation + whitespace."""
    if not text or not text.strip():
        return []
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def semantic_chunker(
    text: str,
    max_chunk_chars: int = 800,
    similarity_threshold: float = 0.10,
) -> list[str]:
    """Group sentences into chunks, breaking at topical shifts."""
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be positive")
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = [sentences[0]]
    current_len = len(sentences[0])

    for prev, nxt in zip(sentences, sentences[1:]):
        sim = cosine_similarity(sparse_tf_vector(prev), sparse_tf_vector(nxt))
        would_overflow = current_len + 1 + len(nxt) > max_chunk_chars
        topic_shift = sim < similarity_threshold
        if current and (would_overflow or topic_shift):
            chunks.append(" ".join(current))
            current, current_len = [nxt], len(nxt)
        else:
            current.append(nxt)
            current_len += 1 + len(nxt)

    if current:
        chunks.append(" ".join(current))
    return chunks
