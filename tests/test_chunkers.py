"""Real tests for the three chunking strategies."""

import pytest

from app.chunkers import fixed_size_chunker, recursive_chunker, semantic_chunker
from app.chunkers.semantic import split_sentences

TEXT = (
    "Retrieval systems split documents into chunks before indexing them.\n\n"
    "A chunk is the unit that gets embedded and stored in the index. "
    "When a question arrives, the retriever compares the question against "
    "every chunk and returns the closest matches.\n\n"
    "Chunk size is a trade-off. Large chunks carry more context but their "
    "embeddings mix several topics together. Small chunks are precise but "
    "may omit the sentence that makes a passage understandable."
)


class TestFixedChunker:
    def test_respects_chunk_size(self):
        chunks = fixed_size_chunker(TEXT, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_overlap_shares_content(self):
        chunks = fixed_size_chunker(TEXT, chunk_size=100, overlap=20)
        # The tail of chunk 0 reappears at the start of chunk 1.
        assert chunks[0][-20:] in chunks[1] or chunks[1][:20] in chunks[0]

    def test_no_text_is_lost(self):
        chunks = fixed_size_chunker(TEXT, chunk_size=80, overlap=0)
        # Every 40-char window of the original appears in some chunk
        # (allowing for strip() at chunk edges).
        joined = "".join(chunks)
        assert len(joined) >= len(TEXT) - 2 * len(chunks)
        assert "embeddings mix several topics" in joined

    def test_empty_and_short_text(self):
        assert fixed_size_chunker("") == []
        assert fixed_size_chunker("   ") == []
        assert fixed_size_chunker("short", chunk_size=100, overlap=10) == ["short"]

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            fixed_size_chunker(TEXT, chunk_size=50, overlap=50)
        with pytest.raises(ValueError):
            fixed_size_chunker(TEXT, chunk_size=0, overlap=0)


class TestRecursiveChunker:
    def test_respects_chunk_size(self):
        chunks = recursive_chunker(TEXT, chunk_size=120, overlap=10)
        assert len(chunks) > 1
        assert all(len(c) <= 120 + 10 for c in chunks)

    def test_keeps_sentences_whole(self):
        chunks = recursive_chunker(TEXT, chunk_size=200, overlap=0)
        joined = " ".join(chunks)
        assert "A chunk is the unit that gets embedded and stored in the index." in joined
        # No chunk should start mid-sentence with a lowercase fragment of
        # the known sentences.
        for chunk in chunks:
            assert "unit that gets embedded" not in chunk or "A chunk is the" in chunk

    def test_paragraph_text_is_preserved(self):
        chunks = recursive_chunker(TEXT, chunk_size=500, overlap=50)
        joined = " ".join(chunks)
        assert "Chunk size is a trade-off." in joined
        assert "embeddings mix several topics together" in joined

    def test_empty_and_invalid(self):
        assert recursive_chunker("") == []
        with pytest.raises(ValueError):
            recursive_chunker(TEXT, chunk_size=10, overlap=10)


class TestSemanticChunker:
    def test_split_sentences(self):
        assert split_sentences("One. Two? Three! Four") == [
            "One.",
            "Two?",
            "Three!",
            "Four",
        ]
        assert split_sentences("") == []

    def test_groups_by_topic_and_breaks_on_shift(self):
        text = (
            "Cats sleep during the day. Cats hunt mice during the night. "
            "Cats groom their fur carefully. "
            "Quantum computers use qubits. Quantum entanglement links qubits. "
            "Quantum algorithms exploit superposition."
        )
        chunks = semantic_chunker(text)
        assert len(chunks) == 2
        assert "Cats" in chunks[0] and "Quantum" not in chunks[0]
        assert "Quantum" in chunks[1]

    def test_max_chunk_chars_is_enforced(self):
        text = " ".join(f"Sentence number {i} about retrieval systems." for i in range(30))
        chunks = semantic_chunker(text, max_chunk_chars=120)
        assert len(chunks) > 1
        assert all(len(c) <= 120 + 40 for c in chunks)

    def test_empty(self):
        assert semantic_chunker("") == []
        with pytest.raises(ValueError):
            semantic_chunker(TEXT, max_chunk_chars=0)
