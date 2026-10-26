"""Tests for embeddings, cosine similarity, and the retriever."""

import pytest

from app.embeddings import HashingEmbedder, TfidfEmbedder, cosine_similarity, tokenize
from app.ingest import ingest_text
from app.retriever import Retriever


def test_tokenize_removes_stopwords_and_lowercases():
    assert tokenize("The Quick brown FOX") == ["quick", "brown", "fox"]


def test_cosine_similarity_properties():
    a = {"cats": 1.0, "dogs": 2.0}
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, {"quantum": 1.0}) == 0.0
    assert cosine_similarity({}, a) == 0.0
    assert cosine_similarity(a, {}) == 0.0


def test_tfidf_requires_fit():
    with pytest.raises(RuntimeError):
        TfidfEmbedder().embed("anything")


def test_tfidf_ranks_topical_match_higher():
    emb = TfidfEmbedder().fit(
        [
            "cats sleep all day and groom their fur",
            "quantum computers use qubits and entanglement",
        ]
    )
    q = emb.embed("cats groom fur")
    assert cosine_similarity(q, emb.embed("cats sleep and groom fur")) > cosine_similarity(
        q, emb.embed("quantum computers use qubits and entanglement")
    )


def test_hashing_embedder_is_deterministic_and_stateless():
    emb = HashingEmbedder(dim=64)
    v1 = emb.embed("retrieval augmented generation")
    v2 = emb.embed("retrieval augmented generation")
    assert v1 == v2
    assert all(0 <= bucket < 64 for bucket in v1)


def _build_retriever() -> Retriever:
    chunks = []
    chunks += ingest_text(
        "Cats are small mammals. Cats sleep for long hours and groom their fur.",
        doc_id="cats",
    )
    chunks += ingest_text(
        "Quantum computers use qubits. Entanglement links qubits together.",
        doc_id="quantum",
    )
    retriever = Retriever()
    retriever.index(chunks)
    return retriever


def test_retriever_returns_expected_document_first():
    retriever = _build_retriever()
    results = retriever.query("How long do cats sleep?", k=2)
    assert results
    assert results[0].chunk.doc_id == "cats"
    assert results[0].score > 0


def test_retriever_metadata_filter():
    retriever = _build_retriever()
    results = retriever.query("qubits and cats", k=5, filters={"doc_id": "quantum"})
    assert results
    assert all(r.chunk.doc_id == "quantum" for r in results)


def test_retriever_no_shared_vocabulary_returns_nothing():
    retriever = _build_retriever()
    assert retriever.query("zzzqqq nonexistentterm", k=5) == []


def test_retriever_empty_index_and_bad_k():
    retriever = Retriever()
    retriever.index([])
    assert retriever.query("cats", k=5) == []
    with pytest.raises(ValueError):
        _build_retriever().query("cats", k=0)
