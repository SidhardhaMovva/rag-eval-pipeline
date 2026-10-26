"""Tests for the extractive generator, abstention, and confidence logic."""

import pytest

from app.confidence import retrieval_confidence, should_abstain
from app.generator import ABSTAIN_MESSAGE, ExtractiveGenerator, LLMGenerator
from app.pipeline import RagPipeline


@pytest.fixture()
def pipeline() -> RagPipeline:
    p = RagPipeline()
    p.add_text(
        "Retrieval-augmented generation retrieves relevant passages before "
        "generating an answer. The answer is grounded in the retrieved "
        "passages and can cite them.",
        doc_id="rag_doc",
    )
    p.add_text(
        "Photosynthesis is how plants convert sunlight into energy. "
        "Chlorophyll absorbs light in plant leaves.",
        doc_id="plants_doc",
    )
    return p


def test_confidence_helpers():
    assert retrieval_confidence([]) == 0.0
    assert retrieval_confidence([0.5]) == 0.5
    assert retrieval_confidence([0.6, 0.2]) > 0.6  # margin bonus
    assert should_abstain(0.01, threshold=0.15) is True
    assert should_abstain(0.5, threshold=0.15) is False
    with pytest.raises(ValueError):
        should_abstain(0.5, threshold=1.5)


def test_answer_is_grounded_and_cited(pipeline):
    ans = pipeline.query("How does retrieval-augmented generation work?", k=3)
    assert not ans.abstained
    assert ans.citations
    cited = {c.chunk_id: c for c in pipeline.chunks}
    # Every sentence in the answer appears verbatim in a cited chunk.
    for chunk_id in ans.citations:
        assert chunk_id in cited
    sentences = [s.strip() for s in ans.text.split("] ") if s.strip()]
    assert sentences  # non-empty answer
    joined_sources = " ".join(cited[c].text for c in ans.citations)
    for sentence in sentences:
        core = sentence.split(" [")[0].strip()
        assert core in joined_sources


def test_abstains_when_no_evidence(pipeline):
    ans = pipeline.query("What is the capital of Atlantis in 1850?", k=3)
    # Either no/weak retrieval -> abstention with the refusal message.
    assert ans.abstained
    assert ans.text == ABSTAIN_MESSAGE
    assert ans.citations == []


def test_generator_abstains_on_empty_retrieval():
    gen = ExtractiveGenerator()
    ans = gen.generate("anything", [])
    assert ans.abstained
    assert ans.confidence == 0.0


def test_llm_generator_without_client_raises():
    with pytest.raises(RuntimeError):
        LLMGenerator().generate("q", [])


def test_conversation_memory_records_turns(pipeline):
    pipeline.query("How does retrieval-augmented generation work?", session_id="s1")
    history = pipeline.memory.get_history("s1")
    assert len(history) == 1
    assert "retrieval-augmented" in history[0].query
    pipeline.memory.clear("s1")
    assert pipeline.memory.get_history("s1") == []
