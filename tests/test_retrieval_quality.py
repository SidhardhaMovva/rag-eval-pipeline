"""Golden-set regression tests.

These guard the headline quality numbers: if a future change to a chunker,
the embedder, or the corpus quietly tanks retrieval quality or abstention
behaviour, these tests fail. Thresholds are set slightly below the
currently measured values (see README) so the tests guard against
regressions rather than enshrining exact numbers.
"""

import json
from pathlib import Path

import pytest

from app.pipeline import RagPipeline
from eval.evaluate_generation import evaluate_generation
from eval.evaluate_retrieval import evaluate_retrieval

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def golden() -> list[dict]:
    return json.loads((REPO_ROOT / "eval" / "golden_set.json").read_text())


@pytest.fixture(scope="module")
def pipeline() -> RagPipeline:
    p = RagPipeline(chunker_name="recursive")
    p.load_corpus(REPO_ROOT / "data" / "corpus")
    return p


def test_golden_set_is_real_and_labeled(golden):
    assert len(golden) >= 20
    assert all(g["query"] and "answerable" in g for g in golden)
    answerable = [g for g in golden if g["answerable"]]
    unanswerable = [g for g in golden if not g["answerable"]]
    assert len(answerable) >= 15
    assert len(unanswerable) >= 3
    assert all(g["expected_doc"] for g in answerable)


def test_retrieval_recall_and_mrr(pipeline, golden):
    result = evaluate_retrieval(pipeline, golden, k=5)
    assert result["recall@5"] >= 0.9
    assert result["mrr"] >= 0.9


def test_generation_citations_and_abstention(pipeline, golden):
    result = evaluate_generation(pipeline, golden, k=5)
    assert result["answer_rate"] >= 0.9
    assert result["citation_precision"] >= 0.9
    assert result["abstention_accuracy"] >= 0.75
