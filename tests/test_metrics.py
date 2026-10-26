"""Exact-value tests for the retrieval metrics."""

from eval.metrics import aggregate_metrics, precision_at_k, recall_at_k, reciprocal_rank, score_query

RANKED = ["alpha", "beta", "alpha", "gamma"]


def test_recall_at_k():
    assert recall_at_k(RANKED, "alpha", 1) == 1.0
    assert recall_at_k(RANKED, "gamma", 3) == 0.0
    assert recall_at_k(RANKED, "gamma", 5) == 1.0
    assert recall_at_k(RANKED, "missing", 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, "alpha") == 1.0
    assert reciprocal_rank(RANKED, "beta") == 0.5
    assert reciprocal_rank(RANKED, "gamma") == 0.25
    assert reciprocal_rank(RANKED, "missing") == 0.0


def test_precision_at_k():
    assert precision_at_k(RANKED, "alpha", 1) == 1.0
    assert precision_at_k(RANKED, "alpha", 3) == 2 / 3
    assert precision_at_k([], "alpha", 5) == 0.0


def test_score_query_and_aggregate():
    per_query = [score_query(RANKED, "alpha"), score_query(RANKED, "gamma")]
    summary = aggregate_metrics(per_query)
    assert summary["queries"] == 2
    assert summary["recall@5"] == 1.0
    assert summary["mrr"] == (1.0 + 0.25) / 2
    assert summary["hit_rate@5"] == summary["recall@5"]
