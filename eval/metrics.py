"""Retrieval metrics, computed at document level.

Each golden query has exactly one relevant document, so a retrieved chunk
is "relevant" when its ``doc_id`` equals the query's ``expected_doc``.
The methodology is deliberately simple and is documented in README.md —
no chunk-level relevance judgments are claimed.
"""

from __future__ import annotations


def _first_relevant_rank(ranked_doc_ids: list[str], expected_doc: str) -> int | None:
    """1-based rank of the first chunk from ``expected_doc``, or None."""
    for i, doc_id in enumerate(ranked_doc_ids):
        if doc_id == expected_doc:
            return i + 1
    return None


def recall_at_k(ranked_doc_ids: list[str], expected_doc: str, k: int) -> float:
    """1.0 if the expected document appears in the top k, else 0.0.

    (With one relevant document per query, per-query recall@k is binary;
    the reported Recall@k is its mean over queries.)
    """
    return 1.0 if expected_doc in ranked_doc_ids[:k] else 0.0


def reciprocal_rank(ranked_doc_ids: list[str], expected_doc: str) -> float:
    rank = _first_relevant_rank(ranked_doc_ids, expected_doc)
    return 0.0 if rank is None else 1.0 / rank


def precision_at_k(ranked_doc_ids: list[str], expected_doc: str, k: int) -> float:
    top = ranked_doc_ids[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d == expected_doc) / len(top)


def aggregate_metrics(per_query: list[dict], ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """Average per-query metric dicts into a summary dict."""
    if not per_query:
        return {"queries": 0}
    n = len(per_query)
    summary: dict = {"queries": n}
    for k in ks:
        summary[f"recall@{k}"] = round(sum(q[f"recall@{k}"] for q in per_query) / n, 4)
        summary[f"precision@{k}"] = round(
            sum(q[f"precision@{k}"] for q in per_query) / n, 4
        )
    summary["mrr"] = round(sum(q["reciprocal_rank"] for q in per_query) / n, 4)
    summary["hit_rate@5"] = summary["recall@5"]
    return summary


def score_query(ranked_doc_ids: list[str], expected_doc: str, ks=(1, 3, 5)) -> dict:
    """All metrics for one query's ranked doc-id list."""
    out: dict = {"reciprocal_rank": round(reciprocal_rank(ranked_doc_ids, expected_doc), 4)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked_doc_ids, expected_doc, k)
        out[f"precision@{k}"] = round(precision_at_k(ranked_doc_ids, expected_doc, k), 4)
    return out
