"""Retrieval evaluation: Recall@k, Precision@k, MRR, hit rate.

Runs every *answerable* golden query through retrieval and scores the
ranked document ids against the labeled ``expected_doc``. Unanswerable
queries are excluded here — they measure abstention, which lives in
``evaluate_generation.py``.

Usage:  python -m eval.evaluate_retrieval [--k 5] [--chunker recursive]
Writes: eval/results/retrieval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import RagPipeline  # noqa: E402
from eval.metrics import aggregate_metrics, score_query  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluate_retrieval(pipeline: RagPipeline, golden: list[dict], k: int = 5) -> dict:
    """Score retrieval for the answerable queries in ``golden``."""
    per_query: list[dict] = []
    details: list[dict] = []
    for item in golden:
        if not item.get("answerable", True):
            continue
        retrieved = pipeline.retriever.query(item["query"], k=k)
        ranked_doc_ids = [r.chunk.doc_id for r in retrieved]
        scores = score_query(ranked_doc_ids, item["expected_doc"])
        per_query.append(scores)
        details.append(
            {
                "id": item.get("id"),
                "query": item["query"],
                "expected_doc": item["expected_doc"],
                "ranked_doc_ids": ranked_doc_ids,
                **scores,
            }
        )
    summary = aggregate_metrics(per_query)
    summary["k"] = k
    summary["chunker"] = pipeline.chunker_name
    summary["chunks_indexed"] = pipeline.stats()["chunks"]
    return {**summary, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunker", default="recursive")
    parser.add_argument("--corpus", default=str(REPO_ROOT / "data" / "corpus"))
    parser.add_argument("--golden", default=str(REPO_ROOT / "eval" / "golden_set.json"))
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "eval" / "results" / "retrieval_results.json")
    )
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    pipeline = RagPipeline(chunker_name=args.chunker)
    pipeline.load_corpus(args.corpus)

    result = evaluate_retrieval(pipeline, golden, k=args.k)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {key: val for key, val in result.items() if key != "details"}
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
