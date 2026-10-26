"""Generation evaluation for the extractive generator.

This is *not* LLM-as-judge (the previous README claimed that; there is no
LLM in the loop to judge with, and none is needed). Three behaviours of
the extractive generator are measurable without one:

- **Citation precision** — for answerable queries the system answered,
  how often is the expected document among the cited chunks?
- **Answer rate** — how often does the system answer an answerable query
  at all (vs. over-abstaining)?
- **Abstention accuracy** — for unanswerable queries, how often does the
  system correctly refuse?

Faithfulness is not scored here because it is structural: the extractive
generator returns verbatim source sentences, so an answer sentence cannot
contain a fact that is absent from a cited chunk. That property, and its
limit (irrelevant sentences can still be selected), is covered by the
citation metric and the unit tests.

Usage:  python -m eval.evaluate_generation
Writes: eval/results/generation_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import RagPipeline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluate_generation(pipeline: RagPipeline, golden: list[dict], k: int = 5) -> dict:
    answerable = [g for g in golden if g.get("answerable", True)]
    unanswerable = [g for g in golden if not g.get("answerable", True)]

    answered = 0
    citation_hits = 0
    details: list[dict] = []
    for item in answerable:
        ans = pipeline.query(item["query"], k=k)
        if ans.abstained:
            details.append({"id": item.get("id"), "outcome": "abstained"})
            continue
        answered += 1
        cited_docs = {
            c.rsplit("#", 1)[0] for c in ans.citations
        }
        hit = item["expected_doc"] in cited_docs
        citation_hits += int(hit)
        details.append(
            {
                "id": item.get("id"),
                "outcome": "answered",
                "expected_doc": item["expected_doc"],
                "cited_docs": sorted(cited_docs),
                "citation_hit": hit,
                "confidence": ans.confidence,
            }
        )

    correct_abstentions = 0
    for item in unanswerable:
        ans = pipeline.query(item["query"], k=k)
        correct_abstentions += int(ans.abstained)
        details.append(
            {
                "id": item.get("id"),
                "outcome": "abstained" if ans.abstained else "answered",
                "expected": "abstain",
                "confidence": ans.confidence,
            }
        )

    return {
        "chunker": pipeline.chunker_name,
        "answerable_queries": len(answerable),
        "answered": answered,
        "answer_rate": round(answered / len(answerable), 4) if answerable else 0.0,
        "citation_precision": round(citation_hits / answered, 4) if answered else 0.0,
        "unanswerable_queries": len(unanswerable),
        "abstention_accuracy": (
            round(correct_abstentions / len(unanswerable), 4) if unanswerable else 0.0
        ),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunker", default="recursive")
    parser.add_argument("--corpus", default=str(REPO_ROOT / "data" / "corpus"))
    parser.add_argument("--golden", default=str(REPO_ROOT / "eval" / "golden_set.json"))
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "eval" / "results" / "generation_results.json")
    )
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    pipeline = RagPipeline(chunker_name=args.chunker)
    pipeline.load_corpus(args.corpus)

    result = evaluate_generation(pipeline, golden, k=args.k)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {key: val for key, val in result.items() if key != "details"}
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
