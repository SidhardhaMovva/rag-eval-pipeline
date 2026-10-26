"""Chunker comparison: fixed vs recursive vs semantic, same golden set.

Rebuilds the pipeline once per chunking strategy and reports retrieval
metrics plus chunk counts, so the trade-off is measured on identical
queries and an identical corpus — not argued from intuition.

Usage:  python -m eval.compare_chunkers
Writes: eval/results/chunker_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunkers import CHUNKERS  # noqa: E402
from app.pipeline import RagPipeline  # noqa: E402
from eval.evaluate_retrieval import evaluate_retrieval  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def compare_chunkers(corpus: str, golden: list[dict], k: int = 5) -> dict:
    results: dict = {}
    for name in CHUNKERS:
        pipeline = RagPipeline(chunker_name=name)
        pipeline.load_corpus(corpus)
        result = evaluate_retrieval(pipeline, golden, k=k)
        results[name] = {
            key: val for key, val in result.items() if key != "details"
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--corpus", default=str(REPO_ROOT / "data" / "corpus"))
    parser.add_argument("--golden", default=str(REPO_ROOT / "eval" / "golden_set.json"))
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "eval" / "results" / "chunker_comparison.json")
    )
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    results = compare_chunkers(args.corpus, golden, k=args.k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    header = f"{'chunker':<12} {'chunks':>6} {'recall@5':>9} {'mrr':>7} {'prec@5':>7}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<12} {r['chunks_indexed']:>6} {r['recall@5']:>9} "
            f"{r['mrr']:>7} {r['precision@5']:>7}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
