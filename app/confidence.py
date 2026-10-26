"""Confidence scoring and abstention ("refusal") logic.

The confidence in an answer is derived from retrieval strength, not from a
model's self-report: it is the cosine score of the best retrieved chunk,
optionally tempered by the margin over the runner-up. When confidence is
below the abstention threshold, the generator refuses to answer instead of
composing something unsupported — the simplest effective hallucination
control there is.
"""

from __future__ import annotations

# Tuned on eval/golden_set.json (26 queries): every answerable query in
# the set scores retrieval confidence >= 0.20 and every unanswerable query
# scores <= 0.09, so 0.15 sits in the middle of the observed gap. It is a
# corpus-specific value, not a universal constant — re-tune it when the
# corpus or embedder changes, using a golden set that includes
# unanswerable questions.
DEFAULT_ABSTAIN_THRESHOLD = 0.15


def retrieval_confidence(scores: list[float]) -> float:
    """Confidence in [0, 1] from ranked retrieval scores (best first).

    Blends the top score with half the margin over the second score, so a
    lone weak match scores lower than a clear winner. Returns 0.0 when
    there are no scores at all.
    """
    if not scores:
        return 0.0
    top = scores[0]
    if len(scores) == 1:
        return round(top, 6)
    margin = max(top - scores[1], 0.0)
    return round(min(top + 0.5 * margin, 1.0), 6)


def should_abstain(confidence: float, threshold: float = DEFAULT_ABSTAIN_THRESHOLD) -> bool:
    """True when ``confidence`` is too low to answer responsibly."""
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be within [0, 1]")
    return confidence < threshold
