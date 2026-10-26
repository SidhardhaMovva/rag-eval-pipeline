"""Local, deterministic text vectors — no API key, no model download.

Vectors are *sparse dicts* (token -> weight), which keeps cosine similarity
trivial and the whole retrieval stack dependency-free.

Two embedders are provided behind one interface:

- ``TfidfEmbedder`` — classic TF-IDF. It must be ``fit`` on the corpus first
  so it can learn inverse document frequencies. This is the default: on a
  small, known corpus it is strong, exact, and fully reproducible.
- ``HashingEmbedder`` — stateless hashing vectors (term counts hashed into
  a fixed number of buckets). No fitting, works on a single document, but
  collisions make it weaker than TF-IDF. Useful for streaming ingestion.

To plug in a real embedding model (e.g. a sentence-transformer or a hosted
embeddings API), implement the same two methods — ``embed(text) -> dict``
and ``embed_many(texts) -> list[dict]`` — returning dense vectors as
``{index: weight}`` dicts, and pass the instance to ``Retriever``.
``cosine_similarity`` works unchanged for any hashable keys.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an the and or of to in on for with by is are was were be been being
    it its this that these those as at from how what why when where which
    who whom can could should would do does did not no yes if then than
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def sparse_tf_vector(text: str) -> dict[str, float]:
    """Raw term-frequency vector for a single text (no IDF)."""
    counts = Counter(tokenize(text))
    return {tok: float(n) for tok, n in counts.items()}


def cosine_similarity(a: dict, b: dict) -> float:
    """Cosine similarity between two sparse vectors. 0.0 if either is empty."""
    if not a or not b:
        return 0.0
    # Iterate over the smaller vector for the dot product.
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b.get(tok, 0.0) for tok, w in a.items())
    if dot == 0.0:
        return 0.0
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class TfidfEmbedder:
    """TF-IDF embedder fitted on a corpus.

    ``fit`` learns document frequencies; ``embed`` then weights terms by
    ``tf * (log((1 + N) / (1 + df)) + 1)`` — the smoothed IDF used by
    scikit-learn, so unseen terms still get a small positive weight.
    """

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._n_docs = 0
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        df: Counter[str] = Counter()
        for text in texts:
            for tok in set(tokenize(text)):
                df[tok] += 1
        self._n_docs = len(texts)
        self._idf = {
            tok: math.log((1 + self._n_docs) / (1 + count)) + 1.0
            for tok, count in df.items()
        }
        self._fitted = True
        return self

    def embed(self, text: str) -> dict[str, float]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fit() before embed()")
        counts = Counter(tokenize(text))
        if not counts:
            return {}
        max_tf = max(counts.values())
        return {
            tok: (0.5 + 0.5 * n / max_tf) * self._idf.get(tok, math.log(1 + self._n_docs) + 1.0)
            for tok, n in counts.items()
        }

    def embed_many(self, texts: list[str]) -> list[dict[str, float]]:
        return [self.embed(t) for t in texts]

    @property
    def vocabulary_size(self) -> int:
        return len(self._idf)


class HashingEmbedder:
    """Stateless hashing embedder: no fitting, fixed ``dim`` buckets."""

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def embed(self, text: str) -> dict[int, float]:
        counts = Counter(tokenize(text))
        vec: dict[int, float] = {}
        for tok, n in counts.items():
            bucket = self._bucket(tok)
            vec[bucket] = vec.get(bucket, 0.0) + float(n)
        return vec

    def embed_many(self, texts: list[str]) -> list[dict[int, float]]:
        return [self.embed(t) for t in texts]
