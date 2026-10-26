"""Grounded answering — extractive by default, LLM optional.

The default ``ExtractiveGenerator`` never invents text: it selects the
sentences from the retrieved chunks that best match the query, returns
them verbatim with chunk citations, and abstains when retrieval confidence
is too low. Faithfulness is therefore structural — every sentence in an
answer exists word-for-word in a cited chunk. That is the trade this
project makes on purpose: less fluent than an LLM, but it cannot
hallucinate a fact that is not in the corpus.

``LLMGenerator`` documents the optional seam for a generative model. It is
not wired up and requires a client implementing ``complete(prompt) -> str``
(e.g. wrapping a hosted LLM API, which needs an API key). Nothing in the
default pipeline calls it.
"""

from __future__ import annotations

from app.chunkers.semantic import split_sentences
from app.confidence import DEFAULT_ABSTAIN_THRESHOLD, retrieval_confidence, should_abstain
from app.embeddings import TfidfEmbedder, cosine_similarity
from app.schemas import Answer, RetrievedChunk

ABSTAIN_MESSAGE = (
    "I don't have enough evidence in the indexed documents to answer that."
)


class ExtractiveGenerator:
    """Answers by extracting the best-matching sentences from retrieved chunks."""

    def __init__(
        self,
        abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
        max_sentences: int = 3,
        min_sentence_score: float = 0.05,
    ) -> None:
        self.abstain_threshold = abstain_threshold
        self.max_sentences = max_sentences
        self.min_sentence_score = min_sentence_score

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> Answer:
        scores = [r.score for r in retrieved]
        confidence = retrieval_confidence(scores)

        if not retrieved or should_abstain(confidence, self.abstain_threshold):
            return Answer(
                text=ABSTAIN_MESSAGE,
                citations=[],
                confidence=confidence,
                abstained=True,
                retrieved=retrieved,
            )

        # Score every sentence of every retrieved chunk against the query,
        # using a TF-IDF space fitted on just those chunks.
        embedder = TfidfEmbedder().fit([r.chunk.text for r in retrieved])
        query_vec = embedder.embed(query)

        candidates: list[tuple[float, int, str, str]] = []
        for rank, r in enumerate(retrieved):
            for sentence in split_sentences(r.chunk.text):
                s_score = cosine_similarity(query_vec, embedder.embed(sentence))
                if s_score >= self.min_sentence_score:
                    candidates.append((s_score, rank, sentence, r.chunk.chunk_id))

        if not candidates:
            return Answer(
                text=ABSTAIN_MESSAGE,
                citations=[],
                confidence=confidence,
                abstained=True,
                retrieved=retrieved,
            )

        candidates.sort(key=lambda c: (-c[0], c[1]))
        chosen = candidates[: self.max_sentences]
        # Present sentences in retrieval-rank order, not score order, so the
        # answer reads in source order.
        chosen.sort(key=lambda c: c[1])

        parts: list[str] = []
        citations: list[str] = []
        for _, _, sentence, chunk_id in chosen:
            parts.append(f"{sentence} [{chunk_id}]")
            if chunk_id not in citations:
                citations.append(chunk_id)

        return Answer(
            text=" ".join(parts),
            citations=citations,
            confidence=confidence,
            abstained=False,
            retrieved=retrieved,
        )


class LLMGenerator:
    """Optional generative answering via an injected LLM client.

    Not used by the default pipeline. To use it, pass any object with
    ``complete(prompt: str) -> str`` — for example a thin wrapper around a
    hosted LLM API (requires that provider's API key). Calling ``generate``
    without a client raises immediately instead of failing silently.
    """

    def __init__(self, client=None) -> None:
        self.client = client

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> Answer:
        if self.client is None:
            raise RuntimeError(
                "LLMGenerator requires an LLM client; the default pipeline "
                "uses ExtractiveGenerator, which needs no API key."
            )
        context = "\n\n".join(
            f"[{r.chunk.chunk_id}] {r.chunk.text}" for r in retrieved
        )
        prompt = (
            "Answer the question using only the context below. Cite chunk "
            "ids like [chunk-id] after each claim. If the context does not "
            f"contain the answer, say so.\n\nContext:\n{context}\n\n"
            f"Question: {query}"
        )
        text = self.client.complete(prompt)
        citations = [r.chunk.chunk_id for r in retrieved]
        return Answer(
            text=text,
            citations=citations,
            confidence=retrieval_confidence([r.score for r in retrieved]),
            abstained=False,
            retrieved=retrieved,
        )
