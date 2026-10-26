"""Fixed-size character chunking with optional overlap.

The simplest strategy: slice the text into windows of ``chunk_size``
characters, advancing by ``chunk_size - overlap`` each step. It is fast and
predictable, but it can split mid-sentence and mid-word — the eval harness
in ``eval/compare_chunkers.py`` measures what that costs.
"""

from __future__ import annotations


def fixed_size_chunker(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> list[str]:
    """Split ``text`` into fixed-size character windows.

    Raises ValueError if ``overlap`` is not smaller than ``chunk_size``
    (otherwise the splitter would not advance).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks
