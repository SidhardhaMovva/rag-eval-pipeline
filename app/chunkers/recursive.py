"""Recursive, separator-aware chunking.

Tries to split on the strongest boundary first (paragraph breaks), then
falls back to lines, sentences, and finally words, recursively splitting
any piece that is still too large. Adjacent small pieces are merged back
together up to ``chunk_size`` so chunks stay coherent instead of
fragmenting at every paragraph break.
"""

from __future__ import annotations

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    if not separators:
        return [
            text[i : i + chunk_size]
            for i in range(0, len(text), chunk_size)
            if text[i : i + chunk_size].strip()
        ]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_recursive(text, chunk_size, rest)

    pieces: list[str] = []
    parts = text.split(sep)
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) <= chunk_size:
            pieces.append(piece)
        else:
            pieces.extend(_split_recursive(piece, chunk_size, rest))
    return [p for p in pieces if p.strip()]


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            current = (current[-overlap:] if overlap else "") + piece
        else:
            current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def recursive_chunker(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """Split ``text`` on paragraph/line/sentence boundaries, recursively."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    if not text or not text.strip():
        return []

    seps = separators if separators is not None else list(DEFAULT_SEPARATORS)
    pieces = _split_recursive(text, chunk_size, seps)
    return _merge_pieces(pieces, chunk_size, overlap)
