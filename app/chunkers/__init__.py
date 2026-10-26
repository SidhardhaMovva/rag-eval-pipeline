"""Chunking strategies: fixed-size, recursive (separator-aware), semantic."""

from app.chunkers.fixed import fixed_size_chunker
from app.chunkers.recursive import recursive_chunker
from app.chunkers.semantic import semantic_chunker

CHUNKERS = {
    "fixed": fixed_size_chunker,
    "recursive": recursive_chunker,
    "semantic": semantic_chunker,
}


def get_chunker(name: str):
    """Look up a chunker by name ("fixed" | "recursive" | "semantic")."""
    try:
        return CHUNKERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown chunker {name!r}; expected one of {sorted(CHUNKERS)}"
        ) from None


__all__ = [
    "CHUNKERS",
    "fixed_size_chunker",
    "get_chunker",
    "recursive_chunker",
    "semantic_chunker",
]
