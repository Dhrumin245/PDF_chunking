from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from .chunker import Chunk


_WORD_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def generate_embeddings(chunks: Sequence[Chunk | str], dimensions: int = 384) -> list[list[float]]:
    """Generate local hashed embeddings without external model/API token cost."""

    return [_embed(_text(item), dimensions) for item in chunks]


def generate_query_embedding(query: str, dimensions: int = 384) -> list[float]:
    return _embed(query, dimensions)


def _text(item: Chunk | str) -> str:
    return item.text if isinstance(item, Chunk) else item


def _embed(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    terms = _tokenize(text)
    if not terms:
        return vector

    for term in terms:
        digest = hashlib.md5(term.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0).lower() for match in _WORD_PATTERN.finditer(text))
        if token not in _STOPWORDS and len(token) > 1
    ]
