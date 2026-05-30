from __future__ import annotations

from dataclasses import dataclass

from .chunker import Chunk
from .embeddings import generate_query_embedding
from .token_tracker import count_tokens
from .vector_store import VectorStore, search_index


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


def retrieve_top_k(
    query: str,
    chunks: list[Chunk],
    vector_store: VectorStore,
    top_k: int = 5,
    context_budget: int = 900,
    embedding_dimensions: int = 384,
) -> list[RetrievalResult]:
    """Retrieve the most relevant chunks and keep only what fits the token budget."""

    query_embedding = generate_query_embedding(query, dimensions=embedding_dimensions)
    candidates = [
        RetrievalResult(chunk=chunks[index], score=score)
        for index, score in search_index(vector_store, query_embedding, top_k)
        if score > 0
    ]

    packed: list[RetrievalResult] = []
    used_tokens = 0
    seen_signatures: set[str] = set()

    for candidate in candidates:
        signature = " ".join(candidate.chunk.text.lower().split()[:32])
        if signature in seen_signatures:
            continue
        token_count = candidate.chunk.token_count or count_tokens(candidate.chunk.text)
        if used_tokens + token_count > context_budget:
            continue
        packed.append(candidate)
        seen_signatures.add(signature)
        used_tokens += token_count

    return packed
