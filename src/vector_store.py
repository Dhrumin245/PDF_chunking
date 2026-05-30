from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VectorStore:
    embeddings: list[list[float]]
    backend: str
    index: object | None = None


def build_faiss_index(embeddings: list[list[float]]) -> VectorStore:
    """Build a FAISS index when available, otherwise use a pure Python fallback."""

    if not embeddings:
        return VectorStore(embeddings=[], backend="empty")

    try:
        import faiss
        import numpy as np

        matrix = np.array(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return VectorStore(embeddings=embeddings, backend="faiss", index=index)
    except Exception:
        return VectorStore(embeddings=embeddings, backend="python")


def search_index(store: VectorStore, query_embedding: list[float], top_k: int) -> list[tuple[int, float]]:
    if store.backend == "faiss" and store.index is not None:
        import numpy as np

        query = np.array([query_embedding], dtype="float32")
        scores, indexes = store.index.search(query, top_k)
        return [
            (int(index), float(score))
            for index, score in zip(indexes[0], scores[0])
            if int(index) >= 0
        ]

    scored = [
        (index, _dot_product(query_embedding, embedding))
        for index, embedding in enumerate(store.embeddings)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def _dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
