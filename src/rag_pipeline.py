from __future__ import annotations

from dataclasses import asdict, dataclass

from .chunker import Chunk, chunk_text
from .embeddings import generate_embeddings
from .pdf_loader import extract_text
from .retriever import RetrievalResult, retrieve_top_k
from .token_tracker import TokenTracker, count_tokens
from .vector_store import build_faiss_index


SYSTEM_PROMPT = (
    "Answer using only the provided PDF context. If the answer is not present, "
    "say that the PDF context is insufficient. Cite chunk ids."
)


@dataclass(frozen=True)
class RagOutput:
    prompt: str
    chunks: list[Chunk]
    retrieved_chunks: list[RetrievalResult]
    token_usage: dict[str, int | str | bool]
    vector_backend: str

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "retrieved_chunks": [
                {
                    "chunk": asdict(result.chunk),
                    "score": result.score,
                }
                for result in self.retrieved_chunks
            ],
            "token_usage": self.token_usage,
            "vector_backend": self.vector_backend,
        }


def run_pipeline(
    pdf_path: str,
    query: str,
    chunk_size: int = 256,
    top_k: int = 5,
    context_budget: int = 900,
    answer_budget: int = 200,
    embedding_dimensions: int = 384,
) -> RagOutput:
    raw_text = extract_text(pdf_path)
    chunks = chunk_text(raw_text, chunk_size=chunk_size)
    embeddings = generate_embeddings(chunks, dimensions=embedding_dimensions)
    vector_store = build_faiss_index(embeddings)
    retrieved = retrieve_top_k(
        query=query,
        chunks=chunks,
        vector_store=vector_store,
        top_k=top_k,
        context_budget=context_budget,
        embedding_dimensions=embedding_dimensions,
    )
    prompt = build_prompt(query, retrieved)

    tracker = TokenTracker()
    tracker.set("document_tokens", sum(chunk.token_count for chunk in chunks))
    tracker.set("chunk_count", len(chunks))
    tracker.set("query_tokens", count_tokens(query))
    tracker.set("retrieved_chunk_count", len(retrieved))
    tracker.set("retrieved_context_tokens", sum(item.chunk.token_count for item in retrieved))
    tracker.set("prompt_tokens", count_tokens(prompt))

    return RagOutput(
        prompt=prompt,
        chunks=chunks,
        retrieved_chunks=retrieved,
        token_usage=tracker.report(answer_budget=answer_budget),
        vector_backend=vector_store.backend,
    )


def build_prompt(query: str, retrieved_chunks: list[RetrievalResult]) -> str:
    context = "\n\n".join(
        f"[{result.chunk.id}, score={result.score:.3f}]\n{result.chunk.text}"
        for result in retrieved_chunks
    )
    if not context:
        context = "No relevant context retrieved."

    return (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"PDF context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
