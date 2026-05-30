from __future__ import annotations

import argparse
import json

from src.rag_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF RAG token optimization POC")
    parser.add_argument("--pdf", default="data/sample.pdf", help="Path to PDF file")
    parser.add_argument("--query", required=True, help="Question to retrieve context for")
    parser.add_argument("--chunk-size", type=int, default=256, choices=[128, 256, 512])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-budget", type=int, default=900)
    parser.add_argument("--answer-budget", type=int, default=200)
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args()

    result = run_pipeline(
        pdf_path=args.pdf,
        query=args.query,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        context_budget=args.context_budget,
        answer_budget=args.answer_budget,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"Vector backend: {result.vector_backend}")
    print("\nRetrieved chunks")
    print("----------------")
    for item in result.retrieved_chunks:
        print(
            f"{item.chunk.id} | score={item.score:.3f} | "
            f"tokens={item.chunk.token_count}"
        )

    print("\nToken usage")
    print("-----------")
    for key, value in result.token_usage.items():
        print(f"{key}: {value}")

    print("\nPrompt")
    print("------")
    print(result.prompt)


if __name__ == "__main__":
    main()
