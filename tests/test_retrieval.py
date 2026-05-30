import unittest

from src.chunker import chunk_text
from src.chunker import Chunk
from src.embeddings import generate_embeddings
from src.retriever import retrieve_top_k
from src.vector_store import build_faiss_index


class RetrievalTests(unittest.TestCase):
    def test_retrieve_top_k_returns_relevant_chunk_under_budget(self):
        text = (
            "Payment terms are net 30 days after invoice receipt.\n\n"
            "Termination requires written notice and a cure period.\n\n"
            "Support hours are Monday through Friday."
        )
        chunks = chunk_text(text, chunk_size=128, overlap=0)
        embeddings = generate_embeddings(chunks)
        store = build_faiss_index(embeddings)

        results = retrieve_top_k(
            query="What are the payment terms?",
            chunks=chunks,
            vector_store=store,
            top_k=3,
            context_budget=128,
        )

        self.assertTrue(results)
        self.assertIn("Payment terms", results[0].chunk.text)
        self.assertLessEqual(sum(item.chunk.token_count for item in results), 128)

    def test_toc_chunk_does_not_beat_answer_chunk(self):
        chunks = [
            Chunk(
                id="chunk-0000",
                text=(
                    "[page 10]\n"
                    "4.\n"
                    "Service Mesh: Service-to-Service Traffic Management\n"
                    ".\n.\n.\n.\n.\n.\n.\n"
                    "87\n"
                    "What Is Service Mesh?\n"
                    "90\n"
                    "What Functionality Does a Service Mesh Provide?\n"
                    "92"
                ),
                token_count=58,
                chunk_size=128,
            ),
            Chunk(
                id="chunk-0001",
                text=(
                    "[page 90]\n"
                    "What Is Service Mesh?\n"
                    "A service mesh is a dedicated infrastructure layer for handling "
                    "service-to-service communication. It provides traffic management, "
                    "security, observability, and policy enforcement between services."
                ),
                token_count=45,
                chunk_size=128,
            ),
            Chunk(
                id="chunk-0002",
                text="Release management covers deployment workflows and rollback strategy.",
                token_count=10,
                chunk_size=128,
            ),
        ]
        embeddings = generate_embeddings(chunks)
        store = build_faiss_index(embeddings)

        results = retrieve_top_k(
            query="What Is Service Mesh",
            chunks=chunks,
            vector_store=store,
            top_k=1,
            context_budget=128,
        )

        self.assertEqual(results[0].chunk.id, "chunk-0001")
        self.assertGreater(results[0].quality_score, 1.0)


if __name__ == "__main__":
    unittest.main()
