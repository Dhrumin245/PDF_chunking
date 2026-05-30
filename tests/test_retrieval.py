import unittest

from src.chunker import chunk_text
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


if __name__ == "__main__":
    unittest.main()
