import unittest

from src.chunker import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_chunk_text_respects_128_token_limit(self):
        text = " ".join(f"policy{i}" for i in range(700))

        chunks = chunk_text(text, chunk_size=128, overlap=16)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 128 for chunk in chunks))

    def test_invalid_chunk_size_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("hello world", chunk_size=300)


if __name__ == "__main__":
    unittest.main()
