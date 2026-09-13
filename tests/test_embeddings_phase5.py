import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embeddings import (
    EMBEDDING_DIMENSION,
    embed_chunk_texts,
    embed_query,
    load_embedding_model,
)


class EmbeddingPhase5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_embedding_model()

    def test_multiple_chunk_shape_dimension_and_norm(self):
        vectors = embed_chunk_texts(
            ["The first short chunk.", "A different second chunk.", "Third chunk for batching."],
            self.model,
            batch_size=2,
        )
        self.assertIsInstance(vectors, np.ndarray)
        self.assertEqual(vectors.shape, (3, EMBEDDING_DIMENSION))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_query_shape_dimension_and_norm(self):
        vector = embed_query("What is chunking?", self.model)
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (EMBEDDING_DIMENSION,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_empty_chunks(self):
        vectors = embed_chunk_texts([], self.model)
        self.assertEqual(vectors.shape, (0, EMBEDDING_DIMENSION))
        self.assertEqual(vectors.dtype, np.float32)

    def test_blank_input_rejected(self):
        with self.assertRaises(ValueError):
            embed_chunk_texts(["valid", "  "], self.model)
        with self.assertRaises(ValueError):
            embed_query("", self.model)

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            embed_chunk_texts(["valid"], self.model, batch_size=0)


if __name__ == "__main__":
    unittest.main()
