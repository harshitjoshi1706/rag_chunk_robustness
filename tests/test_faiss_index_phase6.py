import sys
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embeddings import embed_chunk_texts, embed_query, load_embedding_model
from faiss_index import (
    add_vectors, create_index, load_index, save_index, search_index,
    validate_embedding_dimensions,
)


class FaissIndexPhase6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_embedding_model()
        cls.texts = ["A small page about chunking.", "Another page about finance.",
                     "A manual describing machine setup."]
        cls.vectors = embed_chunk_texts(cls.texts, cls.model)

    def test_index_type_dimension_and_count(self):
        index = create_index()
        self.assertIsInstance(index, faiss.IndexFlatIP)
        self.assertEqual(index.d, 384)
        self.assertEqual(index.ntotal, 0)
        self.assertEqual(add_vectors(index, self.vectors), 3)
        self.assertEqual(index.ntotal, 3)

    def test_save_load_preserves_type_dimension_and_count(self):
        index = create_index()
        add_vectors(index, self.vectors)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "debug.faiss"
            save_index(index, path)
            loaded = load_index(path)
            self.assertIsInstance(loaded, faiss.IndexFlatIP)
            self.assertEqual((loaded.d, loaded.ntotal), (384, 3))

    def test_search_shapes_and_self_match(self):
        index = create_index()
        add_vectors(index, self.vectors)
        query = embed_query(self.texts[1], self.model)
        scores, positions = search_index(index, query, k=2)
        self.assertEqual(scores.shape, (1, 2))
        self.assertEqual(positions.shape, (1, 2))
        self.assertTrue(np.issubdtype(positions.dtype, np.integer))
        self.assertEqual(int(positions[0, 0]), 1)
        self.assertAlmostEqual(float(scores[0, 0]), 1.0, places=5)

    def test_multiple_query_shape(self):
        index = create_index()
        add_vectors(index, self.vectors)
        scores, positions = search_index(index, self.vectors[:2], k=2)
        self.assertEqual(scores.shape, (2, 2))
        self.assertEqual(positions.shape, (2, 2))

    def test_wrong_dimensions_and_nonunit_vectors_raise(self):
        index = create_index()
        with self.assertRaisesRegex(ValueError, "384"):
            add_vectors(index, np.zeros((2, 383), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "384"):
            search_index(index, np.zeros(383, dtype=np.float32), k=1)
        with self.assertRaisesRegex(ValueError, "unit L2 norm"):
            validate_embedding_dimensions(np.ones((1, 384), dtype=np.float32))

    def test_empty_vectors_are_noop(self):
        index = create_index()
        self.assertEqual(add_vectors(index, np.empty((0, 384), dtype=np.float32)), 0)
        self.assertEqual(index.ntotal, 0)

    def test_empty_index_and_invalid_k_raise(self):
        index = create_index()
        with self.assertRaisesRegex(ValueError, "empty index"):
            search_index(index, self.vectors[:1], k=1)
        with self.assertRaisesRegex(ValueError, "positive"):
            search_index(index, self.vectors[:1], k=0)


if __name__ == "__main__":
    unittest.main()
