import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chunkers import semantic_chunk, validate_semantic_chunks


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class AlternatingEncoder:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]
                         for i in range(len(texts))])


class SemanticPhase4ETests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()
        self.encoder = AlternatingEncoder()

    def test_minimum_suppresses_similarity_breaks(self):
        text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
        doc = {"document_id": "d", "paragraphs": [{"paragraph_id": "p", "text": text}]}
        chunks = semantic_chunk(doc, self.tokenizer, self.encoder,
                                similarity_threshold=0.5, min_tokens=7, max_tokens=20)
        self.assertEqual([c["token_count"] for c in chunks], [9, 3])
        self.assertEqual([c["source_paragraph_ids"] for c in chunks], [["p"], ["p"]])

    def test_oversized_sentence_is_split_and_provenance_preserved(self):
        text = " ".join(f"word{i}" for i in range(25)) + "."
        doc = {"document_id": "d", "paragraphs": [{"paragraph_id": "p", "text": text}]}
        chunks = semantic_chunk(doc, self.tokenizer, self.encoder,
                                similarity_threshold=0.5, min_tokens=4, max_tokens=10)
        self.assertTrue(validate_semantic_chunks(chunks, 10))
        self.assertTrue(all(c["token_count"] <= 10 for c in chunks))
        self.assertTrue(all(c["source_paragraph_ids"] == ["p"] for c in chunks))
        self.assertEqual(chunks[0]["start_char"], 0)
        self.assertEqual(chunks[-1]["end_char"], len(text))

    def test_invalid_minimum(self):
        doc = {"document_id": "d", "paragraphs": [{"paragraph_id": "p", "text": "Text."}]}
        for minimum in (0, 11):
            with self.assertRaises(ValueError):
                semantic_chunk(doc, self.tokenizer, self.encoder, min_tokens=minimum, max_tokens=10)


if __name__ == "__main__":
    unittest.main()
