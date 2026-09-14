"""Regression for delimiter-only recursive chunks in clean OHRBench."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunkers import load_tokenizer, recursive_chunk
from ohrbench_adapter import attach_page_provenance, iter_documents, load_pages


class RecursiveResumeTests(unittest.TestCase):
    def test_first_clean_document_has_no_delimiter_only_chunks(self):
        pages = load_pages("clean")
        document = next(iter_documents(pages, "clean"))
        self.assertEqual(document["document_id"], "academic/2305.02437v3")
        chunks = recursive_chunk(document, load_tokenizer())
        attach_page_provenance(chunks)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(chunk["text"].strip() for chunk in chunks))
        self.assertTrue(all(chunk["source_paragraph_ids"] for chunk in chunks))
        self.assertTrue(all(chunk["source_page_ids"] for chunk in chunks))
        self.assertTrue(all(0 < chunk["token_count"] <= 254 for chunk in chunks))
        self.assertEqual([c["start_char"] for c in chunks],
                         sorted(c["start_char"] for c in chunks))
        self.assertEqual([c["chunk_id"] for c in chunks],
                         [f"{document['document_id']}_recursive_{i:04d}"
                          for i in range(len(chunks))])


if __name__ == "__main__":
    unittest.main()
