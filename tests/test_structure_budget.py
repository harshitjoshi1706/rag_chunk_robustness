"""Regression for structure fallback source-index and token-budget failure."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunkers import load_tokenizer, structure_aware_chunk
from document_builder import build_document_text
from ohrbench_adapter import attach_page_provenance, iter_documents, load_pages


class StructureBudgetTests(unittest.TestCase):
    def test_3m_table_fallback_stays_within_embedding_budget(self):
        pages = load_pages("clean")
        document = next(d for d in iter_documents(pages, "clean")
                        if d["document_id"] == "finance/3M_2023Q2_10Q")
        tokenizer = load_tokenizer()
        full_text, _ = build_document_text(document)
        chunks = structure_aware_chunk(document, tokenizer)
        attach_page_provenance(chunks)
        self.assertGreater(len(chunks), 63)
        self.assertEqual(chunks[63]["chunk_id"],
                         "finance/3M_2023Q2_10Q_structure_0063")
        self.assertLessEqual(chunks[63]["token_count"], 254)
        self.assertEqual(chunks[63]["source_page_ids"], [14])
        self.assertTrue(all(c["text"].strip() and c["source_paragraph_ids"]
                            and c["source_page_ids"] for c in chunks))
        self.assertTrue(all(full_text[c["start_char"]:c["end_char"]] == c["text"]
                            for c in chunks))
        self.assertTrue(all(len(tokenizer(c["text"], add_special_tokens=True)["input_ids"])
                            <= 256 for c in chunks))

    def test_structural_markers_remain_recognized(self):
        tokenizer = load_tokenizer()
        document = {"document_id": "debug", "paragraphs": [
            {"paragraph_id": "PAGE_0", "text":
             "# Heading\nBody paragraph.\n\n- first item\n- second item\n\n"
             "| A | B |\n| 1 | 2 |\n\n```python\nprint(1)\n```\n"}
        ]}
        chunks = structure_aware_chunk(document, tokenizer)
        self.assertTrue(chunks)
        self.assertTrue(all(c["source_paragraph_ids"] == ["PAGE_0"] for c in chunks))
        self.assertTrue(all(c["token_count"] <= 254 for c in chunks))
        self.assertIn("# Heading", "".join(c["text"] for c in chunks))
        self.assertIn("```python", "".join(c["text"] for c in chunks))


if __name__ == "__main__":
    unittest.main()
