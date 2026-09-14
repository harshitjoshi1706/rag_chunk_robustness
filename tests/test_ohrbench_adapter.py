import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ohrbench_adapter import attach_page_provenance, iter_documents
from official_qa import load_mapped_qas


class OHRBenchAdapterTests(unittest.TestCase):
    def test_sorted_pages_and_cross_page_provenance(self):
        pages = pd.DataFrame([
            {"domain": "finance", "doc_name": "finance/a", "page_idx": 2,
             "gt_text": "Second page."},
            {"domain": "finance", "doc_name": "finance/a", "page_idx": 1,
             "gt_text": "First page."},
        ]).sort_values(["doc_name", "page_idx"])
        doc = next(iter_documents(pages, "clean"))
        self.assertEqual([p["page_idx"] for p in doc["paragraphs"]], [1, 2])
        chunks = [{"source_paragraph_ids": ["PAGE_1", "PAGE_2"]}]
        attach_page_provenance(chunks)
        self.assertEqual(chunks[0]["source_page_ids"], [1, 2])

    def test_official_qa_counts(self):
        clean = pd.read_parquet("data/raw/OHR-Bench_v2.parquet",
                                columns=["doc_name", "page_idx", "gt_text"])
        mapped, excluded = load_mapped_qas(clean)
        self.assertEqual((len(mapped), len(excluded)), (8456, 42))
        self.assertEqual(sum(q["context_verified"] for q in mapped), 5277)
        self.assertEqual(sum(len(q["evidence_pages"]) for q in mapped), 9217)


if __name__ == "__main__":
    unittest.main()
