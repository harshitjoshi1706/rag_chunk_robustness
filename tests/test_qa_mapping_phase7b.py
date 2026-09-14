import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qa_mapping_audit import audit_qa, evidence_pairs, normalize


class QAMappingPhase7BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = audit_qa()

    def test_scalar_and_multi_page_evidence_pairing(self):
        self.assertEqual(evidence_pairs({"ID": "x", "evidence_page_no": 2,
                                         "evidence_context": ["a", "b"]}), [(2, "a"), (2, "b")])
        self.assertEqual(evidence_pairs({"ID": "y", "evidence_page_no": [1, 2],
                                         "evidence_context": ["a", "b"]}), [(1, "a"), (2, "b")])

    def test_whitespace_normalization_only(self):
        self.assertEqual(normalize("A  B\n C"), "A B C")

    def test_every_mapped_qa_page_exists_without_offset(self):
        result = self.audit
        self.assertEqual(result["total_qa"], result["mapped_docs"] + result["unmapped_docs"])
        self.assertEqual(result["offsets"]["0"]["qa_all_pages_exist"], result["mapped_docs"])
        self.assertEqual(result["offsets"]["0"].get("reference_missing", 0), 0)
        self.assertGreater(result["offsets"]["-1"].get("reference_missing", 0), 0)
        self.assertGreater(result["offsets"]["1"].get("reference_missing", 0), 0)

    def test_context_outcomes_partition_all_mapped_evidence(self):
        counts = self.audit["context_counts"]
        self.assertEqual(counts["total"], counts["exact"] + counts["normalized_only"]
                         + counts["unresolved"])
        self.assertEqual(self.audit["duplicate_ids"], 0)
        self.assertEqual(sum(self.audit["qa_by_domain"].values()), self.audit["mapped_docs"])


if __name__ == "__main__":
    unittest.main()
