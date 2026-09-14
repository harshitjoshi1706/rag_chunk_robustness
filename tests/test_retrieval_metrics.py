import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retrieval_metrics import is_page_relevant, score_ranking


class RetrievalMetricTests(unittest.TestCase):
    def setUp(self):
        self.qa = {"doc_name": "doc", "evidence_pages": (2, 5)}

    def chunk(self, doc, pages):
        return {"document_id": doc, "source_page_ids": pages}

    def test_multi_page_and_cross_page_relevance(self):
        self.assertTrue(is_page_relevant(self.chunk("doc", [1, 2]), self.qa))
        self.assertTrue(is_page_relevant(self.chunk("doc", [5]), self.qa))
        self.assertFalse(is_page_relevant(self.chunk("other", [2]), self.qa))

    def test_rank_metrics(self):
        ranking = [self.chunk("other", [2]), self.chunk("doc", [1]),
                   self.chunk("doc", [5])]
        self.assertEqual(score_ranking(ranking, self.qa), {
            "hit_at_1": 0, "hit_at_5": 1, "reciprocal_rank": 1 / 3,
            "first_relevant_rank": 3,
        })
        self.assertEqual(score_ranking([self.chunk("doc", [2])], self.qa)["hit_at_1"], 1)
        self.assertEqual(score_ranking(ranking[:2], self.qa)["first_relevant_rank"], None)


if __name__ == "__main__":
    unittest.main()
