import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_results import aggregate, degradation


class ResultAnalysisTests(unittest.TestCase):
    def test_aggregate_and_subset_use_same_query_rows(self):
        rows = [
            {"chunker": "fixed", "noise_level": "clean", "query_id": "a",
             "context_verified": 1, "hit_at_1": 1, "hit_at_5": 1, "reciprocal_rank": 1},
            {"chunker": "fixed", "noise_level": "clean", "query_id": "b",
             "context_verified": 0, "hit_at_1": 0, "hit_at_5": 1, "reciprocal_rank": .2},
            {"chunker": "fixed", "noise_level": "mild", "query_id": "a",
             "context_verified": 1, "hit_at_1": 0, "hit_at_5": 0, "reciprocal_rank": 0},
            {"chunker": "fixed", "noise_level": "mild", "query_id": "b",
             "context_verified": 0, "hit_at_1": 0, "hit_at_5": 0, "reciprocal_rank": 0},
        ]
        data = pd.DataFrame(rows)
        primary = aggregate(data)
        subset = aggregate(data, subset=True)
        clean = primary[primary.noise_level == "clean"].iloc[0]
        self.assertEqual((clean.num_queries, clean.hit_at_1, clean.hit_at_5), (2, .5, 1))
        self.assertEqual(subset[subset.noise_level == "clean"].iloc[0].num_queries, 1)
        drop = degradation(primary)
        self.assertEqual(drop[drop.metric == "hit_at_1"].iloc[0].absolute_degradation, .5)
        self.assertEqual(drop[drop.metric == "hit_at_1"].iloc[0].relative_degradation_percent, 100)


if __name__ == "__main__":
    unittest.main()
