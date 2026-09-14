import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as runner


class RunnerResumeTests(unittest.TestCase):
    def test_atomic_query_part_and_aggregate(self):
        old_results, old_indexes = runner.RESULTS, runner.INDEXES
        with tempfile.TemporaryDirectory() as directory:
            runner.RESULTS = Path(directory) / "results"
            runner.INDEXES = Path(directory) / "indexes"
            try:
                paths = runner.paths_for("fixed_clean")
                runner.atomic_json(paths["profile"], {"run_id": runner.RUN_ID,
                    "noise_level": "clean", "chunker": "fixed", "status": "complete"})
                row = {name: "" for name in runner.QUERY_COLUMNS}
                row.update({"run_id": runner.RUN_ID, "noise_level": "clean",
                            "chunker": "fixed", "query_id": "official-1"})
                runner.atomic_csv(paths["parts"] / "batch_000000.csv", runner.QUERY_COLUMNS, [row])
                self.assertEqual(runner.completed_query_ids(paths["parts"]), {"official-1"})
                profiles = runner.aggregate_completed()
                self.assertEqual(len(profiles), 1)
                with (runner.RESULTS / "query_results.csv").open(newline="") as stream:
                    self.assertEqual([item["query_id"] for item in csv.DictReader(stream)], ["official-1"])
                self.assertFalse(any(Path(directory).rglob("*.tmp")))
            finally:
                runner.RESULTS, runner.INDEXES = old_results, old_indexes


if __name__ == "__main__":
    unittest.main()
