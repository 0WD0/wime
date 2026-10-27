"""Benchmark scoring retains multiple answers and complete-string matching."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).resolve().parents[1] / "neural/benchmark-japanese.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class Scoring(unittest.TestCase):
    def test_accepts_alternative_answers_and_nfkc_but_not_partial_text(self):
        self.assertEqual(benchmark.outcome(["描く", "書く"], ["書く"]), 1)
        self.assertEqual(benchmark.outcome(["９８円"], ["98円"]), 1)
        self.assertIsNone(benchmark.outcome(["９８円"], ["98円"], False))
        self.assertIsNone(benchmark.outcome(["実装する"], ["実装"]))

    def test_top5_excludes_a_match_beyond_fifth_place(self):
        rows = [dict(suite="s", variant="v", rank=r, elapsed_ms=i, spellable=True)
                for i, r in enumerate([1, 5, 6, None])]
        result = benchmark.summarize(rows)["s"]["v"]
        self.assertEqual((result["total"], result["top1"], result["top5"]), (4, 1, 2))


if __name__ == "__main__":
    unittest.main()
