"""Frozen dataset order and recovery of interrupted, append-only results."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

directory = Path(__file__).resolve().parents[1] / "neural"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("comparison", directory / "compare-rescorers.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonRecovery(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "rows.jsonl"
        self.rows = [dict(id=str(i), input="nihk", gold=["你好"], document=str(i),
                          context="上文", category="chat") for i in range(3)]

    def result(self, i):
        return dict(self.rows[i], candidates=["你好"], top1=True, top5=True, elapsed_ms=1.)

    def test_full_selection_keeps_every_row_and_is_deterministic(self):
        self.path.write_text(json.dumps(self.rows))
        selected = comparison.select(self.path, 0, "seed")
        self.assertEqual({r["id"] for r in selected}, {"0", "1", "2"})
        self.assertEqual(selected, comparison.select(self.path, 0, "seed"))

    def test_partial_last_write_is_discarded_without_losing_complete_rows(self):
        complete = json.dumps(self.result(0)) + "\n"
        self.path.write_text(complete + '{"id": "1",')
        self.assertEqual(comparison.recover(self.path, self.rows), [self.result(0)])
        self.assertEqual(self.path.read_text(), complete)

    def test_changed_row_order_or_gold_is_rejected(self):
        self.path.write_text(json.dumps(self.result(0)) + "\n")
        with self.assertRaises(ValueError):
            comparison.recover(self.path, self.rows[::-1])
        with self.assertRaises(ValueError):
            comparison.recover(self.path, [dict(self.rows[0], gold=["拟好"])])

    def test_malformed_complete_line_is_not_silently_discarded(self):
        self.path.write_text('{"id":\n')
        with self.assertRaises(json.JSONDecodeError):
            comparison.recover(self.path, self.rows)


if __name__ == "__main__":
    unittest.main()
