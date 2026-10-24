"""Check that generated probes preserve typed readings and honest boundaries."""
import importlib.util
from pathlib import Path
import sqlite3
import unittest

spec = importlib.util.spec_from_file_location("stress_cases", Path(__file__).with_name("make-stress-cases.py"))
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)


class StressCases(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE entry (code, word, aux, weight)")
        self.db.executemany("INSERT INTO entry VALUES (?, ?, ?, ?)",
                            [("vf", "真", "ub", 10), ("qd", "枪", "mv", 5)])
        self.row = dict(id="test", context="之前", input="vfdkvfqd", gold=["真刀真枪"],
                        pinyin="zhen dao zhen qiang")
        self.rows = {r["id"]: r for r in stress.variants(self.row, self.db)}

    def tearDown(self):
        self.db.close()

    def test_commit_only_adds_the_known_prefix_and_keeps_the_original_reading(self):
        row = self.rows["test/split-3"]
        self.assertEqual((row["context"], row["input"], row["gold"]), ("之前真刀真", "qd", ["枪"]))
        self.assertEqual(row["pinyin"], "qiang")
        self.assertEqual(self.row["context"], "之前")

    def test_beyond_window_equivalence_is_explicit_and_exact(self):
        control, other = self.rows["test/window_control"], self.rows["test/outside_window"]
        self.assertEqual(len(control["context"]), 64)
        self.assertEqual(other["context"][-64:], control["context"])
        self.assertEqual(other["equivalent_to"], control["id"])
        self.assertEqual(other["input"], control["input"])

    def test_auxiliary_code_stays_attached_to_its_syllable(self):
        self.assertEqual(self.rows["test/aux-0-1"]["input"], "vfUdkvfqd")
        self.assertEqual(self.rows["test/aux-3-2"]["input"], "vfdkvfqdMV")
        self.assertTrue(all(row["input"].lower() != "vfdkvfqd" for row in self.rows.values()
                            if row["variant"] == "auxiliary"))


if __name__ == "__main__":
    unittest.main()
