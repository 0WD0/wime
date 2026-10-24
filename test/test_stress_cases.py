"""Check that generated probes preserve typed readings and honest boundaries."""
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("stress_cases", Path(__file__).with_name("make-stress-cases.py"))
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)
audit_spec = importlib.util.spec_from_file_location("stress_audit", Path(__file__).with_name("analyze-stress.py"))
audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit)


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

    def test_independent_audit_rejects_wrong_keys_aux_and_promoted_partial_readings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dictionary.db"
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE entry (code, word, aux, weight)")
            db.executemany("INSERT INTO entry VALUES (?, ?, ?, ?)",
                           [("aa", "甲", "a", 10), ("aabb", "甲乙", "a/b", 10)])
            db.commit()
            db.close()
            rows = [dict(id="word", input="aaAbbB", constrained="甲乙", calls=1),
                    dict(id="partial", input="aabb", constrained="甲", calls=0),
                    dict(id="bad_aux", input="aaZbb", constrained="甲乙", calls=1),
                    dict(id="bad_code", input="aacc", constrained="甲乙", calls=1),
                    dict(id="bad_coverage", input="aabb", constrained="甲", calls=1)]
            errors = audit.check_constraints(rows, path)["violations"]
            self.assertEqual([r["id"] for r in errors], ["bad_aux", "bad_code", "bad_coverage"])


if __name__ == "__main__":
    unittest.main()
