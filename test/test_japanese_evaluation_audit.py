"""Frozen-label audits must not quietly rewrite identities or accept stale edits."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "evaluation_audit", Path(__file__).resolve().parents[1] / "neural/audit-japanese-evaluation.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def case(index=0, kana="にせんろねん", answers=None):
    return dict(suite="tatoeba", case_index=index, case_id=str(index + 100),
                kana=kana, context="", answers=answers or ["2006年"])


def correction(row):
    return dict(suite=row["suite"], case_index=row["case_index"], case_id=row["case_id"],
                before={"kana": row["kana"]}, after={"kana": "にせんろくねん"},
                reason="missing ku in source", evidence={"source_transcription": "[2006|にせんろ]年"})


class Audit(unittest.TestCase):
    def test_revision_keeps_original_and_identity(self):
        original = [case(), case(1)]
        before = copy.deepcopy(original)
        updated, changes = audit.apply_corrections(original, {"corrections": [correction(original[0])]})
        self.assertEqual(original, before)
        self.assertEqual([audit.identity(r) for r in updated], [audit.identity(r) for r in original])
        self.assertEqual(updated[0]["kana"], "にせんろくねん")
        self.assertEqual(updated[1], original[1])
        self.assertEqual(len(changes), 1)

    def test_stale_or_unidentified_change_is_rejected(self):
        row = case()
        item = correction(row)
        item["before"]["kana"] = "different"
        with self.assertRaisesRegex(ValueError, "stale"):
            audit.apply_corrections([row], {"corrections": [item]})
        item = correction(row)
        item["case_id"] = "wrong-source"
        with self.assertRaisesRegex(ValueError, "unknown"):
            audit.apply_corrections([row], {"corrections": [item]})
        item = correction(row)
        item["after"]["answers"] = ["a new answer"]
        with self.assertRaisesRegex(ValueError, "guard"):
            audit.apply_corrections([row], {"corrections": [item]})

    def test_duplicates_are_reported_not_removed(self):
        rows = [case(), case(1)]
        report = audit.structure_report(rows)
        self.assertEqual(report["total"], 2)
        self.assertEqual(len(report["duplicate_inputs"][0]), 2)
        with self.assertRaisesRegex(ValueError, "duplicate case identity"):
            audit.structure_report([case(), case()])

    def test_literal_numbers_allowed_but_small_ke_and_controls_flagged(self):
        rows = [case(kana="2006ねん"), case(1, "すうゖかげつ"), case(2, "かな\u200b")]
        report = audit.structure_report(rows)
        self.assertEqual([r["case_index"] for r in report["unusual_characters"]], [1, 2])

    def test_bad_schema_and_empty_answers_fail(self):
        for row in [case(answers=[""]), dict(case(), kana=""), dict(case(), case_index=True)]:
            with self.assertRaises(ValueError):
                audit.validate_cases([row])

    def test_different_existing_artifact_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            audit.write_json(path, {"v": 1})
            audit.write_json(path, {"v": 1})
            with self.assertRaisesRegex(ValueError, "refusing"):
                audit.write_json(path, {"v": 2})
            self.assertIn('"v": 1', path.read_text())


if __name__ == "__main__":
    unittest.main()
