"""Strict final-label provenance, independently of inference and file I/O."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from corpus_tones import SYLLABLES, tone_policy_strict, validate_tones


class CompleteTones(unittest.TestCase):
    def row(self):
        return dict(gold=["你好"], input=SYLLABLES["ni"] + SYLLABLES["hao"],
                    pinyin=["ni", "hao"], tones=[2, 3],
                    tone_sources=["contextual_tone_sandhi", "mandarin_frontend"],
                    reading_rules=[["contextual_tone_sandhi"], []],
                    reading_evidence=dict(frontend_before_sandhi=["ni3", "hao3"], final=["ni2", "hao3"]))

    def test_actual_sandhi_and_complete_frontend_labels(self):
        validate_tones(self.row(), strict=True)

    def test_incomplete_labels_are_legacy_only(self):
        row = self.row()
        row.update(pinyin=["ni", None], tones=[2, None],
                   tone_sources=["contextual_tone_sandhi", "unresolved"])
        for stage in row["reading_evidence"].values():
            stage[1] = None
        validate_tones(row)
        with self.assertRaisesRegex(ValueError, "complete supported"):
            validate_tones(row, strict=True)

    def test_fabricated_or_unexplained_sandhi_is_rejected(self):
        cases = [
            ("reading_evidence", {"frontend_before_sandhi": ["ni2", "hao3"], "final": ["ni2", "hao3"]}),
            ("reading_evidence", {"frontend_before_sandhi": [None, "hao3"], "final": ["ni2", "hao3"]}),
            ("reading_evidence", {"frontend_before_sandhi": ["li3", "hao3"], "final": ["ni2", "hao3"]}),
            ("tone_sources", ["mandarin_frontend", "mandarin_frontend"]),
            ("reading_rules", [[], []]),
            ("reading_rules", [["contextual_tone_sandhi"], ["contextual_tone_sandhi"]]),
            ("reading_rules", [["contextual_tone_sandhi", "unreviewed"], []]),
            ("tone_sources", ["contextual_tone_sandhi", "unresolved"]),
            ("input", self.row()["input"] + "aa"),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                row = self.row()
                row[field] = value
                with self.assertRaises(ValueError):
                    validate_tones(row, strict=True)

    def test_correction_provenance_can_preserve_the_same_reading(self):
        for rule in ("phrase_correction:你好", "character_correction:好", "pos_correction:好:a"):
            row = self.row()
            row["tone_sources"][1] = "phrase_correction"
            row["reading_rules"][1] = [rule]
            validate_tones(row, strict=True)
            missing_rule = copy.deepcopy(row)
            missing_rule["reading_rules"][1] = []
            with self.assertRaisesRegex(ValueError, "correction rules"):
                validate_tones(missing_rule, strict=True)
            wrong_source = copy.deepcopy(row)
            wrong_source["tone_sources"][1] = "mandarin_frontend"
            with self.assertRaisesRegex(ValueError, "correction rules"):
                validate_tones(wrong_source, strict=True)

    def test_policy_must_opt_in_with_a_boolean(self):
        self.assertFalse(tone_policy_strict({"name": "gpt-sovits-mandarin-v1"}))
        self.assertTrue(tone_policy_strict({"require_complete": True}))
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tone_policy_strict({"require_complete": value})


if __name__ == "__main__":
    unittest.main()
