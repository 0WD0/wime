"""Strict final-label provenance, independently of inference and file I/O."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from corpus_tones import SYLLABLES, validate_tones


class CompleteTones(unittest.TestCase):
    def row(self):
        return dict(gold=["你好"], input=SYLLABLES["ni"] + SYLLABLES["hao"],
                    pinyin=["ni", "hao"], tones=[2, 3],
                    tone_sources=["contextual_tone_sandhi", "mandarin_frontend"],
                    reading_rules=[["contextual_tone_sandhi"], []],
                    reading_evidence=dict(frontend_before_sandhi=["ni3", "hao3"], final=["ni2", "hao3"]))

    def test_actual_sandhi_and_complete_frontend_labels(self):
        validate_tones(self.row())

    def test_unknown_is_not_a_valid_training_label(self):
        row = self.row()
        row.update(pinyin=["ni", None], tones=[2, None],
                   tone_sources=["contextual_tone_sandhi", "unresolved"])
        for stage in row["reading_evidence"].values():
            stage[1] = None
        with self.assertRaisesRegex(ValueError, "complete normalized"):
            validate_tones(row)

    def test_fabricated_or_unexplained_sandhi_is_rejected(self):
        cases = [
            ("reading_evidence", {"frontend_before_sandhi": ["ni2", "hao3"], "final": ["ni2", "hao3"]}),
            ("reading_evidence", {"frontend_before_sandhi": ["li3", "hao3"], "final": ["ni2", "hao3"]}),
            ("tone_sources", ["mandarin_frontend", "mandarin_frontend"]),
            ("reading_rules", [[], []]),
            ("reading_rules", [["contextual_tone_sandhi"], ["contextual_tone_sandhi"]]),
            ("input", SYLLABLES["li"] + SYLLABLES["hao"]),
            ("tones", [3, 3]),  # A citation tone cannot replace the final ni2.
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                row = self.row()
                row[field] = value
                with self.assertRaises(ValueError):
                    validate_tones(row)

    def test_correction_provenance_can_preserve_the_same_reading(self):
        for rule in ("phrase_correction:你好", "character_correction:好", "pos_correction:好:a"):
            row = self.row()
            row["tone_sources"][1] = "phrase_correction"
            row["reading_rules"][1] = [rule]
            validate_tones(row)
            missing_rule = copy.deepcopy(row)
            missing_rule["reading_rules"][1] = []
            with self.assertRaisesRegex(ValueError, "actual corrections"):
                validate_tones(missing_rule)
            wrong_source = copy.deepcopy(row)
            wrong_source["tone_sources"][1] = "mandarin_frontend"
            with self.assertRaisesRegex(ValueError, "actual corrections"):
                validate_tones(wrong_source)


if __name__ == "__main__":
    unittest.main()
