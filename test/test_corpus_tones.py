"""Strict final-label provenance, independently of inference and file I/O."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from corpus_tones import SYLLABLES, compact_tones, tone_source_counts, validate_tones


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

    def test_compaction_keeps_only_sparse_provenance(self):
        row = self.row()
        compact = compact_tones(row)
        self.assertEqual(compact, dict(gold=["你好"], input=row["input"], pinyin=["ni", "hao"], tones=[2, 3],
                                       reading_notes=[dict(index=0, before_tone=3,
                                                           rules=["contextual_tone_sandhi"])]))
        self.assertEqual(tone_source_counts(compact), tone_source_counts(row))
        row["tone_sources"][1] = "phrase_correction"
        row["reading_rules"][1] = ["phrase_correction:你好"]
        self.assertEqual(compact_tones(row)["reading_notes"][1], dict(index=1, rules=["phrase_correction:你好"]))
        row["tones"] = [3, 3]
        row["reading_evidence"]["final"] = ["ni3", "hao3"]
        row["tone_sources"] = ["mandarin_frontend"] * 2
        row["reading_rules"] = [[], []]
        self.assertNotIn("reading_notes", compact_tones(row))

    def test_invalid_sparse_evidence_is_rejected(self):
        compact = compact_tones(self.row())
        for notes in ([dict(index=0, rules=["contextual_tone_sandhi"])],
                      [dict(index=0, rules=["contextual_tone_sandhi"], before_tone=2)],
                      [dict(index=0, rules=["contextual_tone_sandhi"], before_tone=0)],
                      [dict(index=0, rules=["contextual_tone_sandhi"], before_tone=True)],
                      [dict(index=2, rules=["character_correction:好"])],
                      [dict(index=0, rules=[])],
                      [dict(index=0, rules=["unknown-rule"])],
                      compact["reading_notes"] * 2,
                      [dict(index=1, rules=["character_correction:好"]), *compact["reading_notes"]]):
            with self.subTest(notes=notes), self.assertRaises(ValueError):
                validate_tones(dict(compact, reading_notes=notes))
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            validate_tones(dict(self.row(), reading_notes=compact["reading_notes"]))


if __name__ == "__main__":
    unittest.main()
