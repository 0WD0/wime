"""Per-layout databases, tone/auxiliary preservation and phonetic features."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen"))
from wime_gen import Converter, SYLLABLE_ENCODING, build_database, load_scheme


class Schemes(unittest.TestCase):
    def test_algebra_descriptor_selects_its_upstream_dialect_and_rejects_missing_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # A small upstream-shaped fixture exercises dispatch without
            # requiring a downloaded word corpus in runtime CI.
            (root / "wanxiang_algebra.yaml").write_text(
                "pro:\n  微软双拼:\n    __append:\n      - xform/ling/l◯/\n"
                "  自然码:\n    __append:\n      - xform/ling/ly/\n"
                "  小鹤双拼:\n    __append:\n      - xform/ling/lk/\n"
                "  搜狗双拼:\n    __append:\n      - xform/ling/l◯/\n")
            for name, code in [("ziranma", "ly"), ("flypy", "lk"), ("sogou", "l;")]:
                with self.subTest(scheme=name):
                    converter, selected = load_scheme(
                        Path(__file__).resolve().parents[1] / "schemes" / f"{name}.json", root)
                    self.assertEqual(selected, name)
                    self.assertEqual(converter.codes("ling"), [code])
            descriptor = root / "missing.json"
            descriptor.write_text(json.dumps(dict(name="missing", algebra="missing")))
            with self.assertRaisesRegex(ValueError, "no pro/missing"):
                load_scheme(descriptor, root)

    def test_table_layout_preserves_auxiliary_tones_and_phonetic_collisions(self):
        scheme = dict(initials={"": "a", "l": "l", "n": "n"},
                      finals={"ing": ",", "in": ",", "v": "v", "ve": "t", "an": "n"})
        converter = Converter(scheme=scheme)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, database = root / "words.dict.yaml", root / "words.db"
            source.write_text("零\tlíng;wx\t100\n林\tlín;mm\t90\n女\tnǚ;vv\t80\n"
                              "略\tlüè;tk\t70\n安\tān;bn\t60\n零女\tlíng;wx nǚ;vv\t50\n")
            build_database([source], converter, database, "fixture")
            with sqlite3.connect(database) as db:
                self.assertEqual(db.execute("SELECT aux, tones FROM entry WHERE code='l,nv'").fetchone(),
                                 ("wx/vv", "2/3"))
                self.assertEqual(db.execute("SELECT word FROM entry WHERE code='an'").fetchone(), ("安",))
                self.assertEqual(db.execute("SELECT word FROM entry WHERE code='lt'").fetchone(), ("略",))
                self.assertEqual(db.execute("SELECT initial, final FROM syllable WHERE code='l,' ORDER BY final").fetchall(),
                                 [("l", "in"), ("l", "ing")])
                meta = dict(db.execute("SELECT key, value FROM meta"))
                self.assertEqual(meta["scheme"], "fixture")
                self.assertEqual(meta["syllable_encoding"], SYLLABLE_ENCODING)
                self.assertIn(",", meta["final_keys"])
                self.assertNotIn(",", meta["initial_keys"])

    def test_table_definitions_remain_supported_and_invalid_keys_fail_early(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("liangti", "liangti-fast"):
            converter, selected = load_scheme(root / "schemes" / f"{name}.json", root)
            self.assertEqual(selected, name)
            self.assertEqual(len(converter.codes("nǐ")[0]), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text(json.dumps(dict(name="bad", initials={"n": "N"}, finals={"i": "i"})))
            with self.assertRaisesRegex(ValueError, "single code keys"):
                load_scheme(path, path.parent)


if __name__ == "__main__":
    unittest.main()
