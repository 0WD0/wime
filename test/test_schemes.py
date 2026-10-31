"""Per-layout databases, tone/auxiliary preservation and phonetic features."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from wime_gen import Converter, SYLLABLE_ENCODING, build_database, load_scheme
import syllables


class Schemes(unittest.TestCase):
    def test_self_contained_layouts_define_zero_initials_finals_and_syllable_aliases(self):
        root = Path(__file__).resolve().parents[1]
        cases = {
            "mspy": {"a": ["oa"], "ai": ["ol"], "ān": ["oj"], "líng": ["l;"],
                     "jū": ["ju", "jy"], "jué": ["jt", "jv"], "lüè": ["lt", "lv"]},
            "ziranma": {"a": ["aa"], "ai": ["ai", "al"], "ān": ["aj", "an"], "líng": ["ly"],
                        "jū": ["ju", "jv"], "jué": ["jt"], "lüè": ["lt"]},
            "flypy": {"a": ["aa"], "ai": ["ad", "ai"], "ān": ["aj", "an"], "líng": ["lk"],
                      "jū": ["ju", "jv"], "jué": ["jt"], "lüè": ["lt"]},
            "sogou": {"a": ["oa"], "ai": ["ol"], "ān": ["oj"], "líng": ["l;"],
                      "jū": ["ju", "jy"], "jué": ["jt"], "lüè": ["lt"]},
        }
        for name, examples in cases.items():
            converter, selected = load_scheme(root / "schemes" / f"{name}.json")
            self.assertEqual(selected, name)
            for pinyin, expected in examples.items():
                with self.subTest(scheme=name, pinyin=pinyin):
                    self.assertEqual(converter.codes(pinyin), expected)
        canonical = json.loads((root / "neural/context-pinyin.json").read_text())["syllables"]
        converter, _ = load_scheme(root / "schemes/mspy.json")
        self.assertEqual({pinyin: converter.primary(pinyin) for pinyin in canonical}, canonical)

    def test_cli_builds_each_layout_without_an_upstream_algebra_file(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / "wanxiang_pro.dict.yaml").write_text("import_tables:\n  - words\n...\n")
            (data / "words.dict.yaml").write_text("零\tlíng;wx\t100\n安\tān;bn\t90\n")
            for name, code in [("mspy", "l;"), ("ziranma", "ly"), ("flypy", "lk"), ("sogou", "l;")]:
                database = data / f"{name}.db"
                command = [sys.executable, str(root / "gen/wime_gen.py"), "--rime-data", str(data),
                           "--database", str(database)]
                if name != "mspy":
                    command += ["--scheme", str(root / "schemes" / f"{name}.json")]
                subprocess.run(command, check=True, capture_output=True)
                with closing(sqlite3.connect(database)) as db:
                    self.assertEqual(db.execute("SELECT code FROM entry WHERE word='零'").fetchall(), [(code,)])

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
            with closing(sqlite3.connect(database)) as db:
                self.assertEqual(db.execute("SELECT aux, tones FROM entry WHERE code='l,nv'").fetchone(),
                                 ("wx/vv", "2/3"))
                self.assertEqual(db.execute("SELECT word FROM entry WHERE code='an'").fetchone(), ("安",))
                self.assertEqual(db.execute("SELECT word FROM entry WHERE code='lt'").fetchone(), ("略",))
                self.assertEqual(db.execute("SELECT initial, final FROM syllable WHERE code='l,' ORDER BY final").fetchall(),
                                 [("l", "ing")])
                meta = dict(db.execute("SELECT key, value FROM meta"))
                self.assertEqual(meta["scheme"], "fixture")
                self.assertEqual(meta["syllable_encoding"], SYLLABLE_ENCODING)
                self.assertIn(",", meta["final_keys"])
                self.assertNotIn(",", meta["initial_keys"])
            layout = syllables.database_layout(database)
            self.assertEqual(syllables.typed("nv", [(0, 3)], layout), [syllables.labelled("nü", 3)])
            self.assertEqual(syllables.typed("l,", layout=layout), [syllables.labelled("ling", 0)])
            self.assertEqual(syllables.typed("l", layout=layout), [(syllables.INITIAL["l"], 0, 0)])

    def test_microsoft_database_preserves_all_trained_features_without_a_scheme_branch(self):
        root = Path(__file__).resolve().parents[1]
        converter, _ = load_scheme(root / "schemes/mspy.json")
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "words.db"
            build_database([], converter, database, "anything")
            keys, half = syllables.database_layout(database)
            self.assertEqual({key: keys[key] for key in syllables.KEYS}, syllables.KEYS)
            self.assertEqual(half, syllables.HALF)
            self.assertEqual(syllables.typed("jv", layout=(keys, half)), [syllables.labelled("jue", 0)])
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("DELETE FROM meta WHERE key='syllable_encoding'")
            with self.assertRaisesRegex(ValueError, "rebuild the database"):
                syllables.database_layout(database)

    def test_invalid_keys_fail_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text(json.dumps(dict(name="bad", initials={"n": "N"}, finals={"i": "i"})))
            with self.assertRaisesRegex(ValueError, "single code keys"):
                load_scheme(path)


if __name__ == "__main__":
    unittest.main()
