"""Corpus reading decisions on a tiny dictionary; no g2pW model required."""
import sqlite3
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from corpus_readings import Lexicon, Reader, key


class FakeG2PW:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, sentences):
        self.calls.append(list(sentences))
        return [[self.answers.get((s, k)) for k in range(len(s))] for s in sentences]


def lexicon(entries):
    path = Path(tempfile.mkdtemp()) / "db.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE entry (code TEXT, word TEXT, aux TEXT, weight INTEGER)")
    db.executemany("INSERT INTO entry VALUES (?, ?, '', ?)", entries)
    db.commit()
    return Lexicon(path)


K = {p: key(p) for p in ("ying", "le", "liao", "jie", "chang", "zhang", "hen", "de", "di", "man")}


class Readings(unittest.TestCase):
    def test_frequent_word_beats_a_rare_corpus_phrase(self):
        lex = lexicon([(K["ying"], "应", 36502), (K["jie"], "解", 3058), (K["le"], "了", 9000), (K["liao"], "了", 900),
                       (K["liao"] + K["jie"], "了解", 157763), (K["ying"] + K["le"], "应了", 700)])
        self.assertEqual([w for _, _, w in lex.segment("应了解")], [None, "了解"])
        g2pw = FakeG2PW({})
        self.assertEqual(Reader(lex, g2pw).keys(["应了解。"])[0][:3], [K["ying"], K["liao"], K["jie"]])
        self.assertEqual(g2pw.calls, [])  # word and pypinyin agree; no model call

    def test_rare_word_reading_is_outvoted(self):
        # A junk phrase with a wrong automatic reading must not decide alone.
        lex = lexicon([(K["ying"], "应", 9), (K["le"], "了", 9000), (K["liao"], "了", 900), (K["jie"], "解", 9),
                       (K["ying"] + K["le"] + K["jie"], "应了解", 35)])
        reader = Reader(lex, FakeG2PW({("应了解", 1): "liao3"}))
        self.assertEqual(reader.keys(["应了解"])[0][1], K["liao"])

    def test_tie_between_dictionary_and_pypinyin_goes_to_g2pw_context(self):
        lex = lexicon([(K["hen"], "很", 9), (K["chang"], "长", 9), (K["zhang"], "长", 9)])
        g2pw = FakeG2PW({("很长", 1): "chang2"})
        self.assertEqual(Reader(lex, g2pw).keys(["很长"])[0], [K["hen"], K["chang"]])
        self.assertEqual(len(g2pw.calls), 1)

    def test_trusted_word_and_g2pw_outvote_pypinyin(self):
        lex = lexicon([(K["man"], "慢", 9), (K["de"], "地", 9), (K["di"], "地", 9),
                       (K["man"] * 2 + K["de"], "慢慢地", 5000)])
        g2pw = FakeG2PW({})
        reader = Reader(lex, g2pw)
        # pypinyin reads 地 as di here, so the trusted word and pypinyin disagree:
        # the dictionary word and g2pW must decide, never pypinyin alone.
        g2pw.answers[("慢慢地", 2)] = "de5"
        self.assertEqual(reader.keys(["慢慢地"])[0], [K["man"], K["man"], K["de"]])

    def test_character_without_legal_keys_is_unusable(self):
        lex = lexicon([(K["hen"], "很", 9)])
        self.assertEqual(Reader(lex).keys(["很好"])[0], [K["hen"], None])


if __name__ == "__main__":
    unittest.main()
