"""Corpus reading decisions on a tiny dictionary; no g2pW model required."""
import sqlite3
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from corpus_readings import HAN, Lexicon, Reader, key, numbered_reading


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

    def test_yijing_symbols_do_not_shift_following_readings(self):
        lex = lexicon([(K["hen"], "很", 9), (K["chang"], "长", 9), (K["zhang"], "长", 9)])
        self.assertIsNone(HAN.search("䷀䷁"))
        g2pw = FakeG2PW({("䷀䷁很长", 3): "chang2"})
        self.assertEqual(Reader(lex, g2pw).keys(["䷀䷁很长"])[0], [None, None, K["hen"], K["chang"]])

    def test_unknown_han_stay_aligned(self):
        lex = lexicon([(key("xi"), "㐂", 9), (key("ma"), "㐃", 9), (K["hen"], "很", 9)])
        self.assertEqual(Reader(lex).keys(["㐂㐃很"])[0], [key("xi"), key("ma"), K["hen"]])

    def test_misaligned_pypinyin_fails(self):
        lex = lexicon([(K["hen"], "很", 9)])
        with patch("corpus_readings.lazy_pinyin", return_value=[]):
            with self.assertRaisesRegex(ValueError, "pypinyin returned"):
                Reader(lex).keys(["很"])

    def test_g2pw_cardinality_and_alignment_fail(self):
        lex = lexicon([(K["chang"], "长", 9), (K["zhang"], "长", 9)])
        for response in ([], [["chang2"], ["zhang3"]], [None], [[]], [["chang2", "chang2"]]):
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, "g2pW"):
                Reader(lex, lambda sentences: response).keys(["长"])

    def test_only_disputed_positions_are_requested(self):
        lex = lexicon([(K["hen"], "很", 9), (K["chang"], "长", 9), (K["zhang"], "长", 9)])

        class PositionalG2PW(FakeG2PW):
            def read_positions(self, sentences, positions):
                self.positions = positions
                return super().__call__(sentences)

            def __call__(self, sentences):
                raise AssertionError("Reader must use the positional interface")

        g2pw = PositionalG2PW({("很长。", 1): "chang2"})
        self.assertEqual(Reader(lex, g2pw).keys(["很", "很长。"]),
                         [[K["hen"]], [K["hen"], K["chang"], None]])
        self.assertEqual(g2pw.calls, [["很长。"]])
        self.assertEqual(g2pw.positions, [[1]])

    def test_model_cannot_break_tie_when_mandarin_reading_is_missing(self):
        lex = lexicon([(key("ei"), "诶", 9), (key("xi"), "诶", 9)])
        g2pw = FakeG2PW({("诶", 0): "xi1"})
        # Real g2pW translates 诶 to 誒, whose trained labels omit ei.
        # Its fallback dictionary does include ei, but is never consulted for
        # a character already in the model's polyphonic label inventory.
        g2pw.enable_non_tradional_chinese = True
        g2pw.s2t_dict = {"诶": "誒"}
        g2pw.polyphonic_chars = [("誒", "ㄒㄧ1")]
        g2pw.char_bopomofo_dict = {"誒": ["ㄟ2", "ㄒㄧ1"]}
        g2pw.style_convert_func = {"ㄟ2": "ei2", "ㄒㄧ1": "xi1"}.get
        reader = Reader(lex, g2pw)
        self.assertEqual(reader.keys(["诶"]), [[None]])
        self.assertEqual(reader.methods["coverage_conflict"], 1)
        # With both readings represented the existing contextual tie-break
        # remains available; abstention is about coverage, not disagreement.
        g2pw.polyphonic_chars.append(("誒", "ㄟ2"))
        self.assertEqual(Reader(lex, g2pw).keys(["诶"]), [[key("xi")]])


class PositionalConverter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from corpus_g2pw import CorpusG2PWConverter
            from g2pw import G2PWConverter
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(f"optional g2pW runtime is unavailable: {error}")
        cls.converter_class, cls.upstream_class = CorpusG2PWConverter, G2PWConverter

    def test_query_filter_preserves_sentence_and_character_alignment(self):
        converter = self.converter_class.__new__(self.converter_class)
        # The first character of the second sentence has a static reading;
        # only polyphonic query records pass through the model.
        partial = [[None, None, None], ["hen3", None]]
        prepared = (["长。长", "长。长", "很长"], [0, 2, 1], [0, 0, 1], partial)
        with patch.object(self.upstream_class, "_prepare_data", return_value=prepared), \
                patch.object(self.upstream_class, "__call__", lambda self, sentences: self._prepare_data(sentences)):
            result = converter.read_positions(["长。长", "很长"], [[2], [0, 1]])
            self.assertEqual(result[:3], (["长。长", "很长"], [2, 1], [0, 1]))
            self.assertIs(result[3], partial)
            self.assertEqual(converter._prepare_data(["长。长", "很长"]), prepared)
            empty = converter.read_positions(["长。长", "很长"], [[], []])
            self.assertEqual(empty[:3], ([], [], []))
            self.assertIs(empty[3], partial)

    def test_query_validation_and_reset_after_failure(self):
        converter = self.converter_class.__new__(self.converter_class)
        for positions in ([], [[-1]], [[1]], [[0.5]], [[True]]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                converter.read_positions(["长"], positions)
        with patch.object(self.upstream_class, "__call__", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                converter.read_positions(["长"], [[0]])
        self.assertIsNone(converter._requested_positions)

    def test_expanding_lowercase_keeps_han_query_indices(self):
        converter = self.converter_class.__new__(self.converter_class)
        original = "EMİN GÜN SİRER的银行行长"
        sentences = [original]
        positions = [[i for i, char in enumerate(original) if char in "行长"]]

        def lower_and_query(instance, model_sentences):
            self.assertEqual(model_sentences, ["EM N GÜN S RER的银行行长"])
            lowered = model_sentences[0].lower()
            self.assertEqual(len(lowered), len(original))
            return [[lowered[i] for i in sorted(instance._requested_positions[0])]]

        with patch.object(self.upstream_class, "__call__", lower_and_query):
            self.assertEqual(converter.read_positions(sentences, positions), [["行", "行", "长"]])
        self.assertEqual(sentences, [original])  # model normalization never edits corpus input


class ToneReadings(unittest.TestCase):
    def test_same_key_tones_are_queried_and_keys_stay_unchanged(self):
        lex = lexicon([(key(p), c, 9) for c, p in (("好", "hao"), ("处", "chu"), ("和", "he"), ("喜", "xi"))])
        sentence = "好处和喜好"

        class PositionalG2PW(FakeG2PW):
            def read_positions(self, sentences, positions):
                self.positions = positions
                return self(sentences)

        g2pw = PositionalG2PW({(sentence, 0): "hao3", (sentence, 4): "hao4"})
        reader = Reader(lex, g2pw)
        previous = reader.keys([sentence])
        self.assertEqual(g2pw.calls, [])  # same-key tones were invisible to the old path
        row = reader.readings([sentence])[0]
        self.assertEqual(row["keys"], previous[0])
        self.assertIn(0, g2pw.positions[0])
        self.assertIn(4, g2pw.positions[0])
        self.assertEqual([row["tones"][0], row["tones"][4]], [3, 4])
        self.assertEqual([row["tone_sources"][0], row["tone_sources"][4]], ["agreement", "agreement"])

    def test_tone_conflict_keeps_both_contextual_predictions(self):
        lex = lexicon([(key("zhi"), "只", 9)])
        row = Reader(lex, FakeG2PW({("只", 0): "zhi1"})).readings(["只"])[0]
        self.assertEqual(row["keys"], [key("zhi")])
        self.assertEqual(row["pinyin"], ["zhi"])
        self.assertEqual(row["tones"], [None])
        self.assertEqual(row["tone_sources"], ["tone_conflict"])
        self.assertEqual(row["reading_evidence"], {"pypinyin": ["zhi3"], "g2pw": ["zhi1"]})

    def test_single_source_ambiguous_tone_is_unknown(self):
        lex = lexicon([(key("hao"), "好", 9)])
        row = Reader(lex).readings(["好"])[0]
        self.assertEqual(row["pinyin"], ["hao"])
        self.assertEqual(row["tones"], [None])
        self.assertEqual(row["tone_sources"], ["pypinyin_ambiguous"])
        self.assertEqual(row["reading_evidence"]["pypinyin"], ["hao3"])

    def test_neutral_tone_is_not_unknown_or_unusable(self):
        lex = lexicon([(key("de"), "的", 9), (key("xi"), "㐂", 9)])
        row = Reader(lex).readings(["的㐂。"])[0]
        self.assertEqual(row["keys"], [key("de"), key("xi"), None])
        self.assertEqual(row["pinyin"], ["de", None, None])
        self.assertEqual(row["tones"], [5, None, None])
        self.assertEqual(row["reading_evidence"]["pypinyin"], ["de5", None, None])
        self.assertEqual(row["tone_sources"][2], "non_han")

    def test_contextual_sandhi_is_not_replaced_with_citation_tone(self):
        lex = lexicon([(key("ni"), "你", 9), (key("hao"), "好", 9)])
        row = Reader(lex).readings(["你好"])[0]
        self.assertEqual(row["reading_evidence"]["pypinyin"], ["ni2", "hao3"])
        self.assertEqual(row["tones"], [2, None])
        self.assertEqual(row["keys"], [key("ni"), key("hao")])

    def test_merged_key_syllable_conflict_does_not_guess_an_inverse(self):
        lex = lexicon([(key("en"), "恩", 9)])
        self.assertEqual(key("en"), key("n"))
        g2pw = FakeG2PW({("恩", 0): "n1"})
        g2pw.polyphonic_chars = [("恩", "en1"), ("恩", "n1")]
        g2pw.style_convert_func = lambda reading: reading
        row = Reader(lex, g2pw).readings(["恩"])[0]
        self.assertEqual(row["keys"], [key("en")])
        self.assertEqual(row["pinyin"], [None])
        self.assertEqual(row["tones"], [None])
        self.assertEqual(row["tone_sources"], ["syllable_conflict"])
        self.assertEqual(row["reading_evidence"], {"pypinyin": ["en1"], "g2pw": ["n1"]})

    def test_numbered_evidence_requires_an_explicit_valid_tone(self):
        self.assertEqual(numbered_reading("lü4"), "lv4")
        self.assertEqual(numbered_reading("nu:e4"), "nve4")
        for reading in (None, "", "de", "de0", "de6", "㐂", "ma55"):
            self.assertIsNone(numbered_reading(reading))


if __name__ == "__main__":
    unittest.main()
