"""Japanese kana decoding with toy models: no GPU, model or dictionary."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
import torch  # noqa: E402

import japanese as ja  # noqa: E402

VOCAB = ["<pad>", "\n", "日本", "語", "後", "誤", "に", "ほ", "ん", "ご", "、", "日本語"]
UNITS = {"にほん": [["日本", [2]]], "ご": [["語", [3]], ["後", [4]], ["誤", [5]], ["ご", [9]]],
         "に": [["に", [6]]], "ほ": [["ほ", [7]]], "ん": [["ん", [8]]], "にほんご": [["日本語", [11]]]}


class Tokenizer:
    def __len__(self):
        return len(VOCAB)

    def __call__(self, text):
        return dict(input_ids=[VOCAB.index(text)] if text in VOCAB else [1])


class Toy(ja.Decoder):
    """The LM likes 日本 then 語; the kana model is indifferent."""

    def __init__(self, units=UNITS, validation=None):
        path = Path(tempfile.mkdtemp()) / "table.json"
        path.write_text(json.dumps(dict(units=units, validation_units=validation), ensure_ascii=False))
        super().__init__(ja.Table(path, Tokenizer()), None, None, weight=1.0)
        self.forwards = 0

    def lm_context(self, context):
        return [1]

    def lm_next(self, context, sequences):
        self.forwards += 1
        rows = torch.full((len(sequences), len(VOCAB)), -5.0)
        for i, s in enumerate(sequences):
            rows[i, 3 if s and s[-1] == 2 else 2] = -0.5
        return rows

    def lm_tails(self, context, prefixes, tails):
        return [-5.0 * len(t) for t in tails]

    def model_score(self, context, kana, texts):
        return [0.0] * len(texts)


class Decoding(unittest.TestCase):
    def test_tokens_spell_the_kana(self):
        texts = [t for t, _ in Toy().decode("", "にほんご")]
        self.assertEqual(texts[0], "日本語")
        self.assertIn("日本後", texts)

    def test_typing_reuses_kept_frontiers(self):
        decoder = Toy()
        decoder.decode("", "にほん")
        before = decoder.forwards
        decoder.decode("", "にほんご")
        self.assertEqual(decoder.forwards - before, 1)
        before = decoder.forwards
        decoder.decode("", "にほん")
        self.assertEqual(decoder.forwards, before)

    def test_pending_keys_give_a_provisional_last_kana(self):
        texts = [t for t, _ in Toy().decode("", "にほん", pending=["ご"])]
        self.assertTrue(texts and all(len(t) >= 2 for t in texts))
        self.assertEqual(texts[0], "日本語")
        self.assertEqual(Toy().decode("", "にほん", pending=["ぱ"]), [])

    def test_explicit_spans_convert_shifted_kana_and_keep_the_rest(self):
        # にほん converted, ご kept: 日本ご; all converted: 日本語; none: the kana themselves.
        self.assertEqual([t for t, _ in Toy().decode("", "にほんご", shift="1110", explicit=True)], ["日本ご"])
        self.assertEqual({t for t, _ in Toy().decode("", "にほんご", shift="1111", explicit=True)},
                         {"日本語", "日本後", "日本誤"})
        self.assertEqual([t for t, _ in Toy().decode("", "にほんご", shift="0000", explicit=True)], ["にほんご"])
        decoder = Toy()
        decoder.decode("", "にほん")
        # Using Shift makes every earlier kana constrained: decoded again from the start.
        self.assertEqual([t for t, _ in decoder.decode("", "にほん", shift="000", explicit=True)], ["にほん"])
        self.assertEqual([t for t, _ in Toy().decode("", "にほん", ["ご"], "111", True, False)], ["日本ご"])

    def test_unknown_punctuation_is_typed_literally(self):
        self.assertTrue(any(t.endswith("、") for t, _ in Toy().decode("", "にほん、")))

    def test_exact_dictionary_readings_can_exceed_inferred_token_limit(self):
        reading = "に" * (ja.MAXREAD + 1)
        decoder = Toy(dict(UNITS, **{reading: [["日本語", [11]]]}))
        self.assertTrue(decoder.table.spells(reading, "日本語"))
        self.assertIn("日本語", [t for t, _ in decoder.decode("", reading)])
        self.assertFalse(decoder.table.spells(reading + "ご", "日本語"))


class FakeZenz:
    """Generates one spellable reading, one that drops ご and one the table cannot write."""

    def __init__(self):
        self.calls = 0

    def generate(self, context, kana):
        self.calls += 1
        return ["日本後", "日本", "ニホンゴ"]

    def score(self, context, kana, texts):
        return [0.0 if t == "日本後" else -10.0 for t in texts]


class Reranking(unittest.TestCase):
    def test_table_spells_only_what_reads_as_the_kana(self):
        table = Toy().table
        self.assertTrue(table.spells("にほんご", "日本語"))
        self.assertTrue(table.spells("にほんご、", "日本ご、"))
        self.assertFalse(table.spells("にほんご", "日本"))
        self.assertTrue(table.spells("にほんご", "日本ご", ("c", "c", "c", "k")))
        self.assertFalse(table.spells("にほんご", "日本語", ("c", "c", "c", "k")))

    def test_zenz_joins_and_ranks_only_spellable_readings_after_pending_keys(self):
        decoder = Toy()
        decoder.zenz = FakeZenz()
        decoder.decode("", "にほん", pending=["ご"])
        self.assertEqual(decoder.zenz.calls, 0)
        ranked = [t for t, _ in decoder.decode("", "にほんご")]
        self.assertEqual(ranked[0], "日本後")
        self.assertIn("日本語", ranked)
        self.assertNotIn("日本", ranked)
        self.assertNotIn("ニホンゴ", ranked)

    def test_extended_coverage_does_not_require_expanding_the_zenz_search_beam(self):
        base = dict(UNITS, ご=[["語", [3]], ["ご", [9]]])
        extra = {"ご": [["後", [4]]]}
        decoder = Toy(base, extra)
        self.assertNotIn(("後", (4,)), decoder.table.search("ご"))
        self.assertTrue(decoder.table.spells("にほんご", "日本後"))
        decoder.zenz = FakeZenz()
        self.assertEqual(decoder.decode("", "にほんご")[0][0], "日本後")
        # Without zenz, the full table remains available to the ordinary beam.
        self.assertIn("日本後", [t for t, _ in Toy(base, extra).decode("", "にほんご")])


class Lexicon(unittest.TestCase):
    def test_each_reading_keeps_its_most_frequent_writings(self):
        lexicon = {"加奈": {"かな": 3}, "仮名": {"かな": 90, "かめい": 5}, "哉": {"かな": 1}}
        self.assertEqual(sorted(ja.frequent(lexicon, 2)), [("仮名", "かな"), ("仮名", "かめい"), ("加奈", "かな")])

    def test_unprioritized_word_keeps_its_reading_across_token_boundaries(self):
        class SplitTokenizer:
            vocab = ["実", "装", "す", "る", "今", "日"]

            def __len__(self):
                return len(self.vocab)

            def decode(self, ids):
                return "".join(self.vocab[i] for i in ids)

            def __call__(self, text):
                return dict(input_ids=[self.vocab.index(c) if c in self.vocab else 0 for c in text])

        # The model splits 実装, and neither word has JMdict's common flag.
        # Its whole-word reading must survive even without a corpus lexicon.
        writings = {"実装": {"じっそう"}, "実装する": {"じっそうする"}, "今日": {"きょう"}}
        chars = {"実": {("じつ", "")}, "装": {("そう", "")}, "今": {("いま", "")}, "日": {("ひ", "")}}
        with patch.object(ja, "jmdict", return_value=(writings, set())), \
                patch.object(ja, "kanjidic", return_value=chars):
            units = ja.build_table(SplitTokenizer(), "unused", "unused")
        self.assertTrue(ja.spells(units.get, "じっそう", "実装"))
        # Unlike regular sound changes, 今日 needs its dictionary word reading.
        self.assertIn(["今日", [4, 5]], units["きょう"])
        self.assertTrue(ja.spells(units.get, "じっそうする", "実装する"))
        # Reuse the shorter word; don't inflate the beam with redundant compounds.
        self.assertNotIn("じっそうする", units)
        self.assertFalse(ja.spells(units.get, "じっそうする", "実装"))
        self.assertFalse(ja.spells(units.get, "じっそうする", "実装する", "kkkkkk"))
        before = sum(map(len, units.values()))
        ja.complete_dictionary(units, SplitTokenizer(), writings)
        self.assertEqual(sum(map(len, units.values())), before)

    def test_byte_split_characters_kana_and_long_dictionary_words_are_covered(self):
        class ByteTokenizer:
            def __len__(self):
                return 256

            def decode(self, ids):
                return bytes(ids).decode("utf-8", errors="replace")

            def __call__(self, text):
                return dict(input_ids=list(text.encode("utf-8")))

        writing, reading = "取締役会設置会社", "とりしまりやくかいせっちがいしゃ"
        writings = {"罠": {"わな"}, writing: {reading}}
        rare = {"髭": {("ひげ", "")}, "叩": {("たた", "く")}}
        with patch.object(ja, "jmdict", return_value=(writings, set())), \
                patch.object(ja, "kanjidic", side_effect=lambda _path, graded=True: {} if graded else rare):
            extra = {}
            search = ja.build_table(ByteTokenizer(), "unused", "unused", validation_units=extra)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.json"
            path.write_text(json.dumps(dict(units=search, validation_units=extra)))
            table = ja.Table(path, ByteTokenizer())
        units = table.index
        required = ja.required_readings(writings, rare)
        self.assertEqual(ja.missing_readings(units, required), [])
        for text, kana in [("罠", "わな"), ("髭", "ひげ"), ("叩いて", "たたいて"),
                           ("ゔ", "ゔ"), ("ヅ", "づ"), (writing, reading)]:
            self.assertTrue(ja.spells(units.get, kana, text), (kana, text))
        self.assertFalse(ja.spells(units.get, "わなでした", "罠"))

    def test_dictionary_reading_restrictions_and_kana_only_forms_are_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dict.db"
            db = sqlite3.connect(path)
            db.executescript("""
                CREATE TABLE entries (id INTEGER, frequency INTEGER);
                CREATE TABLE kanji (id INTEGER, docid INTEGER, priority INTEGER);
                CREATE TABLE kanjiText_content (docid INTEGER, c0reading TEXT);
                CREATE TABLE kana (id INTEGER, docid INTEGER, nokanji INTEGER, restrictedTo TEXT);
                CREATE TABLE kanaText_content (docid INTEGER, c0reading TEXT);
                INSERT INTO entries VALUES (1, 1);
                INSERT INTO kanji VALUES (1, 10, 0), (1, 11, 0);
                INSERT INTO kanjiText_content VALUES (10, '生'), (11, '生き');
                INSERT INTO kana VALUES (1, 20, 0, '0'), (1, 21, 0, '1'),
                                        (1, 22, 1, NULL), (1, 23, 0, NULL);
                INSERT INTO kanaText_content VALUES (20, 'なま'), (21, 'いき'),
                                                    (22, 'ナマ'), (23, 'せい');
            """)
            db.close()
            readings, common = ja.jmdict(path)
        self.assertEqual(dict(readings), {"生": {"なま", "せい"}, "生き": {"いき", "せい"}})
        self.assertEqual(common, {"生", "生き"})

    def test_okurigana_stems_keep_compound_readings_when_the_ending_changes(self):
        required = ja.required_readings({"盗る": {"とる"}, "接する": {"せっする"},
                                         "辛抱強い": {"しんぼうづよい"}}, {})
        index = {r: [(w, ()) for w, rs in required.items() if r in rs]
                 for rs in required.values() for r in rs}
        for kana, text in [("とられちゃった", "盗られちゃった"), ("せっしている", "接している"),
                           ("しんぼうづよく", "辛抱強く")]:
            self.assertTrue(ja.spells(index.get, kana, text), (kana, text))
        self.assertFalse(ja.spells(index.get, "しんぼうつよく", "辛抱強く"))

    def test_sound_changes_cross_token_boundaries_in_productive_compounds(self):
        required = ja.required_readings({"料理": {"りょうり"}},
                                         {"一": {("いち", "")}, "本": {("ほん", "")}})
        index = {r: [(w, ()) for w, rs in required.items() if r in rs]
                 for rs in required.values() for r in rs}
        for kana, text in [("りょうりぼん", "料理本"), ("いっぽん", "一本")]:
            self.assertTrue(ja.spells(index.get, kana, text), (kana, text))
        self.assertFalse(ja.spells(index.get, "りょうりぼん", "料理"))

    def test_capped_composition_is_independent_of_python_hash_seed(self):
        script = """import japanese as ja
chars = {c: {(r, '') for r in ['あ', 'い', 'う', 'え', 'お']} for c in '実装'}
print(sorted(ja.compose('実装', chars, {}, limit=3)))
"""
        outputs = [subprocess.check_output([sys.executable, "-c", script], env=dict(
            os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=str(Path(ja.__file__).parent)))
                   for seed in (1, 7, 31)]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])


class CausalModel:
    """Tiny causal model: each logit depends on all preceding token IDs."""

    def __init__(self):
        self.model = self
        self.embedding = torch.arange(48, dtype=torch.float32).reshape(12, 4).sin()
        self.head = torch.arange(48, dtype=torch.float32).reshape(4, 12).cos()

    def __call__(self, input_ids):
        return SimpleNamespace(last_hidden_state=self.embedding[input_ids].cumsum(dim=1))

    def lm_head(self, hidden):
        return hidden @ self.head

    def logits(self, tokens):
        hidden = self(torch.tensor([tokens])).last_hidden_state
        return self.lm_head(hidden)[0].log_softmax(-1)


class ModelBatches(unittest.TestCase):
    def setUp(self):
        self.model = CausalModel()
        rescorer = SimpleNamespace(model=self.model, pad=11, device="cpu")
        self.decoder = ja.Decoder(None, rescorer, None)

    def test_next_matches_individual_prefixes_with_right_padding(self):
        context, sequences = [1, 2], [(), (3,), (4, 5, 6)]
        expected = torch.stack([self.model.logits(context + list(s))[-1] for s in sequences])
        actual = self.decoder.lm_next(context, sequences)
        torch.testing.assert_close(actual, expected)

    def test_tails_score_each_token_after_its_own_prefix(self):
        context, prefixes, tails = [1, 2], [(), (3,), (4, 5)], [(6, 7), (8,), (9, 3, 4)]
        expected = []
        for prefix, tail in zip(prefixes, tails):
            logits = self.model.logits(context + list(prefix) + list(tail))
            start = len(context) + len(prefix) - 1
            expected.append(sum(float(logits[start + i, t]) for i, t in enumerate(tail)))
        actual = self.decoder.lm_tails(context, prefixes, tails)
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))

    def test_kana_scores_match_individual_variable_length_texts(self):
        model = ja.KanaModel.__new__(ja.KanaModel)
        model.device, model.index = "cpu", {c: i + 3 for i, c in enumerate("日本語ん後")}
        model.model = lambda x: (self.model.lm_head(self.model(x).last_hidden_state), None)
        prompt, texts = [1, 2], ["語", "日本語", "ん後"]
        expected = []
        for text in texts:
            tokens = [model.index[c] for c in text]
            logits = self.model.logits(prompt + tokens)
            expected.append(sum(float(logits[len(prompt) - 1 + i, t]) for i, t in enumerate(tokens)))
        actual = model.score(prompt, texts)
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))


if __name__ == "__main__":
    unittest.main()
