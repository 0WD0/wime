"""Japanese kana decoding with toy models: no GPU, model or dictionary."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

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

    def __init__(self):
        path = Path(tempfile.mkdtemp()) / "table.json"
        path.write_text(json.dumps(dict(units=UNITS), ensure_ascii=False))
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


class Lexicon(unittest.TestCase):
    def test_each_reading_keeps_its_most_frequent_writings(self):
        lexicon = {"加奈": {"かな": 3}, "仮名": {"かな": 90, "かめい": 5}, "哉": {"かな": 1}}
        self.assertEqual(sorted(ja.frequent(lexicon, 2)), [("仮名", "かな"), ("仮名", "かめい"), ("加奈", "かな")])


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
