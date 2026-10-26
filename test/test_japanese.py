"""Japanese kana decoding with toy models: no GPU, model or dictionary."""
import json
from pathlib import Path
import sys
import tempfile
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

    def test_unknown_punctuation_is_typed_literally(self):
        self.assertTrue(any(t.endswith("、") for t, _ in Toy().decode("", "にほん、")))


if __name__ == "__main__":
    unittest.main()
