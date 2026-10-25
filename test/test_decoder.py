"""Constrained decoding with toy models: no GPU, model or database."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
import torch  # noqa: E402

from decoder import Decoder, Units  # noqa: E402

VOCAB = ["<pad>", "\n", "整", "证", "句", "据", "输", "入", "证据", "整句", "政"]
READINGS = {"整": ["vg"], "证": ["vg"], "政": ["vg"], "句": ["ju"], "据": ["ju"],
            "输": ["uu"], "入": ["ru"]}


class Tokenizer:
    def __len__(self):
        return len(VOCAB)

    def decode(self, ids):
        return "".join(VOCAB[i] for i in ids)

    def __call__(self, text):
        return dict(input_ids=[VOCAB.index(text)] if text in VOCAB else [1])


class Toy(Decoder):
    """The LM prefers 证据; the conditional model is indifferent."""

    def __init__(self):
        super().__init__(None, None, Units(Tokenizer(), READINGS))
        self.forwards = 0

    def prompts(self, context, code, anchor, known):
        return [0], [1]

    def window(self):
        return 128

    def char_ids(self, texts):
        return [[ord(c) % 7 for c in text] for text in texts]

    def lm_next(self, context, sequences):
        self.forwards += 1
        row = torch.full((len(VOCAB),), -3.0)
        row[VOCAB.index("证据")] = -1.0
        return row.log_softmax(-1).expand(len(sequences), -1)

    def model_next(self, prompt, texts):
        return torch.zeros(len(texts), 7)

    def model_score(self, prompt, texts):
        return [0.0] * len(texts)


class Decoding(unittest.TestCase):
    def test_the_language_model_spells_the_keys(self):
        texts = [t for t, _ in Toy().decode("", "vgjuuuru")]
        self.assertEqual(texts[0], "证据输入")
        self.assertTrue(all(len(t) == 4 and t[2:] == "输入" for t in texts))
        self.assertIn("整句输入", texts)

    def test_learned_readings_and_pins_steer_the_beam(self):
        self.assertEqual(Toy().decode("", "vgjuuuru", learned=[(0, "整句")], weight=10.0)[0][0], "整句输入")
        texts = [t for t, _ in Toy().decode("", "vgjuuuru", pins=[(0, "政")])]
        self.assertTrue(texts and all(t[0] == "政" for t in texts))

    def test_a_half_syllable_ends_in_a_matching_character(self):
        texts = [t for t, _ in Toy().decode("", "vgjuuur")]
        self.assertTrue(texts and all(len(t) == 4 and t[3] == "入" for t in texts))
        self.assertEqual(Toy().decode("", "v"), [])

    def test_typing_reuses_kept_frontiers(self):
        decoder = Toy()
        first = decoder.decode("", "vgjuuu")
        before = decoder.forwards
        decoder.decode("", "vgjuuuru")
        self.assertEqual(decoder.forwards - before, 1)
        before = decoder.forwards
        self.assertEqual(decoder.decode("", "vgjuuu"), first)
        self.assertEqual(decoder.forwards, before)
        decoder.decode("ctx", "vgjuuu")
        self.assertGreater(decoder.forwards, before)

    def test_a_new_pin_keeps_the_frontiers_before_it(self):
        decoder = Toy()
        decoder.decode("", "vgjuuuru")
        session, before = decoder.session, decoder.forwards
        texts = [t for t, _ in decoder.decode("", "vgjuuuru", pins=[(2, "输")])]
        self.assertTrue(texts and all(t[2] == "输" for t in texts))
        # Same session: re-expanding past the pin only reads memoized LM rows.
        self.assertIs(decoder.session, session)
        self.assertEqual(decoder.forwards, before)
        self.assertEqual(decoder.decode("", "vgjuuuru"), Toy().decode("", "vgjuuuru"))

    def test_unknown_keys_give_nothing(self):
        self.assertEqual(Toy().decode("", "vgxx"), [])


if __name__ == "__main__":
    unittest.main()
