"""Constrained decoding with toy models: no GPU, model or database."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
import torch  # noqa: E402

from decoder import Decoder, Units  # noqa: E402

VOCAB = ["<pad>", "\n", "整", "证", "句", "据", "输", "入", "证据", "整句", "政"]
READINGS = {"整": {"vg": ["sz"]}, "证": {"vg": ["ye", "yd"]}, "政": {"vg": ["zw"]},
            "句": {"ju": ["kp"]}, "据": {"ju": ["tj"]}, "输": {"uu": ["cr"]}, "入": {"ru": ["rr"]}}


class Tokenizer:
    def __len__(self):
        return len(VOCAB)

    def decode(self, ids):
        return "".join(VOCAB[i] for i in ids)

    def __call__(self, text, add_special_tokens=True):
        return dict(input_ids=[VOCAB.index(text)] if text in VOCAB else [1])


class Toy(Decoder):
    """The LM prefers 证据; the conditional model is indifferent."""

    def __init__(self):
        super().__init__(None, None, Units(Tokenizer(), READINGS))
        self.forwards = 0

    def prompts(self, context, code):
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


class Lookahead(Toy):
    """A one-character beam that prefers 证 only once the next keys are ju."""

    def __init__(self):
        super().__init__()
        self.beam = self.per_hypothesis = 1
        # A multi-character token could bypass the earlier pruned frontier.
        self.units.longest = 1
        for index in (self.units.index, self.units.partial):
            for code, units in index.items():
                index[code] = [unit for unit in units if len(unit[0]) == 1]
        self.conditioned_on = []
        self.lm_prefixes = []

    def prompts(self, context, code):
        return [code], [1]

    def char_ids(self, texts):
        return [[VOCAB.index(c) for c in text] for text in texts]

    def lm_next(self, context, sequences):
        self.lm_prefixes.extend(sequences)
        return super().lm_next(context, sequences)

    def preferred(self, prompt):
        self.conditioned_on.append(prompt[0])
        return "证据输入" if prompt[0].startswith("vgju") else "整输入"

    def model_next(self, prompt, texts):
        preferred = self.preferred(prompt)
        rows = torch.full((len(texts), len(VOCAB)), -8.0)
        for i, text in enumerate(texts):
            rows[i, VOCAB.index(preferred[len(text)])] = 0.0
        return rows

    def model_score(self, prompt, texts):
        preferred = self.preferred(prompt)
        return [-8.0 * sum(a != b for a, b in zip(text, preferred)) for text in texts]


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

    def test_typing_reuses_language_model_prefixes(self):
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

    def test_later_syllables_recover_an_earlier_pruned_reading(self):
        decoder = Lookahead()
        self.assertEqual(decoder.decode("", "vg")[0][0], "整")
        decoder.conditioned_on.clear()
        self.assertEqual(decoder.decode("", "vgju")[0][0], "证据")
        # Both the first and second frontier see all current syllables.
        self.assertTrue(decoder.conditioned_on)
        self.assertEqual(set(decoder.conditioned_on), {"vgju"})
        # Existing general-LM rows remain valid despite rebuilding the beam.
        self.assertEqual(decoder.lm_prefixes.count(()), 1)
        decoder.conditioned_on.clear()
        self.assertEqual(decoder.decode("", "vg")[0][0], "整")
        self.assertEqual(set(decoder.conditioned_on), {"vg"})
        self.assertEqual(decoder.lm_prefixes.count(()), 1)

    def test_pasted_input_uses_later_syllables_before_first_pruning(self):
        decoder = Lookahead()
        self.assertEqual(decoder.decode("", "vgju")[0][0], "证据")
        self.assertEqual(set(decoder.conditioned_on), {"vgju"})
        # Replacing the second syllable also revisits the first character.
        self.assertEqual(decoder.decode("", "vguu")[0][0], "整输")

    def test_half_syllables_reuse_complete_frontiers(self):
        decoder = Lookahead()
        decoder.decode("", "vgju")
        frontier = decoder.session.kept[1]
        decoder.conditioned_on.clear()
        self.assertEqual(decoder.decode("", "vgjuu")[0][0], "证据输")
        self.assertIs(decoder.session.kept[1], frontier)
        self.assertEqual(set(decoder.conditioned_on), {"vgju"})
        decoder.conditioned_on.clear()
        decoder.decode("", "vgju")
        self.assertEqual(decoder.conditioned_on, [])

    def test_rebuilding_respects_pins_and_auxiliary_codes(self):
        decoder = Lookahead()
        decoder.decode("", "vg", pins=[(0, "整")])
        self.assertEqual(decoder.decode("", "vgju", pins=[(0, "整")])[0][0], "整据")
        self.assertEqual(decoder.decode("", "vgju")[0][0], "证据")
        decoder.decode("", "vg", aux=[(0, "s")])
        self.assertEqual(decoder.decode("", "vgju", aux=[(0, "s")])[0][0], "整据")

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

    def test_typed_auxiliary_codes_select_characters_by_prefix(self):
        decoder = Toy()
        texts = [t for t, _ in decoder.decode("", "vgjuuuru", aux=[(0, "s"), (1, "kp")])]
        self.assertTrue(texts and all(t.startswith("整句") for t in texts))
        session = decoder.session
        texts = [t for t, _ in decoder.decode("", "vgjuuuru", aux=[(0, "y")])]
        self.assertIs(decoder.session, session)
        self.assertTrue(texts and all(t[0] == "证" for t in texts))
        self.assertEqual(decoder.decode("", "vgjuuuru", aux=[(0, "q")]), [])
        # A half syllable's auxiliary code applies to its provisional character.
        self.assertEqual({t[3] for t, _ in Toy().decode("", "vgjuuur", aux=[(3, "r")])}, {"入"})
        self.assertEqual(Toy().decode("", "vgjuuur", aux=[(3, "x")]), [])

    def test_unknown_keys_give_nothing(self):
        self.assertEqual(Toy().decode("", "vgxx"), [])


if __name__ == "__main__":
    unittest.main()
