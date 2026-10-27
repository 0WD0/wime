"""Character search constraints and scoring without a tokenizer or GPU."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import warnings

import torch

directory = Path(__file__).resolve().parents[1] / "neural"
sys.path.insert(0, str(directory))
from conditional import EXTRA, input_tokens  # noqa: E402

spec = importlib.util.spec_from_file_location("character_proposals", directory / "character-proposals.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

READINGS = {"邯": {"hj": ["ge"]}, "韩": {"hj": ["uw"]}, "钢": {"gh": ["jg"]},
            "刚": {"gh": ["kd"]}, "好": {"hk": ["nz"]}, "坏": {"hy": ["tb"]}}


class Toy(module.CharacterProposals):
    def __init__(self, readings=READINGS, **kwargs):
        vocab = ["<pad>", "<unk>", "<bos>"] + list("邯韩钢刚好坏") + EXTRA
        evaluator = SimpleNamespace(device="cpu", index={c: i for i, c in enumerate(vocab)},
                                    model=SimpleNamespace(cfg=SimpleNamespace(context=128)))
        super().__init__(evaluator, readings, **kwargs)
        self.prompts = []

    def model_next(self, prompt, texts):
        self.prompts.append(prompt)
        # 邯 has a low absolute probability but is best among allowed readings.
        logits = torch.full((len(texts), len(self.evaluator.index)), -20.0)
        for i, text in enumerate(texts):
            for char, score in {"邯": -8., "韩": -9., "钢": -1., "刚": -3.,
                                "好": 0., "坏": 10.}.items():
                logits[i, self.evaluator.index[char]] = score
            if text == "韩":
                logits[i, self.evaluator.index["钢"]] = -5.
        return logits


class CharacterProposalTests(unittest.TestCase):
    def test_known_rare_character_needs_no_general_tokenizer_or_probability_gate(self):
        search = Toy(beam=8)
        candidates = search.decode("", "hjgh")
        self.assertEqual(candidates[0], ("邯钢", -4.5))
        self.assertEqual({t for t, _ in candidates}, {"邯钢", "邯刚", "韩钢", "韩刚"})
        # All syllables are visible before any prefix is pruned.
        input_ids = [search.evaluator.index[c] for c in input_tokens("hjgh")]
        self.assertTrue(all(all(token in prompt for token in input_ids) for prompt in search.prompts))

    def test_pins_aux_and_reading_are_hard_constraints(self):
        search = Toy()
        self.assertEqual(search.decode("", "hjgh", pins=[(0, "韩")], aux=[(1, "jg")]),
                         [("韩钢", -7.0)])
        self.assertEqual(search.decode("", "hjgh", pins=[(0, "好")]), [])
        self.assertEqual(search.decode("", "hjgh", aux=[(0, "x")]), [])
        self.assertEqual(search.decode("", "hk")[0][0], "好")

    def test_unknown_characters_are_reported_instead_of_sharing_unk_scores(self):
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            search = Toy(dict(READINGS, 龘={"hj": ["ll"]}))
        self.assertEqual(len(emitted), 1)
        self.assertEqual(search.unsupported_by_code, {"hj": ["龘"]})
        texts = [t for t, _ in search.decode("", "hj")]
        self.assertNotIn("龘", texts)
        self.assertEqual(search.last_unsupported, {0: ["龘"]})

    def test_invalid_input_and_bounds_are_explicit(self):
        search = Toy(beam=1, per_hypothesis=1)
        self.assertEqual(search.decode("", ""), [])
        self.assertEqual(len(search.decode("", "hjgh")), 1)
        for code in ("h", "Hj", "h1"):
            with self.assertRaises(ValueError):
                search.decode("", code)
        with self.assertRaises(ValueError):
            search.decode("", "hj", pins=[(1, "韩")])


if __name__ == "__main__":
    unittest.main()
