"""Sequence boundaries when a tokenizer automatically inserts BOS."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from decoder import Units
from rescorer import Rescorer


class Tokenizer:
    vocab = ["<s>", "前", "文", "甲", "乙", "丙"] + ["�"] * 8

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text, add_special_tokens=True):
        ids = [6, 7] if text == "龘" else [self.vocab.index(c) for c in text]
        return {"input_ids": ([0] if add_special_tokens else []) + ids}

    def decode(self, ids):
        return "龘" if list(ids) == [6, 7] else "".join(self.vocab[i] for i in ids)


class Model:
    def __call__(self, input_ids):
        self.input_ids = input_ids.tolist()
        logits = torch.full((*input_ids.shape, len(Tokenizer.vocab)), -8.)
        # After 前文甲, prefer 丙 to 乙. The proposal must retain exactly 甲.
        logits[:, -1, 5] = 8.
        return SimpleNamespace(logits=logits)


class SequenceBoundaries(unittest.TestCase):
    def scorer(self):
        scorer = Rescorer.__new__(Rescorer)
        scorer.tokenizer, scorer.model = Tokenizer(), Model()
        scorer.device, scorer.pad = "cpu", 0
        return scorer

    def test_score_keeps_only_context_bos(self):
        scorer = self.scorer()
        scores = scorer.score("前文", ["甲乙"])
        self.assertEqual(scorer.model.input_ids, [[0, 1, 2, 3, 4]])
        self.assertEqual(len(scores), 1)
        self.assertTrue(torch.isfinite(torch.tensor(scores)).all())

    def test_proposal_offset_excludes_bos(self):
        scorer = self.scorer()
        self.assertEqual(scorer.proposals("前文", "甲乙"), ["甲丙"])
        self.assertEqual(scorer.model.input_ids, [[0, 1, 2, 3]])

    def test_multitoken_character_has_no_internal_bos(self):
        units = Units(Tokenizer(), {"龘": {"da": []}})
        self.assertEqual(units.get("da"), [("龘", (6, 7))])


if __name__ == "__main__":
    unittest.main()
