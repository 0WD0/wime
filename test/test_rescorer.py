"""Sequence boundaries when a tokenizer automatically inserts BOS."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from decoder import Units


class Tokenizer:
    vocab = ["<s>", "前", "文", "甲", "乙", "丙"] + ["�"] * 8

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text, add_special_tokens=True):
        ids = [6, 7] if text == "龘" else [self.vocab.index(c) for c in text]
        return {"input_ids": ([0] if add_special_tokens else []) + ids}

    def decode(self, ids):
        return "龘" if list(ids) == [6, 7] else "".join(self.vocab[i] for i in ids)


class SequenceBoundaries(unittest.TestCase):
    def test_multitoken_character_has_no_internal_bos(self):
        units = Units(Tokenizer(), {"龘": {"da": []}})
        self.assertEqual(units.get("da"), [("龘", (6, 7))])


if __name__ == "__main__":
    unittest.main()
