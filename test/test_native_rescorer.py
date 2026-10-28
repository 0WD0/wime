"""Tokenizer and score/proposal contracts; no native library or model weights."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from native_rescorer import NativeRescorer
from rescorer import Rescorer


class Tokenizer:
    vocab = ["<s>", "\n", "前", "文", "甲", "乙", "丙", "中国", "国",
             "�", "�", "丁", "山", "水", "!", "a", "中"]
    pad_token_id = None

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        ids = []
        while text:
            if text.startswith("中国"):
                ids.append(7)
                text = text[2:]
            elif text[0] == "龘":
                ids.extend((9, 10))
                text = text[1:]
            else:
                ids.append(self.vocab.index(text[0]))
                text = text[1:]
        return {"input_ids": ([0] if add_special_tokens else []) + ids}

    def decode(self, ids):
        if list(ids) == [9, 10]:
            return "龘"
        return "".join(self.vocab[i] for i in ids)


def causal_logits(ids):
    row = torch.arange(len(Tokenizer.vocab), dtype=torch.float32) * -0.13
    row[(sum(ids) * 3 + len(ids)) % 5 + 4] += 4.0
    row[5] += 1.0
    return row


class Model:
    def __call__(self, input_ids):
        return SimpleNamespace(logits=torch.stack([
            torch.stack([causal_logits(row[:i + 1]) for i in range(len(row))])
            for row in input_ids.tolist()]))


class Backend:
    def __init__(self):
        self.next_calls, self.tail_calls = [], []
        self.closed = False

    def next(self, context, sequences):
        self.next_calls.append((list(context), sequences))
        return torch.stack([causal_logits(list(context) + list(sequence)).log_softmax(-1)
                            for sequence in sequences])

    def tails(self, context, prefixes, tails):
        self.tail_calls.append((list(context), prefixes, tails))
        return [float(torch.stack([
            causal_logits(list(context) + list(prefix) + list(tail[:i])).log_softmax(-1)[token]
            for i, token in enumerate(tail)]).sum())
            for prefix, tail in zip(prefixes, tails)]

    def close(self):
        self.closed = True


class NativeRescorerContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.native = NativeRescorer.__new__(NativeRescorer)
        self.native.tokenizer, self.native.backend = Tokenizer(), Backend()
        self.native.weight, self.native.device, self.native.pad = 0.7, "cpu", 0
        self.reference = Rescorer.__new__(Rescorer)
        self.reference.tokenizer, self.reference.model = Tokenizer(), Model()
        self.reference.weight, self.reference.device, self.reference.pad = 0.7, "cpu", 0

    def test_score_matches_bos_targets_and_character_normalization(self):
        texts = ["甲乙", "中国甲", "龘", "丁"]
        actual = self.native.score("前文", texts)
        expected = self.reference.score("前文", texts)
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))
        self.assertEqual(self.native.backend.tail_calls,
                         [([0, 2, 3], [()] * 4, [(4, 5), (7, 4), (9, 10), (11,)])])
        self.assertEqual(self.native.tokenizer.calls,
                         [("前文", True)] + [(text, False) for text in texts])

    def test_score_context_truncation_and_empty_context(self):
        for context in ("前" * 150 + "文", ""):
            actual = self.native.score(context, ["甲乙", "中国"])
            expected = self.reference.score(context, ["甲乙", "中国"])
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))
        self.assertEqual(self.native.tokenizer.calls[0], ("前" * 127 + "文", True))
        self.assertEqual(self.native.tokenizer.calls[3], ("\n", True))

    def test_proposals_match_existing_token_offsets_topk_and_margin(self):
        proposals = []
        for draft in ("甲乙丙", "中国甲", "龘乙"):
            for margin in (0.5, 1.5):
                actual = self.native.proposals("前文", draft, margin)
                self.assertEqual(actual, self.reference.proposals("前文", draft, margin))
                proposals.extend(actual)
        self.assertTrue(proposals)

    def test_empty_requests_and_close(self):
        self.assertEqual(self.native.score("前文", []), [])
        self.assertEqual(self.native.proposals("前文", ""), [])
        self.assertFalse(self.native.backend.tail_calls or self.native.backend.next_calls)
        self.native.close()
        self.assertTrue(self.native.backend.closed)


if __name__ == "__main__":
    unittest.main()
