"""Incremental attention and constrained beam scores match full teacher forcing."""
import itertools
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from conditional import CharLM, Config, EXTRA, prompt
from conditional_decode import cached_next, ConditionalDecoder


class DirectDecoding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.vocab = ["<pad>", "<unk>", "<bos>"] + EXTRA + list("上甲乙丙丁")
        self.model = CharLM(Config(vocab=len(self.vocab), context=32, n_layer=2,
                                   n_head=2, n_embd=16, dropout=0)).eval()

    @torch.no_grad()
    def test_cache_reorders_and_duplicates_beam_parents(self):
        tokens = torch.randint(len(self.vocab), (2, 5))
        logp, cache = cached_next(self.model, tokens)
        expected = self.model(tokens)[0][:, -1].log_softmax(-1)
        torch.testing.assert_close(logp, expected)
        for parents in (torch.tensor([1, 1, 0]), torch.tensor([2, 0])):
            added = torch.randint(len(self.vocab), (len(parents), 1))
            tokens = torch.cat((tokens[parents], added), dim=1)
            logp, cache = cached_next(self.model, added, cache, parents)
            expected = self.model(tokens)[0][:, -1].log_softmax(-1)
            torch.testing.assert_close(logp, expected)

    @torch.no_grad()
    def test_beam_matches_exhaustive_legal_paths_without_gold_or_prior(self):
        decoder = ConditionalDecoder(self.model, self.vocab, {"aa": "甲乙", "bb": "丙丁"}, beam=4)
        actual = decoder.decode("上", "aabb")
        prefix = [2] + [decoder.index[c] for c in prompt("上", "aabb", 32)]
        expected = []
        for chars in itertools.product("甲乙", "丙丁"):
            ids = [decoder.index[c] for c in chars]
            values = self.model(torch.tensor([prefix + ids[:-1]]))[0].log_softmax(-1)
            score = sum(values[0, len(prefix) - 1 + i, token].item() for i, token in enumerate(ids))
            expected.append(("".join(chars), score))
        expected.sort(key=lambda pair: -pair[1])
        self.assertEqual([x[0] for x in actual], [x[0] for x in expected])
        for (_, left), (_, right) in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=5)
        self.assertEqual(decoder.decode("", "cc"), [])


if __name__ == "__main__":
    unittest.main()
