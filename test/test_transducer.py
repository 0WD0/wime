"""Verify scoring by the syllable transducer, on a small random model."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from composition import OUTPUT
import transducer
from charlm import Config


def evaluator():
    torch.manual_seed(17)
    vocab = ["<pad>", "<unk>", "<bos>", OUTPUT] + list("前文便宜行事甲乙丙丁")
    ev = object.__new__(transducer.Evaluator)
    ev.model = transducer.Transducer(
        Config(vocab=len(vocab), n_layer=2, n_head=2, n_embd=16, context=128, dropout=0),
        transducer.EncoderConfig(n_layer=1, n_head=2, n_embd=16, cross_layers=1)).eval()
    ev.vocab, ev.index, ev.device = vocab, {c: i for i, c in enumerate(vocab)}, "cpu"
    ev.chinese = torch.arange(4, len(vocab))
    ev.chinese_chars = vocab[4:]
    return ev


class Verify(unittest.TestCase):
    def test_committed_prefix_keeps_its_syllables_and_scores_only_the_rest(self):
        ev = evaluator()
        _, whole = ev.evaluate("前文", "bmyix;ui", ["便宜行事", "便宜甲乙"])
        scores, tail = ev.evaluate("前文", "x;ui", ["行事", "甲乙"], dict(text="便宜", input="bmyi"))
        torch.testing.assert_close(whole[:, 2:], tail)
        expected = tail[0].gather(-1, torch.tensor([[ev.index["行"]], [ev.index["事"]]])).mean()
        self.assertAlmostEqual(scores[0], expected.item(), places=5)

    def test_proposals_replace_one_character_by_a_clearly_likelier_one(self):
        ev = evaluator()
        distribution = torch.full((2, len(ev.vocab)), -10.)
        distribution[0, ev.index["甲"]], distribution[0, ev.index["乙"]] = -1., -.4
        distribution[1, ev.index["甲"]], distribution[1, ev.index["丙"]] = -4., -.2
        self.assertEqual(ev.proposals("甲甲", distribution, .5), ["乙", "甲丙"])

    def test_odd_or_overlong_keys_are_rejected(self):
        ev = evaluator()
        for code in ("bmy", "aa" * 65):
            with self.assertRaises(ValueError):
                ev.prompt("", code)
