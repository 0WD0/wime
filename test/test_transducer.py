"""Batched decode scoring by the syllable transducer, on a small random model."""
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
    return ev


class JointScores(unittest.TestCase):
    def test_batched_prompts_match_independent_scoring_with_encoder_padding(self):
        ev = evaluator()
        prompts = [ev.prompt("前文", "bmyi"), ev.prompt("前", "x;uijxyi", [(1, 4)])]
        texts = ["便宜", "行事甲乙"]
        slots = [(0, 0), (0, 1), (1, 0), (1, 3)]
        with torch.inference_mode():
            batch = ev.joint(prompts, texts, slots)
            for row, (i, position) in enumerate(slots):
                alone = ev.joint([prompts[i]], [texts[i]], [(0, position)])[0]
                torch.testing.assert_close(batch[row], alone)

    def test_odd_or_overlong_keys_are_rejected(self):
        ev = evaluator()
        for code in ("bmy", "aa" * 65):
            with self.assertRaises(ValueError):
                ev.prompt("", code)
