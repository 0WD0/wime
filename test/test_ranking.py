"""Coverage, leakage guards and exact inference/training score agreement."""
import json
import random
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from conditional import CharLM, Config
from inference import ConditionalEvaluator, anchored_prompt
from ranking import encode_candidate, candidate_scores, preference_loss, ranking_composition, spelling_equivalent
from replay import phrases, coverage_additions, Protection


class Ranking(unittest.TestCase):
    def test_mined_compositions_never_hide_a_future_suffix(self):
        row = dict(context="这个", input="bud;yij;heb;le", gold=["补丁已经合并了"])
        for seed in range(100):
            selected = ranking_composition(row, random.Random(seed))
            length = len(selected["gold"][0])
            self.assertEqual(selected["gold"][0], row["gold"][0][-length:])
            self.assertEqual(selected["input"], row["input"][-2*length:])
        self.assertTrue(spelling_equivalent("好哦", "好喔"))
        self.assertFalse(spelling_equivalent("的", "地"))

    def test_mining_keeps_whole_phrases_and_actual_committed_polyphone_keys(self):
        row = dict(context="允许", input="bmyix;ui", gold=["便宜行事"])
        rng = Mock()
        rng.random.side_effect = [.4, .9]
        rng.randint.return_value = 2
        suffix = ranking_composition(row, rng)
        self.assertEqual(suffix["gold"], ["行事"])
        self.assertEqual(suffix["known_reading"], dict(text="便宜", input="bmyi"))
        whole = ranking_composition(row, random.Random(0), whole_share=1)
        self.assertEqual(whole["input"], row["input"])
        self.assertEqual(whole["gold"], row["gold"])
        self.assertEqual(whole["context"], row["context"])
        self.assertEqual(whole["composition"], "whole")

    def test_scores_match_deployed_evaluator_and_mask_fixed_prefix(self):
        row = dict(context="允许便宜", input="x;ui", gold=["行事"],
                   known_reading=dict(text="便宜", input="bmyi"))
        texts = ["行事", "形式"]
        chars = sorted(set(anchored_prompt(row["context"], row["input"], 32, 2, row["known_reading"]) + "".join(texts)))
        vocab = ["<pad>", "<unk>", "<bos>"] + chars
        index = {c:i for i,c in enumerate(vocab)}
        model = CharLM(Config(vocab=len(vocab), n_layer=1, n_head=2, n_embd=8, context=32, dropout=0)).eval()
        evaluator = object.__new__(ConditionalEvaluator)
        evaluator.model, evaluator.index, evaluator.device, evaluator.context_anchor = model, index, "cpu", 2
        expected, _ = evaluator.evaluate(row["context"], row["input"], texts, known_reading=row["known_reading"])
        encoded = [encode_candidate(row, text, index, 32) for text in texts]
        x, y = (torch.tensor([p[i] for p in encoded]) for i in range(2))
        logits = model(x)[0].detach().requires_grad_()
        scores = candidate_scores(logits, y)
        torch.testing.assert_close(scores, torch.tensor(expected))
        scores.sum().backward()
        self.assertEqual(logits.grad[y == 0].abs().sum().item(), 0)
        self.assertEqual((y != 0).sum().item(), 4)  # neither anchor nor EOS

    def test_loss_includes_real_prior_and_prefers_correct_candidate(self):
        scores = torch.tensor([-.5, -.4], requires_grad=True)
        priors = torch.tensor([-.1, -4.])
        plain = preference_loss(scores, torch.zeros(2))
        weighted = preference_loss(scores, priors)
        self.assertLess(weighted.item(), plain.item())
        weighted.backward()
        self.assertLess(scores.grad[0].item(), 0)
        self.assertGreater(scores.grad[1].item(), 0)

    def test_coverage_recovers_supported_phrases_without_merging_readings(self):
        bian = dict(id="a", split="train", context="", input="bmyi", gold=["便宜"])
        pian = dict(bian, id="b", input="pmyi")
        self.assertNotEqual(phrases(bian), phrases(pian))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"train.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in [bian,pian,dict(pian,id="c")]))
            extra, audit = coverage_additions(path, [bian], 2, min_support=2)
            self.assertEqual([r["id"] for r in extra], ["b"])
            self.assertEqual(audit["missing_before"], 1)
            self.assertEqual(audit["missing_after"], 0)
            path.write_text(json.dumps(dict(bian,split="test")))
            with self.assertRaises(ValueError):
                coverage_additions(path, [], 2)

    def test_protection_checks_documents_and_normalized_augmented_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"heldout.json"
            path.write_text(json.dumps([dict(document="test-doc", context="真刀真 ",gold=["枪"])]))
            guard = Protection([path])
            self.assertTrue(guard.contains(dict(document="test-doc",context="别的",gold=["句子"])))
            self.assertTrue(guard.contains(dict(context="真刀真\n",gold=["枪"])))
            self.assertFalse(guard.contains(dict(context="买一把",gold=["枪"])))


if __name__ == "__main__":
    unittest.main()
