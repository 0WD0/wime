"""CPU comparisons against the deployed conditional scoring semantics."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "emacs-packages/wime/neural"))
sys.path.insert(0, str(ROOT / "third_party/chinese-ime-lm/neural"))
from conditional import EOS
from conditional_prefix import ConditionalPrefixScores
from decoder import Decoder
from model import CharLM, Config


class CountingCharLM(CharLM):
    def __init__(self):
        super().__init__(Config(vocab=16, n_layer=2, n_head=2, n_embd=16,
                                context=24, dropout=0.0))
        self.calls = []

    def forward(self, ids, targets=None):
        self.calls.append(ids.detach().clone())
        return super().forward(ids, targets)


class ConditionalPrefixTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.model = CountingCharLM().eval()
        index = {char: i + 3 for i, char in enumerate("甲乙丙丁戊己庚辛" + EOS)}
        self.evaluator = SimpleNamespace(model=self.model, device="cpu", index=index)
        self.legacy = Decoder(self.evaluator, None, None)
        self.scorer = ConditionalPrefixScores(self.evaluator, batch_size=2)
        self.prompt = [2, 6, 4]

    def assert_scores(self, texts, prompt=None, scorer=None):
        prompt = self.prompt if prompt is None else prompt
        scorer = self.scorer if scorer is None else scorer
        nonempty = [text for text in texts if text]
        expected = iter(self.legacy.model_score(prompt, nonempty) if nonempty else [])
        expected = [next(expected) if text else 0.0 for text in texts]
        actual = scorer.score(prompt, texts)
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-6, atol=2e-6)
        return actual

    def test_matches_legacy_variable_lengths_duplicates_and_empty(self):
        self.assert_scores(["甲乙", "甲丙丁", "", "己", "甲乙", "甲乙戊己"])
        self.assertEqual(self.scorer.score(self.prompt, []), [])
        self.assertEqual(self.scorer.score(self.prompt, ["", ""]), [0.0, 0.0])

    def test_parent_rows_shared_and_cached_without_precision_change(self):
        texts = ["甲乙", "甲丙", "甲丁", "甲乙戊"]
        first = self.assert_scores(texts)
        self.assertEqual(self.scorer.stats["rows"], 3)  # (), 甲, 甲乙
        before = len(self.model.calls)
        second = self.scorer.score(self.prompt, texts)
        self.assertEqual(first, second)
        self.assertEqual(len(self.model.calls), before)
        self.assertEqual(self.scorer.stats["cache_hits"], 3)
        for row in self.scorer._rows.values():
            self.assertEqual(row.dtype, torch.float32)
            self.assertIsNone(row._base)

    def test_oov_aliases_and_explicit_eos(self):
        scores = self.assert_scores(["?甲", "!甲", "甲", "甲" + EOS])
        self.assertEqual(scores[0], scores[1])
        self.assertLess(scores[3], scores[2])
        # Unknown characters share encoded parent (UNK), not separate rows.
        parents = [prefix for prompt, prefix in self.scorer._rows if prompt == tuple(self.prompt)]
        self.assertEqual(set(parents), {(), (1,), (self.evaluator.index["甲"],)})

    def test_exact_prompt_is_part_of_key(self):
        first = self.assert_scores(["甲乙"])
        other = [2, 5, 4]
        second = self.assert_scores(["甲乙"], other)
        self.assertEqual(self.scorer.stats["rows"], 4)
        self.assertNotEqual(first, second)
        before = self.scorer.stats["rows"]
        self.assertEqual(self.scorer.score(self.prompt, ["甲乙"]), first)
        self.assertEqual(self.scorer.stats["rows"], before)

    def test_lru_eviction_and_request_larger_than_cache(self):
        scorer = ConditionalPrefixScores(self.evaluator, max_rows=2, batch_size=2)
        self.assert_scores(["甲乙", "甲丙"], scorer=scorer)
        scorer.next(self.prompt, ["乙"])
        self.assertEqual(len(scorer._rows), 2)
        self.assertNotIn((tuple(self.prompt), ()), scorer._rows)
        before = scorer.stats["rows"]
        scorer.next(self.prompt, ["甲"])
        self.assertEqual(scorer.stats["rows"], before)
        scorer.next(self.prompt, [""])
        self.assertEqual(scorer.stats["rows"], before + 1)
        self.assert_scores(["甲乙丙丁", "甲丙丁戊", "己庚辛"], scorer=scorer)
        self.assertEqual(len(scorer._rows), 2)
        scorer.clear()
        self.assertFalse(scorer._rows)

    def test_next_matches_legacy_and_preserves_order(self):
        texts = ["甲乙", "", "甲", "甲乙", "?", "!"]
        expected = self.legacy.model_next(self.prompt, texts)
        actual = self.scorer.next(self.prompt, texts)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(self.scorer.next(self.prompt, []).shape, (0, self.model.cfg.vocab))
        # The returned batch cannot let a caller mutate cached distributions.
        with torch.inference_mode():
            actual.fill_(0)
        torch.testing.assert_close(self.scorer.next(self.prompt, texts), expected, rtol=1e-6, atol=1e-6)

    def test_missing_rows_are_length_batched_and_bounded(self):
        self.scorer.next(self.prompt, ["甲乙", "甲", "乙", "", "丙", "甲丙"])
        self.assertEqual(self.scorer.stats["rows"], 6)
        self.assertEqual(self.scorer.stats["batches"], 4)
        self.assertEqual([tuple(ids.shape) for ids in self.model.calls], [(1, 3), (2, 4), (1, 4), (2, 5)])
        self.assertEqual(self.scorer.stats["input_tokens"], 25)

    def test_window_limits_final_target_and_invalid_inputs(self):
        # Final target need not be fed to the model, as in Decoder.model_score.
        prompt = [2] * self.model.cfg.context
        self.assert_scores(["甲"], prompt)
        with self.assertRaisesRegex(ValueError, "window"):
            self.scorer.score(prompt, ["甲乙"])
        with self.assertRaisesRegex(ValueError, "window"):
            self.scorer.next(prompt, ["甲"])
        for prompt in ([], [True], [-1]):
            with self.assertRaises(ValueError):
                self.scorer.score(prompt, ["甲"])
        with self.assertRaises(ValueError):
            self.scorer.score(self.prompt, [None])
        with self.assertRaises(ValueError):
            ConditionalPrefixScores(self.evaluator, max_rows=0)
        with self.assertRaises(ValueError):
            ConditionalPrefixScores(self.evaluator, batch_size=True)
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            self.scorer.score(self.prompt, ["甲"])


if __name__ == "__main__":
    unittest.main()
