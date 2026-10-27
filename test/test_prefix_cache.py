"""Prefix-cache branching and score alignment, using a CPU causal toy model."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from prefix_cache import PrefixCacheLM  # noqa: E402


class Cache:
    def __init__(self, batch):
        self.state = torch.zeros(batch, 3)
        self.length = 0

    def reorder_cache(self, indices):
        self.state = self.state.index_select(0, indices)


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        batch, size = input_ids.shape
        cache = past_key_values if past_key_values is not None else Cache(batch)
        self.calls.append(dict(ids=input_ids.clone(), mask=attention_mask.clone() if attention_mask is not None else None,
                               cached=past_key_values is not None, previous_length=cache.length))
        if attention_mask is not None:
            assert attention_mask.shape == (batch, cache.length + size)
        hidden = []
        for i in range(size):
            token = input_ids[:, i].float()
            position = cache.length + i + 1
            delta = torch.stack([token, token * position, token.square()], dim=1)
            if attention_mask is not None:
                delta *= attention_mask[:, cache.length + i, None]
            cache.state.add_(delta)
            hidden.append(torch.cat([cache.state.clone(), torch.full((batch, 1), float(position))], dim=1))
        cache.length += size
        return SimpleNamespace(last_hidden_state=torch.stack(hidden, dim=1),
                               past_key_values=cache if use_cache else None)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.lm_head = torch.nn.Linear(4, 11, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(44).reshape(11, 4).float().remainder(7).sub(3) / 100)


class PrefixCaching(unittest.TestCase):
    def setUp(self):
        self.rescorer = SimpleNamespace(model=Model(), device="cpu", pad=0)
        self.cached = PrefixCacheLM(self.rescorer, max_prefixes=2)

    @torch.inference_mode()
    def uncached_next(self, context, sequence):
        hidden = self.rescorer.model.model(input_ids=torch.tensor([list(context) + list(sequence)])).last_hidden_state[0, -1]
        return self.rescorer.model.lm_head(hidden).float().log_softmax(-1)

    @torch.inference_mode()
    def uncached_tail(self, context, prefix, tail):
        if not tail:
            return 0.0
        sequence = list(context) + list(prefix) + list(tail)
        hidden = self.rescorer.model.model(input_ids=torch.tensor([sequence])).last_hidden_state[0]
        probabilities = self.rescorer.model.lm_head(hidden).float().log_softmax(-1)
        return sum(float(probabilities[len(context) + len(prefix) + j - 1, target]) for j, target in enumerate(tail))

    def test_next_matches_full_sequences_including_empty_and_variable_lengths(self):
        context, sequences = [2, 3], [(), (4,), (1, 2, 3), (), (9, 5)]
        actual = self.cached.next(context, sequences)
        expected = torch.stack([self.uncached_next(context, s) for s in sequences])
        torch.testing.assert_close(actual, expected)
        calls = self.rescorer.model.model.calls
        self.assertEqual(calls[1]["mask"].tolist(), [[True, True, True, False, False],
                                                   [True, True, True, True, True],
                                                   [True, True, True, True, False]])

    def test_tail_scores_align_first_target_and_reuse_only_real_prefix(self):
        context = [2, 3]
        prefixes = [(), (4,), (), (1, 2), (7,)]
        tails = [(1, 2), (5,), (), (4, 5, 6), ()]
        actual = self.cached.tails(context, prefixes, tails)
        expected = [self.uncached_tail(context, p, t) for p, t in zip(prefixes, tails)]
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))

    def test_first_target_needs_no_suffix_forward_and_last_token_is_not_fed(self):
        self.cached.tails([2, 3], [()], [(4,)])
        self.assertEqual(self.cached.stats["continuations"], 0)
        self.cached.tails([2, 3], [(5, 6)], [(7, 8)])
        self.assertEqual(self.rescorer.model.model.calls[-1]["ids"].tolist(), [[5, 6, 7]])
        self.assertEqual(self.cached.stats["prefills"], 1)

    def test_branch_mutation_does_not_change_snapshot_or_repeated_scores(self):
        first = self.cached.next([1, 2], [(3, 4), ()])
        snapshot = self.cached._prefixes[(1, 2)].cache
        state, length = snapshot.state.clone(), snapshot.length
        self.cached.tails([1, 2], [(5, 6)], [(7, 8, 9)])
        self.cached.next([1, 2], [(9,), (10, 8, 6, 4)])
        torch.testing.assert_close(self.cached.next([1, 2], [(3, 4), ()]), first)
        torch.testing.assert_close(snapshot.state, state)
        self.assertEqual(snapshot.length, length)
        self.assertEqual(self.cached.stats["prefills"], 1)

    def test_prefix_lru_is_bounded_and_changes_context_scores(self):
        a = self.cached.next([1], [(3,)])
        b = self.cached.next([2], [(3,)])
        self.assertFalse(torch.allclose(a, b))
        self.cached.next([1], [()])
        self.cached.next([4], [()])
        self.assertEqual(list(self.cached._prefixes), [(1,), (4,)])
        self.cached.next([2], [()])
        self.assertEqual(self.cached.stats["prefills"], 4)
        self.assertEqual(len(self.cached._prefixes), 2)

    def test_empty_tails_skip_model_and_empty_next_batch_keeps_shape(self):
        self.assertEqual(self.cached.tails([1], [(), (4,)], [(), ()]), [0.0, 0.0])
        self.assertEqual(self.cached.stats["prefills"], 0)
        self.assertEqual(tuple(self.cached.next([1], []).shape), (0, 11))

    def test_long_tails_bound_lm_head_batches_without_losing_scores(self):
        sizes = []
        hook = self.rescorer.model.lm_head.register_forward_pre_hook(
            lambda module, args: sizes.append(args[0].shape[0]))
        prefixes = [(), (2, 3), (4,)]
        tails = [tuple(i % 7 + 1 for i in range(length)) for length in (90, 80, 70)]
        actual = self.cached.tails([1, 2], prefixes, tails)
        hook.remove()
        expected = [self.uncached_tail([1, 2], p, t) for p, t in zip(prefixes, tails)]
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))
        self.assertEqual(max(sizes), 128)

    def test_invalid_inputs_and_unsupported_cache_fail_explicitly(self):
        with self.assertRaises(ValueError):
            PrefixCacheLM(self.rescorer, max_prefixes=0)
        with self.assertRaises(ValueError):
            self.cached.next([], [()])
        with self.assertRaises(ValueError):
            self.cached.next([1], [(True,)])
        with self.assertRaises(ValueError):
            self.cached.tails([1], [()], [])
        original = self.rescorer.model.model.forward
        def unsupported(**kwargs):
            result = original(**kwargs)
            result.past_key_values = object()
            return result
        self.rescorer.model.model.forward = unsupported
        with self.assertRaisesRegex(RuntimeError, "supporting reorder_cache"):
            self.cached.next([5], [()])


if __name__ == "__main__":
    unittest.main()
