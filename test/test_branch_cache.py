"""Parent-path cache ownership and causal alignment, entirely on CPU."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
import branch_cache
from prefix_cache import PrefixCacheLM


class DynamicLayer:
    def __init__(self):
        self.keys = self.values = None
        self.is_initialized = False


class LinearAttentionLayer:
    def __init__(self):
        self.number_of_states = 1
        self.conv_states = {0: None}
        self.recurrent_states = {0: None}
        self.is_conv_states_initialized = {0: False}
        self.is_recurrent_states_initialized = {0: False}
        self.has_previous_state = {0: False}
        self.conv_kernel_size = {0: None}
        self.device = self.dtype = None
        self.record_past = False


class DynamicCache:
    def __init__(self):
        self.layers = [DynamicLayer(), LinearAttentionLayer()]
        self.layer_class_to_replicate = None
        self.offloading = False

    def reorder_cache(self, indices):
        attention, linear = self.layers
        attention.keys = attention.keys.index_select(0, indices)
        attention.values = attention.values.index_select(0, indices)
        linear.conv_states[0] = linear.conv_states[0].index_select(0, indices)
        linear.recurrent_states[0] = linear.recurrent_states[0].index_select(0, indices)


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.extra_metadata_once = False

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        cache = past_key_values if past_key_values is not None else DynamicCache()
        attention, linear = cache.layers
        batch, size = input_ids.shape
        previous = attention.keys.shape[-2] if attention.is_initialized else 0
        self.calls.append(dict(ids=input_ids.clone(), previous=previous,
                               mask=attention_mask.clone() if attention_mask is not None else None))
        if not attention.is_initialized:
            attention.keys = attention.values = torch.empty(batch, 1, 0, 1)
            attention.dtype, attention.device, attention.is_initialized = torch.float32, torch.device("cpu"), True
            linear.dtype, linear.device = torch.float32, torch.device("cpu")
            linear.conv_states[0] = torch.zeros(batch, 3, 2)
            linear.recurrent_states[0] = torch.zeros(batch, 1, 1, 3)
            linear.is_conv_states_initialized[0] = linear.is_recurrent_states_initialized[0] = True
            linear.has_previous_state[0], linear.conv_kernel_size[0] = True, 2
        if attention_mask is not None:
            assert attention_mask.shape == (batch, previous + size)
        outputs = []
        for j in range(size):
            token = input_ids[:, j].float()
            position = previous + j + 1
            delta = torch.stack([token, token * position, token.square()], dim=1)
            if attention_mask is not None:
                delta = delta * attention_mask[:, previous + j, None]
            # Deliberately mutate both linear states, as real recurrent models
            # do. Padding still advances convolution state and cache length.
            linear.conv_states[0].copy_(torch.cat([linear.conv_states[0][:, :, 1:], delta[:, :, None]], -1))
            linear.recurrent_states[0].add_(delta[:, None, None, :])
            addition = token[:, None, None, None]
            attention.keys = torch.cat([attention.keys, addition], dim=-2)
            attention.values = torch.cat([attention.values, addition.square()], dim=-2)
            outputs.append(torch.cat([linear.recurrent_states[0][:, 0, 0].clone(),
                                      linear.conv_states[0][:, :1].sum(-1)], dim=1))
        if self.extra_metadata_once:
            self.extra_metadata_once = False
            cache.unrecognized_state = torch.ones(batch)
        return SimpleNamespace(last_hidden_state=torch.stack(outputs, 1),
                               past_key_values=cache if use_cache else None)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.lm_head = torch.nn.Linear(4, 11, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(44).reshape(11, 4).float().remainder(7).sub(3) / 100)


class BranchCaching(unittest.TestCase):
    def setUp(self):
        self.types_patch = patch.object(branch_cache, "_cache_types", return_value=(DynamicCache, DynamicLayer, LinearAttentionLayer))
        self.types_patch.start()
        self.addCleanup(self.types_patch.stop)
        self.rescorer = SimpleNamespace(model=Model(), device="cpu", pad=0)
        self.fallback = PrefixCacheLM(self.rescorer)
        self.cached = branch_cache.BranchCacheLM(self.rescorer, self.fallback)

    @torch.inference_mode()
    def reference(self, context, sequences):
        rows = []
        for sequence in sequences:
            ids = torch.tensor([tuple(context) + tuple(sequence)])
            hidden = self.rescorer.model.model(input_ids=ids).last_hidden_state[0, -1]
            rows.append(self.rescorer.model.lm_head(hidden).float().log_softmax(-1))
        return torch.stack(rows)

    def tensors(self, entry):
        if isinstance(entry, branch_cache._PathState):
            entry = entry.batch
        layout, _ = self.cached._layout(entry.cache, entry.length)
        return [t for _, values in layout for t in values.values()] + [entry.rows]

    def test_variable_lengths_duplicates_and_empty_paths_match_full_model(self):
        context = (1, 2)
        paths = [(), (3,), (4, 5), (6, 7, 8), (3,), ()]
        actual = self.cached.next(context, paths)
        calls = self.rescorer.model.model.calls[:]
        torch.testing.assert_close(actual, self.reference(context, paths))
        self.assertEqual(self.cached.stats["entries"], 3)
        self.assertEqual(self.cached.stats["forwarded_tokens"], 6)
        self.assertTrue(all(call["mask"].all() for call in calls))
        self.assertEqual(tuple(self.cached.next(context, []).shape), (0, 11))

    def test_children_merge_equal_length_parents_and_feed_only_last_token(self):
        context = (1, 2)
        self.cached.next(context, [(3,), (4,)])
        parents = [self.cached._paths[(3,)], self.cached._paths[(4,)]]
        before = [[t.clone() for t in self.tensors(p)] for p in parents]
        self.rescorer.model.model.calls.clear()
        paths = [(3, 5), (4, 6)]
        actual = self.cached.next(context, paths)
        calls = self.rescorer.model.model.calls[:]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["ids"].tolist(), [[5], [6]])
        self.assertEqual(calls[0]["previous"], 3)
        self.assertEqual(self.cached.stats["hits"], 2)
        self.assertEqual(self.cached.stats["saved_tokens"], 2)
        torch.testing.assert_close(actual, self.reference(context, paths))
        for parent, old in zip(parents, before):
            for current, previous in zip(self.tensors(parent), old):
                torch.testing.assert_close(current, previous, rtol=0, atol=0)

    def test_different_parent_depths_never_share_padded_cache(self):
        context = (1, 2)
        self.cached.next(context, [(3,), (4, 5)])
        self.rescorer.model.model.calls.clear()
        paths = [(3, 6), (4, 5, 7)]
        actual = self.cached.next(context, paths)
        calls = self.rescorer.model.model.calls[:]
        self.assertEqual([c["previous"] for c in calls], [3, 4])
        self.assertEqual([c["ids"].tolist() for c in calls], [[[6]], [[7]]])
        self.assertTrue(all(c["mask"].all() for c in calls))
        torch.testing.assert_close(actual, self.reference(context, paths))

    def test_mixed_hit_kinds_and_duplicates_preserve_requested_order(self):
        context = (1, 2)
        self.cached.next(context, [(3,), (4, 5)])
        paths = [(4, 5, 6), (), (3,), (7, 8), (3, 9), (4, 5, 6)]
        actual = self.cached.next(context, paths)
        torch.testing.assert_close(actual, self.reference(context, paths))
        self.assertEqual(self.cached.stats["hits"], 2)
        self.assertEqual(self.cached.stats["exact_hits"], 1)

    def test_eviction_falls_back_to_exact_prefix_recompute_without_changing_scores(self):
        self.cached.max_paths = 1
        context = (1, 2)
        self.cached.next(context, [(3,)])
        self.cached.next(context, [(4,)])
        self.assertEqual(list(self.cached._paths), [(4,)])
        actual = self.cached.next(context, [(3, 5)])
        self.assertEqual(self.cached.stats["hits"], 0)
        self.assertEqual(self.rescorer.model.model.calls[-1]["ids"].tolist(), [[3, 5]])
        torch.testing.assert_close(actual, self.reference(context, [(3, 5)]))
        self.assertLessEqual(self.cached.stats["entries"], 1)

    def test_batch_storage_is_charged_once_and_evicted_as_a_whole(self):
        context = (1, 2)
        self.cached.next(context, [(3,)])
        size = self.cached._paths[(3,)].batch.size
        self.cached.clear()
        self.cached.max_bytes = size * 2
        for paths in [[(3,), (4,)], [(6,), (7,)]]:
            actual = self.cached.next(context, paths)
            torch.testing.assert_close(actual, self.reference(context, paths))
            stores = {}
            for batch in self.cached._batches.values():
                for tensor in self.tensors(batch):
                    key = tensor.untyped_storage().data_ptr()
                    stores[key] = tensor.untyped_storage().nbytes()
            self.assertEqual(sum(stores.values()), self.cached.stats["bytes"])
            self.assertLessEqual(sum(stores.values()), self.cached.max_bytes)
            self.assertEqual(self.cached.stats["entries"], 2)
            self.assertEqual(self.cached.stats["batches"], 1)
            self.assertIs(self.cached._paths[paths[0]].batch, self.cached._paths[paths[1]].batch)
        self.assertEqual(set(self.cached._paths), {(6,), (7,)})
        self.assertEqual(self.cached.stats["batch_evictions"], 1)

    def test_oversized_batch_is_not_partially_retained(self):
        context = (1, 2)
        self.cached.max_paths = 2
        paths = [(3,), (4,), (5,)]
        actual = self.cached.next(context, paths)
        torch.testing.assert_close(actual, self.reference(context, paths))
        self.assertEqual(self.cached.stats["entries"], 0)
        self.assertEqual(self.cached.stats["bytes"], 0)
        self.assertEqual(self.cached.stats["oversized_batches"], 1)

    def test_touching_one_path_retains_and_charges_its_entire_batch(self):
        context = (1, 2)
        self.cached.max_paths = 4
        self.cached.next(context, [(3,), (4,)])
        first = self.cached._paths[(3,)].batch
        self.cached.next(context, [(5,), (6,)])
        self.cached.next(context, [(3,)])
        self.cached.next(context, [(7,), (8,)])
        self.assertEqual(set(self.cached._paths), {(3,), (4,), (7,), (8,)})
        self.assertIs(self.cached._paths[(4,)].batch, first)
        self.assertEqual(self.cached.stats["bytes"], sum(b.size for b in self.cached._batches.values()))

    def test_admission_keeps_forward_batch_without_per_path_tensor_copies(self):
        paths = [(i,) for i in range(1, 10)]
        with patch.object(self.cached, "_copy_rows", wraps=self.cached._copy_rows) as copies:
            self.cached.next((1, 2), paths)
        self.assertEqual(copies.call_count, 1)
        self.assertEqual(len(self.cached._batches), 1)
        batch = self.cached._paths[(1,)].batch
        self.assertTrue(all(self.cached._paths[p].batch is batch for p in paths))

    def test_fork_checks_shared_source_layout_once_and_preserves_row_order(self):
        context = (1, 2)
        self.cached.next(context, [(3,), (4,)])
        parent = self.cached._paths[(3,)].batch
        with patch.object(self.cached, "_layout", wraps=self.cached._layout) as checks:
            fork = self.cached._copy_rows([(parent.cache, 1), (parent.cache, 0), (parent.cache, 1)], parent.length)
        self.assertEqual(checks.call_count, 1)
        original = parent.cache.layers[1].recurrent_states[0].clone()
        torch.testing.assert_close(fork.layers[1].recurrent_states[0], original[[1, 0, 1]])
        fork.layers[1].recurrent_states[0].add_(100)
        torch.testing.assert_close(parent.cache.layers[1].recurrent_states[0], original)

    def test_state_larger_than_budget_is_not_retained(self):
        self.cached.max_bytes = 1
        actual = self.cached.next((1, 2), [(3,), (4, 5)])
        torch.testing.assert_close(actual, self.reference((1, 2), [(3,), (4, 5)]))
        self.assertEqual(self.cached.stats["entries"], 0)
        self.assertEqual(self.cached.stats["bytes"], 0)

    def test_exact_path_hits_and_context_change_invalidation(self):
        first = self.cached.next((1, 2), [(3,), (4,)])
        count = len(self.rescorer.model.model.calls)
        torch.testing.assert_close(self.cached.next((1, 2), [(3,), (4,)]), first, rtol=0, atol=0)
        self.assertEqual(len(self.rescorer.model.model.calls), count)
        self.assertEqual(self.cached.stats["exact_hits"], 2)
        second = self.cached.next((5,), [(3,)])
        self.assertEqual(list(self.cached._paths), [(3,)])
        self.assertEqual(self.cached.stats["hits"], 0)
        self.assertFalse(torch.allclose(first[0], second[0]))
        self.cached.clear()
        self.assertEqual(self.cached.stats["bytes"], 0)

    def test_unknown_metadata_disables_branching_and_uses_existing_fallback(self):
        context = (1, 2)
        self.cached.next(context, [(3,)])
        self.fallback._prefill(context).cache.layers[1].unrecognized_counter = 9
        actual = self.cached.next(context, [(3, 4), (5,)])
        torch.testing.assert_close(actual, self.reference(context, [(3, 4), (5,)]))
        self.assertEqual(self.cached.stats["fallback_batches"], 1)
        self.assertEqual(self.cached.stats["entries"], 0)
        self.cached.next(context, [(3, 4)])
        self.assertEqual(self.cached.stats["fallback_batches"], 2)

    def test_unknown_returned_metadata_retries_without_mutating_prefill(self):
        context = (1, 2)
        self.cached.next(context, [(3,)])
        prefill = copy.deepcopy(self.fallback._prefill(context).cache)
        self.rescorer.model.model.extra_metadata_once = True
        actual = self.cached.next(context, [(3, 4)])
        torch.testing.assert_close(actual, self.reference(context, [(3, 4)]))
        current = self.fallback._prefill(context).cache
        torch.testing.assert_close(current.layers[1].recurrent_states[0], prefill.layers[1].recurrent_states[0])
        self.assertEqual(self.cached.stats["fallback_batches"], 1)
        self.assertEqual(self.cached.stats["entries"], 0)

    def test_incompatible_linear_metadata_is_not_silently_merged(self):
        self.cached.next((1, 2), [(3,), (4,)])
        self.cached._paths[(4,)].cache.layers[1].record_past = True
        actual = self.cached.next((1, 2), [(3, 5), (4, 6)])
        torch.testing.assert_close(actual, self.reference((1, 2), [(3, 5), (4, 6)]))
        self.assertEqual(self.cached.stats["fallback_batches"], 1)
        self.assertEqual(self.cached.stats["entries"], 0)

    def test_invalid_inputs_and_unsupported_types(self):
        with self.assertRaises(ValueError):
            branch_cache.BranchCacheLM(self.rescorer, self.fallback, max_paths=0)
        with self.assertRaises(ValueError):
            branch_cache.BranchCacheLM(self.rescorer, self.fallback, max_bytes=True)
        with self.assertRaises(ValueError):
            self.cached.next([], [()])
        with self.assertRaises(ValueError):
            self.cached.next([1], [(True,)])
        self.cached._types = ()
        actual = self.cached.next((1, 2), [(3,)])
        torch.testing.assert_close(actual, self.reference((1, 2), [(3,)]))
        self.assertEqual(self.cached.stats["fallback_batches"], 1)


if __name__ == "__main__":
    unittest.main()
