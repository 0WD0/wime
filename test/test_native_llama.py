"""Native adapter semantics without weights, GPU, or a compiled shared library."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from native_llama import NativeLlamaLM


def logits(tokens):
    features = torch.tensor([sum(tokens), sum((i + 1) * t for i, t in enumerate(tokens)),
                             sum(t * t for t in tokens), len(tokens)], dtype=torch.float32)
    weights = torch.arange(68, dtype=torch.float32).reshape(17, 4).remainder(11).sub(5) / 300
    return weights @ features


class FakeContext:
    def __init__(self, runtime, size, batch, sequences):
        self.runtime, self.size, self.batch, self.sequences = runtime, size, batch, sequences
        self.paths, self.calls, self.copies, self.removed = {}, [], [], []
        self.closed = False

    def clear(self):
        self.paths.clear()

    def copy(self, source, destination, end):
        assert 0 <= destination < self.sequences and destination not in self.paths
        assert end <= len(self.paths[source])
        self.paths[destination] = self.paths[source][:end]
        self.copies.append((source, destination, end))

    def remove(self, sequence):
        assert sequence != 0 and sequence in self.paths
        del self.paths[sequence]
        self.removed.append(sequence)

    def decode(self, tokens, positions, sequences, outputs):
        assert not self.closed and len(tokens) <= self.batch
        self.calls.append((tuple(tokens), tuple(positions), tuple(sequences), tuple(outputs)))
        if self.runtime.fail_once:
            self.runtime.fail_once = False
            self.paths[0] = (999,)  # a failed native call may leave dirty state
            raise RuntimeError("injected native decode failure")
        rows = []
        for token, position, sequence, output in zip(tokens, positions, sequences, outputs):
            previous = self.paths.get(sequence, ())
            assert position == len(previous) < self.size
            self.paths[sequence] = previous + (token,)
            if output:
                rows.append(logits(self.paths[sequence]))
        return torch.stack(rows) if rows else torch.empty((0, self.runtime.vocab_size))

    def close(self):
        self.closed = True
        self.paths.clear()


class FakeRuntime:
    vocab_size, max_sequences = 17, 256

    def __init__(self):
        self.contexts, self.closed, self.fail_once = [], False, False

    def create_context(self, size, batch, sequences, threads):
        assert sequences <= self.max_sequences
        context = FakeContext(self, size, batch, sequences)
        self.contexts.append(context)
        return context

    def token_piece(self, token):
        return bytes([65 + token])

    def close(self):
        self.closed = True


class NativeSemantics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.runtime = FakeRuntime()
        self.model = NativeLlamaLM("", "", _runtime=self.runtime, max_paths=4,
                                   batch_size=7, context_size=400)
        self.addCleanup(self.model.close)

    def reference_next(self, context, paths):
        return torch.stack([logits(tuple(context) + tuple(path)).log_softmax(-1) for path in paths])

    def reference_tails(self, context, prefixes, tails):
        totals = torch.zeros(len(tails), dtype=torch.float32)
        for owner, (prefix, tail) in enumerate(zip(prefixes, tails)):
            for position, target in enumerate(tail):
                row = logits(tuple(context) + tuple(prefix) + tuple(tail[:position])).log_softmax(-1)
                totals[owner] += row[target]
        return totals

    def test_next_variable_lengths_empty_duplicates_and_order(self):
        context = [1, 2]
        paths = [(), (3,), (4, 5), (), (3,), (6, 7, 8)]
        actual = self.model.next(context, paths)
        torch.testing.assert_close(actual, self.reference_next(context, paths))
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual.exp().sum(-1), torch.ones(len(paths)))
        before = self.model.stats["input_tokens"]
        torch.testing.assert_close(self.model.next(context, paths), actual, rtol=0, atol=0)
        self.assertEqual(self.model.stats["input_tokens"], before)

    def test_different_parent_positions_share_one_native_batch_without_padding(self):
        context = [1, 2]
        self.model.next(context, [(3,), (4, 5)])
        ctx = self.runtime.contexts[0]
        parents = {p: (s.sequence_id, ctx.paths[s.sequence_id]) for p, s in self.model._paths.items()}
        ctx.calls.clear()
        paths = [(3, 6), (4, 5, 7)]
        torch.testing.assert_close(self.model.next(context, paths), self.reference_next(context, paths))
        self.assertEqual(len(ctx.calls), 1)
        self.assertEqual(ctx.calls[0][0], (6, 7))
        self.assertEqual(ctx.calls[0][1], (3, 4))
        for sequence_id, original in parents.values():
            self.assertEqual(ctx.paths[sequence_id], original)

    def test_longest_available_parent_reused_and_full_inputs_split(self):
        self.model.next([1, 2], [(3,)])
        before = self.model.stats["input_tokens"]
        path = tuple(range(3, 15))
        torch.testing.assert_close(self.model.next([1, 2], [path]), self.reference_next([1, 2], [path]))
        self.assertEqual(self.model.stats["input_tokens"] - before, len(path) - 1)
        self.assertEqual(self.model.stats["parent_hits"], 1)
        self.assertTrue(all(len(c[0]) <= 7 for c in self.runtime.contexts[0].calls))

    def test_lru_evicts_native_sequences_and_requests_larger_than_capacity(self):
        runtime = FakeRuntime()
        model = NativeLlamaLM("", "", _runtime=runtime, max_paths=1, batch_size=4)
        self.addCleanup(model.close)
        paths = [(3,), (4,), (5, 6), (7, 8)]
        torch.testing.assert_close(model.next([1, 2], paths), self.reference_next([1, 2], paths))
        self.assertEqual(list(model._paths), [(7, 8)])
        self.assertEqual(len(runtime.contexts[0].paths), 2)  # root plus last retained path
        torch.testing.assert_close(model.next([1, 2], [(3, 9)]), self.reference_next([1, 2], [(3, 9)]))

    def test_context_changes_invalidate_search_and_root(self):
        self.model.next([1, 2], [(3,)])
        actual = self.model.next([8], [(3,)])
        torch.testing.assert_close(actual, self.reference_next([8], [(3,)]))
        self.assertEqual(self.runtime.contexts[0].paths[0], (8,))
        self.assertEqual(self.model.stats["prefills"], 2)

    def test_tails_match_full_causal_scores_without_forwarding_last_target(self):
        context = [1, 2]
        prefixes = [(), (3,), (4, 5), (), (8,), ()]
        tails = [(6, 7), (8, 9, 10), (11,), (), (), (12,)]
        actual = self.model.tails(context, prefixes, tails)
        torch.testing.assert_close(torch.tensor(actual), self.reference_tails(context, prefixes, tails))
        ctx = self.runtime.contexts[0]
        self.assertEqual(set(ctx.paths), {0})
        fed = [token for call in ctx.calls[1:] for token in call[0]]
        self.assertEqual(fed, [6, 3, 8, 9, 4, 5])
        before = len(ctx.calls)
        torch.testing.assert_close(torch.tensor(self.model.tails(context, [()], [(12,)])),
                                   self.reference_tails(context, [()], [(12,)]))
        self.assertEqual(len(ctx.calls), before)

    def test_canonical_scoring_has_independent_native_context(self):
        self.model.next([1, 2], [(3,), (4, 5)])
        search = self.runtime.contexts[0]
        before = search.paths.copy()
        self.model.tails([6], [(), (7,)], [(8, 9), (10,)])
        self.assertEqual(search.paths, before)
        self.assertEqual(len(self.runtime.contexts), 2)
        count = len(search.calls)
        self.model.next([1, 2], [(3,), (4, 5)])
        self.assertEqual(len(search.calls), count)

    def test_long_tail_logit_batches_are_bounded(self):
        runtime = FakeRuntime()
        model = NativeLlamaLM("", "", _runtime=runtime, max_paths=2,
                              batch_size=256, context_size=400)
        self.addCleanup(model.close)
        tail = tuple([3, 4, 5] * 70)
        torch.testing.assert_close(torch.tensor(model.tails([1, 2], [()], [tail])),
                                   self.reference_tails([1, 2], [()], [tail]))
        self.assertLessEqual(max(sum(call[3]) for call in runtime.contexts[0].calls), 128)

    def test_failed_decode_clears_state_before_reuse(self):
        self.model.next([1], [(2,)])
        self.runtime.fail_once = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.model.next([1], [(2, 3)])
        self.assertFalse(self.runtime.contexts[0].paths)
        self.assertFalse(self.model._paths)
        torch.testing.assert_close(self.model.next([1], [(2, 3)]), self.reference_next([1], [(2, 3)]))
        self.runtime.fail_once = True
        with self.assertRaises(RuntimeError):
            self.model.tails([1], [()], [(2, 3)])
        self.assertFalse(self.runtime.contexts[1].paths)
        torch.testing.assert_close(torch.tensor(self.model.tails([1], [()], [(2, 3)])),
                                   self.reference_tails([1], [()], [(2, 3)]))

    def test_token_byte_validation_and_native_sequence_limit(self):
        table = [self.runtime.token_piece(i) for i in range(self.runtime.vocab_size)]
        table[0] = None
        self.model.validate_token_bytes(table)
        table[5] = b"wrong"
        with self.assertRaisesRegex(ValueError, "ID 5"):
            self.model.validate_token_bytes(table)
        with self.assertRaises(ValueError):
            self.model.validate_token_bytes(table[:-1])
        runtime = FakeRuntime()
        model = NativeLlamaLM("", "", _runtime=runtime)  # default128 must fit max256
        self.addCleanup(model.close)
        model.next([1], [(2,)])
        self.assertLessEqual(runtime.contexts[0].sequences, 256)

    def test_empty_requests_invalid_inputs_and_close(self):
        self.assertEqual(self.model.next([1], []).shape, (0, 17))
        self.assertEqual(self.model.tails([1], [(), (2,)], [(), ()]), [0.0, 0.0])
        self.assertFalse(self.runtime.contexts)
        for context, paths in [([], [()]), ([True], [()]), ([1], [(17,)])]:
            with self.assertRaises(ValueError):
                self.model.next(context, paths)
        with self.assertRaises(ValueError):
            self.model.tails([1], [()], [])
        with self.assertRaises(ValueError):
            self.model.tails([1], [()], [(17,)])
        self.model.next([1], [(2,)])
        self.model.close()
        self.model.close()
        self.assertTrue(self.runtime.closed)
        self.assertTrue(all(c.closed for c in self.runtime.contexts))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.model.next([1], [(2,)])


if __name__ == "__main__":
    unittest.main()
