"""Immutable context-prefill snapshots for bounded candidate recomputation.

Only an exact, unchanged context token tuple is reused. Generated paths are
recomputed on disposable copies, so no recurrent-state rollback is required.
This supports transformers caches with ``reorder_cache`` (including Qwen3.5's
conv/recurrent layers, which lack ``batch_repeat_interleave`` in 5.17).
"""
from collections import OrderedDict
import copy
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _Prefill:
    cache: object
    length: int
    last_hidden: torch.Tensor
    next_logprobs: torch.Tensor


def _tokens(values):
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError("tokens must be an iterable of nonnegative integer IDs") from error
    if any(type(token) is not int or token < 0 for token in result):
        raise ValueError("tokens must be nonnegative integer IDs")
    return result


class PrefixCacheLM:
    """Next-token and teacher-forced tail scores with a small prefix LRU.

    Cache snapshots are never passed back to the model. Right-padded branches
    are discarded after their real positions have been scored; in particular,
    padding-contaminated recurrent states are never reused. Output scores are
    float32 log probabilities with the model's full vocabulary normalization.
    """

    def __init__(self, rescorer, max_prefixes=2):
        if type(max_prefixes) is not int or max_prefixes < 1:
            raise ValueError("max_prefixes must be a positive integer")
        self.rescorer, self.model = rescorer, rescorer.model
        self.device, self.pad = rescorer.device, rescorer.pad
        self.max_prefixes = max_prefixes
        self._prefixes = OrderedDict()
        self.stats = dict(prefills=0, continuations=0, suffix_tokens=0)

    def clear(self):
        self._prefixes.clear()

    def _prefill(self, context):
        if not context:
            raise ValueError("context must contain at least one causal prefix token")
        if context in self._prefixes:
            self._prefixes.move_to_end(context)
            return self._prefixes[context]
        ids = torch.tensor([context], dtype=torch.long, device=self.device)
        result = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        cache = getattr(result, "past_key_values", None)
        if cache is None or not callable(getattr(cache, "reorder_cache", None)):
            raise RuntimeError("model did not provide a cache supporting reorder_cache")
        last = result.last_hidden_state[0, -1].detach().clone()
        probabilities = self.model.lm_head(last[None]).float().log_softmax(-1)[0]
        entry = _Prefill(cache, len(context), last, probabilities)
        self._prefixes[context] = entry
        self.stats["prefills"] += 1
        while len(self._prefixes) > self.max_prefixes:
            self._prefixes.popitem(last=False)
        return entry

    def _continue(self, entry, sequences):
        """Hidden states of nonempty generated paths, from one fresh copy."""
        size = max(map(len, sequences))
        inputs = torch.full((len(sequences), size), self.pad, dtype=torch.long)
        mask = torch.zeros((len(sequences), entry.length + size), dtype=torch.bool)
        for i, sequence in enumerate(sequences):
            inputs[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            mask[i, :entry.length + len(sequence)] = True
        try:
            branch = copy.deepcopy(entry.cache)
            branch.reorder_cache(torch.zeros(len(sequences), dtype=torch.long, device=self.device))
        except Exception as error:
            raise RuntimeError("prefill cache cannot be independently copied and expanded") from error
        result = self.model.model(input_ids=inputs.to(self.device),
                                  attention_mask=mask.to(self.device),
                                  past_key_values=branch, use_cache=True)
        self.stats["continuations"] += 1
        self.stats["suffix_tokens"] += sum(map(len, sequences))
        # No reference to the mutable, padded branch survives this call.
        return result.last_hidden_state

    @torch.inference_mode()
    def next(self, context, sequences):
        """Return log p(next token | context + sequence), in input order."""
        context = _tokens(context)
        sequences = [_tokens(sequence) for sequence in sequences]
        entry = self._prefill(context)
        if not sequences:
            return entry.next_logprobs.new_empty((0, entry.next_logprobs.numel()))
        rows = [entry.next_logprobs if not sequence else None for sequence in sequences]
        nonempty = [(i, sequence) for i, sequence in enumerate(sequences) if sequence]
        if nonempty:
            hidden = self._continue(entry, [sequence for _, sequence in nonempty])
            lengths = torch.tensor([len(sequence) - 1 for _, sequence in nonempty], device=hidden.device)
            selected = hidden[torch.arange(len(nonempty), device=hidden.device), lengths]
            for offset in range(0, len(nonempty), 128):
                probabilities = self.model.lm_head(selected[offset:offset + 128]).float().log_softmax(-1)
                for (index, _), row in zip(nonempty[offset:offset + 128], probabilities):
                    rows[index] = row
        return torch.stack(rows)

    @torch.inference_mode()
    def tails(self, context, prefixes, tails):
        """Sum log p(tail | context + prefix) for each aligned pair.

        The final target token never needs a forward pass: its probability is
        determined by its predecessor. Empty tails have zero score, and an
        empty prefix's first target is scored from the saved prefill output.
        """
        context = _tokens(context)
        if not context:
            raise ValueError("context must contain at least one causal prefix token")
        prefixes, tails = [_tokens(p) for p in prefixes], [_tokens(t) for t in tails]
        if len(prefixes) != len(tails):
            raise ValueError("prefixes and tails must have equal lengths")
        if not any(tails):
            return [0.0] * len(tails)
        entry = self._prefill(context)
        totals = torch.zeros(len(tails), dtype=torch.float32, device=self.device)
        work, owners = [], []
        for i, (prefix, tail) in enumerate(zip(prefixes, tails)):
            if not tail:
                continue
            if not prefix:
                totals[i] += entry.next_logprobs[tail[0]]
            sequence = prefix + tail[:-1]
            if sequence:
                work.append(sequence)
                owners.append(i)
        if work:
            hidden = self._continue(entry, work)
            batch, positions, targets, destinations = [], [], [], []
            for row, owner in enumerate(owners):
                prefix, tail = prefixes[owner], tails[owner]
                for j, target in enumerate(tail):
                    position = len(prefix) + j - 1
                    if position >= 0:
                        batch.append(row), positions.append(position)
                        targets.append(target), destinations.append(owner)
            for offset in range(0, len(targets), 128):
                row = torch.tensor(batch[offset:offset + 128], device=hidden.device)
                position = torch.tensor(positions[offset:offset + 128], device=hidden.device)
                target = torch.tensor(targets[offset:offset + 128], device=hidden.device)
                destination = torch.tensor(destinations[offset:offset + 128], device=hidden.device)
                probabilities = self.model.lm_head(hidden[row, position]).float().log_softmax(-1)
                selected = probabilities.gather(1, target[:, None]).squeeze(1)
                totals.index_add_(0, destination, selected)
        return totals.tolist()
