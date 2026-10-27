"""Bounded conditional next-character rows shared by candidate prefixes.

This is a row cache, not a KV cache: each missing *distinct parent prefix*
still runs CharLM normally. Siblings reuse one distribution, and later search
frontiers reuse earlier rows. Keys include the entire encoded conditional
prompt, so changed pinyin/context cannot reuse stale probabilities. OOV text
aliases intentionally share rows because Decoder.char_ids maps both to UNK.

Scores match Decoder.model_score: sum character log probabilities, with no
implicit EOS or length normalization. An explicitly supplied EOS character is
scored just like any other character. The model must be in eval mode; clear
the cache after changing its weights, and recreate it after vocabulary/device
changes. Batched forward kernels may differ by ordinary FP32 roundoff.
"""
from collections import OrderedDict, defaultdict

import torch


class ConditionalPrefixScores:
    def __init__(self, evaluator, max_rows=512, batch_size=32):
        if (type(max_rows) is not int or max_rows < 1
                or type(batch_size) is not int or batch_size < 1):
            raise ValueError("conditional cache sizes must be positive integers")
        self.evaluator = evaluator
        self.model, self.device = evaluator.model, evaluator.device
        self.max_rows, self.batch_size = max_rows, batch_size
        self._rows = OrderedDict()
        self.stats = dict(rows=0, batches=0, input_tokens=0, cache_hits=0)

    def clear(self):
        """Drop cached distributions; cumulative counters are unchanged."""
        self._rows.clear()

    def _inputs(self, prompt, texts):
        if self.model.training:
            raise ValueError("conditional prefix caching requires model.eval()")
        prompt = tuple(prompt)
        if not prompt or any(type(token) is not int or token < 0 for token in prompt):
            raise ValueError("conditional prompt must contain nonnegative integer token IDs")
        if len(prompt) > self.model.cfg.context:
            raise ValueError("conditional prompt exceeds model window")
        texts = list(texts)
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("conditional candidates must be strings")
        index = self.evaluator.index
        return prompt, [tuple(index.get(char, 1) for char in text) for text in texts]

    def _batches(self, prompt, prefixes):
        """Yield distinct prefixes and their rows, retaining only a bounded LRU."""
        missing, hits = defaultdict(list), []
        for prefix in dict.fromkeys(prefixes):
            if len(prompt) + len(prefix) > self.model.cfg.context:
                raise ValueError("conditional candidate prefix exceeds model window")
            key = (prompt, prefix)
            if key in self._rows:
                hits.append((prefix, self._rows[key]))
                self._rows.move_to_end(key)
                self.stats["cache_hits"] += 1
            else:
                missing[len(prefix)].append(prefix)
        # Hold hits for this request before inserting/evicting any new rows.
        # This also works when the request is larger than the cache budget.
        for offset in range(0, len(hits), self.batch_size):
            chunk = hits[offset:offset + self.batch_size]
            yield [prefix for prefix, _ in chunk], torch.stack([row for _, row in chunk])
        del hits
        for length in sorted(missing):
            group = missing[length]
            for offset in range(0, len(group), self.batch_size):
                chunk = group[offset:offset + self.batch_size]
                ids = torch.tensor([prompt + prefix for prefix in chunk], device=self.device)
                logits, _ = self.model(ids)
                rows = logits[:, -1].float().log_softmax(-1)
                del logits
                self.stats["rows"] += len(chunk)
                self.stats["batches"] += 1
                self.stats["input_tokens"] += ids.numel()
                for prefix, row in zip(chunk, rows):
                    # A view would keep all sibling rows alive after eviction.
                    self._rows[(prompt, prefix)] = row.clone()
                    if len(self._rows) > self.max_rows:
                        self._rows.popitem(last=False)
                yield chunk, rows

    @torch.inference_mode()
    def score(self, prompt, texts):
        """Sum log p(text | prompt), preserving candidate order and duplicates."""
        prompt, encoded = self._inputs(prompt, texts)
        owners, requests = [], defaultdict(list)
        for owner, tokens in enumerate(encoded):
            for position, target in enumerate(tokens):
                requests[tokens[:position]].append((len(owners), target))
                owners.append(owner)
        if not owners:
            return [0.0] * len(encoded)
        picked = torch.empty(len(owners), dtype=torch.float32, device=self.device)
        for prefixes, rows in self._batches(prompt, requests):
            source, positions, targets = [], [], []
            for row, prefix in enumerate(prefixes):
                for position, target in requests[prefix]:
                    source.append(row)
                    positions.append(position)
                    targets.append(target)
            picked[torch.tensor(positions, device=self.device)] = rows[
                torch.tensor(source, device=self.device), torch.tensor(targets, device=self.device)]
        # Match the legacy FP32 accumulation and original per-text token order;
        # Python double sums would introduce a second rounding convention.
        scores = torch.zeros(len(encoded), dtype=torch.float32, device=self.device)
        scores.index_add_(0, torch.tensor(owners, device=self.device), picked)
        return scores.tolist()

    @torch.inference_mode()
    def next(self, prompt, texts):
        """log p(next char | prompt + text), as an FP32 [N, vocab] tensor."""
        prompt, encoded = self._inputs(prompt, texts)
        if not encoded:
            return torch.empty((0, self.model.cfg.vocab), dtype=torch.float32, device=self.device)
        owners = defaultdict(list)
        for owner, prefix in enumerate(encoded):
            owners[prefix].append(owner)
        result = None
        for prefixes, rows in self._batches(prompt, owners):
            if result is None:
                result = torch.empty((len(encoded), rows.shape[1]), dtype=torch.float32,
                                     device=self.device)
            source, positions = [], []
            for row, prefix in enumerate(prefixes):
                for owner in owners[prefix]:
                    source.append(row)
                    positions.append(owner)
            result[torch.tensor(positions, device=self.device)] = rows[
                torch.tensor(source, device=self.device)]
        return result
