"""Bounded, immutable parent-path states for exact search-path continuation.

Only the explicitly supported Hugging Face DynamicCache layout is reused.
Paths share immutable forward-batch snapshots, including hybrid recurrent
states. Storage is charged and evicted by whole batch, never by a view's size.
Unfamiliar cache metadata uses the existing prefix-only implementation.
No padded or rolled-back state is admitted.
"""
from collections import OrderedDict, defaultdict
import copy
from dataclasses import dataclass

import torch

from prefix_cache import _tokens


class _UnsupportedCache(Exception):
    pass


def _cache_types():
    try:
        from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer
    except ImportError:
        return ()
    return DynamicCache, DynamicLayer, LinearAttentionLayer


@dataclass(frozen=True)
class _BatchState:
    cache: object
    length: int
    rows: torch.Tensor
    size: int
    sequences: tuple


@dataclass(frozen=True)
class _PathState:
    batch: _BatchState
    index: int

    @property
    def cache(self):
        return self.batch.cache

    @property
    def length(self):
        return self.batch.length

    @property
    def row(self):
        return self.batch.rows[self.index]


class BranchCacheLM:
    """Reuse only exact token parents, without changing the decoder's search.

    ``max_bytes`` bounds retained tensor storage, not temporary forward-pass
    allocations. Touching one path keeps its entire batch alive and charged.
    An individual batch exceeding either limit is not retained. Context
    changes discard all path states. The caller's score memo and the fallback's
    context prefill remain separate caches.
    """

    def __init__(self, rescorer, fallback, max_paths=128, max_bytes=512 * 1024**2):
        if (type(max_paths) is not int or max_paths < 1 or
                type(max_bytes) is not int or max_bytes < 1):
            raise ValueError("branch cache limits must be positive integers")
        self.rescorer, self.model, self.fallback = rescorer, rescorer.model, fallback
        self.device = rescorer.device
        self.max_paths, self.max_bytes = max_paths, max_bytes
        self._types = _cache_types()
        self._context = None
        self._paths = {}
        self._batches = OrderedDict()
        self._disabled = False
        self.stats = dict(hits=0, misses=0, exact_hits=0, fallback_batches=0,
                          forwarded_tokens=0, saved_tokens=0, continuations=0,
                          entries=0, bytes=0, evictions=0, batches=0,
                          batch_evictions=0, oversized_batches=0)

    def clear(self):
        self._paths.clear()
        self._batches.clear()
        self._context = None
        self._disabled = False
        self.stats.update(entries=0, bytes=0, batches=0)

    def _layout(self, cache, length):
        """Validate every mutable field before copying or combining states."""
        if not self._types or type(cache) is not self._types[0]:
            raise _UnsupportedCache("cache type")
        if (set(vars(cache)) != {"layers", "layer_class_to_replicate", "offloading"}
                or cache.offloading is not False
                or cache.layer_class_to_replicate not in (None, self._types[1])
                or type(cache.layers) is not list or not cache.layers):
            raise _UnsupportedCache("cache metadata")
        layouts, batch = [], None
        attention = False
        for layer in cache.layers:
            if type(layer) is self._types[1]:
                names = {"keys", "values", "is_initialized", "dtype", "device"}
                if set(vars(layer)) != names or layer.is_initialized is not True:
                    raise _UnsupportedCache("attention metadata")
                tensors = {"keys": layer.keys, "values": layer.values}
                metadata = {k: getattr(layer, k) for k in names - tensors.keys()}
                if any(not isinstance(t, torch.Tensor) or t.ndim != 4 or
                       t.shape[-2] != length for t in tensors.values()):
                    raise _UnsupportedCache("attention dimensions")
                if layer.keys.shape != layer.values.shape:
                    raise _UnsupportedCache("attention key/value dimensions")
                attention = True
            elif type(layer) is self._types[2]:
                names = {"number_of_states", "conv_states", "recurrent_states",
                         "is_conv_states_initialized", "is_recurrent_states_initialized",
                         "has_previous_state", "conv_kernel_size", "device", "dtype", "record_past"}
                if (set(vars(layer)) != names or layer.record_past is not False or
                        type(layer.number_of_states) is not int or layer.number_of_states < 1):
                    raise _UnsupportedCache("linear metadata")
                keys = set(range(layer.number_of_states))
                for name in ("conv_states", "recurrent_states", "is_conv_states_initialized",
                             "is_recurrent_states_initialized", "has_previous_state", "conv_kernel_size"):
                    value = getattr(layer, name)
                    if type(value) is not dict or set(value) != keys:
                        raise _UnsupportedCache("linear state indices")
                for name in ("is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state"):
                    if any(v is not True for v in getattr(layer, name).values()):
                        raise _UnsupportedCache("uninitialized linear state")
                tensors = {}
                for index in keys:
                    conv, recurrent = layer.conv_states[index], layer.recurrent_states[index]
                    if (not isinstance(conv, torch.Tensor) or conv.ndim != 3 or
                            not isinstance(recurrent, torch.Tensor) or recurrent.ndim != 4 or
                            type(layer.conv_kernel_size[index]) is not int or
                            conv.shape[-1] != layer.conv_kernel_size[index]):
                        raise _UnsupportedCache("linear dimensions")
                    tensors[("conv_states", index)] = conv
                    tensors[("recurrent_states", index)] = recurrent
                metadata = {k: getattr(layer, k) for k in names - {"conv_states", "recurrent_states"}}
            else:
                raise _UnsupportedCache("layer type")
            for tensor in tensors.values():
                if tensor.shape[0] < 1 or tensor.device != layer.device:
                    raise _UnsupportedCache("state device or batch")
                if batch is None:
                    batch = tensor.shape[0]
                if tensor.shape[0] != batch:
                    raise _UnsupportedCache("inconsistent state batch")
            layouts.append((metadata, tensors))
        if not attention:
            raise _UnsupportedCache("no position-tracking attention layer")
        return layouts, batch

    def _copy_rows(self, sources, length):
        """Fork a batch with independent tensors; validate each source once."""
        first = sources[0][0]
        checked = {}
        for cache, row in sources:
            if id(cache) not in checked:
                checked[id(cache)] = (cache, *self._layout(cache, length))
            _, layout, batch = checked[id(cache)]
            if (type(row) is not int or not 0 <= row < batch or
                    cache.layer_class_to_replicate is not first.layer_class_to_replicate or
                    len(layout) != len(first.layers)):
                raise _UnsupportedCache("incompatible cache rows")
        layouts = [checked[id(cache)][1] for cache, _ in sources]
        unique = list(checked.values())
        result = copy.copy(first)
        result.layers = []
        for index, original in enumerate(first.layers):
            metadata, tensors = layouts[0][index]
            for cache, layout, _ in unique:
                other_meta, other_tensors = layout[index]
                if (type(cache.layers[index]) is not type(original) or
                        metadata != other_meta or tensors.keys() != other_tensors.keys() or
                        any(t.shape[1:] != other_tensors[k].shape[1:] or
                            t.dtype != other_tensors[k].dtype or t.device != other_tensors[k].device
                            for k, t in tensors.items())):
                    raise _UnsupportedCache("incompatible layer metadata")
            layer = copy.copy(original)
            # The validated metadata contains only scalar values and dicts
            # of scalar values. No tensor or nested mutable object is shared.
            layer.__dict__ = {k: v.copy() if type(v) is dict else v for k, v in metadata.items()}
            for key in tensors:
                value = torch.cat([layout[index][1][key][row:row + 1]
                                   for (_, row), layout in zip(sources, layouts)], dim=0)
                if isinstance(key, tuple):
                    name, state = key
                    layer.__dict__.setdefault(name, {})[state] = value
                else:
                    setattr(layer, key, value)
            result.layers.append(layer)
        return result

    @staticmethod
    def _size(layouts, rows):
        tensors = [t for _, ts in layouts for t in ts.values()] + [rows]
        stores = {(t.device, t.untyped_storage().data_ptr()): t.untyped_storage().nbytes() for t in tensors}
        return sum(stores.values())

    def _admit(self, sequences, cache, length, rows, layout):
        # This is a fresh forward result from an independent fork. Retain it
        # directly: no row copies and no uncharged slices of a larger batch.
        size = self._size(layout, rows)
        if len(sequences) > self.max_paths or size > self.max_bytes:
            self.stats["oversized_batches"] += 1
            return
        while self._batches and (len(self._paths) + len(sequences) > self.max_paths or
                                 self.stats["bytes"] + size > self.max_bytes):
            _, previous = self._batches.popitem(last=False)
            for sequence in previous.sequences:
                del self._paths[sequence]
            self.stats["bytes"] -= previous.size
            self.stats["evictions"] += len(previous.sequences)
            self.stats["batch_evictions"] += 1
        snapshot = _BatchState(cache, length, rows, size, tuple(sequences))
        self._batches[id(snapshot)] = snapshot
        for index, sequence in enumerate(sequences):
            self._paths[sequence] = _PathState(snapshot, index)
        self.stats["bytes"] += size
        self.stats["entries"] = len(self._paths)
        self.stats["batches"] = len(self._batches)

    def _fallback(self, context, sequences):
        self._paths.clear()
        self._batches.clear()
        self.stats.update(entries=0, bytes=0, batches=0)
        self._disabled = True
        self.stats["fallback_batches"] += 1
        return self.fallback.next(context, sequences)

    @torch.inference_mode()
    def next(self, context, sequences):
        context = _tokens(context)
        sequences = [_tokens(sequence) for sequence in sequences]
        if self._context != context:
            self.clear()
            self._context = context
        if self._disabled:
            return self._fallback(context, sequences)
        entry = self.fallback._prefill(context)
        try:
            _, batch = self._layout(entry.cache, len(context))
            if batch != 1:
                raise _UnsupportedCache("prefill batch")
            rows, groups = {}, defaultdict(list)
            for sequence in dict.fromkeys(sequences):
                if not sequence:
                    rows[sequence] = entry.next_logprobs
                elif sequence in self._paths:
                    path = self._paths[sequence]
                    rows[sequence] = path.row
                    self._batches.move_to_end(id(path.batch))
                    self.stats["exact_hits"] += 1
                else:
                    parent = self._paths.get(sequence[:-1])
                    if parent is not None:
                        self._batches.move_to_end(id(parent.batch))
                        suffix, cache, length = sequence[-1:], parent.cache, parent.length
                        index = parent.index
                        self.stats["hits"] += 1
                        self.stats["saved_tokens"] += len(sequence) - 1
                    else:
                        suffix, cache, length = sequence, entry.cache, len(context)
                        index = 0
                        self.stats["misses"] += 1
                    groups[(length, len(suffix))].append((sequence, suffix, cache, index))
            # Equal parent length and suffix length makes every state real:
            # there is no padding to remove from any recurrent/cache layer.
            for (length, suffix_length), work in groups.items():
                for start in range(0, len(work), 16):
                    chunk = work[start:start + 16]
                    branch = self._copy_rows([(cache, index) for _, _, cache, index in chunk], length)
                    inputs = torch.tensor([suffix for _, suffix, _, _ in chunk], dtype=torch.long, device=self.device)
                    mask = torch.ones((len(chunk), length + suffix_length), dtype=torch.bool, device=self.device)
                    result = self.model.model(input_ids=inputs, attention_mask=mask,
                                              past_key_values=branch, use_cache=True)
                    cache = getattr(result, "past_key_values", None)
                    layout, batch = self._layout(cache, length + suffix_length)
                    if batch != len(chunk):
                        raise _UnsupportedCache("returned batch")
                    probabilities = self.model.lm_head(result.last_hidden_state[:, -1]).float().log_softmax(-1)
                    self.stats["continuations"] += 1
                    self.stats["forwarded_tokens"] += len(chunk) * suffix_length
                    for index in range(len(chunk)):
                        sequence = chunk[index][0]
                        rows[sequence] = probabilities[index]
                    self._admit([item[0] for item in chunk], cache,
                                length + suffix_length, probabilities, layout)
            if not sequences:
                return entry.next_logprobs.new_empty((0, entry.next_logprobs.numel()))
            return torch.stack([rows[sequence] for sequence in sequences])
        except _UnsupportedCache:
            return self._fallback(context, sequences)
