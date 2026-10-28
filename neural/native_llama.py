"""Opt-in ctypes adapter for an independently built llama_bridge shared library.

``next`` uses a bounded token-path LRU backed by native shared sequence cells.
``tails`` has a separate native context, so canonical rescoring never removes
or rewinds search state. Both return full-vocabulary-normalized FP32 scores;
there is no sampler, chat template, grammar mask, temperature, or added BOS.

The bridge accepts standard Llama GGUF models only. Pass the HF raw token-byte
table at construction to reject GGUF/HF token-ID mismatches before searching.
``context_size`` bounds each native KV pool, not each retained path. If the
pool cannot hold a batch, the adapter clears that context and raises cleanly.
This object serializes calls; close it before unloading its shared library.
"""
from collections import OrderedDict
import ctypes
from dataclasses import dataclass
import threading

import torch


def _tokens(values):
    result = tuple(values)
    if any(type(value) is not int or value < 0 for value in result):
        raise ValueError("tokens must be nonnegative integer IDs")
    return result


class _CtypesRuntime:
    def __init__(self, library_path, model_path, gpu_layers):
        self.lib = ctypes.CDLL(str(library_path))
        void, integer = ctypes.c_void_p, ctypes.c_int32
        iptr, fptr = ctypes.POINTER(integer), ctypes.POINTER(ctypes.c_float)
        signatures = {
            "error": ([], ctypes.c_char_p), "abi_version": ([], integer),
            "max_sequences": ([], integer),
            "load": ([ctypes.c_char_p, integer], void), "free_model": ([void], None),
            "vocab_size": ([void], integer),
            "token_piece": ([void, integer, void, integer], integer),
            "create_context": ([void, integer, integer, integer, integer], void),
            "free_context": ([void], None), "clear": ([void], None),
            "seq_copy": ([void, integer, integer, integer], integer),
            "seq_remove": ([void, integer], integer),
            "decode": ([void, integer, iptr, iptr, iptr, iptr, fptr, ctypes.c_int64], integer),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, "wime_llama_" + name)
            function.argtypes, function.restype = args, result
        if self.lib.wime_llama_abi_version() != 1:
            raise RuntimeError("unsupported Wime llama bridge ABI")
        self.handle = self.lib.wime_llama_load(str(model_path).encode(), gpu_layers)
        if not self.handle:
            self.fail()
        self.vocab_size = self.lib.wime_llama_vocab_size(self.handle)
        self.max_sequences = self.lib.wime_llama_max_sequences()

    def fail(self):
        raise RuntimeError(self.lib.wime_llama_error().decode("utf-8", errors="replace"))

    def create_context(self, size, batch, sequences, threads):
        handle = self.lib.wime_llama_create_context(self.handle, size, batch, sequences, threads)
        if not handle:
            self.fail()
        return _CtypesContext(self, handle)

    def token_piece(self, token):
        buffer = ctypes.create_string_buffer(256)
        size = self.lib.wime_llama_token_piece(self.handle, token, buffer, len(buffer))
        if size < 0:
            buffer = ctypes.create_string_buffer(-size)
            size = self.lib.wime_llama_token_piece(self.handle, token, buffer, len(buffer))
        if size < 0:
            raise RuntimeError("native token piece did not fit its reported size")
        return buffer.raw[:size]

    def close(self):
        if self.handle:
            self.lib.wime_llama_free_model(self.handle)
            self.handle = None


class _CtypesContext:
    def __init__(self, runtime, handle):
        self.runtime, self.handle = runtime, handle

    def clear(self):
        self.runtime.lib.wime_llama_clear(self.handle)

    def copy(self, source, destination, end):
        if self.runtime.lib.wime_llama_seq_copy(self.handle, source, destination, end):
            self.runtime.fail()

    def remove(self, sequence):
        if self.runtime.lib.wime_llama_seq_remove(self.handle, sequence):
            self.runtime.fail()

    def decode(self, tokens, positions, sequences, outputs):
        arrays = [torch.tensor(values, dtype=torch.int32) for values in (tokens, positions, sequences, outputs)]
        result = torch.empty((sum(outputs), self.runtime.vocab_size), dtype=torch.float32)
        pointers = [ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_int32)) for t in arrays]
        out = ctypes.cast(result.data_ptr(), ctypes.POINTER(ctypes.c_float))
        if self.runtime.lib.wime_llama_decode(self.handle, len(tokens), *pointers, out, result.numel()):
            self.runtime.fail()
        return result

    def close(self):
        if self.handle:
            self.runtime.lib.wime_llama_free_context(self.handle)
            self.handle = None


@dataclass(frozen=True)
class _Path:
    sequence_id: int
    row: torch.Tensor


class NativeLlamaLM:
    def __init__(self, library_path, model_path, *, gpu_layers=0, context_size=8192,
                 max_paths=128, batch_size=256, threads=4, token_bytes=None,
                 _runtime=None):
        if any(type(v) is not int or v < 1 for v in (context_size, max_paths, batch_size, threads)):
            raise ValueError("native cache, batch and thread sizes must be positive integers")
        if type(gpu_layers) is not int:
            raise ValueError("gpu_layers must be an integer")
        self._runtime = _runtime or _CtypesRuntime(library_path, model_path, gpu_layers)
        self.vocab_size, self.device = self._runtime.vocab_size, "cpu"
        self.context_size, self.max_paths = context_size, max_paths
        self.batch_size, self.threads = batch_size, threads
        self._pending_limit = min(max_paths, 32, self._runtime.max_sequences - max_paths - 1)
        if self._pending_limit < 1:
            self._runtime.close()
            raise ValueError("max_paths leaves no native sequence slot for continuation")
        self._sequence_slots = max_paths + self._pending_limit + 1
        self._lock = threading.RLock()
        self._search, self._scoring = None, None
        self._context = self._score_context = None
        self._root = self._score_root = None
        self._paths = OrderedDict()
        self._free = set(range(1, self._sequence_slots))
        self._closed = False
        self.stats = dict(prefills=0, decode_calls=0, input_tokens=0, logit_rows=0,
                          copies=0, hits=0, parent_hits=0, evictions=0)
        if token_bytes is not None:
            try:
                self.validate_token_bytes(token_bytes)
            except Exception:
                self.close()
                raise

    def validate_token_bytes(self, token_bytes):
        if len(token_bytes) != self.vocab_size:
            raise ValueError("HF and GGUF vocabulary sizes differ")
        for token, expected in enumerate(token_bytes):
            if expected is not None and self._runtime.token_piece(token) != expected:
                raise ValueError(f"HF and GGUF token bytes differ at ID {token}")

    def _check(self, context, sequences):
        if self._closed:
            raise RuntimeError("native model is closed")
        if not context:
            raise ValueError("context must contain at least one causal token")
        for sequence in (context, *sequences):
            if any(token >= self.vocab_size for token in sequence):
                raise ValueError("token ID exceeds native vocabulary")
        if len(context) + max(map(len, sequences), default=0) > self.context_size:
            raise ValueError("sequence exceeds native context capacity")

    def _new_context(self):
        # A full pending batch can coexist with the bounded retained paths.
        return self._runtime.create_context(self.context_size, self.batch_size,
                                             self._sequence_slots, self.threads)

    def _forward(self, ctx, records):
        """Yield selected rows, with at most 128 full-vocab rows live per decode."""
        begin = 0
        while begin < len(records):
            end, outputs = begin, 0
            while end < len(records) and end - begin < self.batch_size:
                wanted = records[end][3] is not None
                if wanted and outputs == 128:
                    break
                outputs += wanted
                end += 1
            chunk = records[begin:end]
            raw = ctx.decode([r[0] for r in chunk], [r[1] for r in chunk],
                             [r[2] for r in chunk], [int(r[3] is not None) for r in chunk])
            self.stats["decode_calls"] += 1
            self.stats["input_tokens"] += len(chunk)
            self.stats["logit_rows"] += outputs
            if outputs:
                yield [r[3] for r in chunk if r[3] is not None], raw.float().log_softmax(-1)
            begin = end

    def _prefill(self, ctx, context):
        records = [(token, i, 0, 0 if i == len(context) - 1 else None)
                   for i, token in enumerate(context)]
        row = None
        for _, rows in self._forward(ctx, records):
            row = rows[0].clone()
        self.stats["prefills"] += 1
        return row

    def _reset_search(self):
        if self._search is not None:
            self._search.clear()
        self._context, self._root = None, None
        self._paths.clear()
        self._free = set(range(1, self._sequence_slots))

    @torch.inference_mode()
    def next(self, context, sequences):
        with self._lock:
            context, sequences = _tokens(context), [_tokens(s) for s in sequences]
            self._check(context, sequences)
            if not sequences:
                return torch.empty((0, self.vocab_size), dtype=torch.float32)
            if self._search is None:
                self._search = self._new_context()
            try:
                if self._context != context:
                    self._reset_search()
                    self._root = self._prefill(self._search, context)
                    self._context = context
                rows, missing = {}, []
                for sequence in dict.fromkeys(sequences):
                    if not sequence:
                        rows[sequence] = self._root
                    elif sequence in self._paths:
                        rows[sequence] = self._paths[sequence].row
                        self._paths.move_to_end(sequence)
                        self.stats["hits"] += 1
                    else:
                        missing.append(sequence)
                for offset in range(0, len(missing), self._pending_limit):
                    pending, records = [], []
                    for sequence in missing[offset:offset + self._pending_limit]:
                        parent, parent_id = (), 0
                        for length in range(len(sequence) - 1, 0, -1):
                            candidate = sequence[:length]
                            if candidate in self._paths:
                                parent, parent_id = candidate, self._paths[candidate].sequence_id
                                self._paths.move_to_end(candidate)
                                self.stats["parent_hits"] += 1
                                break
                        sequence_id = self._free.pop()
                        self._search.copy(parent_id, sequence_id, len(context) + len(parent))
                        self.stats["copies"] += 1
                        pending.append((sequence, sequence_id))
                        records.extend((token, len(context) + i, sequence_id,
                                        sequence if i == len(sequence) - 1 else None)
                                       for i, token in enumerate(sequence) if i >= len(parent))
                    for tags, probabilities in self._forward(self._search, records):
                        for sequence, row in zip(tags, probabilities):
                            rows[sequence] = row.clone()
                    for sequence, sequence_id in pending:
                        self._paths[sequence] = _Path(sequence_id, rows[sequence])
                        while len(self._paths) > self.max_paths:
                            _, old = self._paths.popitem(last=False)
                            self._search.remove(old.sequence_id)
                            self._free.add(old.sequence_id)
                            self.stats["evictions"] += 1
                return torch.stack([rows[sequence] for sequence in sequences])
            except Exception:
                self._reset_search()
                raise

    @torch.inference_mode()
    def tails(self, context, prefixes, tails):
        with self._lock:
            context = _tokens(context)
            prefixes, tails = [_tokens(p) for p in prefixes], [_tokens(t) for t in tails]
            if len(prefixes) != len(tails):
                raise ValueError("prefixes and tails must have equal lengths")
            work = [p + t[:-1] if t else () for p, t in zip(prefixes, tails)]
            self._check(context, work)
            if any(token >= self.vocab_size for sequence in prefixes + tails for token in sequence):
                raise ValueError("target ID exceeds native vocabulary")
            if not any(tails):
                return [0.0] * len(tails)
            if self._scoring is None:
                self._scoring = self._new_context()
            try:
                if self._score_context != context:
                    self._scoring.clear()
                    self._score_context = None
                    self._score_root = self._prefill(self._scoring, context)
                    self._score_context = context
                scores = torch.zeros(len(tails), dtype=torch.float32)
                for owner, (prefix, tail) in enumerate(zip(prefixes, tails)):
                    if tail and not prefix:
                        scores[owner] = self._score_root[tail[0]]
                active = [i for i, sequence in enumerate(work) if sequence and tails[i]]
                for offset in range(0, len(active), self._pending_limit):
                    chunk, records = active[offset:offset + self._pending_limit], []
                    for sequence_id, owner in enumerate(chunk, 1):
                        self._scoring.copy(0, sequence_id, len(context))
                        self.stats["copies"] += 1
                        prefix, tail = prefixes[owner], tails[owner]
                        for i, token in enumerate(work[owner]):
                            target = i - len(prefix) + 1
                            tag = (owner, tail[target]) if 0 <= target < len(tail) else None
                            records.append((token, len(context) + i, sequence_id, tag))
                    for tags, probabilities in self._forward(self._scoring, records):
                        owners = torch.tensor([tag[0] for tag in tags])
                        targets = torch.tensor([tag[1] for tag in tags])
                        picked = probabilities[torch.arange(len(tags)), targets]
                        scores.index_add_(0, owners, picked)
                    for sequence_id in range(1, len(chunk) + 1):
                        self._scoring.remove(sequence_id)
                return scores.tolist()
            except Exception:
                self._scoring.clear()
                self._score_context, self._score_root = None, None
                raise

    def close(self):
        with self._lock:
            if not self._closed:
                for context in (self._search, self._scoring):
                    if context is not None:
                        context.close()
                self._runtime.close()
                self._paths.clear()
                self._root = self._score_root = None
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
