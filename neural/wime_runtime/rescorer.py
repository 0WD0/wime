"""General causal language model shared by Chinese and Japanese decoding."""
import copy
import sys

import torch
from transformers import DynamicCache


class SharedContext(DynamicCache):
    """LM states after a left context, read by every row of a batch.

    Attention and recurrent states share the read-only context. Convolution
    states are private because single-token GatedDeltaNet updates them in place."""

    def __init__(self, prefix, rows):
        self.__dict__.update(prefix.__dict__)
        self.layers = [copy.copy(layer) for layer in prefix.layers]
        for layer in self.layers:
            if getattr(layer, "is_initialized", False):
                layer.keys = layer.keys.expand(rows, *layer.keys.shape[1:])
                layer.values = layer.values.expand(rows, *layer.values.shape[1:])
            for states in ("conv_states", "recurrent_states"):
                if getattr(layer, states, {}).get(0) is not None:
                    value = getattr(layer, states)[0].expand(rows, *getattr(layer, states)[0].shape[1:])
                    setattr(layer, states, {0: value.clone() if states == "conv_states" else value})

    def update(self, keys, values, layer_idx, *args, **kwargs):
        layer = self.layers[layer_idx]
        return torch.cat([layer.keys, keys], -2), torch.cat([layer.values, values], -2)

    def update_conv_state(self, conv_states, layer_idx, state_idx=0, **kwargs):
        return torch.cat([self.layers[layer_idx].conv_states[0], conv_states], -1)

    def update_recurrent_state(self, recurrent_states, layer_idx, state_idx=0, **kwargs):
        return recurrent_states


# Qwen3.5's linear-attention kernels, which transformers binds at import to
# flash-linear-attention or causal-conv1d when installed.
KERNELS = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update")


def reference_kernels(model):
    """Run MODEL's linear attention in PyTorch: the Triton kernels cannot read CPU tensors."""
    module = sys.modules[type(model.model).__module__]
    for name in KERNELS:
        if hasattr(module, name):
            function = getattr(module, name)
            setattr(module, name, getattr(function, "__wrapped__", function))


class Rescorer:
    def __init__(self, path, weight, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # CPUs without native BF16 matmul run FP32 several times faster.
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        self.weight, self.device = weight, device
        self.pad = self.tokenizer.pad_token_id or 0
        if device == "cpu":
            reference_kernels(self.model)

    def run(self, context, sequences):
        """Final hidden states over [last context token] + each sequence.

        All but the last context token run once per context and are shared:
        a long context then costs one prefill, not one per row."""
        model, head = self.model, context[:-1]
        if not head:
            past = None
        else:
            key = tuple(head)
            if getattr(self, "prefix", (None,))[0] != key:
                cache = DynamicCache(config=model.config)
                model.model(input_ids=torch.tensor([head], device=self.device), past_key_values=cache, use_cache=True)
                self.prefix = key, cache
            past = SharedContext(self.prefix[1], len(sequences))
        rows = [context[-1:] + list(s) for s in sequences]
        ids = torch.full((len(rows), max(map(len, rows))), self.pad)
        for i, row in enumerate(rows):
            # Right padding follows every scored position.
            ids[i, :len(row)] = torch.tensor(row)
        return model.model(input_ids=ids.to(self.device), past_key_values=past, use_cache=past is not None).last_hidden_state

    @torch.inference_mode()
    def next(self, context, sequences):
        """log p(next token | context + sequence): [len(sequences), V]."""
        hidden = self.run(context, sequences)
        last = torch.tensor([len(s) for s in sequences], device=self.device)
        return self.model.lm_head(hidden[torch.arange(len(sequences), device=self.device), last]).float().log_softmax(-1)

    @torch.inference_mode()
    def tails(self, context, prefixes, tails):
        """Sum log p(tail | context + prefix) for each pair."""
        hidden = self.run(context, [list(p) + list(t) for p, t in zip(prefixes, tails)])
        rows, columns, targets = [], [], []
        for i, (p, t) in enumerate(zip(prefixes, tails)):
            for j, token in enumerate(t):
                rows.append(i), columns.append(len(p) + j), targets.append(token)
        rows = torch.tensor(rows, device=self.device)
        picked = self.model.lm_head(hidden[rows, torch.tensor(columns, device=self.device)]).float().log_softmax(-1)
        picked = picked[torch.arange(len(targets), device=self.device), torch.tensor(targets, device=self.device)]
        return torch.zeros(len(tails), device=self.device).index_add_(0, rows, picked).tolist()
