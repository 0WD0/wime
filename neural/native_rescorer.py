"""Tokenizer-only Rescorer facade for the optional native causal LM backend.

HF remains responsible for tokenization, including its BOS post-processor.
The native backend receives token IDs and returns full-vocabulary-normalized
scores. No transformers model weights are loaded by this facade.
"""
import torch

from native_llama import NativeLlamaLM
from rescorer import Rescorer, chinese
from token_bytes import raw_token_bytes


class NativeRescorer(Rescorer):
    def __init__(self, tokenizer_path, weight, library_path, model_path, gpu_layers=99):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.weight, self.device = weight, "cpu"
        self.pad = self.tokenizer.pad_token_id or 0
        self.backend = NativeLlamaLM(library_path, model_path, gpu_layers=gpu_layers,
                                     token_bytes=raw_token_bytes(self.tokenizer))

    @torch.inference_mode()
    def score(self, context, texts):
        """Preserve Rescorer's separate target tokenization and character mean."""
        if not texts:
            return []
        prefix = self.tokenizer(context[-128:] or "\n")["input_ids"]
        targets = [tuple(self.tokenizer(text, add_special_tokens=False)["input_ids"])
                   for text in texts]
        totals = self.backend.tails(prefix, [()] * len(targets), targets)
        return [total / len(text) for total, text in zip(totals, texts)]

    @torch.inference_mode()
    def proposals(self, context, draft, margin=0.5):
        """Use the same token boundaries, top 12 alternatives and edit order."""
        prefix = self.tokenizer(context[-128:] or "\n")["input_ids"]
        target = tuple(self.tokenizer(draft, add_special_tokens=False)["input_ids"])
        if not target:
            return []
        logits = self.backend.next(prefix, [target[:k] for k in range(len(target))])
        edits, offset = [], 0
        for k, token in enumerate(target):
            own = float(logits[k, token])
            values, indices = logits[k].topk(12)
            for value, index in zip(values.tolist(), indices.tolist()):
                text = self.tokenizer.decode([index])
                if index != token and chinese(text) and value > own + margin:
                    edits.append((value - own, (draft[:offset] + text)[:len(draft)]))
            offset += len(self.tokenizer.decode([token]))
        edits.sort(key=lambda edit: -edit[0])
        return [p for _, p in edits if p != draft[:len(p)]]

    def close(self):
        self.backend.close()
