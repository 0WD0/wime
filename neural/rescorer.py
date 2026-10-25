"""Optional final-stage rescoring with a general pretrained causal LM.

The conditional model still proposes prefixes; this only scores complete,
dictionary-verified candidates. It needs no pinyin: every candidate already
matches the typed keys. Requires transformers with the model's architecture.
"""
import torch


class Rescorer:
    def __init__(self, path, weight, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        self.weight, self.device = weight, device
        self.pad = self.tokenizer.pad_token_id or 0

    @torch.inference_mode()
    def score(self, context, texts):
        """Mean per-character log p(text | committed context), one batch."""
        prefix = self.tokenizer(context[-128:] or "\n")["input_ids"]
        targets = [self.tokenizer(text)["input_ids"] for text in texts]
        ids = torch.full((len(texts), len(prefix) + max(map(len, targets))), self.pad)
        for i, target in enumerate(targets):
            # Right padding follows every scored position, so neither causal nor
            # linear-attention layers see it.
            ids[i, :len(prefix) + len(target)] = torch.tensor(prefix + target)
        logits = self.model(input_ids=ids.to(self.device)).logits.float().log_softmax(-1)
        scores = []
        for i, target in enumerate(targets):
            positions = torch.arange(len(prefix) - 1, len(prefix) - 1 + len(target))
            total = logits[i, positions, torch.tensor(target)].sum()
            scores.append(float(total) / len(texts[i]))
        return scores

    def combine(self, context, texts, scores):
        return [s + self.weight * r for s, r in zip(scores, self.score(context, texts))]
