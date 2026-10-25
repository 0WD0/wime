"""Optional final-stage rescoring with a general pretrained causal LM.

It scores complete, dictionary-verified candidates and proposes alternative
tokens as prefixes, interleaved with the conditional model's proposals; the
dictionary still checks every prefix against the typed keys, so the LM needs
no pinyin. Requires transformers with the model's architecture.
"""
import torch


def interleave(first, second, limit=16):
    """Alternate two proposal lists, dropping repeats, as one bounded list."""
    result = []
    for i in range(max(len(first), len(second))):
        for prefix in first[i:i + 1] + second[i:i + 1]:
            if prefix not in result:
                result.append(prefix)
    return result[:limit]


def chinese(text):
    return bool(text) and all("\u3400" <= c <= "\u9fff" for c in text)


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

    @torch.inference_mode()
    def proposals(self, context, draft, margin=0.5):
        """Prefixes ending in a token the LM prefers to the draft's own by MARGIN.

        Alternatives start at the draft's token boundaries; strongest first.
        """
        prefix = self.tokenizer(context[-128:] or "\n")["input_ids"]
        target = self.tokenizer(draft)["input_ids"]
        ids = torch.tensor([prefix + target[:-1]], device=self.device)
        logits = self.model(input_ids=ids).logits[0, len(prefix) - 1:].float().log_softmax(-1)
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

    def combine(self, context, texts, scores):
        return [s + self.weight * r for s, r in zip(scores, self.score(context, texts))]
