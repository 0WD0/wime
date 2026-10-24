"""Shared FP32 conditional inference for evaluation and the optional GPU worker."""
import torch

from conditional import FORMAT, load_model, prompt


class ConditionalEvaluator:
    def __init__(self, path, device="cuda"):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model, self.vocab, self.metadata = load_model(path)
        if self.metadata.get("conditional") != FORMAT:
            raise ValueError("expected a wime-ms-pinyin-v1 conditional model")
        self.model.to(device).eval()
        self.device = device
        self.index = {c: i for i, c in enumerate(self.vocab)}
        chinese = [i for i, c in enumerate(self.vocab)
                   if len(c) == 1 and "\u3400" <= c <= "\u9fff"]
        self.chinese = torch.tensor(chinese, device=device)
        self.chinese_chars = [self.vocab[i] for i in chinese]

    @torch.inference_mode()
    def evaluate(self, context, code, texts):
        pre = [2] + [self.index.get(c, 1) for c in prompt(context, code, self.model.cfg.context)]
        encoded = [[self.index.get(c, 1) for c in text] for text in texts]
        if any(len(pre) + len(t) > self.model.cfg.context for t in encoded):
            raise ValueError("candidate exceeds model window")
        size = max(map(len, encoded)) + len(pre) - 1
        inputs = [pre + tail[:-1] + [0] * (size - len(pre) - len(tail) + 1) for tail in encoded]
        # Match the deployed float32 CPU worker. BF16 can flip close candidate
        # comparisons and prefix thresholds even with the same saved weights.
        logits, _ = self.model(torch.tensor(inputs, device=self.device))
        distributions = logits[:, len(pre) - 1:].float().log_softmax(-1)
        scores = [float(distributions[i, torch.arange(len(t)), t].mean()) for i, t in enumerate(encoded)]
        return scores, distributions

    def proposals(self, draft, distribution, margin):
        edits = []
        values, indices = distribution[:len(draft), self.chinese].topk(4, dim=-1)
        target = torch.tensor([self.index.get(c, 1) for c in draft], device=self.device)
        current = distribution[torch.arange(len(draft), device=self.device), target]
        # One transfer for probabilities and one for indices, not a GPU sync
        # for every character/alternative in the Python loop.
        probabilities = torch.cat([current[:, None], values], dim=1).cpu().tolist()
        alternatives = indices.cpu().tolist()
        for position, (row, tokens) in enumerate(zip(probabilities, alternatives)):
            for score, token in zip(row[1:], tokens):
                replacement = self.chinese_chars[token]
                if score > row[0] + margin and replacement != draft[position]:
                    edits.append(draft[:position] + replacement)
        return edits[:16]

