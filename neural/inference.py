"""Shared FP32 conditional inference for evaluation and the optional GPU worker."""
import torch
from functools import lru_cache
import json
import math
from pathlib import Path
import re

from conditional import FORMAT, load_model, prompt


def effective_prior_weight(scores, syllables, weight, confidence=1.0):
    """Use the model alone when its full-candidate probability is high enough.

    Select once from the initial complete pool, then keep it fixed for all
    verification rounds. One disables the gate for compatibility.
    """
    if not 0 < confidence <= 1:
        raise ValueError("confidence must be in (0, 1]")
    if confidence < 1 and max(scores) * syllables >= math.log(confidence):
        return 0.0
    return weight


@lru_cache(maxsize=4096)
def context_reading(text):
    """Infer only the visible committed suffix; never read candidate answers."""
    from pypinyin import lazy_pinyin
    mapping = reading_map()
    codes = [mapping.get(py) for py in lazy_pinyin(text)]
    return "".join(codes) if all(codes) else None


@lru_cache(maxsize=1)
def reading_map():
    return json.loads(Path(__file__).with_name("context-pinyin.json").read_text())["syllables"]


def anchored_prompt(context, code, window, limit=0, known_reading=None):
    """Keep committed characters inside the decoder; return only a prompt.

    Actual committed keystrokes take precedence over inferred pronunciation.
    The current composition's keys are always appended verbatim. The caller
    scores only NEW characters, and never proposes edits to this fixed prefix.
    """
    if not limit:
        return prompt(context, code, window)
    match = re.search(r"[\u4e00-\u9fff]{1,16}$", context)
    if not match:
        return prompt(context, code, window)
    text = match[0]
    size = min(limit, len(text), max(0, (window - 4) // 2 - len(code) // 2))
    if not size:
        return prompt(context, code, window)
    anchor = text[-size:]
    known_size = (min(size, len(known_reading["text"]))
                  if known_reading and context.endswith(known_reading["text"]) else 0)
    if known_size == size:
        reading = known_reading["input"][-2 * size:]
    else:
        reading = context_reading(text)
        if reading is None:
            return prompt(context, code, window)
        reading = reading[-2 * size:]
        if known_size:
            reading = reading[:-2 * known_size] + known_reading["input"][-2 * known_size:]
    return prompt(context[:-size], reading + code, window) + anchor


class ConditionalEvaluator:
    def __init__(self, path, device="cuda", context_anchor=0, proposal_order="position"):
        if proposal_order not in ("position", "gain"):
            raise ValueError("unknown proposal order")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model, self.vocab, self.metadata = load_model(path)
        if self.metadata.get("conditional") != FORMAT:
            raise ValueError("expected a wime-ms-pinyin-v1 conditional model")
        self.model.to(device).eval()
        self.device = device
        self.context_anchor = context_anchor
        self.proposal_order = proposal_order
        self.index = {c: i for i, c in enumerate(self.vocab)}
        chinese = [i for i, c in enumerate(self.vocab)
                   if len(c) == 1 and "\u3400" <= c <= "\u9fff"]
        self.chinese = torch.tensor(chinese, device=device)
        self.chinese_chars = [self.vocab[i] for i in chinese]

    @torch.inference_mode()
    def evaluate(self, context, code, texts, context_anchor=None, known_reading=None):
        limit = self.context_anchor if context_anchor is None else context_anchor
        prefix = anchored_prompt(context, code, self.model.cfg.context, limit, known_reading)
        pre = [2] + [self.index.get(c, 1) for c in prefix]
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
                    edits.append((score - row[0], draft[:position] + replacement))
        if self.proposal_order == "gain":
            # Prioritize stronger local evidence without changing the model,
            # dictionary constraints, threshold or verification budget.
            edits.sort(key=lambda edit: edit[0], reverse=True)
        return [prefix for _, prefix in edits[:16]]

