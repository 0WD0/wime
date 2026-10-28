"""Shared dictionary readings and uncached model scoring primitives."""
import collections
import sqlite3

import torch

from inference import anchored_prompt


def readings(database):
    """Single-character readings: character -> {two-key code: auxiliary groups}."""
    result = collections.defaultdict(dict)
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        for code, char, aux in db.execute("SELECT code, word, aux FROM entry "
                                          "WHERE length(word) = 1 AND length(code) = 2 ORDER BY code"):
            result[char].setdefault(code, []).extend(g for g in aux.split(",") if g)
    return dict(result)


def aux_match(groups, typed):
    """Typed auxiliary keys select a character when they begin one of its groups."""
    return any(group.startswith(typed) for group in groups)


class DecodingModel:
    """Model interface shared by byte search and offline reference experiments."""

    def __init__(self, evaluator, rescorer, beam=16, per_hypothesis=24):
        self.evaluator, self.rescorer = evaluator, rescorer
        self.beam, self.per_hypothesis = beam, per_hypothesis
        self.weight = rescorer.weight if rescorer else 1.0
        self.device = evaluator.device if evaluator else "cpu"

    def prompts(self, context, code, anchor, known):
        """Conditional-model prompt ids and LM context ids."""
        ev = self.evaluator
        prefix = anchored_prompt(context, code, ev.model.cfg.context, anchor, known)
        return [2] + [ev.index.get(c, 1) for c in prefix], self.lm_ids(context)

    def lm_ids(self, context):
        return self.rescorer.tokenizer(context[-128:] or "\n")["input_ids"]

    def window(self):
        return self.evaluator.model.cfg.context

    def char_ids(self, texts):
        index = self.evaluator.index
        return [[index.get(c, 1) for c in text] for text in texts]

    @torch.inference_mode()
    def lm_next(self, context, sequences):
        """log p(next token | context + sequence): [len(sequences), V]."""
        model = self.rescorer.model
        ids = torch.full((len(sequences), len(context) + max(map(len, sequences))), self.rescorer.pad)
        for i, s in enumerate(sequences):
            # Right padding follows every scored position.
            ids[i, :len(context) + len(s)] = torch.tensor(context + list(s))
        hidden = model.model(input_ids=ids.to(self.device)).last_hidden_state
        last = torch.tensor([len(context) + len(s) - 1 for s in sequences], device=self.device)
        return model.lm_head(hidden[torch.arange(len(sequences), device=self.device), last]).float().log_softmax(-1)

    @torch.inference_mode()
    def lm_tails(self, context, prefixes, tails):
        """Sum log p(tail | context + prefix) for each pair."""
        model, sequences = self.rescorer.model, [list(p) + list(t) for p, t in zip(prefixes, tails)]
        ids = torch.full((len(sequences), len(context) + max(map(len, sequences))), self.rescorer.pad)
        for i, s in enumerate(sequences):
            ids[i, :len(context) + len(s)] = torch.tensor(context + s)
        hidden = model.model(input_ids=ids.to(self.device)).last_hidden_state
        rows, columns, targets = [], [], []
        for i, (p, t) in enumerate(zip(prefixes, tails)):
            for j, token in enumerate(t):
                rows.append(i), columns.append(len(context) + len(p) - 1 + j), targets.append(token)
        rows = torch.tensor(rows, device=self.device)
        picked = model.lm_head(hidden[rows, torch.tensor(columns, device=self.device)]).float().log_softmax(-1)
        picked = picked[torch.arange(len(targets), device=self.device), torch.tensor(targets, device=self.device)]
        return torch.zeros(len(tails), device=self.device).index_add_(0, rows, picked).tolist()

    @torch.inference_mode()
    def model_next(self, prompt, texts):
        """Conditional log p(next char | prompt + text): [len(texts), V]."""
        encoded = self.char_ids(texts)
        size = len(prompt) + max(map(len, encoded))
        inputs = [prompt + e + [0] * (size - len(prompt) - len(e)) for e in encoded]
        logits, _ = self.evaluator.model(torch.tensor(inputs, device=self.device))
        last = torch.tensor([len(prompt) + len(e) - 1 for e in encoded], device=self.device)
        return logits[torch.arange(len(texts), device=self.device), last].float().log_softmax(-1)

    @torch.inference_mode()
    def model_score(self, prompt, texts):
        """Conditional sum log p(text | prompt) for each text."""
        encoded = self.char_ids(texts)
        size = len(prompt) + max(map(len, encoded)) - 1
        inputs = [prompt + e[:-1] + [0] * (size - len(prompt) - len(e) + 1) for e in encoded]
        logits, _ = self.evaluator.model(torch.tensor(inputs, device=self.device))
        rows, columns, targets = [], [], []
        for i, e in enumerate(encoded):
            for j, char in enumerate(e):
                rows.append(i), columns.append(len(prompt) - 1 + j), targets.append(char)
        rows = torch.tensor(rows, device=self.device)
        picked = logits[rows, torch.tensor(columns, device=self.device)].float().log_softmax(-1)
        picked = picked[torch.arange(len(targets), device=self.device), torch.tensor(targets, device=self.device)]
        return torch.zeros(len(texts), device=self.device).index_add_(0, rows, picked).tolist()

