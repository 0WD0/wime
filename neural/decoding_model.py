"""Shared dictionary readings and uncached model scoring primitives."""
import collections
import sqlite3

import torch

from composition import prompt


def readings(database):
    """Single-character readings: character -> {two-key code: auxiliary groups}."""
    result = collections.defaultdict(dict)
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        for code, char, aux in db.execute("SELECT code, word, aux FROM entry "
                                          "WHERE length(word) = 1 AND length(code) = 2 ORDER BY code"):
            result[char].setdefault(code, []).extend(g for g in aux.split(",") if g)
    return dict(result)


def tones(database):
    """Single-character tones: character -> {two-key code: tone digits}."""
    result = collections.defaultdict(lambda: collections.defaultdict(str))
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        for code, char, digits in db.execute("SELECT code, word, tones FROM entry "
                                             "WHERE length(word) = 1 AND length(code) = 2"):
            result[char][code] += digits
    return {char: dict(codes) for char, codes in result.items()}


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

    def prompts(self, context, code, tones=()):
        """Conditional-model prompt ids and LM context ids."""
        return self.model_prompt(context, code, tones), self.lm_ids(context)

    def model_prompt(self, context, code, tones=()):
        """The conditional model reads no tones; a syllable transducer's does."""
        ev = self.evaluator
        return [2] + [ev.index.get(c, 1) for c in prompt(context, code, ev.model.cfg.context)]

    def lm_ids(self, context):
        return self.rescorer.tokenizer(context[-128:] or "\n")["input_ids"]

    def literal_ids(self, text):
        return tuple(self.rescorer.tokenizer(text, add_special_tokens=False)["input_ids"])

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

    def model_next(self, prompt, texts):
        """Conditional log p(next char | prompt + text): [len(texts), V]."""
        return self.model_next_each([prompt] * len(texts), texts)

    def model_score(self, prompt, texts):
        """Conditional sum log p(text | prompt) for each text."""
        return self.model_score_each([prompt] * len(texts), texts)

    @torch.inference_mode()
    def model_next_each(self, prompts, texts):
        """model_next with its own prompt for each text."""
        rows = [p + e for p, e in zip(prompts, self.char_ids(texts))]
        size = max(map(len, rows))
        logits, _ = self.evaluator.model(torch.tensor([r + [0] * (size - len(r)) for r in rows], device=self.device))
        last = torch.tensor([len(r) - 1 for r in rows], device=self.device)
        return logits[torch.arange(len(texts), device=self.device), last].float().log_softmax(-1)

    @torch.inference_mode()
    def model_score_each(self, prompts, texts):
        """model_score with its own prompt for each text."""
        encoded = self.char_ids(texts)
        size = max(len(p) + len(e) - 1 for p, e in zip(prompts, encoded))
        inputs = [p + e[:-1] + [0] * (size - len(p) - len(e) + 1) for p, e in zip(prompts, encoded)]
        logits, _ = self.evaluator.model(torch.tensor(inputs, device=self.device))
        rows, columns, targets = [], [], []
        for i, (p, e) in enumerate(zip(prompts, encoded)):
            for j, char in enumerate(e):
                rows.append(i), columns.append(len(p) - 1 + j), targets.append(char)
        rows = torch.tensor(rows, device=self.device)
        picked = logits[rows, torch.tensor(columns, device=self.device)].float().log_softmax(-1)
        picked = picked[torch.arange(len(targets), device=self.device), torch.tensor(targets, device=self.device)]
        return torch.zeros(len(texts), device=self.device).index_add_(0, rows, picked).tolist()

