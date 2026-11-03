"""Dictionary readings and scoring by the syllable transducer and the LM."""
import collections
import copy
import sqlite3

import torch
from transformers import DynamicCache

# Left-context characters the LM reads; the transducer reads its own 64.
LM_CONTEXT = 1024


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
    """The decoder's model calls, separate so tests can replace the models."""

    def __init__(self, evaluator, rescorer, beam=16, per_hypothesis=24):
        self.evaluator, self.rescorer = evaluator, rescorer
        self.beam, self.per_hypothesis = beam, per_hypothesis
        self.weight = rescorer.weight if rescorer else 1.0
        self.device = evaluator.device if evaluator else "cpu"

    def prompts(self, context, code, tones=()):
        """The transducer's prompt and LM context ids."""
        return self.model_prompt(context, code, tones), self.lm_ids(context)

    def model_prompt(self, context, code, tones=()):
        return self.evaluator.prompt(context, code, tones)

    def lm_ids(self, context):
        return self.rescorer.tokenizer(context[-LM_CONTEXT:] or "\n")["input_ids"]

    def lm_run(self, context, sequences):
        """Final hidden states over [last context token] + each sequence.

        All but the last context token run once per context and are shared:
        a 1,024-character context then costs one prefill, not one per row."""
        model, head = self.rescorer.model, context[:-1]
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
        ids = torch.full((len(rows), max(map(len, rows))), self.rescorer.pad)
        for i, row in enumerate(rows):
            # Right padding follows every scored position.
            ids[i, :len(row)] = torch.tensor(row)
        return model.model(input_ids=ids.to(self.device), past_key_values=past, use_cache=past is not None).last_hidden_state

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
        hidden = self.lm_run(context, sequences)
        last = torch.tensor([len(s) for s in sequences], device=self.device)
        return self.rescorer.model.lm_head(hidden[torch.arange(len(sequences), device=self.device), last]).float().log_softmax(-1)

    @torch.inference_mode()
    def lm_tails(self, context, prefixes, tails):
        """Sum log p(tail | context + prefix) for each pair."""
        hidden = self.lm_run(context, [list(p) + list(t) for p, t in zip(prefixes, tails)])
        rows, columns, targets = [], [], []
        for i, (p, t) in enumerate(zip(prefixes, tails)):
            for j, token in enumerate(t):
                rows.append(i), columns.append(len(p) + j), targets.append(token)
        rows = torch.tensor(rows, device=self.device)
        picked = self.rescorer.model.lm_head(hidden[rows, torch.tensor(columns, device=self.device)]).float().log_softmax(-1)
        picked = picked[torch.arange(len(targets), device=self.device), torch.tensor(targets, device=self.device)]
        return torch.zeros(len(tails), device=self.device).index_add_(0, rows, picked).tolist()

    def model_next(self, prompt, texts):
        """log p(next char | prompt + text): [len(texts), V]."""
        return self.model_next_each([prompt] * len(texts), texts)

    def model_score(self, prompt, texts):
        """Sum log p(text | prompt) for each text."""
        return self.model_score_each([prompt] * len(texts), texts)

    @torch.inference_mode()
    def model_next_each(self, prompts, texts):
        """model_next with its own prompt for each text."""
        return self.evaluator.joint(prompts, texts, [(i, len(t)) for i, t in enumerate(texts)])

    @torch.inference_mode()
    def model_score_each(self, prompts, texts):
        """model_score with its own prompt for each text."""
        slots = [(i, j) for i, t in enumerate(texts) for j in range(len(t))]
        if not slots:
            return [0.0] * len(texts)
        values = self.evaluator.joint(prompts, texts, slots)
        targets = torch.tensor([self.evaluator.index.get(texts[i][j], 1) for i, j in slots], device=self.device)
        picked = values[torch.arange(len(slots), device=self.device), targets]
        rows = torch.tensor([i for i, _ in slots], device=self.device)
        return torch.zeros(len(texts), device=self.device).index_add_(0, rows, picked).tolist()
