"""Direct double-pinyin beam search using only the conditional character model."""
from collections import defaultdict

import torch
import torch.nn.functional as F

from composition import composition_prompt
from decoding_model import readings


def cached_next(model, tokens, cache=None, parents=None):
    """Prefill or append one token, optionally reordering KV rows after beam selection."""
    past = 0 if cache is None else cache[0][0].shape[2]
    if past and tokens.shape[1] != 1:
        raise ValueError("incremental decoding appends exactly one token")
    positions = torch.arange(past, past + tokens.shape[1], device=tokens.device)
    hidden = model.drop(model.tok(tokens) + model.pos(positions))
    updated = []
    for i, block in enumerate(model.blocks):
        batch, length, width = hidden.shape
        shape = (batch, length, block.n_head, width // block.n_head)
        q, k, v = (part.view(shape).transpose(1, 2)
                   for part in block.qkv(block.ln1(hidden)).split(width, dim=2))
        if cache is not None:
            old_k, old_v = cache[i]
            if parents is not None:
                old_k, old_v = old_k[parents], old_v[parents]
            k, v = torch.cat((old_k, k), dim=2), torch.cat((old_v, v), dim=2)
        updated.append((k, v))
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=not past)
        hidden = hidden + block.drop(block.proj(attention.transpose(1, 2).reshape(batch, length, width)))
        hidden = hidden + block.drop(block.out(F.gelu(block.fc(block.ln2(hidden)))))
    return model.head(model.ln_f(hidden[:, -1])).float().log_softmax(-1), updated


class ConditionalDecoder:
    def __init__(self, model, vocab, legal, beam=16):
        if beam < 1:
            raise ValueError("beam must be positive")
        self.model, self.vocab, self.beam = model.eval(), vocab, beam
        self.index = {c: i for i, c in enumerate(vocab)}
        self.device = next(model.parameters()).device
        self.legal = {code: torch.tensor(sorted({self.index[c] for c in chars if c in self.index}),
                                         device=self.device, dtype=torch.long)
                      for code, chars in legal.items()}

    @classmethod
    def from_database(cls, model, vocab, database, beam=16):
        legal = defaultdict(set)
        for char, codes in readings(database).items():
            for code in codes:
                legal[code].add(char)
        return cls(model, vocab, legal, beam)

    @torch.inference_mode()
    def decode(self, context, code, committed=None):
        prefix = [2] + [self.index.get(c, 1) for c in composition_prompt(context, code, self.model.cfg.context, committed)]
        syllables = [code[i:i+2] for i in range(0, len(code), 2)]
        if not syllables:
            return []
        if any(s not in self.legal or not len(self.legal[s]) for s in syllables):
            return []
        probabilities, cache = cached_next(self.model, torch.tensor([prefix], device=self.device))
        scores = torch.zeros(1, device=self.device)
        sequences = torch.empty((1, 0), dtype=torch.long, device=self.device)
        for step, syllable in enumerate(syllables):
            allowed = self.legal[syllable]
            # Normalize over the entire vocabulary BEFORE applying hard keys.
            # Renormalizing each legal subset would change comparisons between parents.
            expanded = scores[:, None] + probabilities[:, allowed]
            scores, selected = expanded.flatten().topk(min(self.beam, expanded.numel()))
            parents = selected // len(allowed)
            next_tokens = allowed[selected % len(allowed)]
            sequences = torch.cat((sequences[parents], next_tokens[:, None]), dim=1)
            if step + 1 < len(syllables):
                probabilities, cache = cached_next(self.model, next_tokens[:, None], cache, parents)
        return [("".join(self.vocab[token] for token in sequence), score)
                for sequence, score in zip(sequences.tolist(), scores.tolist())]
