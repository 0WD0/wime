"""Experimental character proposals independent of a general LM tokenizer.

The dictionary supplies hard pronunciation/pin/auxiliary constraints. The
conditional model supplies only beam scores, with every complete syllable in
its prompt before the first expansion. No token-based rare-character gate is
used. These proposals are not connected to the production worker; partial
syllables and learned-reading bonuses are intentionally unsupported here.
"""
import collections
import warnings

import torch

from conditional import KEYS
from decoder import Decoder, aux_match, readings
from inference import anchored_prompt


class CharacterProposals(Decoder):
    """Bounded character beam with no general LM or tokenizer dependency.

    Characters absent from the conditional vocabulary cannot be distinguished
    by this model. Exclude them explicitly, warn once, and expose them through
    ``unsupported_by_code`` and ``last_unsupported`` for coverage audits rather
    than scoring every such character as the same UNK token.
    """

    def __init__(self, evaluator, readings_map, beam=64, per_hypothesis=32):
        if beam < 1 or per_hypothesis < 1:
            raise ValueError("beam and per_hypothesis must be positive")
        # Reuse the small model's batched next-character scoring only.
        super().__init__(evaluator, None, None, beam, per_hypothesis)
        self.readings = readings_map
        supported, unsupported = collections.defaultdict(list), collections.defaultdict(list)
        for char, codes in sorted(readings_map.items()):
            if len(char) != 1:
                continue
            destination = supported if evaluator.index.get(char, 1) != 1 else unsupported
            for code in codes:
                if len(code) == 2 and all(key in KEYS for key in code):
                    destination[code].append(char)
        self.by_code = dict(supported)
        self.unsupported_by_code = dict(unsupported)
        self.last_unsupported = {}
        if unsupported:
            count = len({c for chars in unsupported.values() for c in chars})
            warnings.warn(f"Character proposals exclude {count} dictionary characters absent "
                          "from the conditional vocabulary; inspect unsupported_by_code.",
                          RuntimeWarning, stacklevel=2)

    @classmethod
    def from_database(cls, evaluator, database, **kwargs):
        return cls(evaluator, readings(database), **kwargs)

    @torch.inference_mode()
    def decode(self, context, code, anchor=2, known=None, pins=(), aux=()):
        """Return ``[(text, mean conditional log probability)]`` for full keys.

        ``pins`` and ``aux`` use the production decoder's (position, value)
        convention. Scores retain the original vocabulary normalization; they
        are not renormalized over the characters allowed by the typed keys.
        """
        if len(code) % 2 or any(key not in KEYS for key in code):
            raise ValueError("character proposals require complete double-pinyin syllables")
        syllables = len(code) // 2
        pins, aux = dict(pins), dict(aux)
        if any(not isinstance(p, int) or not 0 <= p < syllables for p in (*pins, *aux)):
            raise ValueError("constraint position is outside the composition")
        if any(len(char) != 1 for char in pins.values()):
            raise ValueError("each pin must be one character")
        self.last_unsupported = {p: self.unsupported_by_code[code[2*p:2*p+2]]
                                 for p in range(syllables)
                                 if code[2*p:2*p+2] in self.unsupported_by_code}
        if not syllables:
            return []
        prefix = anchored_prompt(context, code, self.window(), anchor, known)
        prompt = [2] + [self.evaluator.index.get(char, 1) for char in prefix]
        if len(prompt) + syllables > self.window():
            raise ValueError("composition exceeds conditional model window")
        texts, scores = [""], torch.zeros(1, device=self.device)
        for position in range(syllables):
            keys = code[2*position:2*position+2]
            chars = [char for char in self.by_code.get(keys, ())
                     if (position not in pins or char == pins[position])
                     and (position not in aux or aux_match(self.readings[char][keys], aux[position]))]
            if not chars:
                return []
            ids = torch.tensor([self.evaluator.index[char] for char in chars], device=self.device)
            probabilities = self.model_next(prompt, texts).index_select(1, ids)
            values, choices = probabilities.topk(min(self.per_hypothesis, len(chars)), dim=1)
            accumulated = values + scores[:, None]
            scores, selected = accumulated.flatten().topk(min(self.beam, accumulated.numel()))
            width = choices.shape[1]
            owners = selected.div(width, rounding_mode="floor")
            chosen = choices[owners, selected.remainder(width)]
            texts = [texts[parent] + chars[char]
                     for parent, char in zip(owners.tolist(), chosen.tolist())]
        return list(zip(texts, (scores / syllables).tolist()))
