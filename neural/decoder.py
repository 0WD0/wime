"""Legacy fused decoding: the general LM spells the keys, fused with the 25M.

No dictionary lattice. At character position p a unit (an LM token, or the
byte tokens of a character the LM vocabulary lacks) may follow when its
characters' readings spell the next keys. The beam is character-synchronous,
so hypotheses compared at one frontier cover the same characters, each
ranked by  m + w * q + bonus:  m the pinyin-conditioned model's log
probability of the text given every key typed so far, q the LM's, and the
bonus learned readings earn per character they cover.

A Session revisits the composition whenever its complete syllables change:
every frontier is pruned with all currently typed complete syllables in the
conditional model's prompt, so later keys can recover an earlier reading.
LM distributions are memoized by token prefix and reused across these
searches. A trailing half syllable is decoded provisionally: its last
character's reading must start with the typed key. Pinned characters and
typed auxiliary codes are hard constraints at their position; changing one
keeps the frontiers before it when the complete input is unchanged.
"""
import collections
import itertools
import math
import sqlite3

import torch

from decoding_model import DecodingModel, aux_match, readings

BEAM, PER_HYPOTHESIS = 16, 24
MEMO = 4096  # memoized LM rows, ~0.5 MB each
# Only this many expansions, best by their incremental estimate, are scored
# again under the current keys; that rescoring dominates a syllable's cost.
RESCORED = 4 * BEAM
# A byte token's probability is shared by every character it starts, so a
# character the LM lacks competes only by the conditional model's score.
RARE, RARE_FLOOR = 2, math.log(0.02)


def chinese(text):
    return bool(text) and all("㐀" <= c <= "鿿" for c in text)


class Units:
    """Keys -> allowed units [(text, ids)]; odd keys index half syllables."""

    def __init__(self, tokenizer, readings, combinations=16):
        self.readings = readings
        self.index = collections.defaultdict(list)
        single, self.longest = set(), 1
        for i in range(len(tokenizer)):
            text = tokenizer.decode([i])
            if not chinese(text) or any(c not in readings for c in text):
                continue
            single.update(text if len(text) == 1 else ())
            self.longest = max(self.longest, len(text))
            for codes in itertools.islice(itertools.product(*(readings[c] for c in text)), combinations):
                self.index["".join(codes)].append((text, (i,)))
        for char, codes in readings.items():
            if char not in single:
                # This character continues the context; an inserted BOS
                # would change its probability and every following token.
                ids = tuple(tokenizer(char, add_special_tokens=False)["input_ids"])
                for code in codes:
                    self.index[code].append((char, ids))
        self.partial = collections.defaultdict(list)
        for code, units in list(self.index.items()):
            self.partial[code[:-1]].extend(units)

    def get(self, keys):
        return (self.partial if len(keys) % 2 else self.index).get(keys, ())

    def aux_allows(self, char, keys, typed):
        """Whether TYPED auxiliary keys allow CHAR read as KEYS (one key: any reading starting with it)."""
        return any(aux_match(groups, typed) for code, groups in self.readings.get(char, {}).items()
                   if code == keys or len(keys) == 1 and code.startswith(keys))


Hypothesis = collections.namedtuple("Hypothesis", "text ids q m")


class Decoder(DecodingModel):
    """Model-facing methods are separate so tests can replace the models."""

    def __init__(self, evaluator, rescorer, units, beam=BEAM, per_hypothesis=PER_HYPOTHESIS):
        super().__init__(evaluator, rescorer, beam, per_hypothesis)
        self.units, self.session = units, None

    # --- search -----------------------------------------------------------

    def decode(self, context, code, anchor=0, known=None, pins=(), learned=(), weight=0.0, aux=()):
        """Best complete readings of CODE as [(text, per-character score)].

        PINS fix characters by position and AUX (position, keys) restricts a
        position to characters whose auxiliary codes start with the keys;
        LEARNED (start, text) readings earn WEIGHT per covered character. An
        odd CODE ends in half a syllable.
        """
        key = (context, anchor, repr(known), tuple(learned), weight)
        constraints = {p: ("pin", c) for p, c in pins}
        for p, typed in aux:
            constraints[p] = constraints.get(p, ()) + ("aux", typed)
        session = self.session
        if session is None or session.key != key:
            session = self.session = Session(self, key, context, anchor, known, constraints, learned, weight)
        elif session.constraints != constraints:
            # Frontiers before the first changed position cover no constrained one.
            first = min(p for p in set(constraints) | set(session.constraints)
                        if constraints.get(p) != session.constraints.get(p))
            session.kept = session.kept[:first + 1]
            session.constraints = constraints
        return session.decode(code)


class Session:
    def __init__(self, decoder, key, context, anchor, known, constraints, learned, weight):
        self.decoder, self.key = decoder, key
        self.context, self.anchor, self.known = context, anchor, known
        self.constraints, self.learned, self.weight = constraints, learned, weight
        self.code, self.kept = "", [[Hypothesis("", (), 0.0, 0.0)]]
        self.memo, self.tails = {}, {}
        self.lm_context = None

    def bonus(self, text):
        covered = set()
        for start, reading in self.learned:
            end = min(len(text), start + len(reading))
            if start < end and text[start:end] == reading[:end - start]:
                covered.update(range(start, end))
        return self.weight * len(covered)

    def rank(self, h):
        return h.m + self.decoder.weight * h.q + self.bonus(h.text)

    def allowed(self, code, start, text):
        """Whether TEXT read from START keeps every pin and typed auxiliary code."""
        for i, char in enumerate(text):
            rule = self.constraints.get(start + i, ())
            for kind, value in zip(rule[::2], rule[1::2]):
                if (char != value if kind == "pin" else not self.decoder.units.aux_allows(
                        char, code[2 * (start + i):2 * (start + i) + 2], value)):
                    return False
        return True

    def lm_next(self, sequences):
        decoder = self.decoder
        missing = [s for s in dict.fromkeys(sequences) if s not in self.memo]
        if missing:
            for s, row in zip(missing, decoder.lm_next(self.lm_context, missing).to(torch.bfloat16)):
                self.memo[s] = row
        return torch.stack([self.memo[s] for s in sequences]).float()

    def lm_tails(self, prefixes, tails):
        keys = [(tuple(p), tuple(t)) for p, t in zip(prefixes, tails)]
        missing = [k for k in dict.fromkeys(keys) if k not in self.tails]
        if missing:
            self.tails.update(zip(missing, self.decoder.lm_tails(
                self.lm_context, [k[0] for k in missing], [k[1] for k in missing])))
        return [self.tails[k] for k in keys]

    def expand(self, code, end, prompt):
        """Units ending at position END (half a syllable when CODE is odd),
        from every kept frontier they can start at, before pruning."""
        decoder, w = self.decoder, self.decoder.weight
        hypotheses, choices, owners = [], [], []
        for start in range(max(0, end - decoder.units.longest), end):
            units = decoder.units.get(code[2 * start:])
            if self.constraints:
                units = [u for u in units if self.allowed(code, start, u[0])]
            if units and self.kept[start]:
                first = len(hypotheses)
                hypotheses += self.kept[start]
                for unit in units:
                    choices.append(unit), owners.append((first, len(hypotheses)))
        if not choices:
            return []
        q = self.lm_next([h.ids for h in hypotheses])[:, torch.tensor([u[1][0] for u in choices], device=decoder.device)]
        half = len(code) % 2
        # The conditional model has seen only complete syllables of a half one.
        m = (decoder.model_next(prompt, [h.text for h in hypotheses])[
            :, torch.tensor(decoder.char_ids([u[0][0] for u in choices]), device=decoder.device).flatten()]
             if not half and prompt is not None else torch.zeros_like(q))
        valid = torch.zeros(len(hypotheses), len(choices), dtype=torch.bool)
        for c, (a, b) in enumerate(owners):
            valid[a:b, c] = True
        valid = valid.to(q.device)
        rare = torch.tensor([len(u[1]) > 1 for u in choices], device=q.device)
        estimate = (m + w * q).masked_fill(~valid | rare, -math.inf)
        top = estimate.topk(min(decoder.per_hypothesis, len(choices)), dim=1)
        picks = [[c for c, v in zip(row, values) if v > -math.inf]
                 for row, values in zip(top.indices.tolist(), top.values.tolist())]
        if not half and prompt is not None and bool(rare.any()):
            extra = m.masked_fill(~valid | ~rare | (m < RARE_FLOOR), -math.inf).topk(min(RARE, len(choices)), dim=1)
            for row, more, values in zip(picks, extra.indices.tolist(), extra.values.tolist()):
                row.extend(c for c, v in zip(more, values) if v > -math.inf)
        q, m = q.tolist(), m.tolist()
        new, pending = [], []
        for i, h in enumerate(hypotheses):
            for c in picks[i]:
                text, ids = choices[c]
                if len(ids) > 1:
                    pending.append((len(new), h, ids))
                new.append([h.text + text, h.ids + ids, h.q + q[i][c], h.m + m[i][c]])
        if pending:
            for (i, h, _), value in zip(pending, self.lm_tails([h.ids for _, h, _ in pending],
                                                               [ids for _, _, ids in pending])):
                new[i][2] = h.q + value
        return new

    def finish(self, new, prompt, complete):
        """Score every text's first COMPLETE characters under the current
        keys, then keep the best text of each surface form."""
        if not new:
            return []
        w = self.decoder.weight
        new = sorted(new, key=lambda e: e[3] + w * e[2] + self.bonus(e[0]), reverse=True)[:RESCORED]
        scores = ([0.0] * len(new) if prompt is None
                  else self.decoder.model_score(prompt, [e[0][:complete] for e in new]))
        best = {}
        for (text, ids, q, _), m in zip(new, scores):
            h = Hypothesis(text, ids, q, m)
            if text not in best or self.rank(h) > self.rank(best[text]):
                best[text] = h
        return sorted(best.values(), key=self.rank, reverse=True)[:self.decoder.beam]

    def prompt(self, code, characters):
        """The conditional model's prompt for CODE and up to CHARACTERS
        characters, or None past its window: then the LM alone ranks."""
        decoder = self.decoder
        try:
            prompt, self.lm_context = decoder.prompts(self.context, code, self.anchor, self.known)
        except ValueError:
            prompt = None
        if self.lm_context is None:
            self.lm_context = decoder.lm_ids(self.context)
        return prompt if prompt is not None and len(prompt) + characters <= decoder.window() else None

    def decode(self, code):
        syllables, half = len(code) // 2, len(code) % 2
        if not syllables:
            return []
        complete = code[:2 * syllables]
        if complete != self.code:
            # Earlier pruning depended on the old complete input. Revisit it
            # even on append/delete so newly informative suffixes can change
            # the first character, including readings no longer in the beam.
            self.kept, self.code = self.kept[:1], complete
        if len(self.memo) > MEMO:
            # General-LM rows depend only on the unchanged left context and
            # output token prefix, so key edits do not invalidate them.
            self.memo, self.tails = {}, {}
        prompt = self.prompt(complete, syllables)
        for end in range(len(self.kept), syllables + 1):
            self.kept.append(self.finish(self.expand(code[:2 * end], end, prompt), prompt, end))
        if half:
            prompt = self.prompt(complete, syllables + 1)
            final = self.finish(self.expand(code, syllables + 1, prompt), prompt, syllables)
        else:
            final = self.kept[syllables]
        n = syllables + half
        return [(h.text, self.rank(h) / n) for h in final if len(h.text) == n]
