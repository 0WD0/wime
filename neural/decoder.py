"""Fused decoding: the general LM spells the keys, fused with the syllable model.

No dictionary lattice. At character position p a unit (an LM token, or the
byte tokens of a character the LM vocabulary lacks) may follow when its
characters' readings spell the next keys. The beam is character-synchronous,
so hypotheses compared at one frontier cover the same characters, each
ranked by  m + w * q + bonus:  m the syllable transducer's log
probability of the text given every syllable typed so far, q the LM's, and the
bonus learned readings earn per character they cover.

A Session revisits the composition whenever its complete syllables change:
every frontier is pruned with all currently typed complete syllables in the
transducer's prompt, so later keys can recover an earlier reading.
LM distributions are memoized by token prefix and reused across these
searches. A trailing half syllable is decoded provisionally: its last
character's reading must start with the typed key. Pinned characters and
typed auxiliary codes are hard constraints at their position; changing one
keeps the frontiers before it when the complete input is unchanged.

Literal text typed inside the composition (English, punctuation) sits
between syllables. The LM's token path includes it, so the LM scores the
sentence as typed. The transducer never saw literals among syllables;
each literal starts a new composition for it, whose context is
the hypothesis text before it plus the literal, as in training.
"""
import collections
import itertools
import math
import sqlite3

import torch

from decoding_model import DecodingModel, aux_match, readings, tones

BEAM, PER_HYPOTHESIS = 16, 24
MEMO = 4096  # memoized LM rows, ~0.5 MB each
# Only this many expansions, best by their incremental estimate, are scored
# again under the current keys; that rescoring dominates a syllable's cost.
RESCORED = 4 * BEAM
# A byte token's probability is shared by every character it starts, so a
# character the LM lacks competes only by the transducer's score.
RARE, RARE_FLOOR = 2, math.log(0.02)


def chinese(text):
    return bool(text) and all("㐀" <= c <= "鿿" for c in text)


class Units:
    """Keys -> allowed units [(text, ids)]; odd keys index half syllables."""

    def __init__(self, tokenizer, readings, tones=None, combinations=16):
        self.readings, self.tones = readings, tones or {}
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

    @classmethod
    def from_database(cls, tokenizer, database):
        return cls(tokenizer, readings(database), tones(database))

    def get(self, keys):
        return (self.partial if len(keys) % 2 else self.index).get(keys, ())

    def aux_allows(self, char, keys, typed):
        """Whether TYPED auxiliary keys allow CHAR read as KEYS (one key: any reading starting with it)."""
        return any(aux_match(groups, typed) for code, groups in self.readings.get(char, {}).items()
                   if code == keys or len(keys) == 1 and code.startswith(keys))

    def tone_allows(self, char, keys, tone):
        """Whether CHAR read as KEYS has typed TONE; the neutral tone keeps every reading.

        Character readings, not word ones, decide: a person types the
        dictionary tone of 一 or 不, while words record their sandhi."""
        return tone == 5 or any(str(tone) in digits for code, digits in self.tones.get(char, {}).items()
                                if code == keys or len(keys) == 1 and code.startswith(keys))


Hypothesis = collections.namedtuple("Hypothesis", "text ids q m")


class Decoder(DecodingModel):
    """Model-facing methods are separate so tests can replace the models."""

    def __init__(self, evaluator, rescorer, units, beam=BEAM, per_hypothesis=PER_HYPOTHESIS):
        super().__init__(evaluator, rescorer, beam, per_hypothesis)
        self.units, self.session = units, None

    # --- search -----------------------------------------------------------

    def decode(self, context, code, pins=(), learned=(), weight=0.0, aux=(), literals=(), tones=(), candidates=()):
        """Best complete readings of CODE as [(text, per-character score)].

        PINS fix characters by position and AUX (position, keys) restricts a
        position to characters whose auxiliary codes start with the keys;
        LEARNED (start, text) readings earn WEIGHT per covered character.
        LITERALS (position, text) follow the first POSITION characters; the
        returned texts omit them. TONES (position, tone) restrict a position
        to characters with that tone and reach the model's prompts. An odd
        CODE ends in half a syllable. CANDIDATES, the lattice's complete
        readings, join the ranking when the beam missed them.
        """
        key = (context, tuple(learned), weight)
        constraints = {p: ("pin", c) for p, c in pins}
        for p, typed in aux:
            constraints[p] = constraints.get(p, ()) + ("aux", typed)
        for p, tone in tones:
            constraints[p] = constraints.get(p, ()) + ("tone", tone)
        texts = dict(literals)
        literals = {p: self.literal_ids(text) for p, text in texts.items()}
        session = self.session
        if session is None or session.key != key:
            session = self.session = Session(self, key, context, constraints, learned, weight, literals)
        elif session.constraints != constraints or session.literals != literals:
            # A constraint at p changes frontiers after it; a literal at p
            # changes frontier p itself. Earlier frontiers stay valid.
            cut = min([p + 1 for p in set(constraints) | set(session.constraints)
                       if constraints.get(p) != session.constraints.get(p)]
                      + [p for p in set(literals) | set(session.literals)
                         if literals.get(p) != session.literals.get(p)])
            session.kept = session.kept[:max(cut, 1)]
            session.constraints, session.literals = constraints, literals
        session.texts = texts
        result = session.decode(code)
        return session.merge(result, code, candidates) if candidates and not literals and len(code) % 2 == 0 else result


class Session:
    def __init__(self, decoder, key, context, constraints, learned, weight, literals):
        self.decoder, self.key = decoder, key
        self.context = context
        self.constraints, self.learned, self.weight = constraints, learned, weight
        self.literals, self.texts, self.segment_prompts = literals, {}, {}
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
                keys = code[2 * (start + i):2 * (start + i) + 2]
                if (char != value if kind == "pin" else not self.decoder.units.aux_allows(char, keys, value)
                        if kind == "aux" else not self.decoder.units.tone_allows(char, keys, value)):
                    return False
        return True

    def tones(self, start, stop):
        """Typed tones of positions START..STOP, relative to START."""
        return tuple((p - start, rule[rule.index("tone") + 1]) for p, rule in sorted(self.constraints.items())
                     if start <= p < stop and "tone" in rule[::2])

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

    def segment(self, position):
        """Start of the transducer composition holding character POSITION."""
        return max([0] + [p for p in self.literals if p <= position])

    def spans(self, complete):
        bounds = [0] + sorted(p for p in self.literals if 0 < p < complete) + [complete]
        return list(zip(bounds, bounds[1:]))

    def segment_prompts_for(self, start, texts):
        """Transducer prompts for the composition from START after a literal,
        one per text: its characters before START, with the literals, extend
        the context. None when any exceeds the window: the LM alone ranks."""
        decoder, syllables = self.decoder, len(self.code) // 2
        stop = min([p for p in self.literals if p > start] + [syllables])
        code = self.code[2 * start:2 * stop]
        characters = stop - start + (self.half and stop == syllables)
        result = []
        for text in texts:
            tones = self.tones(start, stop)
            key = (text[:start], code, characters, tones)
            if key not in self.segment_prompts:
                surface = "".join(self.texts.get(i, "") + c for i, c in enumerate(text[:start]))
                try:
                    prompt = decoder.model_prompt(self.context + surface + self.texts[start], code, tones)
                except ValueError:
                    prompt = None
                self.segment_prompts[key] = (prompt if prompt is not None
                                             and len(prompt) + characters <= decoder.window() else None)
            result.append(self.segment_prompts[key])
        return None if any(p is None for p in result) else result

    def expand(self, code, end, prompt):
        """Units ending at position END (half a syllable when CODE is odd),
        from every kept frontier they can start at, before pruning."""
        decoder, w = self.decoder, self.decoder.weight
        hypotheses, choices, owners, forced = [], [], [], []
        # A unit never spans a literal.
        segment = self.segment(end - 1)
        for start in range(max(segment, end - decoder.units.longest), end):
            units = decoder.units.get(code[2 * start:])
            if self.constraints:
                units = [u for u in units if self.allowed(code, start, u[0])]
            if units and self.kept[start]:
                first = len(hypotheses)
                hypotheses += self.kept[start]
                for unit in units:
                    choices.append(unit), owners.append((first, len(hypotheses)))
                    forced.append(all("pin" in self.constraints.get(start+i, ())[::2]
                                      for i in range(len(unit[0]))))
        if not choices:
            return []
        q = self.lm_next([h.ids for h in hypotheses])[:, torch.tensor([u[1][0] for u in choices], device=decoder.device)]
        half = len(code) % 2
        # The transducer reads complete syllables only; a half one has no score.
        prompts = (None if half else [prompt] * len(hypotheses) if segment == 0 or prompt is None
                   else self.segment_prompts_for(segment, [h.text for h in hypotheses]))
        columns = torch.tensor(decoder.char_ids([u[0][0] for u in choices]), device=decoder.device).flatten()
        m = (torch.zeros_like(q) if prompts is None or prompts[0] is None
             else decoder.model_next(prompt, [h.text for h in hypotheses])[:, columns] if segment == 0
             else decoder.model_next_each(prompts, [h.text[segment:] for h in hypotheses])[:, columns])
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
        # A confirmed character is an input constraint. Its byte fallback
        # must survive even when the model assigns it a tiny probability.
        for row, legal in zip(picks, valid.tolist()):
            row.extend(c for c, fixed in enumerate(forced) if fixed and legal[c] and c not in row)
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
        scores = [0.0] * len(new)
        if prompt is not None:
            for start, stop in self.spans(complete):
                texts = [e[0] for e in new]
                prompts = None if start == 0 else self.segment_prompts_for(start, texts)
                if start == 0 or prompts is not None:
                    values = (self.decoder.model_score(prompt, [t[:stop] for t in texts]) if start == 0
                              else self.decoder.model_score_each(prompts, [t[start:stop] for t in texts]))
                    scores = [a + b for a, b in zip(scores, values)]
        best = {}
        for (text, ids, q, _), m in zip(new, scores):
            h = Hypothesis(text, ids, q, m)
            if text not in best or self.rank(h) > self.rank(best[text]):
                best[text] = h
        return sorted(best.values(), key=self.rank, reverse=True)[:self.decoder.beam]

    def attach(self, kept, end):
        """Append the literal typed after END characters to every LM path."""
        ids = self.literals.get(end)
        if not ids or not kept:
            return kept
        values = self.lm_tails([h.ids for h in kept], [ids] * len(kept))
        return sorted((h._replace(ids=h.ids + ids, q=h.q + v) for h, v in zip(kept, values)),
                      key=self.rank, reverse=True)

    def prompt(self, code, characters):
        """The transducer's prompt for CODE and up to CHARACTERS
        characters, or None past its window: then the LM alone ranks."""
        decoder = self.decoder
        try:
            prompt, self.lm_context = decoder.prompts(self.context, code, self.tones(0, len(code) // 2))
        except ValueError:
            prompt = None
        if self.lm_context is None:
            self.lm_context = decoder.lm_ids(self.context)
        return prompt if prompt is not None and len(prompt) + characters <= decoder.window() else None

    def merge(self, result, code, candidates):
        """RESULT with the CANDIDATES it lacks, ranked alike: a dictionary
        word reaches the LM whole even when its first character was pruned."""
        n = len(code) // 2
        seen = {text for text, _ in result}
        texts = [t for t in dict.fromkeys(candidates) if len(t) == n and t not in seen and self.allowed(code, 0, t)]
        prompt = self.prompt(code, n) if texts else None
        if prompt is None:
            return result
        ids = [self.decoder.literal_ids(t) for t in texts]
        q = self.lm_tails([()] * len(texts), ids)
        m = self.decoder.model_score(prompt, texts)
        extra = [(t, self.rank(Hypothesis(t, i, a, b)) / n) for t, i, a, b in zip(texts, ids, q, m)]
        return sorted(result + extra, key=lambda e: e[1], reverse=True)

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
            self.memo, self.tails, self.segment_prompts = {}, {}, {}
        self.half = bool(half)
        # The first composition ends at the first literal inside the input.
        first = min([p for p in self.literals if p < syllables] + [syllables])
        prompt = self.prompt(complete[:2 * first], first)
        for end in range(len(self.kept), syllables + 1):
            self.kept.append(self.attach(self.finish(self.expand(code[:2 * end], end, prompt), prompt, end), end))
        if half and first == syllables:
            prompt = self.prompt(complete, syllables + 1)
            final = self.finish(self.expand(code, syllables + 1, prompt), prompt, syllables)
        else:
            final = self.kept[syllables]
        n = syllables + half
        return [(h.text, self.rank(h) / n) for h in final if len(h.text) == n]
