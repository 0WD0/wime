"""Japanese: the general LM spells typed kana, fused with a kana-conditioned model.

Units are LM tokens (plus common dictionary words the LM splits, rare kanji,
numbers and Latin letters) with every reading they may have, from a table
build_table() derives from JMdict and KANJIDIC2. A Session decodes one
composition a kana at a time, character-synchronous on kana: hypotheses
compared at frontier p have consumed the same p kana. Each new kana expands
into the newest position only; the best expansions are then scored by the
kana model given every typed kana, and ranked by  m + w * q  (m the kana
model's log probability, q the LM's). Keys still pending in the romaji
front end constrain a provisional last kana to the kana they can start.

With explicit spans (the user held Shift), every kana is constrained: 'c'
kana must be written in something other than hiragana (kanji, katakana),
'k' kana stay the hiragana typed. A unit's text must match its kana's runs.
"""
import collections
import functools
import itertools
import json
import math
import re
import sqlite3
from pathlib import Path

import torch

BEAM, PER_HYPOTHESIS, RESCORED = 16, 24, 64
MEMO = 4096  # memoized LM rows, ~0.5 MB each
MAXREAD = 14
FORMAT = "wime-ja-kana-v1"
INPUT, OUTPUT, EOS = "", "", ""
# Without Japanese before point the LM drifts into Chinese; this line stands
# in for the Japanese text an editor buffer would usually hold.
NEUTRAL = "これは日本語の文章です。\n"


HIRAGANA = "\u3041-\u3096\u309d\u309e"


@functools.lru_cache(maxsize=65536)
def span_pattern(kana, constraints):
    """Text a unit read as KANA may have under CONSTRAINTS ('c'/'k' per kana)."""
    parts = []
    for kind, run in itertools.groupby(zip(constraints, kana), key=lambda x: x[0]):
        run = "".join(c for _, c in run)
        parts.append(f"[^{HIRAGANA}]+" if kind == "c" else re.escape(run))
    return re.compile("".join(parts))


def hira(text):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def is_kana(c):
    return "ぁ" <= c <= "ゖ" or "ァ" <= c <= "ヺ" or c in "ーゝゞヽヾ"


def is_kanji(c):
    return "㐀" <= c <= "鿿" or c in "々〆ヶ"


# --- reading table ---------------------------------------------------------

VOICED = dict(zip("かきくけこさしすせそたちつてとはひふへほ", "がぎぐげござじずぜぞだぢづでどばびぶべぼ"))
SEMI = dict(zip("はひふへほ", "ぱぴぷぺぽ"))
NUMERALS = {0: ["ぜろ", "れい"], 1: ["いち"], 2: ["に"], 3: ["さん"], 4: ["よん", "し"], 5: ["ご"],
            6: ["ろく"], 7: ["なな", "しち"], 8: ["はち"], 9: ["きゅう", "く"], 10: ["じゅう"]}
LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", ["えー", "びー", "しー", "でぃー", "いー", "えふ", "じー",
               "えいち", "あい", "じぇい", "けい", "える", "えむ", "えぬ", "おー", "ぴー", "きゅー", "あーる",
               "えす", "てぃー", "ゆー", "ぶい", "だぶりゅー", "えっくす", "わい", "ぜっと"]))


def jmdict(path):
    """Kanji writing -> kana readings, and the common writings."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    writings, common = collections.defaultdict(set), set()
    for writing, reading, priority, frequency in db.execute(
            "SELECT kt.c0reading, nt.c0reading, k.priority, e.frequency FROM kanji k "
            "JOIN kanjiText_content kt ON kt.docid = k.docid JOIN kana n ON n.id = k.id "
            "JOIN kanaText_content nt ON nt.docid = n.docid JOIN entries e ON e.id = k.id"):
        writings[writing].add(hira(reading))
        if (priority or 0) > 0 or (frequency or 0) > 0:
            common.add(writing)
    return writings, common


def kanjidic(path):
    """Jouyou/jinmeiyou kanji -> (stem, okurigana) readings; か.の reads か before の."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    result = collections.defaultdict(set)
    for entry, reading in db.execute("SELECT r.entry, t.c0reading FROM reading r "
                                     "JOIN readingText_content t ON t.docid = r.docid "
                                     "JOIN entries e ON e.id = r.entry WHERE e.grade IS NOT NULL"):
        stem, _, okurigana = hira(reading).strip("-").partition(".")
        if stem:
            result[chr(entry)].add((stem, okurigana))
    return result


def char_readings(text, i, chars):
    """TEXT[i] alone: kana itself, else kanji readings with sokuon/rendaku variants.
    A kun stem needs its okurigana next, unless the token ends there."""
    c, rest = text[i], hira(text[i + 1:])
    if is_kana(c):
        return {hira(c)}
    if not is_kanji(c):
        return {c}
    variants = {stem for stem, okurigana in chars.get(c, ()) if not okurigana or not rest or rest.startswith(okurigana[:1])}
    for r in list(variants):
        if r[-1] in "くきちつ" and i + 1 < len(text):
            variants.add(r[:-1] + "っ")
        if i > 0 and r[0] in VOICED:
            variants.add(VOICED[r[0]] + r[1:])
        if i > 0 and r[0] in SEMI:
            variants.add(SEMI[r[0]] + r[1:])
    return variants


def compose(text, chars, writings, limit=64):
    """Readings of TEXT over every split into dictionary words and characters,
    longest pieces first, so 今日は reads きょうは."""
    prefix = [set() for _ in range(len(text) + 1)]
    prefix[0] = {""}
    for i in range(len(text)):
        for j in range(len(text), i, -1):
            pieces = set(writings.get(text[i:j], ()))
            if j - i == 1:
                pieces |= char_readings(text, i, chars)
            for head in list(prefix[i]):
                for r in pieces:
                    if len(prefix[j]) < limit:
                        prefix[j].add(head + r)
    return prefix[len(text)]


def number_readings(n):
    if n <= 10:
        return NUMERALS[n]
    tens, ones = divmod(n, 10)
    head = ["じゅう"] if tens == 1 else [r + "じゅう" for r in NUMERALS[tens]]
    return [h + r for h in head for r in (NUMERALS[ones] if ones else [""])]


def build_table(tokenizer, jmdict_path, kanjidic_path):
    """{reading: [[text, [token ids]], ...]} for TOKENIZER's vocabulary."""
    writings, common = jmdict(jmdict_path)
    chars = kanjidic(kanjidic_path)
    index, single = collections.defaultdict(list), set()
    for i in range(len(tokenizer)):
        t = tokenizer.decode([i])
        if not t or "�" in t or t.isspace():
            continue
        if any(is_kanji(c) for c in t):
            readings = set(writings.get(t, ())) | compose(t, chars, writings)
        elif any(is_kana(c) for c in t) or len(t) == 1 or not any(c.isascii() for c in t):
            readings = {hira(t)}
        else:
            continue  # ASCII words: typed literally, one character at a time
        if len(t) == 1:
            single.add(t)
        for r in readings:
            if 0 < len(r) <= MAXREAD:
                index[r].append([t, [i]])
    for c, readings in chars.items():
        if c not in single:
            ids = tokenizer(c)["input_ids"]
            for r in {stem for stem, _ in readings}:
                index[r].append([c, ids])
    tokens = {u[0] for units in index.values() for u in units if len(u[1]) == 1}
    for w in common:
        if w not in tokens and len(w) > 1:
            ids = tokenizer(w)["input_ids"]
            if len(ids) > 1:
                for r in writings[w]:
                    if len(r) <= MAXREAD:
                        index[r].append([w, ids])
    for n in range(100):
        for r in number_readings(n):
            index[r].append([str(n), tokenizer(str(n))["input_ids"]])
    for letter, r in LETTERS.items():
        for text in (letter, letter.lower()):
            index[r].append([text, tokenizer(text)["input_ids"]])
    return dict(index)


class Table:
    def __init__(self, path, tokenizer):
        data = json.loads(Path(path).read_text())
        self.index = {r: [(t, tuple(ids)) for t, ids in units] for r, units in data["units"].items()}
        self.tokenizer = tokenizer

    def get(self, reading):
        return self.index.get(reading, ())

    def literal(self, c):
        return [(c, tuple(self.tokenizer(c)["input_ids"]))]


# --- kana model --------------------------------------------------------------

class KanaModel:
    """Character LM over <bos> context <input> kana <output> text <eos>."""

    def __init__(self, path, device="cuda"):
        import sys
        from safetensors import safe_open
        here = Path(__file__).resolve().parent
        if not (here / "model.py").is_file():
            sys.path.insert(0, str(here.parents[2] / "third_party/chinese-ime-lm/neural"))
        from model import CharLM, Config
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata()
            weights = {k: f.get_tensor(k).float() for k in f.keys()}
        if meta.get("conditional") != FORMAT:
            raise ValueError("expected a " + FORMAT + " model")
        self.vocab = json.loads(meta["vocab"])
        cfg = Config(**json.loads(meta["config"]))
        weights["head.weight"] = weights["tok.weight"]
        self.model = CharLM(cfg)
        self.model.load_state_dict(weights)
        self.model.to(device).eval()
        self.index = {c: i for i, c in enumerate(self.vocab)}
        self.window, self.device = cfg.context, device

    def prompt(self, context, kana, longest):
        body = [self.index.get(c, 1) for c in INPUT + kana + OUTPUT]
        room = self.window - 1 - len(body) - longest
        if room < 0:
            raise ValueError("composition exceeds the kana model window")
        return [2] + ([self.index.get(c, 1) for c in context[-min(room, 32):]] if room else []) + body

    @torch.inference_mode()
    def score(self, prompt, texts):
        """Sum log p(text | prompt) for each text."""
        encoded = [[self.index.get(c, 1) for c in t] for t in texts]
        size = len(prompt) + max(map(len, encoded))
        x = torch.zeros(len(texts), size, dtype=torch.long)
        rows, cols, targets = [], [], []
        for i, e in enumerate(encoded):
            x[i, :len(prompt) + len(e)] = torch.tensor(prompt + e)
            rows += [i] * len(e); cols += range(len(prompt) - 1, len(prompt) - 1 + len(e)); targets += e
        logits, _ = self.model(x.to(self.device))
        rows = torch.tensor(rows, device=self.device)
        lp = logits[rows, torch.tensor(cols, device=self.device)].float().log_softmax(-1)
        picked = lp[torch.arange(len(targets), device=self.device), torch.tensor(targets, device=self.device)]
        return torch.zeros(len(texts), device=self.device).index_add_(0, rows, picked).tolist()


# --- decoding ----------------------------------------------------------------

Hypothesis = collections.namedtuple("Hypothesis", "text ids q m")


class Decoder:
    """Model-facing methods are separate so tests can replace the models."""

    def __init__(self, table, rescorer, kana_model, weight=0.6, beam=BEAM, per_hypothesis=PER_HYPOTHESIS):
        self.table, self.rescorer, self.kana_model = table, rescorer, kana_model
        self.weight, self.beam, self.per_hypothesis = weight, beam, per_hypothesis
        self.device = rescorer.device if rescorer else "cpu"
        self.session = None

    def lm_context(self, context):
        context = context[-64:]
        return self.rescorer.tokenizer(context if any(is_kana(c) or is_kanji(c) for c in context)
                                       else NEUTRAL + context)["input_ids"]

    @torch.inference_mode()
    def lm_next(self, context, sequences):
        model = self.rescorer.model
        ids = torch.full((len(sequences), len(context) + max(map(len, sequences))), self.rescorer.pad)
        for i, s in enumerate(sequences):
            ids[i, :len(context) + len(s)] = torch.tensor(context + list(s))
        hidden = model.model(input_ids=ids.to(self.device)).last_hidden_state
        last = torch.tensor([len(context) + len(s) - 1 for s in sequences], device=self.device)
        return model.lm_head(hidden[torch.arange(len(sequences), device=self.device), last]).float().log_softmax(-1)

    @torch.inference_mode()
    def lm_tails(self, context, prefixes, tails):
        model, sequences = self.rescorer.model, [list(p) + list(t) for p, t in zip(prefixes, tails)]
        ids = torch.full((len(sequences), len(context) + max(map(len, sequences))), self.rescorer.pad)
        for i, s in enumerate(sequences):
            ids[i, :len(context) + len(s)] = torch.tensor(context + s)
        hidden = model.model(input_ids=ids.to(self.device)).last_hidden_state
        rows, cols, targets = [], [], []
        for i, (p, t) in enumerate(zip(prefixes, tails)):
            for j, token in enumerate(t):
                rows.append(i), cols.append(len(context) + len(p) - 1 + j), targets.append(token)
        rows = torch.tensor(rows, device=self.device)
        picked = model.lm_head(hidden[rows, torch.tensor(cols, device=self.device)]).float().log_softmax(-1)
        picked = picked[torch.arange(len(targets), device=self.device), torch.tensor(targets, device=self.device)]
        return torch.zeros(len(tails), device=self.device).index_add_(0, rows, picked).tolist()

    def model_score(self, context, kana, texts):
        model = self.kana_model
        return model.score(model.prompt(context, kana, max(map(len, texts))), texts)

    def decode(self, context, kana, pending=(), shift=None, explicit=False, pending_shift=False):
        """Best readings of KANA as [(text, score)]. PENDING lists the kana the
        still-pending keys can start; the last kana is then provisional. With
        EXPLICIT, SHIFT ('1'/'0' per kana, PENDING_SHIFT for the provisional
        one) says which kana to convert and which stay hiragana."""
        if self.session is None or self.session.context != context:
            self.session = Session(self, context)
        constraints = (tuple("c" if f == "1" else "k" for f in (shift or "0" * len(kana)))
                       + ("c" if pending_shift else "k",) if explicit else None)
        try:
            return self.session.decode(kana, tuple(sorted(set(pending))), constraints)
        except ValueError:
            self.session = None
            return []


class Session:
    def __init__(self, decoder, context):
        self.decoder, self.context = decoder, context
        self.kana, self.kept = "", [[Hypothesis("", (), 0.0, 0.0)]]
        self.memo, self.tails = {}, {}
        self.constraints = None
        self.lm = decoder.lm_context(context)

    def rank(self, h):
        return h.m + self.decoder.weight * h.q

    def lm_next(self, sequences):
        missing = [s for s in dict.fromkeys(sequences) if s not in self.memo]
        if missing:
            for s, row in zip(missing, self.decoder.lm_next(self.lm, missing).to(torch.bfloat16)):
                self.memo[s] = row
        return torch.stack([self.memo[s] for s in sequences]).float()

    def lm_tails(self, prefixes, tails):
        keys = [(tuple(p), tuple(t)) for p, t in zip(prefixes, tails)]
        missing = [k for k in dict.fromkeys(keys) if k not in self.tails]
        if missing:
            self.tails.update(zip(missing, self.decoder.lm_tails(self.lm, [k[0] for k in missing],
                                                                 [k[1] for k in missing])))
        return [self.tails[k] for k in keys]

    def expand(self, kana, end):
        """Hypotheses whose last unit ends at kana position END of KANA, pruned."""
        decoder, table = self.decoder, self.decoder.table
        hypotheses, choices, owners = [], [], []
        for start in range(max(0, end - MAXREAD), end):
            if not self.kept[start]:
                continue
            units = list(table.get(kana[start:end]))
            if end - start == 1 and not is_kana(kana[start]) and not any(u[0] == kana[start] for u in units):
                units += table.literal(kana[start])
            if self.constraints:
                pattern = span_pattern(kana[start:end], self.constraints[start:end])
                units = [u for u in units if pattern.fullmatch(u[0])]
            if units:
                first = len(hypotheses)
                hypotheses += self.kept[start]
                for unit in units:
                    choices.append(unit), owners.append((first, len(hypotheses)))
        if not choices:
            return []
        q = self.lm_next([h.ids for h in hypotheses])[:, torch.tensor([u[1][0] for u in choices], device=decoder.device)]
        valid = torch.zeros(len(hypotheses), len(choices), dtype=torch.bool)
        for c, (a, b) in enumerate(owners):
            valid[a:b, c] = True
        top = q.masked_fill(~valid.to(q.device), -math.inf).topk(min(decoder.per_hypothesis, len(choices)), dim=1)
        new, pending = [], []
        for i, (row, values) in enumerate(zip(top.indices.tolist(), top.values.tolist())):
            h = hypotheses[i]
            for c, v in zip(row, values):
                if v == -math.inf:
                    continue
                text, ids = choices[c]
                if len(ids) > 1:
                    pending.append((len(new), h, ids))
                new.append([h.text + text, h.ids + ids, h.q + v, h.m])
        if pending:
            for (i, h, _), value in zip(pending, self.lm_tails([h.ids for _, h, _ in pending],
                                                               [ids for _, _, ids in pending])):
                new[i][2] = h.q + value
        w = decoder.weight
        new = sorted(new, key=lambda e: e[3] + w * e[2], reverse=True)[:RESCORED]
        try:
            scores = decoder.model_score(self.context, kana[:end], [e[0] for e in new])
        except ValueError:
            scores = [0.0] * len(new)  # past the kana model's window the LM alone ranks
        best = {}
        for (text, ids, q, _), m in zip(new, scores):
            h = Hypothesis(text, ids, q, m)
            if text not in best or self.rank(h) > self.rank(best[text]):
                best[text] = h
        return sorted(best.values(), key=self.rank, reverse=True)[:decoder.beam]

    def decode(self, kana, pending, constraints):
        common = 0
        while common < min(len(kana), len(self.kana)) and kana[common] == self.kana[common]:
            common += 1
        if constraints != self.constraints:
            # Frontiers before the first changed constraint stay valid.
            old = self.constraints or (None,) * len(self.kana)
            new = constraints or (None,) * len(kana)
            first = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
            common = min(common, first)
            self.constraints = constraints
        self.kept, self.kana = self.kept[:common + 1], kana[:common]
        if common == 0 or len(self.memo) > MEMO:
            # A new composition (or a very long one): LM rows of earlier prefixes go.
            self.memo, self.tails = {}, {}
        for end in range(common + 1, len(kana) + 1):
            self.kept.append(self.expand(kana, end))
            self.kana = kana[:end]
        if not pending:
            final = self.kept[len(kana)]
        else:
            # Provisional: one more kana the pending keys can start; nothing is kept.
            final = [h for first in pending for h in self.expand(kana + first, len(kana) + 1)]
            best = {h.text: h for h in sorted(final, key=self.rank)}  # the best of each text last
            final = sorted(best.values(), key=self.rank, reverse=True)[:self.decoder.beam]
        return [(h.text, self.rank(h)) for h in final]
