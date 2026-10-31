"""Japanese: the general LM spells typed kana, fused with a kana-conditioned model.

Units are LM tokens (plus dictionary words the LM splits, kanji it has no
token for, numbers and Latin letters) with every reading they may have, from
a table build_table() derives from JMdict, KANJIDIC2 and a corpus lexicon
(build-japanese-lexicon.py: UniDic and JMnedict words seen in text). A Session decodes one
composition a kana at a time, character-synchronous on kana: hypotheses
compared at frontier p have consumed the same p kana. Each new kana expands
into the newest position only; the best expansions are then scored by the
kana model given every typed kana, and ranked by  m + w * q  (m the kana
model's log probability, q the LM's). Keys still pending in the romaji
front end constrain a provisional last kana to the kana they can start.

With explicit spans (the user held Shift), every kana is constrained: 'c'
kana must be written in something other than hiragana (kanji, katakana),
'k' kana stay the hiragana typed. A unit's text must match its kana's runs.

With zenz (azooKey's GPT-2 kana-kanji converter), a composition without
pending keys is finally reranked: zenz's own beam search joins the decoded
beam, but only what the table can spell from the typed kana (free
generation may drop typed material), and all are ranked by
zenz's log P(text </s> | context, kana) + RERANK * the LM's log P(text).
"""
import collections
import functools
import itertools
import json
import math
import re
import sqlite3
import unicodedata
from pathlib import Path

import torch

BEAM, PER_HYPOTHESIS, RESCORED = 16, 24, 64
MEMO = 4096  # memoized LM rows, ~0.5 MB each
MAXREAD = 14  # cap inferred token readings; exact dictionary readings may be longer
RERANK = 0.7  # the LM's weight beside zenz in the final ranking, chosen on dev
FORMAT = "wime-ja-kana-v1"
INPUT, OUTPUT, EOS = "", "", ""
# Without Japanese before point the LM drifts into Chinese; this line stands
# in for the Japanese text an editor buffer would usually hold.
NEUTRAL = "これは日本語の文章です。\n"


HIRAGANA = "\u3041-\u3096\u309d\u309e"
KANA = ("".join(chr(i) for i in range(0x3041, 0x3097))
        + "".join(chr(i) for i in range(0x30A1, 0x30FB)) + "ーゝゞヽヾ")


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
    """Kanji writing -> permitted kana readings, and the common writings."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    writings, common, entries = collections.defaultdict(set), set(), collections.defaultdict(list)
    # restrictedTo contains zero-based writing positions within this entry.
    for entry, writing, priority, frequency in db.execute(
            "SELECT k.id, kt.c0reading, k.priority, e.frequency FROM kanji k "
            "JOIN kanjiText_content kt ON kt.docid = k.docid JOIN entries e ON e.id = k.id "
            "ORDER BY k.id, k.docid"):
        entries[entry].append(writing)
        if (priority or 0) > 0 or (frequency or 0) > 0:
            common.add(writing)
    for entry, reading, nokanji, restricted in db.execute(
            "SELECT n.id, nt.c0reading, n.nokanji, n.restrictedTo FROM kana n "
            "JOIN kanaText_content nt ON nt.docid = n.docid"):
        if nokanji:
            continue
        allowed = {int(i) for i in restricted.split(",")} if restricted else None
        for i, writing in enumerate(entries[entry]):
            if allowed is None or i in allowed:
                writings[writing].add(hira(reading))
    db.close()
    return writings, common


def kanjidic(path, graded=True):
    """Kanji (only jouyou/jinmeiyou if GRADED) -> (stem, okurigana) readings; か.の reads か before の."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    result = collections.defaultdict(set)
    for entry, reading in db.execute("SELECT r.entry, t.c0reading FROM reading r "
                                     "JOIN readingText_content t ON t.docid = r.docid JOIN entries e ON e.id = r.entry"
                                     + (" WHERE e.grade IS NOT NULL" if graded else "")):
        stem, _, okurigana = hira(reading).strip("-").partition(".")
        if stem:
            result[chr(entry)].add((stem, okurigana))
    db.close()
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
    return sound_variants(variants, before=i > 0, after=i + 1 < len(text))


def sound_variants(readings, before=True, after=True):
    """Composable kanji readings, including the sound changes used by tokens."""
    variants = set(readings)
    for r in list(variants):
        if r[-1] in "くきちつ" and after:
            variants.add(r[:-1] + "っ")
        if before and r[0] in VOICED:
            variants.add(VOICED[r[0]] + r[1:])
        if before and r[0] in SEMI:
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
            for head in sorted(prefix[i]):
                for r in sorted(pieces):
                    if len(prefix[j]) < limit:
                        prefix[j].add(head + r)
    return prefix[len(text)]


def number_readings(n):
    if n <= 10:
        return NUMERALS[n]
    tens, ones = divmod(n, 10)
    head = ["じゅう"] if tens == 1 else [r + "じゅう" for r in NUMERALS[tens]]
    return [h + r for h in head for r in (NUMERALS[ones] if ones else [""])]


def frequent(lexicon, top):
    """LEXICON's TOP most frequent writings of each reading: [(writing, reading)]."""
    by_reading = collections.defaultdict(list)
    for w, readings in lexicon.items():
        for r, n in readings.items():
            by_reading[r].append((n, w))
    return [(w, r) for r, ws in by_reading.items() for _, w in sorted(ws, reverse=True)[:top]]


def spells(get_units, kana, text, constraints=None, max_read=None):
    """Whether GET_UNITS can cover KANA and TEXT (NFKC-equal), without omissions."""
    text = unicodedata.normalize("NFKC", text)

    @functools.lru_cache(maxsize=None)
    def reach(i, j):
        if i == len(kana):
            return j == len(text)
        for end in range(i + 1, min(len(kana), i + (max_read or len(kana))) + 1):
            units = [u for u, _ in (get_units(kana[i:end]) or ())]
            if end == i + 1 and not is_kana(kana[i]):
                units.append(kana[i])
            for u in units:
                if constraints and not span_pattern(kana[i:end], constraints[i:end]).fullmatch(u):
                    continue
                n = unicodedata.normalize("NFKC", u)
                if n and text.startswith(n, j) and reach(end, j + len(n)):
                    return True
        return False
    return reach(0, 0)


def complete_dictionary(index, tokenizer, writings):
    """Add dictionary readings the existing units cannot spell.

    Token boundaries can hide sound changes: 実 + 装 reads じつ + そう,
    while 実装 reads じっそう. JMdict's priority flag must not decide whether
    that word is possible. Shorter words first let compounds reuse them.
    """
    added = 0
    for text in sorted(writings, key=lambda text: (len(text), text)):
        missing = [r for r in sorted(writings[text])
                   if r and not spells(index.get, r, text)]
        if missing:
            ids = tokenizer(text)["input_ids"]
            for r in missing:
                index.setdefault(r, []).append([text, ids])
                added += 1
    return added


def required_readings(writings, characters):
    """Exact dictionary words, kanji stems/okurigana and every basic kana.

    Single-character words and kana must survive byte-split tokenization.
    Rare kanji are included here without using their readings to synthesize
    arbitrary multi-kanji tokens.
    """
    result = collections.defaultdict(set, {w: set(rs) for w, rs in writings.items()})
    # Preserve a dictionary word's reading before its written okurigana.
    # Character readings alone cannot recover ateji or changes inside a
    # compound: 盗る -> 盗/と, 接する -> 接/せっ, 辛抱強い -> 辛抱強/しんぼうづよ.
    # Literal kana then complete inflections such as 盗られ and 辛抱強く.
    for text, readings in writings.items():
        for reading in readings:
            stem, kana = text, reading
            while stem and kana and stem[-1] == kana[-1] and "ぁ" <= stem[-1] <= "ゖ":
                stem, kana = stem[:-1], kana[:-1]
                if kana and any(is_kanji(c) for c in stem):
                    result[stem].add(kana)
    for c, readings in characters.items():
        # Fragments such as 一/いっ and 本/ぼん must remain usable when the
        # tokenizer splits a productive compound (一本, 料理本) between kanji.
        result[c].update(sound_variants(stem for stem, _ in readings))
        for stem, okurigana in readings:
            result[c].add(stem)
            result[c + okurigana].add(stem + okurigana)
    for c in KANA:
        result[c].add(hira(c))
    return result


def missing_readings(index, writings):
    """All (writing, reading) pairs the generated units cannot spell."""
    return [(w, r) for w in sorted(writings) for r in sorted(writings[w])
            if not spells(index.get, r, w)]


def build_table(tokenizer, jmdict_path, kanjidic_path, lexicon=None, top=4, validation_units=None):
    """{reading: [[text, [token ids]], ...]} for TOKENIZER's vocabulary.
    LEXICON ({writing: {reading: corpus count}}) adds the TOP most frequent
    writings of each reading; the LM splits most of them. More crowd the
    beam with names and slow decoding down. Every KANJIDIC kanji the LM has
    a token for reads alone; multi-character tokens compose readings from
    graded kanji only: rarer readings let Chinese tokens spell Japanese kana.
    Dictionary words also supply readings that cannot cross token boundaries.
    If VALIDATION_UNITS is a dict, put additional character/inflection paths
    there so zenz can use them without expanding them at every beam frontier."""
    writings, common = jmdict(jmdict_path)
    for w, r in frequent(lexicon or {}, top):
        writings[w].add(r)
        common.add(w)
    chars, every = kanjidic(kanjidic_path), kanjidic(kanjidic_path, graded=False)
    index, single = collections.defaultdict(list), set()
    for i in range(len(tokenizer)):
        t = tokenizer.decode([i])
        if not t or "�" in t or t.isspace():
            continue
        if any(is_kanji(c) for c in t):
            readings = set(writings.get(t, ())) | compose(t, every if len(t) == 1 else chars, writings)
        elif any(is_kana(c) for c in t) or len(t) == 1 or not any(c.isascii() for c in t):
            readings = {hira(t)}
        else:
            continue  # ASCII words: typed literally, one character at a time
        if len(t) == 1:
            single.add(t)
        for r in sorted(readings):
            if 0 < len(r) <= MAXREAD:
                index[r].append([t, [i]])
    for c, readings in chars.items():  # rarer kanji the LM splits only come in lexicon words
        if c not in single:
            ids = tokenizer(c)["input_ids"]
            for r in sorted({stem for stem, _ in readings}):
                index[r].append([c, ids])
    tokens = {u[0] for units in index.values() for u in units if len(u[1]) == 1}
    for w in sorted(common):
        if w not in tokens and len(w) > 1:
            ids = tokenizer(w)["input_ids"]
            if len(ids) > 1:
                for r in sorted(writings[w]):
                    if len(r) <= MAXREAD:
                        index[r].append([w, ids])
    for n in range(100):
        for r in number_readings(n):
            index[r].append([str(n), tokenizer(str(n))["input_ids"]])
    for letter, r in LETTERS.items():
        for text in (letter, letter.lower()):
            index[r].append([text, tokenizer(text)["input_ids"]])
    core = {w: set(rs) for w, rs in writings.items()}
    for c in KANA:
        core.setdefault(c, set()).add(hira(c))
    complete_dictionary(index, tokenizer, core)
    search = {r: list(units) for r, units in index.items()} if validation_units is not None else None
    required = required_readings(writings, every)
    complete_dictionary(index, tokenizer, required)
    if missing := missing_readings(index, required):
        raise ValueError(f"incomplete reading table: {missing[:5]}")
    if search is not None:
        validation_units.update({r: units[len(search.get(r, ())):] for r, units in index.items()
                                 if len(units) > len(search.get(r, ()))})
        return search
    return dict(index)


class Table:
    def __init__(self, path, tokenizer):
        data = json.loads(Path(path).read_text())
        self.index = {r: [(t, tuple(ids)) for t, ids in units] for r, units in data["units"].items()}
        self.search_index = self.index
        if extra := data.get("validation_units"):
            self.index = {r: list(units) for r, units in self.index.items()}
            for r, units in extra.items():
                self.index.setdefault(r, []).extend((t, tuple(ids)) for t, ids in units)
        self.max_read = max(map(len, self.index), default=MAXREAD)
        self.tokenizer = tokenizer

    def get(self, reading):
        return self.index.get(reading, ())

    def search(self, reading):
        return self.search_index.get(reading, ())

    def literal(self, c):
        return [(c, tuple(self.tokenizer(c)["input_ids"]))]

    def spells(self, kana, text, constraints=None):
        """Whether decoding could write KANA as TEXT (NFKC-equal), under CONSTRAINTS."""
        return spells(self.get, kana, text, constraints, self.max_read)


# --- zenz --------------------------------------------------------------------

class Zenz:
    """zenz converted by convert-zenz.py; prompt [EE02 context] EE00 katakana EE01 text."""

    def __init__(self, path, device="cuda", beams=8):
        from tokenizers import Tokenizer
        from transformers import GPT2LMHeadModel
        self.model = GPT2LMHeadModel.from_pretrained(path).to(device).eval()
        self.tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        self.eos, self.device, self.beams = self.model.config.eos_token_id, device, beams

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def prompt(self, context, kana):
        katakana = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in kana)
        context = context[-40:]
        return self.encode(("\uee02" + context if context else "") + "\uee00" + katakana + "\uee01")

    @torch.inference_mode()
    def generate(self, context, kana):
        prompt = self.prompt(context, kana)
        out = self.model.generate(torch.tensor([prompt], device=self.device), max_new_tokens=len(kana) + 8,
                                  num_beams=self.beams, num_return_sequences=self.beams, do_sample=False,
                                  eos_token_id=self.eos, pad_token_id=self.eos)
        return [self.tokenizer.decode([i for i in o[len(prompt):].tolist() if i != self.eos]).split("\n")[0]
                for o in out]

    @torch.inference_mode()
    def score(self, context, kana, texts):
        """Sum log p(text </s> | prompt) for each text."""
        prompt = self.prompt(context, kana)
        encoded = [self.encode(t) + [self.eos] for t in texts]
        x = torch.full((len(texts), len(prompt) + max(map(len, encoded))), self.eos)
        rows, cols, targets = [], [], []
        for i, e in enumerate(encoded):
            x[i, :len(prompt) + len(e)] = torch.tensor(prompt + e)
            rows += [i] * len(e); cols += range(len(prompt) - 1, len(prompt) - 1 + len(e)); targets += e
        logits = self.model(x.to(self.device)).logits
        rows = torch.tensor(rows, device=self.device)
        lp = logits[rows, torch.tensor(cols, device=self.device)].float().log_softmax(-1)
        picked = lp[torch.arange(len(targets), device=self.device), torch.tensor(targets, device=self.device)]
        return torch.zeros(len(texts), device=self.device).index_add_(0, rows, picked).tolist()


# --- kana model --------------------------------------------------------------

class KanaModel:
    """Character LM over <bos> context <input> kana <output> text <eos>."""

    def __init__(self, path, device="cuda"):
        from safetensors import safe_open
        from charlm import CharLM, Config
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

    def __init__(self, table, rescorer, kana_model, weight=0.6, beam=BEAM, per_hypothesis=PER_HYPOTHESIS, zenz=None):
        self.table, self.rescorer, self.kana_model, self.zenz = table, rescorer, kana_model, zenz
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
            ranked = self.session.decode(kana, tuple(sorted(set(pending))), constraints)
        except ValueError:
            self.session = None
            return []
        if self.zenz is None or pending or not ranked:
            return ranked
        return self.rerank(context, kana, [t for t, _ in ranked], constraints and constraints[:len(kana)])

    def rerank(self, context, kana, texts, constraints):
        texts = list(dict.fromkeys(texts + [t for t in self.zenz.generate(context, kana)
                                            if t and self.table.spells(kana, t, constraints)]))
        z = self.zenz.score(context, kana, texts)
        q = self.lm_tails(self.session.lm, [()] * len(texts),
                          [self.table.tokenizer(t)["input_ids"] for t in texts])
        return sorted(zip(texts, (a + RERANK * b for a, b in zip(z, q))), key=lambda pair: -pair[1])


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
        for start in range(max(0, end - table.max_read), end):
            if not self.kept[start]:
                continue
            units = list((table.search if decoder.zenz else table.get)(kana[start:end]))
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
