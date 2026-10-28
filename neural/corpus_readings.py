"""Text -> Microsoft double-pinyin keys for corpus construction.

Readings come from three sources: frequent wanxiang dictionary words (the same
lattice wime types against), pypinyin, and g2pW for disputed polyphones. The
dictionary also carries corpus-derived phrases with automatic readings, so
rare entries are not trusted alone. Every character must end on a key pair the
database accepts for it; otherwise its span is unusable.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sqlite3

from pypinyin import Style, lazy_pinyin

# U+4DC0..U+4DFF are Yijing symbols, between Extension A and unified Han.
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
SYLLABLES = json.loads(Path(__file__).with_name("context-pinyin.json").read_text())["syllables"]
TRUSTED_WEIGHT = 500


def key(pinyin):
    return SYLLABLES.get(pinyin)


def model_key(pinyin):
    return key(re.sub(r"\d", "", pinyin).replace("u:", "v").replace("ü", "v")) if pinyin else None


class Lexicon:
    """Dictionary readings in canonical keys, plus per-character legal keys."""

    def __init__(self, database):
        inverse = defaultdict(set)
        for pinyin, code in SYLLABLES.items():
            inverse[code].add(pinyin)
        self.words, self.legal, self.char_weight = {}, defaultdict(set), Counter()
        for code, word, weight in sqlite3.connect(database).execute("SELECT code, word, weight FROM entry"):
            if len(code) != 2 * len(word) or not HAN.fullmatch(word):
                continue
            pairs = [code[i:i + 2] for i in range(0, len(code), 2)]
            if len(word) == 1:
                self.legal[word].add(code)
                self.char_weight[word] = max(self.char_weight[word], weight)
            elif len(word) <= 8 and all(p in inverse for p in pairs):
                # Derived spellings (e.g. jv for jt) have a canonical twin entry.
                options = self.words.setdefault(word, {})
                options[tuple(pairs)] = max(options.get(tuple(pairs), 0), weight)
        self.log_total = math.log(sum(self.char_weight.values()) + sum(max(v.values()) for v in self.words.values()))

    def segment(self, run):
        """Unigram max-probability segmentation; returns [(start, end, word or None)]."""
        n = len(run)
        best = [(-math.inf, None)] * (n + 1)
        best[0] = (0.0, None)
        for i in range(n):
            if best[i][0] == -math.inf:
                continue
            options = [(i + 1, None, math.log(1 + self.char_weight[run[i]]) - self.log_total)]
            for j in range(i + 2, min(n, i + 8) + 1):
                word = run[i:j]
                if word in self.words:
                    options.append((j, word, math.log(1 + max(self.words[word].values())) - self.log_total))
            for j, word, value in options:
                if best[i][0] + value > best[j][0]:
                    best[j] = (best[i][0] + value, (i, word))
        spans, j = [], n
        while j:
            i, word = best[j][1]
            spans.append((i, j, word))
            j = i
        return spans[::-1]


class Reader:
    """Batch reader. ``g2pw`` is a G2PWConverter (style='pinyin') or None."""

    def __init__(self, lexicon, g2pw=None):
        self.lexicon, self.g2pw = lexicon, g2pw
        self.methods = Counter()
        self.model_keys = None
        if g2pw is not None and hasattr(g2pw, "polyphonic_chars"):
            # Mirror G2PWConverter._prepare_data's source precedence. In
            # particular, its trained label inventory can omit a common
            # Mandarin reading even when its fallback dictionary knows it.
            self.model_keys = {}
            convert = g2pw.style_convert_func
            for char, readings in getattr(g2pw, "char_bopomofo_dict", {}).items():
                self.model_keys[char] = {model_key(convert(readings[0]))} if readings else set()
            for char, reading in getattr(g2pw, "monophonic_chars", []):
                self.model_keys[char] = {model_key(convert(reading))}
            polyphonic = defaultdict(set)
            for char, reading in g2pw.polyphonic_chars:
                polyphonic[char].add(model_key(convert(reading)))
            self.model_keys.update(polyphonic)

    def supported_model_keys(self, char):
        if self.model_keys is None:
            return None
        if getattr(self.g2pw, "enable_non_tradional_chinese", False):
            char = self.g2pw.s2t_dict.get(char, char)
        return self.model_keys.get(char, set())

    def keys(self, sentences):
        """Per sentence, a list aligned with characters: key pair or None for non-Han/unusable."""
        drafts, disputed = [], []
        for index, sentence in enumerate(sentences):
            py = [None] * len(sentence)
            row = [None] * len(sentence)
            pending = []
            for match in HAN.finditer(sentence):
                # Unknown characters must remain separate, never shrink the
                # aligned list and shift readings onto following characters.
                readings = lazy_pinyin(match.group(), style=Style.NORMAL, errors=lambda chars: list(chars))
                if len(readings) != len(match.group()):
                    raise ValueError(f"pypinyin returned {len(readings)} readings for {len(match.group())} characters")
                py[match.start():match.end()] = readings
                for i, j, word in self.lexicon.segment(match.group()):
                    i, j = i + match.start(), j + match.start()
                    trusted = None
                    if word is not None:
                        options = self.lexicon.words[word]
                        reading, weight = max(options.items(), key=lambda item: item[1])
                        if len(options) == 1 and weight >= TRUSTED_WEIGHT:
                            trusted = reading
                    for k in range(i, j):
                        char, legal = sentence[k], self.lexicon.legal.get(sentence[k], set())
                        guess = key(py[k]) if py[k] else None
                        word_key = trusted[k - i] if trusted else None
                        if len(legal) == 1:
                            row[k] = (next(iter(legal)), "mono")
                        elif word_key and word_key == guess and word_key in legal:
                            row[k] = (word_key, "agree")
                        else:
                            row[k] = (word_key, guess)
                            pending.append(k)
            drafts.append(row)
            if pending:
                disputed.append((index, pending))
        predictions = [None] * len(disputed)
        if disputed and self.g2pw is not None:
            texts = [sentences[i] for i, _ in disputed]
            if hasattr(self.g2pw, "read_positions"):
                predictions = self.g2pw.read_positions(texts, [pending for _, pending in disputed])
            else:
                predictions = self.g2pw(texts)
        if not isinstance(predictions, (list, tuple)) or len(predictions) != len(disputed):
            raise ValueError("g2pW must return one prediction row per disputed sentence")
        for (index, pending), predicted in zip(disputed, predictions):
            sentence, row = sentences[index], drafts[index]
            if self.g2pw is not None and (not isinstance(predicted, (list, tuple)) or len(predicted) != len(sentence)):
                raise ValueError(f"g2pW prediction must align with all {len(sentence)} sentence characters")
            for k in pending:
                word_key, guess = row[k]
                legal = self.lexicon.legal.get(sentence[k], set())
                g = predicted[k] if predicted and predicted[k] else None
                g = model_key(g)
                votes = Counter(v for v in (word_key, guess, g) if v in legal)
                if not votes:
                    row[k] = (None, "unreachable")
                    continue
                (first, count), *rest = votes.most_common()
                if rest and rest[0][1] == count:
                    supported = self.supported_model_keys(sentence[k])
                    if g in votes and guess in votes and g != guess and supported is not None and guess not in supported:
                        # The model cannot adjudicate a reading it was never
                        # trained to emit. Drop this span instead of silently
                        # replacing, for example, colloquial 咋 (za) with ze.
                        row[k] = (None, "coverage_conflict")
                        continue
                    # A tie means disagreeing sources; g2pW reads the sentence context.
                    first = g if g in votes else first
                row[k] = (first, "vote" if count > 1 else "g2pw" if first == g else "single")
        result = []
        for row in drafts:
            for item in row:
                if item is not None:
                    self.methods[item[1]] += 1
            result.append([item[0] if item is not None else None for item in row])
        return result
