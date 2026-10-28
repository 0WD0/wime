"""Text -> Microsoft double-pinyin keys for corpus construction.

Readings come from three sources: frequent wanxiang dictionary words (the same
lattice wime types against), pypinyin, and g2pW for disputed polyphones. The
dictionary also carries corpus-derived phrases with automatic readings, so
rare entries are not trusted alone. Every character must end on a key pair the
database accepts for it; otherwise its span is unusable.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import sqlite3

from pypinyin import Style, lazy_pinyin, pinyin

# U+4DC0..U+4DFF are Yijing symbols, between Extension A and unified Han.
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
SYLLABLES = json.loads(Path(__file__).with_name("context-pinyin.json").read_text())["syllables"]
TRUSTED_WEIGHT = 500


def key(pinyin):
    return SYLLABLES.get(pinyin)


def model_key(pinyin):
    return key(re.sub(r"\d", "", pinyin).replace("u:", "v").replace("ü", "v")) if pinyin else None


def numbered_reading(value):
    """Normalize explicit numbered readings; absent tones are unknown, not 5."""
    if not value:
        return None
    value = value.lower().replace("u:", "v").replace("ü", "v")
    return value if re.fullmatch(r"[a-z]+[1-5]", value) else None


@lru_cache(maxsize=65536)
def character_readings(char):
    """Candidate inventory for detecting ambiguity, never the output tone."""
    readings = pinyin(char, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
    return frozenset(value for reading in readings if (value := numbered_reading(reading)))


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
        self.model_readings = None
        if g2pw is not None and hasattr(g2pw, "polyphonic_chars"):
            # Mirror G2PWConverter._prepare_data's source precedence. In
            # particular, its trained label inventory can omit a common
            # Mandarin reading even when its fallback dictionary knows it.
            self.model_keys = {}
            self.model_readings = {}
            convert = g2pw.style_convert_func
            for char, readings in getattr(g2pw, "char_bopomofo_dict", {}).items():
                converted = convert(readings[0]) if readings else None
                self.model_keys[char] = {model_key(converted)} if readings else set()
                self.model_readings[char] = {numbered_reading(converted)} if readings else set()
            for char, reading in getattr(g2pw, "monophonic_chars", []):
                converted = convert(reading)
                self.model_keys[char] = {model_key(converted)}
                self.model_readings[char] = {numbered_reading(converted)}
            polyphonic = defaultdict(set)
            numbered = defaultdict(set)
            for char, reading in g2pw.polyphonic_chars:
                converted = convert(reading)
                polyphonic[char].add(model_key(converted))
                numbered[char].add(numbered_reading(converted))
            self.model_keys.update(polyphonic)
            self.model_readings.update(numbered)

    def supported_model_keys(self, char):
        if self.model_keys is None:
            return None
        if getattr(self.g2pw, "enable_non_tradional_chinese", False):
            char = self.g2pw.s2t_dict.get(char, char)
        return self.model_keys.get(char, set())

    def supported_model_readings(self, char):
        if self.model_readings is None:
            return set()
        if getattr(self.g2pw, "enable_non_tradional_chinese", False):
            char = self.g2pw.s2t_dict.get(char, char)
        return self.model_readings.get(char, set()) - {None}

    def _predict(self, sentences, positions):
        if hasattr(self.g2pw, "read_positions"):
            predictions = self.g2pw.read_positions(sentences, positions)
        else:
            predictions = self.g2pw(sentences)
        if not isinstance(predictions, (list, tuple)) or len(predictions) != len(sentences):
            raise ValueError("g2pW must return one prediction row per disputed sentence")
        for sentence, predicted in zip(sentences, predictions):
            if not isinstance(predicted, (list, tuple)) or len(predicted) != len(sentence):
                raise ValueError(f"g2pW prediction must align with all {len(sentence)} sentence characters")
        return predictions

    def _key_details(self, sentences):
        """Retain the original toneless decisions and their requested evidence."""
        drafts, disputed = [], []
        evidence = [[None] * len(sentence) for sentence in sentences]
        queried = [set() for _ in sentences]
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
            predictions = self._predict(texts, [pending for _, pending in disputed])
        for (index, pending), predicted in zip(disputed, predictions):
            sentence, row = sentences[index], drafts[index]
            for k in pending:
                word_key, guess = row[k]
                legal = self.lexicon.legal.get(sentence[k], set())
                g = predicted[k] if predicted and predicted[k] else None
                if predicted is not None:
                    evidence[index][k] = numbered_reading(g)
                    queried[index].add(k)
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
        return result, evidence, queried

    def keys(self, sentences):
        """Per sentence, aligned key pairs or None; tone annotation never changes them."""
        return self._key_details(sentences)[0]

    def readings(self, sentences):
        """Keys plus contextual tone evidence, aligned before corpus span clipping.

        pypinyin applies its contextual sandhi rules; g2pW predicts contextual
        readings but does not implement all surface sandhi. Conflicts remain
        unknown. Character inventories only identify ambiguity; they never
        replace contextual predictions with citation tones.
        """
        keys, model, queried = self._key_details(sentences)
        phonetic = [[None] * len(sentence) for sentence in sentences]
        extra = []
        for index, sentence in enumerate(sentences):
            for match in HAN.finditer(sentence):
                values = lazy_pinyin(match.group(), style=Style.TONE3, neutral_tone_with_five=True,
                                     tone_sandhi=True, errors=lambda chars: list(chars))
                if len(values) != len(match.group()):
                    raise ValueError(f"pypinyin returned {len(values)} readings for {len(match.group())} characters")
                phonetic[index][match.start():match.end()] = [numbered_reading(value) for value in values]
            pending = []
            for k, code in enumerate(keys[index]):
                if code is None or k in queried[index]:
                    continue
                candidates = character_readings(sentence[k]) | self.supported_model_readings(sentence[k])
                compatible = {value for value in candidates if model_key(value) == code}
                if len(compatible) > 1:
                    pending.append(k)
            if pending and self.g2pw is not None:
                extra.append((index, pending))
        if extra:
            predicted = self._predict([sentences[i] for i, _ in extra], [pending for _, pending in extra])
            for (index, pending), values in zip(extra, predicted):
                for k in pending:
                    model[index][k] = numbered_reading(values[k])

        results = []
        for index, sentence in enumerate(sentences):
            syllables, tones, sources = [], [], []
            for k, code in enumerate(keys[index]):
                if code is None:
                    syllables.append(None)
                    tones.append(None)
                    sources.append("unusable" if HAN.fullmatch(sentence[k]) else "non_han")
                    continue
                py, gp = phonetic[index][k], model[index][k]
                compatible = [(name, value) for name, value in (("pypinyin", py), ("g2pw", gp))
                              if value and model_key(value) == code]
                syllable, tone, source = None, None, "unknown"
                if len(compatible) == 2:
                    if py == gp:
                        syllable, tone, source = py[:-1], int(py[-1]), "agreement"
                    elif py[:-1] == gp[:-1]:
                        syllable, source = py[:-1], "tone_conflict"
                    else:
                        source = "syllable_conflict"
                elif compatible:
                    name, value = compatible[0]
                    syllable = value[:-1]
                    candidates = character_readings(sentence[k]) | self.supported_model_readings(sentence[k])
                    candidate_tones = {int(candidate[-1]) for candidate in candidates if candidate[:-1] == syllable}
                    if len(candidate_tones) == 1:
                        # Keep the source's contextual tone, even when sandhi
                        # differs from the inventory's single citation tone.
                        tone, source = int(value[-1]), name + "_contextual_unique"
                    else:
                        source = name + "_ambiguous"
                syllables.append(syllable)
                tones.append(tone)
                sources.append(source)
            results.append(dict(keys=keys[index], pinyin=syllables, tones=tones, tone_sources=sources,
                                reading_evidence=dict(pypinyin=phonetic[index], g2pw=model[index])))
        return results
