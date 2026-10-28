"""One Mandarin frontend supplies readings; wanxiang validates their key reachability."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sqlite3

from mandarin_frontend import HAN

SYLLABLES = json.loads(Path(__file__).with_name("context-pinyin.json").read_text())["syllables"]


def key(pinyin):
    return SYLLABLES.get(pinyin)


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
    """Adapt the final Mandarin reading to the exact wime lattice keys.

    The dictionary provides segmentation and legal key pairs, never another
    pronunciation vote. All annotations and input keys share one final reading.
    """

    def __init__(self, lexicon, frontend):
        self.lexicon, self.frontend = lexicon, frontend
        self.methods = Counter()

    def readings(self, sentences):
        result = []
        for sentence, row in zip(sentences, self.frontend.readings(sentences)):
            keys, syllables, tones = [], [], []
            sources = list(row["tone_sources"])
            for i, (char, reading) in enumerate(zip(sentence, row["reading_evidence"]["final"])):
                code = key(reading[:-1]) if reading else None
                if reading and code not in self.lexicon.legal.get(char, ()):
                    code = None
                    sources[i] = "unreachable"
                keys.append(code)
                syllables.append(reading[:-1] if code else None)
                tones.append(int(reading[-1]) if code else None)
            self.methods.update(sources)
            result.append(dict(row, keys=keys, pinyin=syllables, tones=tones, tone_sources=sources))
        return result

    def keys(self, sentences):
        return [row["keys"] for row in self.readings(sentences)]
