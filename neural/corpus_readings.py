"""One Mandarin frontend supplies readings; wanxiang validates their key reachability."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sqlite3

from mandarin_frontend import HAN, numbered_reading

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
        rows = self.frontend.readings(sentences)
        if not isinstance(rows, (list, tuple)) or len(rows) != len(sentences):
            raise ValueError("frontend must return one reading row per sentence")
        result = []
        for sentence, row in zip(sentences, rows):
            evidence = row.get("reading_evidence", {})
            if set(evidence) != {"frontend_before_sandhi", "final"}:
                raise ValueError("frontend must provide before-sandhi and final reading evidence")
            aligned = [row.get("tone_sources"), row.get("reading_rules"), *evidence.values()]
            if any(not isinstance(values, (list, tuple)) or len(values) != len(sentence) for values in aligned):
                raise ValueError("frontend reading fields must align with original characters")
            keys, syllables, tones = [], [], []
            sources = list(row["tone_sources"])
            for i, (char, value) in enumerate(zip(sentence, evidence["final"])):
                normalized = numbered_reading(value)
                if value is not None and normalized != value:
                    raise ValueError("frontend final evidence must be normalized numbered syllables")
                code = key(normalized[:-1]) if normalized else None
                if not HAN.fullmatch(char):
                    if value is not None:
                        raise ValueError("frontend may not annotate non-Han characters")
                    code = None
                    sources[i] = "non_han"
                elif code not in self.lexicon.legal.get(char, set()):
                    code = None
                    sources[i] = "unreachable" if normalized else "unresolved"
                keys.append(code)
                syllables.append(normalized[:-1] if code else None)
                tones.append(int(normalized[-1]) if code else None)
                self.methods[sources[i]] += 1
            result.append(dict(keys=keys, pinyin=syllables, tones=tones, tone_sources=sources,
                               reading_rules=[list(rules) for rules in row["reading_rules"]],
                               reading_evidence={name: list(values) for name, values in evidence.items()}))
        return result

    def keys(self, sentences):
        return [row["keys"] for row in self.readings(sentences)]
