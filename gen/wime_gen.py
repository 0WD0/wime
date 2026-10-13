#!/usr/bin/env python3
"""Generate wime's tables from local Rime data.

Both the single-character table and the word table are keyed by the Microsoft
double-pinyin code.  The conversion is a faithful port of the `pro/微软双拼`
rules in `wanxiang_algebra.yaml`; it is not guessed from a hand-written table,
so it follows the installed wanxiang release.
"""

import argparse
import re
from pathlib import Path

import yaml


def load_rules(algebra_path):
    document = yaml.safe_load(algebra_path.read_text())
    rules = document["pro"]["微软双拼"]["__append"]
    compiled = []
    for rule in rules:
        parts = rule.split("/")
        name = parts[0]
        if name in ("xform", "derive", "abbrev"):
            compiled.append((name, parts[1], _rime_replacement(parts[2])))
        elif name == "xlit":
            compiled.append((name, parts[1], parts[2]))
        elif name == "erase":
            compiled.append((name, parts[1], ""))
        else:
            raise ValueError("unsupported algebra rule: %s" % rule)
    return compiled


def _rime_replacement(value):
    """Translate Rime's $1 backreferences to Python's \\1 form."""
    return re.sub(r"\$(\d)", r"\\\1", value)


def apply_rules(code, rules):
    spellings = {code}
    for name, pattern, replacement in rules:
        if name == "xform":
            spellings = {re.sub(pattern, replacement, s) for s in spellings}
        elif name in ("derive", "abbrev"):
            derived = set(spellings)
            for s in spellings:
                if re.search(pattern, s):
                    derived.add(re.sub(pattern, replacement, s))
            spellings = derived
        elif name == "xlit":
            table = str.maketrans(pattern, replacement)
            spellings = {s.translate(table) for s in spellings}
        elif name == "erase":
            spellings = {re.sub(pattern, "", s) for s in spellings}
    return spellings


def _extract_code(spelling):
    """Pull the typed double-pinyin code out of an algebra spelling.

    The rules append a tone digit and keep the trailing ';' separator we
    introduced.  ``◯`` is the placeholder for the ``ing`` key, which is
    literally ``;`` on the keyboard, so it is restored before the separator is
    removed.
    """
    spelling = spelling.replace("◯", ";")
    if spelling.endswith(";"):
        spelling = spelling[:-1]
    spelling = re.sub(r"\d+$", "", spelling)
    return spelling


class Converter:
    """Double-pinyin conversion with a per-syllable cache."""

    def __init__(self, rules):
        self.rules = rules
        self.cache = {}

    def codes(self, pinyin):
        if pinyin not in self.cache:
            result = set()
            for spelling in apply_rules(pinyin + ";", self.rules):
                code = _extract_code(spelling)
                if code and all(ch.isascii() for ch in code):
                    result.add(code)
            self.cache[pinyin] = sorted(result)
        return self.cache[pinyin]

    def primary(self, pinyin):
        codes = self.codes(pinyin)
        return codes[0] if codes else None

    def word(self, code_field):
        """Return ``(code, aux)`` for a dictionary code field, or None.

        The auxiliary part concatenates each syllable's first Moqi group; it is
        stored lowercase and matched case-insensitively at runtime.
        """
        syllables = code_field.split()
        codes = []
        auxes = []
        for syllable in syllables:
            pinyin, _, aux = syllable.partition(";")
            code = self.primary(pinyin)
            if code is None:
                return None
            codes.append(code)
            auxes.append(re.sub(r"[^a-z]", "", aux.split(",")[0].lower()))
        return "".join(codes), "".join(auxes)


def parse_zi(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            continue
        word, code, weight = fields
        if ";" not in code:
            continue
        pinyin, _, aux = code.partition(";")
        try:
            weight = int(weight)
        except ValueError:
            continue
        yield word, pinyin, aux, weight


def build_single(zi_path, converter):
    table = {}
    for word, pinyin, aux, weight in parse_zi(zi_path):
        for code in converter.codes(pinyin):
            table.setdefault(code, []).append((word, aux, weight))
    for code, entries in table.items():
        seen = set()
        ordered = []
        for word, aux, weight in sorted(entries, key=lambda e: -e[2]):
            if (word, aux) in seen:
                continue
            seen.add((word, aux))
            ordered.append((word, aux))
        table[code] = ordered
    return table


def build_words(words_path, converter, min_weight, max_candidates):
    table = {}
    with words_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#") or "\t" not in line:
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                continue
            word, code_field, weight = fields
            if not 2 <= len(word) <= 4:
                continue
            if not all("\u4e00" <= ch <= "\u9fff" for ch in word):
                continue
            parsed = converter.word(code_field)
            if parsed is None:
                continue
            try:
                weight = int(weight)
            except ValueError:
                continue
            code, aux = parsed
            table.setdefault(code, []).append((word, aux, weight))
    result = {}
    for code, entries in table.items():
        ordered = sorted(entries, key=lambda e: -e[2])[:max_candidates]
        if ordered and ordered[0][2] >= min_weight:
            result[code] = [(word, aux) for word, aux, _ in ordered]
    return result


def elisp_string(value):
    return '"%s"' % value.replace("\\", "\\\\").replace('"', '\\"')


def emit(feature, const, title, table, stream):
    stream.write(";;; %s.el --- Generated wanxiang %s -*- lexical-binding: t; -*-\n"
                 % (feature, title))
    stream.write(";;; Commentary:\n")
    stream.write(";; Generated by gen/wime_gen.py; do not edit by hand.\n")
    stream.write(";;; Code:\n\n")
    stream.write("(defconst %s\n  '(\n" % const)
    for code in sorted(table):
        stream.write("    (%s" % elisp_string(code))
        for word, aux in table[code]:
            stream.write(" (%s %s)" % (elisp_string(word), elisp_string(aux)))
        stream.write(")\n")
    stream.write("    ))\n\n(provide '%s)\n;;; %s.el ends here\n" % (feature, feature))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rime-data", type=Path, default=Path("/usr/share/rime-data"))
    parser.add_argument("--output", type=Path, help="single-character table")
    parser.add_argument("--words-output", type=Path, help="word table")
    parser.add_argument("--min-weight", type=int, default=1000,
                        help="minimum weight for a word code to be kept")
    parser.add_argument("--max-candidates", type=int, default=8,
                        help="maximum words kept per code")
    args = parser.parse_args()
    if not args.output and not args.words_output:
        parser.error("at least one of --output or --words-output is required")
    converter = Converter(load_rules(args.rime_data / "wanxiang_algebra.yaml"))
    if args.output:
        table = build_single(args.rime_data / "dicts" / "zi.pro.dict.yaml", converter)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            emit("wime-data", "wime-wanxiang-table", "single-character table",
                 table, stream)
        print("single: %d codes, %d entries"
              % (len(table), sum(len(v) for v in table.values())))
    if args.words_output:
        table = build_words(args.rime_data / "dicts" / "jichu.pro.dict.yaml",
                            converter, args.min_weight, args.max_candidates)
        args.words_output.parent.mkdir(parents=True, exist_ok=True)
        with args.words_output.open("w", encoding="utf-8") as stream:
            emit("wime-words", "wime-wanxiang-words", "word table", table, stream)
        print("words: %d codes, %d entries"
              % (len(table), sum(len(v) for v in table.values())))


if __name__ == "__main__":
    main()
