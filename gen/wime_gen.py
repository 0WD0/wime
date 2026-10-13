#!/usr/bin/env python3
"""Generate wime's SQLite word database from the pinned rime-wanxiang sources.

The database is keyed by the Microsoft double-pinyin code.  The conversion is
a faithful port of the `pro/微软双拼` rules in `wanxiang_algebra.yaml`; it is
not guessed from a hand-written table, so it follows upstream.
"""

import argparse
import itertools
import re
import sqlite3
from pathlib import Path


def load_rules(algebra_path):
    """Extract the ``pro/微软双拼`` algebra rules without a YAML library."""
    lines = algebra_path.read_text(encoding="utf-8").splitlines()
    rules = []
    in_pro = in_dialect = in_append = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            in_pro = stripped == "pro:"
            in_dialect = in_append = False
            continue
        if not in_pro:
            continue
        if indent == 2:
            in_dialect = stripped == "微软双拼:"
            in_append = False
            continue
        if not in_dialect:
            continue
        if indent == 4:
            in_append = stripped.startswith("__append:")
            continue
        if in_append and stripped.startswith("- "):
            rule = re.split(r"\s+#", stripped[2:], maxsplit=1)[0].strip()
            rules.append(compile_rule(rule))
    return rules


def compile_rule(rule):
    parts = rule.split("/")
    name = parts[0]
    if name in ("xform", "derive", "abbrev"):
        return (name, parts[1], _rime_replacement(parts[2]))
    if name == "xlit":
        return (name, parts[1], parts[2])
    if name == "erase":
        return (name, parts[1], "")
    raise ValueError("unsupported algebra rule: %s" % rule)


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

    def word_codes(self, code_field):
        """Return every ``(code, aux)`` spelling for a dictionary code field.

        A syllable can have more than one valid double-pinyin code (for
        example ``jue`` is both ``jt`` and ``jv``), so a word is stored under
        every combination.  The auxiliary part keeps each syllable's Moqi
        groups and joins the syllables with ``/`` so the engine can match
        them positionally; it is stored lowercase and matched
        case-insensitively at runtime.
        """
        choices = []
        auxes = []
        for syllable in code_field.split():
            pinyin, _, aux = syllable.partition(";")
            codes = self.codes(pinyin)
            if not codes:
                return []
            choices.append(codes)
            auxes.append(re.sub(r"[^a-z,]", "", aux.lower()))
        aux = "/".join(auxes)
        return [("".join(combination), aux)
                for combination in itertools.product(*choices)]


def iter_entries(path):
    """Yield ``(word, code_field, weight)`` from a Rime dictionary file."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#") or "\t" not in line:
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                continue
            word, code_field, weight = fields
            try:
                weight = int(weight)
            except ValueError:
                continue
            yield word, code_field, weight


def imported_dicts(dict_path):
    """Return the dictionary paths imported by a top-level dictionary."""
    paths = []
    collect = False
    for line in dict_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("import_tables:"):
            collect = True
            continue
        if collect:
            if not line.startswith("  "):
                break
            name = line.strip().lstrip("- ").split()[0]
            paths.append(dict_path.parent / (name + ".dict.yaml"))
    return paths


def build_database(dict_paths, converter, output):
    """Write a SQLite database keyed by double-pinyin code."""
    if output.exists():
        output.unlink()
    connection = sqlite3.connect(str(output))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE entry (code TEXT NOT NULL, "
                       "word TEXT NOT NULL, aux TEXT NOT NULL, "
                       "weight INTEGER NOT NULL)")
    best = {}
    total = 0
    for path in dict_paths:
        for word, code_field, weight in iter_entries(path):
            if not word:
                continue
            parsed = converter.word_codes(code_field)
            if not parsed:
                continue
            for code, aux in parsed:
                key = (code, word, aux)
                if weight > best.get(key, -1):
                    best[key] = weight
                total += 1
    connection.executemany(
        "INSERT INTO entry VALUES (?, ?, ?, ?)",
        [(code, word, aux, weight) for (code, word, aux), weight in best.items()])
    connection.execute("CREATE INDEX entry_code ON entry(code, weight DESC)")
    connection.commit()
    connection.close()
    return total, len(best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rime-data", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    converter = Converter(load_rules(args.rime_data / "wanxiang_algebra.yaml"))
    dict_paths = imported_dicts(args.rime_data / "wanxiang_pro.dict.yaml")
    total, unique = build_database(dict_paths, converter, args.database)
    print("database: %d input entries, %d unique" % (total, unique))


if __name__ == "__main__":
    main()
