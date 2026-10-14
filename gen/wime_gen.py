#!/usr/bin/env python3
"""Generate wime's SQLite word database from the pinned rime-wanxiang sources.

By default the database is keyed by the Microsoft double-pinyin code, a
faithful port of the `pro/微软双拼` rules in `wanxiang_algebra.yaml`.  With
``--scheme`` the code comes from a JSON table that assigns every initial and
final to one key (see ``schemes/``).

Either way a ``meta`` table records which keys occur in the initial and final
position of a syllable, so the engine can tell a code key from punctuation.
"""

import argparse
import itertools
import json
import re
import sqlite3
import unicodedata
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


INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k",
            "h", "j", "q", "x", "r", "z", "c", "s", "y", "w")


def split_pinyin(pinyin):
    """Toned pinyin -> (initial, final), ü written v; None if not a syllable.

    After j/q/x/y a written ``u`` is ü, so it is spelled ``v`` (ju -> j + v,
    yuan -> y + van), matching the scheme tables.
    """
    plain = unicodedata.normalize("NFD", pinyin).replace("u\u0308", "v")
    plain = "".join(c for c in plain if c.isascii()).lower()
    if not re.fullmatch(r"[a-z]+", plain):
        return None
    # Syllabic nasals, spelled the way the wanxiang algebra does (嗯 ń = en).
    plain = {"n": "en", "ng": "eng", "m": "me"}.get(plain, plain)
    for initial in INITIALS:
        if plain.startswith(initial) and len(plain) > len(initial):
            final = plain[len(initial):]
            if initial in "jqxy" and final[0] == "u":
                final = "v" + final[1:]
            return initial, final
    return "", plain


class Converter:
    """Double-pinyin conversion with a per-syllable cache.

    ``rules`` is a list of compiled algebra rules; ``scheme`` a table dict
    with ``initials`` and ``finals``.  Exactly one of them is used.
    """

    def __init__(self, rules=None, scheme=None):
        self.rules = rules
        self.scheme = scheme
        self.cache = {}

    def codes(self, pinyin):
        if pinyin not in self.cache:
            result = set()
            if self.scheme is not None:
                parts = split_pinyin(pinyin)
                if parts:
                    initial = self.scheme["initials"].get(parts[0])
                    final = self.scheme["finals"].get(parts[1])
                    if initial and final:
                        result.add(initial + final)
            else:
                for spelling in apply_rules(pinyin + ";", self.rules):
                    code = _extract_code(spelling)
                    if code and all(ch.isascii() for ch in code):
                        result.add(code)
            self.cache[pinyin] = sorted(result)
        return self.cache[pinyin]

    def position_keys(self):
        """Keys seen in the first and second position of syllable codes."""
        first, second = set(), set()
        for codes in self.cache.values():
            for code in codes:
                if len(code) == 2:
                    first.add(code[0])
                    second.add(code[1])
        return "".join(sorted(first)), "".join(sorted(second))

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


def build_database(dict_paths, converter, output, name):
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
    initial_keys, final_keys = converter.position_keys()
    connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany("INSERT INTO meta VALUES (?, ?)",
                           [("scheme", name), ("initial_keys", initial_keys),
                            ("final_keys", final_keys)])
    connection.commit()
    connection.close()
    return total, len(best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rime-data", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--scheme", type=Path,
                        help="JSON scheme table; default: wanxiang 微软双拼 rules")
    args = parser.parse_args()
    if args.scheme:
        scheme = json.loads(args.scheme.read_text(encoding="utf-8"))
        converter, name = Converter(scheme=scheme), scheme["name"]
    else:
        converter = Converter(load_rules(args.rime_data / "wanxiang_algebra.yaml"))
        name = "mspy"
    dict_paths = imported_dicts(args.rime_data / "wanxiang_pro.dict.yaml")
    total, unique = build_database(dict_paths, converter, args.database, name)
    print("database %s: %d input entries, %d unique; keys %s | %s"
          % (name, total, unique, *converter.position_keys()))


if __name__ == "__main__":
    main()
