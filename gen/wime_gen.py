#!/usr/bin/env python3
"""Generate wime's SQLite word database using a JSON double-pinyin scheme.

By default the database uses schemes/mspy.json. ``--scheme`` selects another
self-contained JSON definition with initial/final keys, zero initials and
complete-syllable aliases (see ``schemes/``).

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

SYLLABLE_ENCODING = "wime-syllable-features-v1"


def syllable_inventory():
    """Read the inventory shipped beside this generator and in worker bundles."""
    return json.loads(Path(__file__).with_name("context-pinyin.json").read_text())["syllables"]


def load_scheme(path):
    """Return a converter and identifier from a build-time JSON definition."""
    scheme = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(scheme, dict) or not isinstance(scheme.get("name"), str)
            or not re.fullmatch(r"[a-z][a-z0-9-]*", scheme["name"])):
        raise ValueError("scheme needs an ASCII name")
    for field in ("initials", "finals"):
        table = scheme.get(field)
        if not isinstance(table, dict) or not table or any(
                not isinstance(k, str) or not isinstance(v, str) or len(v) != 1
                or v not in "abcdefghijklmnopqrstuvwxyz;,./" for k, v in table.items()):
            raise ValueError(f"{field} must map sounds to single code keys")
    for field in ("zero_initials", "syllables"):
        table = scheme.get(field, {})
        if not isinstance(table, dict) or any(
                not isinstance(k, str) or split_pinyin(k) is None or not isinstance(v, list) or not v
                or any(not isinstance(code, str) or len(code) != 2
                       or set(code) - set("abcdefghijklmnopqrstuvwxyz;,./") for code in v)
                for k, v in table.items()):
            raise ValueError(f"{field} must map syllables to complete two-key codes")
    return Converter(scheme=scheme), scheme["name"]


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
    # Syllabic nasals (嗯 ń = en).
    plain = {"n": "en", "ng": "eng", "m": "me"}.get(plain, plain)
    for initial in INITIALS:
        if plain.startswith(initial) and len(plain) > len(initial):
            final = plain[len(initial):]
            if initial in "jqxy" and final[0] == "u":
                final = "v" + final[1:]
            return initial, final
    return "", plain


class Converter:
    """Double-pinyin conversion from a JSON table, with a per-syllable cache."""

    def __init__(self, scheme):
        self.scheme = scheme
        self.cache = {}
        self.aliases = {}
        for pinyin, codes in scheme.get("syllables", {}).items():
            parts = split_pinyin(pinyin)
            if parts in self.aliases and self.aliases[parts] != codes:
                raise ValueError(f"conflicting syllable aliases for {pinyin}")
            self.aliases[parts] = codes

    def codes(self, pinyin):
        if pinyin not in self.cache:
            result = []
            parts = split_pinyin(pinyin)
            if parts:
                aliases = self.aliases.get(parts)
                if aliases is None and not parts[0]:
                    aliases = self.scheme.get("zero_initials", {}).get(parts[1])
                if aliases is not None:
                    result = aliases
                else:
                    initial = self.scheme["initials"].get(parts[0])
                    final = self.scheme["finals"].get(parts[1])
                    if initial and final:
                        result = [initial + final]
            self.cache[pinyin] = sorted(set(result))
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


TONE_MARKS = {"\u0304": "1", "\u0301": "2", "\u030c": "3", "\u0300": "4"}


def syllable_tones(code_field):
    """Tone digit of each syllable of CODE_FIELD; an unmarked one is neutral (5)."""
    return [next((TONE_MARKS[c] for c in unicodedata.normalize("NFD", syllable.partition(";")[0])
                  if c in TONE_MARKS), "5")
            for syllable in code_field.split()]


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
            # Releases ship optional tables commented out: "  #- dicts/yixue.pro".
            if line.lstrip().startswith("#"):
                continue
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
    # TONES joins, per syllable, every tone digit the readings of a row use.
    connection.execute("CREATE TABLE entry (code TEXT NOT NULL, "
                       "word TEXT NOT NULL, aux TEXT NOT NULL, "
                       "weight INTEGER NOT NULL, tones TEXT NOT NULL)")
    best, tones = {}, {}
    total = 0
    for path in dict_paths:
        for word, code_field, weight in iter_entries(path):
            if not word:
                continue
            parsed = converter.word_codes(code_field)
            if not parsed:
                continue
            marks = syllable_tones(code_field)
            for code, aux in parsed:
                key = (code, word, aux)
                if weight > best.get(key, -1):
                    best[key] = weight
                for seen, tone in zip(tones.setdefault(key, [set() for _ in marks]), marks):
                    seen.add(tone)
                total += 1
    connection.executemany(
        "INSERT INTO entry VALUES (?, ?, ?, ?, ?)",
        [(code, word, aux, weight, "/".join("".join(sorted(seen)) for seen in tones[key]))
         for key, weight in best.items() for code, word, aux in [key]])
    connection.execute("CREATE INDEX entry_code ON entry(code, weight DESC)")
    # Use the model's fixed syllable inventory and collision order for every
    # layout. Primary codes take precedence over aliases of other syllables.
    inventory = syllable_inventory()
    features = {code: split_pinyin(pinyin) for pinyin in inventory
                if (code := converter.primary(pinyin))}
    for pinyin in inventory:
        for code in converter.codes(pinyin):
            features.setdefault(code, split_pinyin(pinyin))
    connection.execute("CREATE TABLE syllable (code TEXT PRIMARY KEY, initial TEXT, final TEXT)")
    connection.executemany("INSERT INTO syllable VALUES (?, ?, ?)",
                           [(code, *parts) for code, parts in sorted(features.items())])
    initial_keys, final_keys = converter.position_keys()
    connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany("INSERT INTO meta VALUES (?, ?)",
                           [("scheme", name), ("initial_keys", initial_keys),
                            ("final_keys", final_keys), ("syllable_encoding", SYLLABLE_ENCODING)])
    connection.commit()
    connection.close()
    return total, len(best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rime-data", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--scheme", type=Path,
                        default=Path(__file__).resolve().parents[1] / "schemes/mspy.json",
                        help="JSON scheme definition; default: schemes/mspy.json")
    args = parser.parse_args()
    converter, name = load_scheme(args.scheme)
    dict_paths = imported_dicts(args.rime_data / "wanxiang_pro.dict.yaml")
    total, unique = build_database(dict_paths, converter, args.database, name)
    print("database %s: %d input entries, %d unique; keys %s | %s"
          % (name, total, unique, *converter.position_keys()))


if __name__ == "__main__":
    main()
