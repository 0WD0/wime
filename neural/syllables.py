"""Syllables as (initial, final, tone) indices, from labels or typed keys.

Index 0 of every field means unknown: a toneless syllable, the final of a half
typed one, or anything outside the inventory. The inventory comes from
context-pinyin.json, which also maps each syllable to its Microsoft
double-pinyin code; other layouts or full pinyin reach the same features.
"""
import ast
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

# The worker bundle ships the generator beside this file.
if not Path(__file__).with_name("wime_gen.py").is_file():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen"))
from wime_gen import SYLLABLE_ENCODING, split_pinyin, syllable_inventory  # noqa: E402

CODES = syllable_inventory()
SPLIT = {syllable: split_pinyin(syllable) for syllable in CODES}
INITIALS = [None] + sorted({initial for initial, _ in SPLIT.values()})
FINALS = [None] + sorted({final for _, final in SPLIT.values()})
TONES = 6  # unknown, 1-4, neutral (5)
INITIAL = {value: i for i, value in enumerate(INITIALS)}
FINAL = {value: i for i, value in enumerate(FINALS)}
KEYS = {code: SPLIT[syllable] for syllable, code in CODES.items()}
# A lone first key names an initial when every syllable it starts agrees.
HALF = {}
for code, (initial, _) in KEYS.items():
    HALF.setdefault(code[0], set()).add(initial)
HALF = {key: initials.pop() for key, initials in HALF.items() if len(initials) == 1}


def labelled(pinyin, tone):
    """Features of a labelled syllable such as ("zhong", 1); tone 0 is unknown."""
    initial, final = SPLIT.get(pinyin) or split_pinyin(pinyin) or (None, None)
    return INITIAL.get(initial, 0), FINAL.get(final, 0), tone


def database_layout(database):
    """Load the selected database's mapping from typed keys to model features."""
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as db:
        meta = dict(db.execute("SELECT key, value FROM meta"))
        if meta.get("syllable_encoding") != SYLLABLE_ENCODING:
            raise ValueError("rebuild the database to include syllable features")
        keys = {code: (initial, final) for code, initial, final
                in db.execute("SELECT code, initial, final FROM syllable")}
    initials = {}
    for code, (initial, _) in keys.items():
        initials.setdefault(code[0], set()).add(initial)
    half = {key: next(iter(values)) for key, values in initials.items() if len(values) == 1}
    return keys, half


def typed(code, tones=(), layout=None):
    """Features of double-pinyin keys, a trailing half syllable included.

    TONES gives (syllable index, tone) pairs typed with the keys."""
    keys, half = (KEYS, HALF) if layout is None else layout
    tone = dict(tones)
    result = []
    for i in range(0, len(code), 2):
        pair = code[i:i + 2]
        initial, final = keys.get(pair, (None, None)) if len(pair) == 2 else (half.get(pair), None)
        result.append((INITIAL.get(initial, 0), FINAL.get(final, 0), tone.get(i // 2, 0)))
    return result


def citation_tones(row):
    """The tones a person types: contextual sandhi undone from reading_notes."""
    tones = list(row["tones"])
    notes = row.get("reading_notes", [])
    # Evaluation cases store the notes as their Python repr.
    for note in ast.literal_eval(notes) if isinstance(notes, str) else notes:
        if "contextual_tone_sandhi" in note["rules"]:
            tones[note["index"]] = note["before_tone"]
    return tones
