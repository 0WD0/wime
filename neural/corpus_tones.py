"""Validate optional character-aligned contextual tone annotations."""
from collections import Counter
import json
from pathlib import Path
import re

SYLLABLES = json.loads(Path(__file__).with_name("context-pinyin.json").read_text())["syllables"]
NUMBERED = re.compile(r"[a-z]+[1-5]")


def validate_tones(row):
    """Unknown tones are valid; every known tone must retain supporting evidence."""
    size = len(row["gold"][0])
    for field in ("pinyin", "tones", "tone_sources"):
        if not isinstance(row.get(field), list) or len(row[field]) != size:
            raise ValueError(f"{field} must have one entry per gold character")
    evidence = row.get("reading_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("reading_evidence must contain pypinyin and g2pw lists")
    for source in ("pypinyin", "g2pw"):
        values = evidence.get(source)
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"reading_evidence.{source} must align with gold characters")
        for value in values:
            if value is not None and (not isinstance(value, str) or not NUMBERED.fullmatch(value)):
                raise ValueError(f"reading_evidence.{source} must use normalized numbered syllables or null")
    for index, (pinyin, tone, source) in enumerate(zip(row["pinyin"], row["tones"], row["tone_sources"])):
        if pinyin is not None:
            if not isinstance(pinyin, str) or pinyin not in SYLLABLES:
                raise ValueError(f"pinyin[{index}] is not a canonical syllable")
            if SYLLABLES[pinyin] != row["input"][2 * index:2 * index + 2]:
                raise ValueError(f"pinyin[{index}] does not match its input keys")
        if tone is not None and (type(tone) is not int or not 1 <= tone <= 5):
            raise ValueError(f"tones[{index}] must be 1..5 or null")
        if not isinstance(source, str) or not source:
            raise ValueError(f"tone_sources[{index}] must name an annotation source")
        if tone is not None:
            if pinyin is None:
                raise ValueError(f"tones[{index}] needs a selected pinyin syllable")
            if pinyin + str(tone) not in (evidence[name][index] for name in ("pypinyin", "g2pw")):
                raise ValueError(f"tones[{index}] has no matching reading evidence")


def tone_counts(row):
    """Count labels after validation; neutral 5 stays distinct from unknown null."""
    tones = row["tones"]
    known = sum(tone is not None for tone in tones)
    return Counter(total=len(tones), known=known, unknown=len(tones) - known,
                   neutral=sum(tone == 5 for tone in tones))
