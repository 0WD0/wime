"""Validate complete character-aligned labels from the Mandarin frontend."""
from collections import Counter
import json
from pathlib import Path
import re

SYLLABLES = json.loads(Path(__file__).with_name("context-pinyin.json").read_text())["syllables"]
NUMBERED = re.compile(r"[a-z]+[1-5]")
TONE_POLICY = dict(name="gpt-sovits-mandarin-v1", neutral=5)
SANDHI = "contextual_tone_sandhi"
CORRECTIONS = ("phrase_correction:", "character_correction:", "pos_correction:")


def validate_tones(row):
    """Validate one frontend's before/final stages and aligned rule provenance."""
    size = len(row["gold"][0])
    if not isinstance(row.get("input"), str) or len(row["input"]) != 2 * size:
        raise ValueError("input must contain exactly one key pair per gold character")
    for field in ("pinyin", "tones", "tone_sources", "reading_rules"):
        if not isinstance(row.get(field), list) or len(row[field]) != size:
            raise ValueError(f"{field} must have one entry per gold character")
    evidence = row.get("reading_evidence")
    stages = ("frontend_before_sandhi", "final")
    if not isinstance(evidence, dict) or set(evidence) != set(stages):
        raise ValueError("reading_evidence must contain frontend_before_sandhi and final lists")
    for stage in stages:
        values = evidence.get(stage)
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"reading_evidence.{stage} must align with gold characters")
        for value in values:
            if not isinstance(value, str) or not NUMBERED.fullmatch(value):
                raise ValueError(f"reading_evidence.{stage} must use complete normalized numbered syllables")
    for index, (pinyin, tone, source) in enumerate(zip(row["pinyin"], row["tones"], row["tone_sources"])):
        if not isinstance(pinyin, str) or pinyin not in SYLLABLES:
            raise ValueError(f"pinyin[{index}] is not a canonical syllable")
        if SYLLABLES[pinyin] != row["input"][2 * index:2 * index + 2]:
            raise ValueError(f"pinyin[{index}] does not match its input keys")
        if type(tone) is not int or not 1 <= tone <= 5:
            raise ValueError(f"tones[{index}] must be 1..5")
        rules = row["reading_rules"][index]
        if not isinstance(rules, list) or any(not isinstance(rule, str) or not rule for rule in rules):
            raise ValueError(f"reading_rules[{index}] must be a list of rule names")
        before, final = (evidence[stage][index] for stage in stages)
        if pinyin + str(tone) != final:
            raise ValueError(f"tones[{index}] does not match the final frontend reading")
        if before[:-1] != pinyin:
            raise ValueError(f"reading_evidence[{index}] changes syllables during tone sandhi")
        if any(rule != SANDHI and not any(rule.startswith(prefix) and len(rule) > len(prefix)
                                         for prefix in CORRECTIONS) for rule in rules):
            raise ValueError(f"reading_rules[{index}] contains an unsupported rule")
        changed = before != final
        expected_source = SANDHI if changed else "phrase_correction" if rules else "mandarin_frontend"
        if source != expected_source or (SANDHI in rules) != changed:
            raise ValueError(f"tone_sources[{index}] and reading_rules must match actual corrections and tone changes")


def tone_counts(row):
    """Count complete labels and the neutral-tone subset after validation."""
    tones = row["tones"]
    return Counter(total=len(tones), neutral=tones.count(5))
