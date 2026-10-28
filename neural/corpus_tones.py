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
READING_SCHEMA = "wime-reading-notes-v1"
LEGACY_FIELDS = ("tone_sources", "reading_rules", "reading_evidence")


def _validate_legacy_tones(row):
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


def reading_notes(reading, start, end):
    """Store only applied rules and changed tones, relative to the sampled span.

    Dictionary matches can confirm a reading without changing it. Preserve that
    provenance too. The dense frontend stages are transient inference results,
    not additional copies of the final labels in the corpus.
    """
    notes = []
    evidence = reading["reading_evidence"]
    for index in range(start, end):
        rules = reading["reading_rules"][index]
        if rules:
            note = dict(index=index - start, rules=rules)
            before, final = evidence["frontend_before_sandhi"][index], evidence["final"][index]
            if before != final:
                note["before_tone"] = int(before[-1])
            notes.append(note)
    return notes


def compact_tones(row):
    """Validate archived dense labels before losslessly migrating their evidence."""
    if "reading_notes" in row:
        raise ValueError("dense and sparse reading evidence cannot be mixed")
    _validate_legacy_tones(row)
    result = {key: value for key, value in row.items() if key not in LEGACY_FIELDS}
    notes = reading_notes(row, 0, len(row["tones"]))
    if notes:
        result["reading_notes"] = notes
    validate_tones(result)
    return result


def validate_tones(row, schema=None):
    """Check final labels and sparse evidence; archived dense rows remain readable."""
    if schema not in (None, READING_SCHEMA):
        raise ValueError(f"unsupported reading schema: {schema}")
    if any(field in row for field in LEGACY_FIELDS):
        if schema == READING_SCHEMA or "reading_notes" in row:
            raise ValueError("dense and sparse reading evidence cannot be mixed")
        return _validate_legacy_tones(row)
    size = len(row["gold"][0])
    if not isinstance(row.get("input"), str) or len(row["input"]) != 2 * size:
        raise ValueError("input must contain exactly one key pair per gold character")
    for field in ("pinyin", "tones"):
        if not isinstance(row.get(field), list) or len(row[field]) != size:
            raise ValueError(f"{field} must have one entry per gold character")
    for index, (pinyin, tone) in enumerate(zip(row["pinyin"], row["tones"])):
        if not isinstance(pinyin, str) or pinyin not in SYLLABLES:
            raise ValueError(f"pinyin[{index}] is not a canonical syllable")
        if SYLLABLES[pinyin] != row["input"][2 * index:2 * index + 2]:
            raise ValueError(f"pinyin[{index}] does not match its input keys")
        if type(tone) is not int or not 1 <= tone <= 5:
            raise ValueError(f"tones[{index}] must be 1..5")
    notes = row.get("reading_notes", [])
    if not isinstance(notes, list):
        raise ValueError("reading_notes must be a list")
    previous = -1
    for note in notes:
        if not isinstance(note, dict) or not {"index", "rules"} <= note.keys() <= {"index", "rules", "before_tone"}:
            raise ValueError("reading note requires index, rules and optional before_tone")
        index, rules = note["index"], note["rules"]
        if type(index) is not int or not previous < index < size:
            raise ValueError("reading note indices must be ordered, unique and inside the target")
        previous = index
        if (not isinstance(rules, list) or not rules
                or any(not isinstance(rule, str) or
                       (rule != SANDHI and not any(rule.startswith(prefix) and len(rule) > len(prefix)
                                                  for prefix in CORRECTIONS)) for rule in rules)
                or len(set(rules)) != len(rules)):
            raise ValueError("reading note requires distinct supported rules")
        if (SANDHI in rules) != ("before_tone" in note):
            raise ValueError("sandhi requires a before_tone and vice versa")
        if "before_tone" in note:
            before = note["before_tone"]
            if type(before) is not int or not 1 <= before <= 5 or before == row["tones"][index]:
                raise ValueError("before_tone must be 1..5 and differ from the final tone")


def tone_source_counts(row):
    """Derive source statistics instead of storing another per-character array."""
    if "tone_sources" in row:
        return Counter(row["tone_sources"])
    counts = Counter(mandarin_frontend=len(row["tones"]))
    for note in row.get("reading_notes", []):
        counts["mandarin_frontend"] -= 1
        counts[SANDHI if "before_tone" in note else "phrase_correction"] += 1
    return +counts


def tone_counts(row):
    """Count complete labels and the neutral-tone subset after validation."""
    tones = row["tones"]
    return Counter(total=len(tones), neutral=tones.count(5))
