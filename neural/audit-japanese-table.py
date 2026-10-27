#!/usr/bin/env python3
"""Check a Japanese reading table against complete local dictionaries.

Reports missing paths, not model ranking. Optional sentence files are measured
separately and never used to add readings. A missing dictionary path exits 1.
"""
import argparse
import hashlib
import json
from pathlib import Path

from japanese import KANA, Table, frequent, hira, jmdict, kanjidic, missing_readings, required_readings


def audit(table, groups, cases=()):
    reports = {}
    for name, readings in groups.items():
        missing = missing_readings(table.index, readings)
        reports[name] = dict(pairs=sum(map(len, readings.values())), missing=len(missing),
                             failures=[dict(text=w, reading=r) for w, r in missing])
    sentences = []
    for path in cases:
        rows = json.loads(path.read_text())
        missing = [row for row in rows if not table.spells(row["input"], row["gold"])]
        sentences.append(dict(file=str(path), total=len(rows), missing=len(missing), failures=missing))
    return dict(sources=reports, sentences=sentences,
                missing=sum(row["missing"] for row in reports.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--jmdict", default="/usr/share/tagainijisho/jmdict.db")
    ap.add_argument("--kanjidic", default="/usr/share/tagainijisho/kanjidic2.db")
    ap.add_argument("--lexicon", type=Path, help="also audit the selected corpus readings")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--cases", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    writings, _ = jmdict(args.jmdict)
    characters = required_readings({}, kanjidic(args.kanjidic, graded=False))
    kana = {c: {hira(c)} for c in KANA}
    for c in kana:
        characters.pop(c, None)
    groups = dict(jmdict=writings, kanjidic=characters, kana=kana)
    stems = required_readings(writings, {})
    # Report additional okurigana stems separately from full dictionary words.
    groups["okurigana"] = {w: rs - writings.get(w, set()) - kana.get(w, set())
                           for w, rs in stems.items() if rs - writings.get(w, set()) - kana.get(w, set())}
    paths = [args.table, Path(args.jmdict), Path(args.kanjidic), *args.cases]
    if args.lexicon:
        selected = {}
        for w, r in frequent(json.loads(args.lexicon.read_text()), args.top):
            selected.setdefault(w, set()).add(r)
        groups["lexicon"] = selected
        paths.append(args.lexicon)
    report = audit(Table(args.table, None), groups, args.cases)
    report["sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in row.items() if k != "failures"}
                      for name, row in report["sources"].items()}, ensure_ascii=False))
    for row in report["sentences"]:
        print(json.dumps({k: v for k, v in row.items() if k != "failures"}, ensure_ascii=False))
    raise SystemExit(bool(report["missing"]))


if __name__ == "__main__":
    main()
