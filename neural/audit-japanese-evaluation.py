#!/usr/bin/env python3
"""Audit frozen Japanese evaluation labels and make a separate corrected version.

No model predictions are used to create or apply corrections. Analyzer and
reading-table disagreements are evidence for review, not automatic label edits.
The original case identities and ordering remain intact, including duplicates.
"""
import argparse
import collections
import copy
import csv
import difflib
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import unicodedata


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(row):
    return row["suite"], row["case_index"], str(row["case_id"])


def nfkc(text):
    return unicodedata.normalize("NFKC", text)


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    seen = set()
    for row in cases:
        if not isinstance(row, dict) or any(k not in row for k in
                ("suite", "case_index", "case_id", "kana", "context", "answers")):
            raise ValueError("case is missing an identity, input, context or answers")
        if not isinstance(row["suite"], str) or not row["suite"] or type(row["case_index"]) is not int:
            raise ValueError("invalid suite or case index")
        key = row["suite"], row["case_index"]
        if key in seen:
            raise ValueError(f"duplicate case identity: {key}")
        seen.add(key)
        if not isinstance(row["kana"], str) or not row["kana"] or not isinstance(row["context"], str):
            raise ValueError(f"invalid input/context: {key}")
        if not isinstance(row["answers"], list) or not row["answers"] or not all(
                isinstance(answer, str) and answer for answer in row["answers"]):
            raise ValueError(f"invalid answers: {key}")


def apply_corrections(cases, review):
    validate_cases(cases)
    updated = copy.deepcopy(cases)
    by_key = {identity(row): row for row in updated}
    changed, seen = [], set()
    for item in review.get("corrections", []):
        key = identity(item)
        if key in seen or key not in by_key:
            raise ValueError(f"unknown or repeated correction: {key}")
        seen.add(key)
        row = by_key[key]
        before, after = item.get("before", {}), item.get("after", {})
        if not after or set(before) != set(after) or not set(after) <= {"kana", "answers"}:
            raise ValueError(f"correction must explicitly guard all changed labels: {key}")
        if not item.get("reason") or not item.get("evidence"):
            raise ValueError(f"correction needs a reason and source evidence: {key}")
        if any(row[field] != value for field, value in before.items()):
            raise ValueError(f"stale correction does not match original: {key}")
        if before == after:
            raise ValueError(f"correction has no change: {key}")
        row.update(copy.deepcopy(after))
        changed.append(copy.deepcopy(item))
    validate_cases(updated)
    return updated, changed


def structure_report(cases):
    validate_cases(cases)
    groups, unusual, normalized = collections.defaultdict(list), [], []
    for row in cases:
        key = dict(suite=row["suite"], case_index=row["case_index"], case_id=row["case_id"])
        groups[(nfkc(row["context"]), nfkc(row["kana"]))].append(key)
        for field, texts in (("kana", [row["kana"]]), ("context", [row["context"]]),
                             ("answers", row["answers"])):
            chars = sorted({c for text in texts for c in text
                            if unicodedata.category(c).startswith("C") and c not in "\n\t"
                            or field == "kana" and c in "ゕゖ"})
            if chars:
                unusual.append(dict(key, field=field, characters=chars,
                                    codepoints=[f"U+{ord(c):04X}" for c in chars]))
        if len({nfkc(a) for a in row["answers"]}) < len(row["answers"]):
            normalized.append(key)
    return dict(total=len(cases), suites=dict(collections.Counter(r["suite"] for r in cases)),
                duplicate_inputs=[members for members in groups.values() if len(members) > 1],
                unusual_characters=unusual, nfkc_equivalent_answer_lists=normalized)


def read_transcriptions(path):
    result = {}
    with Path(path).open() as stream:
        for row in csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 5:
                result[row[0]] = row[4]
    return result


def source_report(cases, transcriptions):
    # Reproduce the original source parser, including its known small-ke behavior.
    path = Path(__file__).with_name("prepare-japanese.py")
    spec = importlib.util.spec_from_file_location("prepare_japanese_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    for row in cases:
        if row["suite"] != "tatoeba":
            continue
        raw = transcriptions.get(str(row["case_id"]))
        item = dict(suite=row["suite"], case_index=row["case_index"], case_id=row["case_id"])
        if raw is None:
            records.append(dict(item, issue="source_transcription_missing"))
            continue
        try:
            surface, reading, _ = module.parse(raw)
        except (ValueError, IndexError) as error:
            records.append(dict(item, issue="source_parse_failure", raw=raw, error=str(error)))
            continue
        if surface not in row["answers"] or reading != row["kana"]:
            records.append(dict(item, issue="source_differs", raw=raw, surface=surface, reading=reading))
        if "|]" in raw:
            records.append(dict(item, issue="empty_source_reading", raw=raw))
    return records


def analyzer_report(cases):
    import fugashi
    tagger = fugashi.Tagger()
    hira = lambda text: "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)
    records = []
    for row in cases:
        # Analyzer output is only a second opinion. Preserve visible kana and
        # literal punctuation; the analyzer cannot supply all foreign-name and
        # numerical readings and frequently chooses inappropriate alternatives.
        text = row["answers"][0]
        reading = "".join(hira((word.feature.kana or word.surface)
                             if any("㐀" <= c <= "鿿" for c in word.surface) else word.surface)
                          for word in tagger(text))
        if reading != row["kana"]:
            edits = [dict(input=row["kana"][a:b], analyzer=reading[c:d])
                     for op, a, b, c, d in difflib.SequenceMatcher(
                         None, row["kana"], reading, autojunk=False).get_opcodes() if op != "equal"]
            records.append(dict(suite=row["suite"], case_index=row["case_index"], case_id=row["case_id"],
                                kana=row["kana"], answer=text, analyzer_reading=reading, differences=edits))
    return dict(method="UniDic kana on kanji-bearing tokens; no automatic corrections",
                fugashi=importlib.metadata.version("fugashi"),
                dictionaries=tagger.dictionary_info, differences=records)


def coverage_report(cases, path):
    from japanese import Table
    table = Table(path, None)
    missing = [dict(suite=r["suite"], case_index=r["case_index"], case_id=r["case_id"],
                    kana=r["kana"], answers=r["answers"])
               for r in cases if not any(table.spells(r["kana"], answer) for answer in r["answers"])]
    return dict(checked=len(cases), unsupported=len(missing), cases=missing,
                meaning="Lack of a table path is not proof of an incorrect reference reading")


def write_json(path, data):
    encoded = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError(f"refusing to replace a different audit artifact: {path}")
    path.write_text(encoded)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--corrections", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--transcriptions", type=Path)
    ap.add_argument("--table", type=Path, help="optional production reading-table coverage check; needs torch")
    ap.add_argument("--unidic", action="store_true", help="optional independent reading disagreement inventory")
    args = ap.parse_args()
    cases, review = json.loads(args.cases.read_text()), json.loads(args.corrections.read_text())
    if review.get("source_cases_sha256") != digest(args.cases):
        raise ValueError("correction review is for a different frozen case file")
    updated, changed = apply_corrections(cases, review)
    report = dict(version=1, original=structure_report(cases), corrected=structure_report(updated),
                  changes=len(changed), review_scope=review.get("review_scope"),
                  pending_review=review.get("pending_review", []),
                  reviewed_coverage=review.get("reviewed_coverage", []))
    paths = [args.cases, args.corrections, Path(__file__)]
    if args.transcriptions:
        transcriptions = read_transcriptions(args.transcriptions)
        report["source_checks"] = dict(original=source_report(cases, transcriptions),
                                       corrected=source_report(updated, transcriptions))
        paths.extend([args.transcriptions, Path(__file__).with_name("prepare-japanese.py")])
    if args.unidic:
        report["analyzer"] = analyzer_report(cases)
    if args.table:
        report["table_coverage"] = dict(original=coverage_report(cases, args.table),
                                        corrected=coverage_report(updated, args.table))
        paths.extend([args.table, Path(__file__).with_name("japanese.py")])
    args.out.mkdir(parents=True, exist_ok=True)
    if args.out.resolve() == args.cases.parent.resolve():
        raise ValueError("use a separate directory; original benchmark remains frozen")
    write_json(args.out / "corrected-cases.json", updated)
    write_json(args.out / "changes.json", changed)
    write_json(args.out / "audit.json", report)
    manifest = dict(version=1, purpose="Versioned label audit, not a new blind evaluation set",
                    identity_policy="Preserve suite, case_index, case_id, order and repeated inputs",
                    normalization="NFKC complete-string match against any accepted answer",
                    inputs={str(p.resolve()): digest(p) for p in paths},
                    outputs={name: digest(args.out / name) for name in
                             ("corrected-cases.json", "changes.json", "audit.json")},
                    count=len(cases), changed_cases=len(changed),
                    review_limit="Agent linguistic review, not certification by a native human annotator")
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(dict(cases=len(cases), changed=len(changed),
                          pending_review=len(report["pending_review"]),
                          duplicate_inputs=len(report["corrected"]["duplicate_inputs"]),
                          out=str(args.out)), ensure_ascii=False))


if __name__ == "__main__":
    main()
