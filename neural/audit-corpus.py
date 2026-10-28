#!/usr/bin/env python3
"""Audit completed corpus v2 files and sample rows for manual reading review.

Dictionary membership checks key reachability, not contextual pronunciation.
The sampled syllables are possible spellings of each key pair, not adjudicated
readings. Identity checks use temporary SQLite storage rather than RAM sets.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import sys
import tempfile

from corpus_integrity import file_metadata, sample_key, validate_annotation_window
from corpus_tones import tone_counts, tone_policy_strict, validate_tones
from replay import Protection

SPLITS = ("train", "dev", "test")
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def legal_keys(database):
    result = defaultdict(set)
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for word, code in db.execute("SELECT word, code FROM entry WHERE length(word)=1 AND length(code)=2"):
            if HAN.fullmatch(word):
                result[word].add(code)
    return result


def audit(corpus, database, exclude_cases=(), sample_per_source=20, seed="wime-corpus-v2-audit"):
    errors, examples = Counter(), defaultdict(list)
    counts, lengths, blanks = Counter(), Counter(), Counter()
    tones, tone_sources = defaultdict(Counter), defaultdict(Counter)
    samples, sampled, randoms = defaultdict(list), Counter(), {}
    outputs, total = {}, 0

    def error(kind, detail):
        errors[kind] += 1
        if len(examples[kind]) < 5:
            examples[kind].append(detail)

    manifest_path = corpus / "manifest.json"
    manifest = {}
    try:
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
    except (OSError, ValueError) as exc:
        error("manifest", str(exc))
        manifest = {}
    if manifest and manifest.get("format") != "wime-corpus-v2":
        error("manifest_format", str(manifest.get("format")))
    has_tones = "tone_policy" in manifest
    strict_tones = False
    if has_tones:
        try:
            strict_tones = tone_policy_strict(manifest["tone_policy"])
        except ValueError as exc:
            error("tone_policy", str(exc))
    metadata = manifest.get("outputs", {})
    if not isinstance(metadata, dict):
        metadata = {}
        error("manifest_outputs", "outputs must be an object")

    keys = legal_keys(database)
    database_metadata = file_metadata(database)
    provenance = manifest.get("provenance", {})
    has_cleaned = isinstance(provenance, dict) and "cleaned_corpus" in provenance
    expected_cleaned = provenance.get("cleaned_corpus") if has_cleaned else None
    expected_database = provenance.get("database", {}) if isinstance(provenance, dict) else {}
    if not isinstance(expected_database, dict) or expected_database.get("sha256") != database_metadata["sha256"]:
        error("database_fingerprint", "database differs from the completed corpus manifest or lacks a fingerprint")
    expected_assets = provenance.get("reading_assets", {}) if isinstance(provenance, dict) else {}
    reading_assets = {}
    if not isinstance(expected_assets, dict) or not expected_assets:
        error("reading_assets_unpinned", "provenance.reading_assets must be a nonempty mapping")
        expected_assets = {}
    for name, expected in expected_assets.items():
        if (not isinstance(expected, dict)
                or not isinstance(expected.get("path"), str) or not expected["path"]
                or not isinstance(expected.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])
                or type(expected.get("bytes")) is not int or expected["bytes"] < 0):
            error("reading_asset_unpinned", name)
            continue
        try:
            actual = file_metadata(Path(expected["path"]))
        except (OSError, ValueError) as exc:
            error("reading_asset_missing", dict(asset=name, path=expected["path"], message=str(exc)))
            continue
        reading_assets[name] = actual
        if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
            error("reading_asset_changed", dict(asset=name, expected=expected, actual=actual))
    expected_protection = manifest.get("protection", {})
    if not isinstance(expected_protection, dict):
        error("manifest_protection", "protection must map paths to fingerprints")
        expected_protection = {}
    protection = Protection([])
    paths = {Path(path).resolve() for path in exclude_cases}
    for name, expected in expected_protection.items():
        path = Path(name).resolve()
        paths.add(path)
        try:
            actual = file_metadata(path)["sha256"]
        except OSError as exc:
            error("protection_fingerprint", dict(path=str(path), message=str(exc)))
        else:
            if actual != expected:
                error("protection_fingerprint", dict(path=str(path), expected=expected, actual=actual))
    for path in sorted(paths):
        try:
            part = Protection([path])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            error("protection_file", dict(path=str(path), message=str(exc)))
            continue
        protection.documents.update(part.documents)
        protection.exact.update(part.exact)
        protection.fragments.update(part.fragments)
        protection.files.update(part.files)
    inverse = defaultdict(list)
    syllables = json.loads(Path(__file__).with_name("context-pinyin.json").read_text())["syllables"]
    for syllable, code in sorted(syllables.items()):
        inverse[code].append(syllable)

    with ExitStack() as resources, tempfile.TemporaryDirectory(prefix="wime-corpus-audit-") as temporary:
        cleaned = None
        if has_cleaned:
            try:
                from corpus_cleaned import CleanedDocuments
                if (not isinstance(expected_cleaned, dict)
                        or not isinstance(expected_cleaned.get("path"), str) or not expected_cleaned["path"]
                        or not isinstance(expected_cleaned.get("sha256"), str)
                        or not re.fullmatch(r"[0-9a-f]{64}", expected_cleaned["sha256"])
                        or type(expected_cleaned.get("bytes")) is not int or expected_cleaned["bytes"] < 0):
                    raise ValueError("provenance.cleaned_corpus must pin the cleaned manifest path, hash and bytes")
                clean_manifest = Path(expected_cleaned["path"])
                actual = file_metadata(clean_manifest)
                if any(actual[key] != expected_cleaned[key] for key in ("sha256", "bytes")):
                    raise ValueError("cleaned corpus manifest differs from its recorded fingerprint")
                cleaned = resources.enter_context(CleanedDocuments(clean_manifest.parent))
            except (OSError, ValueError, sqlite3.Error) as exc:
                error("cleaned_text", str(exc))
        with closing(sqlite3.connect(str(Path(temporary) / "identities.sqlite3"))) as seen:
            seen.execute("CREATE TABLE identity (kind TEXT, key TEXT, split TEXT, PRIMARY KEY(kind,key)) WITHOUT ROWID")

            def identity(kind, key, split, where):
                inserted = seen.execute("INSERT OR IGNORE INTO identity VALUES (?,?,?)", (kind, key, split)).rowcount
                if inserted:
                    return
                if kind in ("id", "sample"):
                    error("duplicate_" + kind, where)
                else:
                    original = seen.execute("SELECT split FROM identity WHERE kind=? AND key=?", (kind, key)).fetchone()[0]
                    if original != split:
                        error(kind + "_cross_split", dict(location=where, first_split=original, split=split))

            for split in SPLITS:
                path = corpus / f"{split}.jsonl"
                expected = metadata.get(split)
                if not isinstance(expected, dict) or not expected.get("sha256") or "bytes" not in expected:
                    error("output_metadata", split)
                    expected = {}
                if not path.is_file():
                    error("missing_output", str(path))
                    continue
                digest, size, lines = hashlib.sha256(), 0, 0
                with path.open("rb") as stream:
                    for lines, raw in enumerate(stream, 1):
                        digest.update(raw)
                        size += len(raw)
                        total += 1
                        where = f"{split}:{lines}"
                        try:
                            row = json.loads(raw)
                        except (ValueError, UnicodeDecodeError) as exc:
                            error("invalid_json", dict(location=where, message=str(exc)))
                            continue
                        fields = ("id", "document", "group", "source", "split", "context", "input")
                        if (not isinstance(row, dict)
                                or any(not isinstance(row.get(k), str) for k in fields)
                                or any(not row[k] for k in fields if k != "context")
                                or not isinstance(row.get("gold"), list) or len(row["gold"]) != 1
                                or not isinstance(row["gold"][0], str)):
                            error("row_schema", where)
                            continue
                        source, text = row["source"], row["gold"][0]
                        counts[f"{source}/{split}"] += 1
                        lengths[str(len(text))] += 1
                        blanks[f"{source}/{split}"] += not bool(row["context"])
                        if row["split"] != split:
                            error("split_label", where)
                        for kind in ("id", "group", "document"):
                            identity(kind, row[kind], split, where)
                        if "content" in row:
                            if not isinstance(row["content"], str) or not row["content"]:
                                error("content_schema", where)
                            else:
                                identity("content", row["content"], split, where)
                        identity("sample", sample_key(row), split, where)
                        if not 1 <= len(text) <= 24 or not HAN.fullmatch(text):
                            error("gold_shape", where)
                        if len(row["context"]) > 64:
                            error("context_length", where)
                        if len(row["input"]) != 2 * len(text):
                            error("input_length", where)
                        elif any(row["input"][2*i:2*i+2] not in keys.get(char, ()) for i, char in enumerate(text)):
                            error("illegal_reading_key", where)
                        if has_tones:
                            try:
                                validate_tones(row, strict=strict_tones)
                            except ValueError as exc:
                                error("tone_schema", dict(location=where, message=str(exc)))
                            else:
                                tones[source].update(tone_counts(row))
                                tone_sources[source].update(row["tone_sources"])
                        if "annotation_window" in manifest:
                            try:
                                validate_annotation_window(row, manifest["annotation_window"])
                            except ValueError as exc:
                                error("annotation_window", dict(location=where, message=str(exc)))
                        if cleaned is not None:
                            try:
                                cleaned.validate_row(row)
                            except ValueError as exc:
                                error("cleaned_text", dict(location=where, message=str(exc)))
                        if protection.contains(row):
                            error("protected_overlap", where)
                        sampled[source] += 1
                        if source not in randoms:
                            randoms[source] = random.Random(seed + ":" + source)
                        position = randoms[source].randrange(sampled[source])
                        if len(samples[source]) < sample_per_source:
                            samples[source].append(row)
                        elif position < sample_per_source:
                            samples[source][position] = row
                        if lines % 10000 == 0:
                            seen.commit()
                outputs[split] = dict(sha256=digest.hexdigest(), bytes=size, rows=lines)
                if expected.get("sha256") != outputs[split]["sha256"] or expected.get("bytes") != size:
                    error("output_fingerprint", split)
                seen.commit()

    stats = manifest.get("stats", {})
    if isinstance(stats, dict):
        expected_counts = {key: value for key, value in stats.items() if key.rsplit("/", 1)[-1] in SPLITS}
        for key in sorted(counts.keys() | expected_counts.keys()):
            if counts[key] != expected_counts.get(key, 0):
                error("manifest_count", dict(key=key, observed=counts[key], expected=expected_counts.get(key, 0)))
        if "blank_context" in stats and stats["blank_context"] != sum(blanks.values()):
            error("manifest_blank_count", dict(observed=sum(blanks.values()), expected=stats["blank_context"]))
        if has_tones:
            for kind in ("known", "unknown", "neutral"):
                key = f"tone/{kind}"
                observed = sum(source[kind] for source in tones.values())
                if key in stats and stats[key] != observed:
                    error("manifest_tone_count", dict(key=key, observed=observed, expected=stats[key]))
    else:
        error("manifest_stats", "stats must be an object")
    for source_rows in samples.values():
        for row in source_rows:
            row["key_syllable_options"] = [inverse.get(row["input"][i:i+2], []) for i in range(0, len(row["input"]), 2)]
    report = dict(format="wime-corpus-v2-audit", ok=not errors, rows=total,
                  counts=dict(sorted(counts.items())), lengths=dict(sorted(lengths.items(), key=lambda item: int(item[0]))),
                  blank_context=dict(sorted(blanks.items())), outputs=outputs,
                  errors=dict(sorted(errors.items())), error_examples=dict(examples),
                  database=database_metadata, reading_assets=reading_assets, protection=protection.files,
                  reading_accuracy="Not measured; dictionary key legality does not establish contextual reading correctness.")
    if has_tones:
        report["tone_policy"] = manifest["tone_policy"]
        report["tones"] = {source: dict(counts, sources=dict(sorted(tone_sources[source].items())))
                           for source, counts in sorted(tones.items())}
    if "annotation_window" in manifest:
        report["annotation_window"] = manifest["annotation_window"]
    if has_cleaned:
        report["cleaned_corpus"] = expected_cleaned
    if manifest_path.is_file():
        report["manifest"] = file_metadata(manifest_path)
    review = dict(seed=seed, sample_per_source=sample_per_source,
                  note="Fixed-seed reservoir samples for manual review. Syllable options decode keys, not contextual pronunciation.",
                  sources=dict(sorted(samples.items())))
    return report, review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="report directory; defaults to the corpus directory")
    parser.add_argument("--exclude-cases", type=Path, action="append", default=[],
                        help="additional held-out cases; manifest protection files are always checked")
    parser.add_argument("--sample-per-source", type=int, default=20)
    parser.add_argument("--seed", default="wime-corpus-v2-audit")
    args = parser.parse_args(argv)
    if args.sample_per_source < 0:
        parser.error("sample-per-source must be nonnegative")
    report, samples = audit(args.corpus, args.database, args.exclude_cases, args.sample_per_source, args.seed)
    out = args.out or args.corpus
    out.mkdir(parents=True, exist_ok=True)
    for filename, value in (("audit.json", report), ("audit-samples.json", samples)):
        (out / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(ok=report["ok"], rows=report["rows"], errors=report["errors"], report=str(out / "audit.json"))))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
