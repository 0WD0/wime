#!/usr/bin/env python3
"""Create a new corpus version with one final reading and sparse provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from corpus_integrity import empty_output, file_metadata
from corpus_tones import LEGACY_FIELDS, READING_SCHEMA, compact_tones
from dataset_registry import Registry

SPLITS = ("train", "dev", "test")


def verify_row(original, compact):
    """Reconstruct the old evidence to check the complete migration, not just labels."""
    labels = {key: value for key, value in original.items() if key not in LEGACY_FIELDS}
    if labels != {key: value for key, value in compact.items() if key != "reading_notes"}:
        raise ValueError("compaction changed sample identity, text or final labels")
    final = [pinyin + str(tone) for pinyin, tone in zip(compact["pinyin"], compact["tones"])]
    before, rules = list(final), [[] for _ in final]
    sources = ["mandarin_frontend"] * len(final)
    for note in compact.get("reading_notes", []):
        index = note["index"]
        rules[index] = note["rules"]
        sources[index] = "phrase_correction"
        if "before_tone" in note:
            before[index] = compact["pinyin"][index] + str(note["before_tone"])
            sources[index] = "contextual_tone_sandhi"
    if (original["reading_evidence"] != dict(frontend_before_sandhi=before, final=final)
            or original["reading_rules"] != rules or original["tone_sources"] != sources):
        raise ValueError("compaction lost reading provenance")
    return labels


def compact_corpus(source, out):
    manifest_path = source / "manifest.json"
    parent = json.loads(manifest_path.read_text())
    if parent["format"] != "wime-corpus-v2" or "reading_schema" in parent:
        raise ValueError("expected a corpus-v2 build with dense reading evidence")
    empty_output(out)
    started, verified = time.monotonic(), {}
    for split in SPLITS:
        original_hash, labels_hash = hashlib.sha256(), hashlib.sha256()
        counts = Counter()
        with (source / f"{split}.jsonl").open("rb") as stream, (out / f"{split}.jsonl.partial").open("w") as output:
            for line in stream:
                original_hash.update(line)
                counts["input_bytes"] += len(line)
                row = json.loads(line)
                compact = compact_tones(row)
                labels = verify_row(row, compact)
                labels_hash.update((json.dumps(labels, ensure_ascii=False, sort_keys=True) + "\n").encode())
                output.write(json.dumps(compact, ensure_ascii=False) + "\n")
                notes = compact.get("reading_notes", [])
                counts.update(rows=1, characters=len(row["tones"]), notes=len(notes),
                              rows_without_notes=not notes,
                              changed_tones=sum("before_tone" in note for note in notes))
                if counts["rows"] % 100000 == 0:
                    print(json.dumps(dict(split=split, rows=counts["rows"])), flush=True)
        expected = parent["outputs"][split]
        if original_hash.hexdigest() != expected["sha256"] or counts["input_bytes"] != expected["bytes"]:
            raise ValueError(f"source fingerprint changed: {split}")
        verified[split] = dict(counts, labels_sha256=labels_hash.hexdigest(),
                              input_sha256=original_hash.hexdigest(), evidence_reconstruction="exact")
    for split in SPLITS:
        (out / f"{split}.jsonl.partial").rename(out / f"{split}.jsonl")
    result = {key: value for key, value in parent.items() if key not in ("parts", "elapsed_s")}
    result.update(operation="compact-reading-evidence", reading_schema=READING_SCHEMA,
                  arguments=dict(parent["arguments"], out=str(out)),
                  outputs={split: file_metadata(out / f"{split}.jsonl") for split in SPLITS},
                  elapsed_s=time.monotonic() - started,
                  compaction=dict(source_manifest=file_metadata(manifest_path), verified=verified,
                                  code={name: file_metadata(Path(__file__).with_name(name)) for name in
                                        ("compact-corpus.py", "corpus_tones.py", "corpus_integrity.py")}))
    (out / "manifest.json.partial").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (out / "manifest.json.partial").rename(out / "manifest.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="registered source corpus ID")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    registry = Registry()
    if registry.entry(args.dataset)["kind"] != "corpus":
        parser.error("source must be a corpus")
    registry.verify(args.dataset)
    result = compact_corpus(registry.directory(args.dataset), args.out)
    print(json.dumps(result["compaction"]["verified"], ensure_ascii=False))


if __name__ == "__main__":
    main()
