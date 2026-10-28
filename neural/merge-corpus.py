#!/usr/bin/env python3
"""Merge completed corpus-v2 source builds in a fixed, audited part order."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import time

from corpus_integrity import UniqueRows, empty_output, file_metadata, sample_key

SPLITS = ("train", "dev", "test")


def hashes(metadata):
    """Compare asset content, independently of absolute build paths."""
    if "sha256" in metadata:
        return metadata["sha256"]
    return {name: hashes(value) for name, value in metadata.items()}


def compatibility(manifest):
    provenance = manifest["provenance"]
    return dict(
        provenance={name: hashes(provenance[name]) for name in ("database", "reading_assets", "code")},
        source_manifest=provenance["manifest_sha256"],
        protection=manifest["protection"], sampling=manifest["sampling"],
        arguments={key: value for key, value in manifest["arguments"].items()
                   if key not in ("sources", "out", "max_chars")},
        providers=manifest["providers"], packages=manifest["packages"],
    )


def load_parts(parts):
    loaded, expected = [], None
    for directory in parts:
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != "wime-corpus-v2":
            raise ValueError(f"not a completed corpus-v2 build: {directory}")
        actual = compatibility(manifest)
        if expected is not None and actual != expected:
            differences = [key for key in expected if actual[key] != expected[key]]
            raise ValueError(f"incompatible corpus part {directory}: {', '.join(differences)}")
        expected = actual
        for split in SPLITS:
            output = directory / f"{split}.jsonl"
            metadata = file_metadata(output)
            recorded = manifest["outputs"][split]
            if any(metadata[key] != recorded[key] for key in ("sha256", "bytes")):
                raise ValueError(f"changed corpus part output: {output}")
        loaded.append((directory, manifest, file_metadata(path)))
    if not loaded:
        raise ValueError("at least one corpus part is required")
    return loaded


def merge(out, parts):
    started = time.time()
    loaded = load_parts(parts)
    empty_output(out)
    unique = UniqueRows(out / "identities.sqlite3")
    unique.db.execute("CREATE TABLE owners (kind TEXT, key TEXT, value TEXT, PRIMARY KEY(kind, key)) WITHOUT ROWID")

    def owner(kind, key, value):
        inserted = unique.db.execute("INSERT OR IGNORE INTO owners VALUES (?, ?, ?)", (kind, key, value)).rowcount
        if inserted:
            return value
        return unique.db.execute("SELECT value FROM owners WHERE kind = ? AND key = ?", (kind, key)).fetchone()[0]

    stats, readings = Counter(), Counter()
    baseline = loaded[0][1]
    provenance = dict(baseline["provenance"], files={})
    protection = {}
    sources = []
    try:
        with ExitStack() as stack:
            outputs = {split: stack.enter_context((out / f"{split}.jsonl.partial").open("w", encoding="utf-8"))
                       for split in SPLITS}
            for directory, manifest, _ in loaded:
                readings.update(manifest["readings"])
                for path, digest in manifest["provenance"]["files"].items():
                    if path in provenance["files"] and provenance["files"][path] != digest:
                        raise ValueError(f"conflicting source hash: {path}")
                    provenance["files"][path] = digest
                protection.update(manifest["protection"])
                for source in manifest["arguments"]["sources"].split(","):
                    if source not in sources:
                        sources.append(source)
                for split in SPLITS:
                    with (directory / f"{split}.jsonl").open(encoding="utf-8") as stream:
                        for line in stream:
                            row = json.loads(line)
                            if row["split"] != split:
                                raise ValueError(f"row in wrong split file: {row['id']}")
                            if not unique.add("id", row["id"]):
                                raise ValueError(f"duplicate row id: {row['id']}")
                            for kind in ("group", "document"):
                                if owner(kind, row[kind], split) != split:
                                    raise ValueError(f"{kind} crosses splits: {row[kind]}")
                            if owner("document_content", row["document"], row["content"]) != row["content"]:
                                raise ValueError(f"document has conflicting content: {row['document']}")
                            source = row["source"]
                            stats["input_rows"] += 1
                            if owner("content", row["content"], row["document"]) != row["document"]:
                                stats[f"{source}/duplicate_document_rows"] += 1
                            elif not unique.add("sample", sample_key(row)):
                                stats[f"{source}/duplicate_sample"] += 1
                            else:
                                outputs[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                                stats[f"{source}/{split}"] += 1
                                stats["blank_context"] += not bool(row["context"])
                                length = len(row["gold"][0])
                                stats[f"length/{length if length < 13 else '13+'}"] += 1
                                stats[f"{source}/documents"] += unique.add("retained_document", row["document"])
                            if stats["input_rows"] % 10000 == 0:
                                unique.commit()
                            if stats["input_rows"] % 100000 == 0:
                                print(json.dumps(dict(event="merge_progress", input_rows=stats["input_rows"],
                                                      elapsed_s=round(time.time() - started))), flush=True)
                unique.commit()
                print(json.dumps(dict(event="part_complete", path=str(directory),
                                      input_rows=stats["input_rows"])), flush=True)
    finally:
        unique.close()

    for split in SPLITS:
        (out / f"{split}.jsonl.partial").rename(out / f"{split}.jsonl")
    arguments = dict(baseline["arguments"], sources=",".join(sources), out=str(out),
                     max_chars="per-part", parts=[str(directory) for directory, _, _ in loaded])
    provenance["merge_code"] = file_metadata(Path(__file__))
    result = dict(format="wime-corpus-v2", operation="merge", stats=dict(stats), readings=dict(readings),
                  sampling=baseline["sampling"], arguments=arguments, protection=protection,
                  provenance=provenance, providers=baseline["providers"], packages=baseline["packages"],
                  parts=[dict(manifest=metadata, arguments=manifest["arguments"], stats=manifest["stats"])
                         for _, manifest, metadata in loaded],
                  outputs={split: file_metadata(out / f"{split}.jsonl") for split in SPLITS},
                  elapsed_s=time.time() - started)
    temporary = out / "manifest.json.partial"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.rename(out / "manifest.json")
    print(json.dumps(result["stats"], ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    merge(args.out, args.parts)


if __name__ == "__main__":
    main()
