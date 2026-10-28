#!/usr/bin/env python3
"""Materialize immutable dataset releases as ordered, queryable Parquet shards."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from dataset_registry import Registry
from replay import rows

SHARD_ROWS = 131072


def row_digest(records):
    digest = hashlib.sha256()
    count = 0
    for row in records:
        digest.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode())
        count += 1
    return count, digest.hexdigest()


def export(registry, dataset, out):
    registry.verify(dataset)
    entry = registry.entry(dataset)
    if entry["kind"] != "corpus":
        raise ValueError("Parquet releases require a training corpus")
    out.mkdir(parents=True, exist_ok=False)
    db = duckdb.connect()
    db.execute("SET threads=4")
    db.execute("SET memory_limit='2GB'")
    verified = {}
    for split in ("train", "dev", "test"):
        source = registry.path(dataset, split)
        directory = out / split
        directory.mkdir()
        # A bounded temporary table avoids reparsing JSON for every shard.
        staging = out / (split + ".staging.parquet")
        db.execute("COPY (SELECT row_number() OVER () AS _row, * FROM read_json(?, "
                   "format='newline_delimited', sample_size=-1)) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                   [str(source), str(staging)])
        for index, batch in enumerate(pq.ParquetFile(staging).iter_batches(batch_size=SHARD_ROWS)):
            pq.write_table(pa.Table.from_batches([batch]),
                           directory / f"part-{index:05d}.parquet", compression="zstd", row_group_size=32768)
        staging.unlink()
        original = row_digest(rows(source))
        actual = row_digest(rows(directory))
        if original != actual:
            raise ValueError(f"Parquet roundtrip changed rows in {split}")
        verified[split] = dict(rows=actual[0], row_sha256=actual[1],
                               bytes=sum(file.stat().st_size for file in directory.glob('*.parquet')))
        print(json.dumps(dict(split=split, **verified[split])), flush=True)
    source_manifest = json.loads(registry.path(dataset, "manifest").read_text())
    manifest = dict(format="wime-parquet-corpus-v1", source=registry.snapshot(dataset, entry["files"]),
                    provenance=source_manifest.get("provenance", {}),
                    reading_schema=source_manifest.get("reading_schema"),
                    tone_policy=source_manifest.get("tone_policy"),
                    annotation_window=source_manifest.get("annotation_window"),
                    outputs=verified, row_order="original split order; _row is one-based browse ordinal",
                    recipe=dict(program="dataset_store.py", dataset=dataset, shard_rows=SHARD_ROWS))
    if "audit" in entry["files"]:
        shutil.copyfile(registry.path(dataset, "audit"), out / "source-audit.json")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    export(Registry(), args.dataset, args.out)


if __name__ == "__main__":
    main()
