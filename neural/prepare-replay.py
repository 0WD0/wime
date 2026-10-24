#!/usr/bin/env python3
"""Bounded replay of existing chat/technical splits, keeping their documents."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import random

from conditional import augment_composition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat", type=Path, required=True)
    parser.add_argument("--technical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(sampling="lowest SHA256 of source example id; original document splits retained",
                    sources={}, counts={})
    for split in ("train", "dev"):
        rows = []
        for name, directory, limit in (("chat", args.chat, 195000 if split == "train" else 2600),
                                       ("technical", args.technical, 105000 if split == "train" else 1400)):
            with (directory / (split + ".jsonl")).open() as f:
                selected = heapq.nsmallest(limit, map(json.loads, f),
                                           key=lambda r:hashlib.sha256(r["id"].encode()).digest())
            for row in selected:
                row["category"] = name
                if split == "dev":
                    row = augment_composition(row, random.Random(row["id"]), .3, .15, .1)
                rows.append(row)
            manifest["counts"][name + "/" + split] = len(selected)
            manifest["sources"][name] = json.loads((directory / "manifest.json").read_text())
        rows.sort(key=lambda r:r["id"])
        with (args.out / (split + ".jsonl")).open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest["files"] = {p.name:hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in args.out.glob("*.jsonl")}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
