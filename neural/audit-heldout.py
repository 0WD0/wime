#!/usr/bin/env python3
"""Find source and text overlap between proposed tests and registered training data."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import ahocorasick

from dataset_registry import Registry, fingerprint
from replay import rows


def automaton(values):
    result = ahocorasick.Automaton()
    for text, identities in values.items():
        result.add_word(text, (text, tuple(identities)))
    result.make_automaton()
    return result


class Overlap:
    def __init__(self, proposals):
        self.documents, self.groups = defaultdict(set), defaultdict(set)
        needles, seeds = defaultdict(set), defaultdict(set)
        for row in proposals:
            identity, target = row["id"], row["text"]
            self.documents[row["document"]].add(identity)
            self.groups[row["group"]].add(identity)
            width = min(12, len(target))
            for start in range(len(target) - width + 1):
                needles[target[start:start + width]].add(identity)
            # At least half a matching span is on one side of context/target.
            # These cheap seeds let us skip parsing most archived JSON rows.
            for start in range(len(target) - width // 2 + 1):
                seeds[target[start:start + width // 2]].add(identity)
        for field in (self.documents, self.groups):
            for value, identities in field.items():
                seeds[value].update(identities)
        self.needles, self.seeds = automaton(needles), automaton(seeds)

    def scan(self, row):
        matches = defaultdict(set)
        for field, index in (("document", self.documents), ("group", self.groups)):
            for identity in index.get(row.get(field), ()):
                matches[identity].add(field + ":" + row[field])
        for gold in row["gold"]:
            for _, (text, identities) in self.needles.iter(row.get("context", "") + gold):
                for identity in identities:
                    matches[identity].add("text:" + text)
        return matches

    def archived_rows(self, path, expected_hash):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for raw in stream:
                digest.update(raw)
                line = raw.decode()
                if "\\u" in line or next(self.seeds.iter(line), None) is not None:
                    yield json.loads(line)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"changed registered training file: {path}")


def audit(proposals, registry):
    overlap, hits, scans = Overlap(proposals), {}, []
    for name, entry in registry.datasets.items():
        if entry["kind"] not in ("corpus", "mixture"):
            continue
        start = time.monotonic()
        path = registry.path(name, "train")
        if path.is_dir():
            registry.verify(name, ["train"])
            source = rows(path)
        else:
            if "storage" in entry:
                registry.verify(name, ["train"])
                expected = fingerprint(path)
            else:
                expected = entry["files"]["train"]["sha256"]
            source = overlap.archived_rows(path, expected)
        count = 0
        for row in source:
            matches = overlap.scan(row)
            count += bool(matches)
            for identity, reasons in matches.items():
                hit = hits.setdefault(identity, dict(datasets=set(), reasons=set(), examples=[]))
                hit["datasets"].add(name)
                hit["reasons"].update(reasons)
                if len(hit["examples"]) < 2:
                    hit["examples"].append(dict(dataset=name, **{key: row[key] for key in
                        ("id", "document", "group", "context", "gold") if key in row}))
        scans.append(dict(dataset=name, snapshot=registry.snapshot(name, ["train"]),
                          matching_rows=count, seconds=time.monotonic() - start))
        print(json.dumps(dict(dataset=name, matching_rows=count,
                              seconds=scans[-1]["seconds"])), flush=True)
    for hit in hits.values():
        for key in ("datasets", "reasons"):
            hit[key] = sorted(hit[key])
    return dict(policy="Reject shared source document/group or a contiguous 12-character target span "
                "in a training context+target (whole target if shorter). All registered training "
                "releases scanned; upstream pretrained-model overlap is unknown.", scans=scans, hits=hits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.proposals.read_text()), Registry())
    report["proposals_sha256"] = hashlib.sha256(args.proposals.read_bytes()).hexdigest()
    report["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
