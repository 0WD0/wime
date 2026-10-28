#!/usr/bin/env python3
"""Versioned local dataset catalog: list, show and verify immutable artifacts.

WIME_CORPUS_ROOT, WIME_CONDITIONAL_ROOT and WIME_INVESTIGATION_ROOT relocate
storage without changing dataset identities. The catalog itself lives in jjosh.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


CATALOG = Path(__file__).with_name("datasets.json")


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Registry:
    def __init__(self, catalog=CATALOG, roots=None):
        self.catalog = Path(catalog)
        contents = json.loads(self.catalog.read_text())
        if contents["version"] != 1:
            raise ValueError("unsupported dataset catalog version")
        self.datasets = contents["datasets"]
        self.roots = {name: Path(os.environ.get(f"WIME_{name.upper()}_ROOT", str(Path.home() / ".cache" / folder)))
                      for name, folder in (("corpus", "wime-data"), ("conditional", "wime-conditional"),
                                           ("investigation", "wime-minicpm-investigation"))}
        self.roots["repo"] = Path(__file__).resolve().parents[3]
        self.roots.update(roots or {})

    def entry(self, dataset):
        if dataset not in self.datasets:
            raise ValueError(f"unknown dataset: {dataset}")
        return self.datasets[dataset]

    def directory(self, dataset):
        location = self.entry(dataset)["location"]
        return (self.roots[location["root"]] / location["path"]).resolve()

    def path(self, dataset, artifact):
        return self._path(self.entry(dataset), artifact)

    def _path(self, entry, artifact):
        file = entry["files"][artifact]
        location = entry["location"]
        base = (self.roots[file["root"]] if "root" in file else
                self.roots[location["root"]] / location["path"])
        return (base / file["path"]).resolve()

    def verify(self, dataset, artifacts=None):
        entry = self.entry(dataset)
        files = artifacts if artifacts is not None else entry["files"]
        checked = 0
        for key in files:
            path, expected = self.path(dataset, key), entry["files"][key]["sha256"]
            actual = fingerprint(path)
            if actual != expected:
                raise ValueError(f"{dataset}/{key}: SHA256 mismatch ({path}); expected {expected}, got {actual}")
            checked += 1
        # Raw corpus files are already enumerated in the pinned source catalog.
        # Reuse it instead of duplicating hundreds of hashes in this registry.
        if artifacts is None and "inventory" in entry:
            inventory = json.loads(self.path(dataset, entry["inventory"]).read_text())
            for source in inventory["sources"].values():
                for relative, expected in source["files"].items():
                    path = self.directory(dataset) / relative
                    if fingerprint(path) != expected:
                        raise ValueError(f"{dataset}: SHA256 mismatch ({path})")
                    checked += 1
        return checked

    def snapshot(self, dataset, artifacts):
        entry = self.entry(dataset)
        return dict(id=dataset, entry_sha256=hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest(),
                    kind=entry["kind"], status=entry["status"], annotation=entry["annotation"],
                    parents=entry.get("parents", []), files={key: entry["files"][key] for key in artifacts})

    def training(self, dataset):
        entry = self.entry(dataset)
        if entry["kind"] not in ("corpus", "mixture"):
            raise ValueError(f"{dataset}: {entry['kind']} is not a training corpus")
        required = ("manifest", "train", "dev")
        for key in required:
            filename = "manifest.json" if key == "manifest" else key + ".jsonl"
            if self.path(dataset, key) != self.directory(dataset) / filename:
                raise ValueError(f"{dataset}: training layout requires {filename}")
        self.verify(dataset, required)
        excluded = []
        protection = []
        for reference in entry.get("exclude", []):
            self.verify(reference["dataset"], reference["files"])
            excluded.extend(self.path(reference["dataset"], key) for key in reference["files"])
            protection.append(self.snapshot(reference["dataset"], reference["files"]))
        snapshot = self.snapshot(dataset, required)
        snapshot["exclude"] = protection
        return self.directory(dataset), excluded, snapshot

    def evaluation(self, dataset):
        entry = self.entry(dataset)
        if entry["kind"] not in ("evaluation", "diagnostic"):
            raise ValueError(f"{dataset}: {entry['kind']} is not an evaluation suite")
        self.verify(dataset, ["cases"])
        return self.path(dataset, "cases"), self.snapshot(dataset, ["cases"])

    def register(self, specification):
        """Pin a new version's files. Existing IDs cannot be repointed."""
        dataset = specification["id"]
        if dataset in self.datasets:
            raise ValueError(f"{dataset} already exists; register changed data under a new version ID")
        entry = {key: value for key, value in specification.items() if key != "id"}
        for parent in entry.get("parents", []):
            self.entry(parent)
        for key, file in entry["files"].items():
            path = self._path(entry, key)
            file["sha256"] = fingerprint(path)
            if file.get("format") == "jsonl":
                with path.open("rb") as stream:
                    file["rows"] = sum(1 for line in stream if line.strip())
            elif file.get("format") == "json-array":
                file["rows"] = len(json.loads(path.read_text()))
        updated = dict(self.datasets, **{dataset: entry})
        temporary = self.catalog.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(version=1, datasets=updated), ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.catalog)
        self.datasets = updated
        return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    show = commands.add_parser("show")
    show.add_argument("dataset")
    register = commands.add_parser("register", help="pin a JSON dataset specification under a new version ID")
    register.add_argument("specification", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("datasets", nargs="+", help="dataset IDs, or 'all'")
    args = parser.parse_args()
    registry = Registry(args.catalog)
    if args.command == "list":
        print("ID\tKIND\tSTATUS\tANNOTATION\tROWS")
        for name, entry in registry.datasets.items():
            rows = ",".join(f"{key}={value['rows']}" for key, value in entry["files"].items() if "rows" in value)
            print("\t".join((name, entry["kind"], entry["status"], entry["annotation"], rows)))
    elif args.command == "show":
        entry = registry.entry(args.dataset)
        print(json.dumps(dict(id=args.dataset, **entry, resolved=str(registry.directory(args.dataset))),
                         ensure_ascii=False, indent=2))
    elif args.command == "register":
        print(registry.register(json.loads(args.specification.read_text())))
    else:
        datasets = registry.datasets if args.datasets == ["all"] else args.datasets
        for dataset in datasets:
            count = registry.verify(dataset)
            print(json.dumps(dict(id=dataset, verified_files=count)), flush=True)


if __name__ == "__main__":
    main()
