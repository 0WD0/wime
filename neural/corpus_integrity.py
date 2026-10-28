"""Disk-backed identity checks shared by corpus construction and auditing."""
import hashlib
import json
from pathlib import Path
import sqlite3

from replay import Protection, fingerprint

SOURCE_PATTERNS = {
    "wildchat": ("wildchat/data/*.parquet",),
    "zhwiki": ("zhwiki/zhwiki-20261001-pages-articles.xml.bz2",),
    "fineweb": ("fineweb-edu/4_5/*.parquet",),
    "zhihu": ("mnbvc/qa/20230196/zhihu/*.jsonl.gz",),
    "forum": ("mnbvc/forum/20230253/*/*.jsonl.gz", "mnbvc/forum/20260107/*/*.zip"),
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sample_key(row):
    # Normalizing punctuation also catches template copies across sources.
    return digest(json.dumps([Protection.normalize(row["context"]), row["input"],
                              Protection.normalize(row["gold"][0])], ensure_ascii=False))


def validate_annotation_window(row, policy):
    """Check cleaned-document character offsets for a declared reading window."""
    maximum = policy.get("max_characters") if isinstance(policy, dict) else None
    if type(maximum) is not int or maximum < 1:
        raise ValueError("annotation_window.max_characters must be a positive integer")
    offset, window = row.get("offset"), row.get("reading_window")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative character index")
    if (not isinstance(window, list) or len(window) != 2
            or any(type(value) is not int for value in window)):
        raise ValueError("reading_window must contain two integer character offsets")
    start, end = window
    if not 0 <= start < end:
        raise ValueError("reading_window must be a nonempty nonnegative interval")
    if not start <= offset or offset + len(row["gold"][0]) > end:
        raise ValueError("sample interval must be contained in reading_window")
    if end - start > maximum:
        raise ValueError("reading_window exceeds annotation_window.max_characters")


class UniqueRows:
    """Bound memory even for tens of millions of corpus rows."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (kind TEXT, key TEXT, PRIMARY KEY(kind, key)) WITHOUT ROWID")

    def add(self, kind, key):
        return self.db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (kind, key)).rowcount == 1

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()


def checked_row(row, protection, unique, rng, blank_share):
    """Exclude held-out text before context dropout, then deduplicate samples."""
    if protection.contains(row):
        return None, "protected"
    row = dict(row)
    if rng.random() < blank_share:
        row["context"] = ""
    if protection.contains(row):
        return None, "protected"
    if not unique.add("sample", sample_key(row)):
        return None, "duplicate_sample"
    if not unique.add("id", row["id"]):
        raise ValueError(f"duplicate row id: {row['id']}")
    return row, None


def verify_sources(raw, manifest_path, names):
    manifest = json.loads(manifest_path.read_text())
    files = {}
    for name in names:
        source = manifest["sources"][name]
        if not source["files"]:
            raise ValueError(f"no pinned files for {name}")
        expected_files = set(source["files"])
        actual_files = {path.relative_to(raw).as_posix()
                        for pattern in SOURCE_PATTERNS[name]
                        for path in raw.glob(pattern) if path.is_file()}
        if actual_files != expected_files:
            raise ValueError(f"corpus source inventory differs for {name}: "
                             f"unpinned={sorted(actual_files - expected_files)}, "
                             f"missing={sorted(expected_files - actual_files)}")
        for relative, expected in source["files"].items():
            path = raw / relative
            if not path.is_file() or fingerprint(path) != expected:
                raise ValueError(f"missing or changed corpus source: {path}")
            files[relative] = expected
        print(json.dumps(dict(event="verified", source=name, files=len(source["files"]))), flush=True)
    return dict(manifest_sha256=fingerprint(manifest_path), files=files)


def empty_output(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"output directory must be empty: {path}")


def file_metadata(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=fingerprint(path), bytes=path.stat().st_size)


def reading_assets(root, tokenizer=None):
    """Hash model/tokenizer inputs, ignoring caches only below the asset root."""
    assets = {}
    for directory in (root / "model/G2PWModel", tokenizer or root / "bert-base-chinese"):
        files = [p for p in sorted(directory.rglob("*"))
                 if p.is_file() and not {".cache", "__pycache__"}.intersection(p.relative_to(directory).parts)]
        if not files:
            raise ValueError(f"no reading assets found below {directory}")
        for path in files:
            relative = path.relative_to(root) if path.is_relative_to(root) else Path("tokenizer") / path.relative_to(directory)
            assets[str(relative)] = file_metadata(path)
    return assets
