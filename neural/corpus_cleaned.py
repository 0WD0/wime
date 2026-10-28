"""Read completed cleaned corpora and check annotations against their documents."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from corpus_integrity import file_metadata


FORMAT = "wime-cleaned-corpus-v1"
SOURCE = re.compile(r"[a-z][a-z0-9_-]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
COLUMNS = ("id", "group_id", "source", "category", "split", "content", "raw_content", "text")


def _source_name(name):
    if not isinstance(name, str) or not SOURCE.fullmatch(name):
        raise ValueError(f"invalid cleaned corpus source name: {name!r}")
    return name


def _manifest(root, names=None):
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"missing or invalid completed cleaned manifest: {root / 'manifest.json'}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError(f"cleaned manifest format must be {FORMAT}")
    if manifest.get("complete") is not True:
        raise ValueError("cleaned manifest is not complete")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("cleaned manifest outputs must be a nonempty mapping")
    for name in outputs:
        _source_name(name)
    requested = [names] if isinstance(names, str) else names
    for name in requested or ():
        _source_name(name)
        if name not in outputs:
            raise ValueError(f"cleaned manifest has no source: {name}")
    return manifest


def _verify_file(path, metadata):
    if (not isinstance(metadata, dict)
            or not isinstance(metadata.get("path"), str)
            or Path(metadata["path"]).name != path.name
            or not isinstance(metadata.get("sha256"), str)
            or not SHA256.fullmatch(metadata["sha256"])
            or type(metadata.get("bytes")) is not int or metadata["bytes"] < 0):
        raise ValueError(f"cleaned file lacks valid fingerprint metadata: {path.name}")
    # Resolve files within this artifact. Recorded absolute provenance paths
    # must not redirect reads to a different corpus after an artifact is moved.
    try:
        actual = file_metadata(path)
    except OSError as exc:
        raise ValueError(f"missing cleaned file: {path}") from exc
    if any(actual[key] != metadata[key] for key in ("sha256", "bytes")):
        raise ValueError(f"changed cleaned file: {path.name}")


def load_cleaned(root, names=None):
    """Verify every completed output and the index, then return the manifest."""
    root = Path(root)
    manifest = _manifest(root, names)
    for name, metadata in manifest["outputs"].items():
        _verify_file(root / f"{name}.jsonl", metadata)
    database = root / "documents.sqlite3"
    _verify_file(database, manifest.get("documents"))
    # A WAL or journal would contain state not covered by the database hash.
    for suffix in ("-wal", "-journal"):
        if Path(str(database) + suffix).exists():
            raise ValueError(f"cleaned document index has an unfinished journal: {database.name}{suffix}")
    return manifest


def iter_cleaned(root, name):
    """Stream the internal document format after load_cleaned accepts it."""
    with (Path(root) / f"{name}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


class CleanedDocuments:
    """Read-only document index with a bounded cache for adjacent sample rows."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = load_cleaned(self.root)
        self._cache = OrderedDict()
        self.db = sqlite3.connect((self.root / "documents.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("SELECT " + ",".join(COLUMNS) + " FROM documents LIMIT 0")
        except sqlite3.Error as exc:
            self.db.close()
            self.db = None
            raise ValueError("cleaned document index lacks the required documents schema") from exc

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        self._cache.clear()

    def __enter__(self):
        if self.db is None:
            raise ValueError("cleaned document index is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def _document(self, identity):
        if self.db is None:
            raise ValueError("cleaned document index is closed")
        if not isinstance(identity, str) or not identity:
            raise ValueError("annotation document must be a nonempty identity")
        if identity in self._cache:
            self._cache.move_to_end(identity)
            return self._cache[identity]
        record = self.db.execute("SELECT " + ",".join(COLUMNS) + " FROM documents WHERE id=?", (identity,)).fetchone()
        if record is None:
            raise ValueError(f"annotation document is absent from cleaned corpus: {identity}")
        document = dict(record)
        if any(not isinstance(document[key], str) for key in COLUMNS):
            raise ValueError(f"cleaned document has invalid field types: {identity}")
        if document["source"] not in self.manifest["outputs"]:
            raise ValueError(f"cleaned document source is absent from manifest: {identity}")
        content = hashlib.sha256(document["text"].encode("utf-8")).hexdigest()
        if document["content"] != content:
            raise ValueError(f"cleaned document content fingerprint differs from its text: {identity}")
        self._cache[identity] = document
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return document

    def validate_row(self, row):
        """Raise ValueError unless an annotation exactly matches its document."""
        if not isinstance(row, dict):
            raise ValueError("annotation must be an object")
        document = self._document(row.get("document"))
        for field in ("content", "source", "category", "split", "group"):
            expected = document["group_id" if field == "group" else field]
            if row.get(field) != expected:
                raise ValueError(f"annotation {field} differs from cleaned document")
        text = document["text"]
        offset, gold = row.get("offset"), row.get("gold")
        if type(offset) is not int or offset < 0:
            raise ValueError("annotation offset must be a nonnegative integer")
        if (not isinstance(gold, list) or len(gold) != 1
                or not isinstance(gold[0], str) or not gold[0]):
            raise ValueError("annotation gold must contain exactly one nonempty text")
        end = offset + len(gold[0])
        if end > len(text) or text[offset:end] != gold[0]:
            raise ValueError("annotation gold differs from cleaned document slice")
        dropped, context = row.get("context_dropped"), row.get("context")
        if type(dropped) is not bool:
            raise ValueError("annotation context_dropped must be a boolean")
        expected_context = "" if dropped else text[max(0, offset - 64):offset]
        if not isinstance(context, str) or context != expected_context:
            raise ValueError("annotation context differs from cleaned document or dropout policy")
        window = row.get("reading_window")
        if (not isinstance(window, list) or len(window) != 2
                or any(type(value) is not int for value in window)
                or not 0 <= window[0] <= offset < end <= window[1] <= len(text)):
            raise ValueError("annotation reading_window must contain its slice within the cleaned document")
