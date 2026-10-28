"""Pinned cleaned artifacts and exact annotation-to-document checks."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from corpus_cleaned import CleanedDocuments, iter_cleaned, load_cleaned
from corpus_integrity import file_metadata


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Cleaned(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "cleaned"
        self.root.mkdir()
        self.documents = []

    def build(self, count=2):
        for index in range(count):
            text = "前文" * 35 + "İ눈银行行长正在开会。"
            self.documents.append(dict(document=f"doc-{index}", group=f"group-{index // 2}",
                                       source="forum", category="forum", split="train", text=text,
                                       content=digest(text), raw_content=digest("raw:" + text)))
        with sqlite3.connect(self.root / "documents.sqlite3") as db:
            db.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, group_id TEXT, source TEXT, "
                       "category TEXT, split TEXT, content TEXT, raw_content TEXT, text TEXT)")
            db.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)", [
                tuple(row[key] for key in ("document", "group", "source", "category", "split",
                                          "content", "raw_content", "text")) for row in self.documents])
        (self.root / "forum.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                                       for row in self.documents), encoding="utf-8")
        self.manifest = dict(format="wime-cleaned-corpus-v1",
                             outputs={"forum": file_metadata(self.root / "forum.jsonl")},
                             documents=file_metadata(self.root / "documents.sqlite3"))
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def row(self, index=0, offset=72, length=2, dropped=False):
        document = self.documents[index]
        text = document["text"]
        return dict(document=document["document"], group=document["group"], source=document["source"],
                    category=document["category"], split=document["split"], content=document["content"],
                    offset=offset, gold=[text[offset:offset + length]], context_dropped=dropped,
                    context="" if dropped else text[max(0, offset - 64):offset],
                    reading_window=[max(0, offset - 64), len(text)])

    def test_load_and_stream_completed_artifact_without_model(self):
        self.build()
        self.assertEqual(load_cleaned(self.root, ["forum"]), self.manifest)
        self.assertEqual(list(iter_cleaned(self.root, "forum")), self.documents)
        with CleanedDocuments(self.root) as documents:
            self.assertIsNone(documents.validate_row(self.row()))

    def test_serial_slices_and_explicit_context_dropout_match_unicode_characters(self):
        self.build()
        with CleanedDocuments(self.root) as documents:
            for offset in (0, 1, 63, 64, 65, 70, 72, 74, 77):
                for dropped in (False, True):
                    with self.subTest(offset=offset, dropped=dropped):
                        documents.validate_row(self.row(offset=offset, dropped=dropped))
            self.assertEqual(self.row()["gold"], ["银行"])
            self.assertEqual(len(self.row()["context"]), 64)
            self.assertEqual(self.row(offset=0)["context"], "")

    def test_tampered_source_bytes_and_database_are_rejected(self):
        for target in ("forum.jsonl", "documents.sqlite3"):
            with self.subTest(target=target):
                self.build()
                path = self.root / target
                original = path.read_bytes()
                # Same byte count still must fail its SHA-256 fingerprint.
                path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                with self.assertRaisesRegex(ValueError, "changed cleaned file"):
                    load_cleaned(self.root)
                for child in self.root.iterdir():
                    child.unlink()
                self.documents.clear()

    def test_manifest_completion_metadata_and_source_membership_are_required(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "no source"):
            load_cleaned(self.root, ["wildchat"])
        with self.assertRaisesRegex(ValueError, "invalid cleaned corpus source name"):
            list(iter_cleaned(self.root, "../outside"))
        original = json.loads(json.dumps(self.manifest))
        for mutate, pattern in (
            (lambda m: m.update(format="another-format"), "format"),
            (lambda m: m.update(complete=False), "not complete"),
            (lambda m: m.update(outputs={}), "nonempty mapping"),
            (lambda m: m["outputs"]["forum"].pop("sha256"), "fingerprint metadata"),
            (lambda m: m["outputs"]["forum"].update(bytes=True), "fingerprint metadata"),
            (lambda m: m["documents"].update(bytes=0), "changed cleaned file"),
        ):
            self.manifest = json.loads(json.dumps(original))
            mutate(self.manifest)
            self.write_manifest()
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                load_cleaned(self.root)
        (self.root / "manifest.json").rename(self.root / "manifest.json.partial")
        with self.assertRaisesRegex(ValueError, "completed cleaned manifest"):
            load_cleaned(self.root)

    def test_relocation_uses_local_artifact_and_never_external_metadata_path(self):
        self.build()
        moved = self.directory / "moved"
        shutil.copytree(self.root, moved)
        self.assertEqual(load_cleaned(moved), self.manifest)
        # Metadata still points to the original, untampered corpus. It cannot
        # cause verification to ignore the changed local file being consumed.
        (moved / "forum.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "changed cleaned file"):
            load_cleaned(moved)

    def test_unpinned_sqlite_journal_is_rejected(self):
        self.build()
        (self.root / "documents.sqlite3-wal").write_bytes(b"unrecorded changes")
        with self.assertRaisesRegex(ValueError, "unfinished journal"):
            CleanedDocuments(self.root)

    def test_identity_metadata_gold_and_offset_tampering_are_rejected(self):
        self.build()
        with CleanedDocuments(self.root) as documents:
            for field, value in (("document", "absent"), ("content", "0" * 64), ("source", "wildchat"),
                                 ("category", "other"), ("group", "other"), ("split", "test"),
                                 ("gold", ["银航"]), ("gold", []), ("gold", [""]),
                                 ("offset", -1), ("offset", True), ("offset", 10000), ("offset", 73)):
                row = self.row()
                row[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    documents.validate_row(row)

    def test_context_cannot_be_replaced_or_dropped_without_explicit_boolean(self):
        self.build()
        with CleanedDocuments(self.root) as documents:
            for changes in ({"context": ""}, {"context": "别的上下文"}, {"context_dropped": True},
                            {"context_dropped": None}, {"context_dropped": 0}, {"context_dropped": "true"}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "context"):
                    documents.validate_row(dict(self.row(), **changes))
            documents.validate_row(self.row(dropped=True))
            documents.validate_row(self.row(offset=0))

    def test_reading_window_stays_inside_original_and_contains_entire_gold(self):
        self.build()
        with CleanedDocuments(self.root) as documents:
            for window in (None, [0], [0, 10000], [-1, 80], [73, 80], [0, 73], [True, 80]):
                with self.subTest(window=window), self.assertRaisesRegex(ValueError, "reading_window"):
                    documents.validate_row(dict(self.row(), reading_window=window))

    def test_index_content_hash_is_recomputed_even_when_manifest_was_rehashed(self):
        self.build()
        with sqlite3.connect(self.root / "documents.sqlite3") as db:
            db.execute("UPDATE documents SET text='被替换的原文' WHERE id='doc-0'")
        self.manifest["documents"] = file_metadata(self.root / "documents.sqlite3")
        self.write_manifest()
        with CleanedDocuments(self.root) as documents:
            with self.assertRaisesRegex(ValueError, "content fingerprint"):
                documents.validate_row(self.row())

    def test_readonly_index_bounded_lru_and_close(self):
        self.build(130)
        before = file_metadata(self.root / "documents.sqlite3")
        with CleanedDocuments(self.root) as documents:
            with self.assertRaises(sqlite3.OperationalError):
                documents.db.execute("DELETE FROM documents")
            for index in range(128):
                documents.validate_row(self.row(index))
            documents.validate_row(self.row(0))
            documents.validate_row(self.row(128))
            self.assertEqual(len(documents._cache), 128)
            self.assertIn("doc-0", documents._cache)
            self.assertNotIn("doc-1", documents._cache)
            documents.validate_row(self.row(129))
            self.assertEqual(len(documents._cache), 128)
        documents.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            documents.validate_row(self.row())
        self.assertEqual(file_metadata(self.root / "documents.sqlite3"), before)

    def test_iterator_is_lazy_and_reports_invalid_line(self):
        self.build()
        with (self.root / "forum.jsonl").open("a") as stream:
            stream.write("not-json\n")
        iterator = iter_cleaned(self.root, "forum")
        self.assertEqual(next(iterator), self.documents[0])
        self.assertEqual(next(iterator), self.documents[1])
        with self.assertRaisesRegex(ValueError, "forum:3"):
            next(iterator)


if __name__ == "__main__":
    unittest.main()
