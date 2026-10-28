"""Corpus audit invariants on tiny, completed synthetic datasets."""
from collections import Counter
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

NEURAL = Path(__file__).resolve().parents[1] / "neural"
sys.path.insert(0, str(NEURAL))
spec = importlib.util.spec_from_file_location("audit_corpus", NEURAL / "audit-corpus.py")
corpus_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus_audit)


class AuditCorpus(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        self.database = self.root / "dictionary.sqlite3"
        self.syllables = json.loads((NEURAL / "context-pinyin.json").read_text())["syllables"]
        with contextlib.closing(sqlite3.connect(self.database)) as db:
            db.execute("CREATE TABLE entry (word TEXT, code TEXT)")
            db.executemany("INSERT INTO entry VALUES (?, ?)", [
                ("你", self.syllables["ni"]), ("好", self.syllables["hao"]),
                ("行", self.syllables["xing"]), ("行", self.syllables["hang"])])
            db.commit()

    def row(self, index=0, split="train", **updates):
        row = dict(id=f"id-{index}", document=f"doc-{index}", group=f"group-{index}",
                   source="synthetic", split=split, context=f"语境{index}",
                   input=self.syllables["ni"] + self.syllables["hao"], gold=["你好"])
        row.update(updates)
        return row

    def write(self, rows):
        counts = Counter(f"{r['source']}/{r['split']}" for r in rows)
        counts["blank_context"] = sum(not r["context"] for r in rows)
        for split in corpus_audit.SPLITS:
            (self.corpus / f"{split}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r["split"] == split))
        manifest = dict(format="wime-corpus-v2", stats=dict(counts), protection={},
                        provenance=dict(database=corpus_audit.file_metadata(self.database)),
                        outputs={s: corpus_audit.file_metadata(self.corpus / f"{s}.jsonl") for s in corpus_audit.SPLITS})
        (self.corpus / "manifest.json").write_text(json.dumps(manifest))

    def test_valid_corpus_counts_and_bounded_reproducible_samples(self):
        # ASCII digits alone are stripped by the dedup normalizer; use distinct
        # Chinese context strings to make these truly distinct examples.
        rows = [self.row(i, context="语境" + chr(0x4e00 + i)) for i in range(20)]
        rows += [self.row(20, split="dev", context=""), self.row(21, split="test", context="测试")]
        self.write(rows)
        report, samples = corpus_audit.audit(self.corpus, self.database, sample_per_source=3)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["rows"], 22)
        self.assertEqual(report["counts"], {"synthetic/dev": 1, "synthetic/test": 1, "synthetic/train": 20})
        self.assertEqual(report["lengths"], {"2": 22})
        self.assertEqual(report["blank_context"]["synthetic/dev"], 1)
        sample_rows = samples["sources"]["synthetic"]
        self.assertEqual(len(sample_rows), 3)
        self.assertIn("ni", sample_rows[0]["key_syllable_options"][0])
        self.assertEqual(samples, corpus_audit.audit(self.corpus, self.database, sample_per_source=3)[1])

    def test_duplicate_id_normalized_sample_and_split_leakage(self):
        first = self.row(context="上下文。")
        second = self.row(split="test", context="上下文！")
        self.write([first, second])
        report, _ = corpus_audit.audit(self.corpus, self.database)
        for kind in ("duplicate_id", "duplicate_sample", "group_cross_split", "document_cross_split"):
            self.assertEqual(report["errors"].get(kind), 1, report)

    def test_illegal_keys_lengths_and_gold_are_rejected(self):
        self.write([self.row(0, input="zzzz"), self.row(1, input="ni", context="长" * 65),
                    self.row(2, gold=["你好a"], input="nihaaa"),
                    self.row(3, gold=["你" * 25], input="ni" * 25)])
        report, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertFalse(report["ok"])
        self.assertEqual(report["errors"]["illegal_reading_key"], 2)
        self.assertEqual(report["errors"]["input_length"], 1)
        self.assertEqual(report["errors"]["context_length"], 1)
        self.assertEqual(report["errors"]["gold_shape"], 2)

    def test_changed_files_and_missing_completion_manifest_fail(self):
        self.write([self.row()])
        with (self.corpus / "train.jsonl").open("a") as stream:
            stream.write("not json\n")
        report, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(report["errors"]["output_fingerprint"], 1)
        self.assertEqual(report["errors"]["invalid_json"], 1)
        (self.corpus / "manifest.json").unlink()
        report, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertIn("manifest", report["errors"])
        self.assertFalse(report["ok"])

    def test_wrong_database_and_protected_overlap_fail_cli(self):
        row = self.row()
        self.write([row])
        protected = self.root / "heldout.json"
        protected.write_text(json.dumps([row]))
        with contextlib.closing(sqlite3.connect(self.database)) as db:
            db.execute("INSERT INTO entry VALUES ('新', 'xb')")
            db.commit()
        with contextlib.redirect_stdout(io.StringIO()):
            status = corpus_audit.main(["--corpus", str(self.corpus), "--database", str(self.database),
                                        "--exclude-cases", str(protected)])
        self.assertEqual(status, 1)
        report = json.loads((self.corpus / "audit.json").read_text())
        self.assertEqual(report["errors"]["protected_overlap"], 1)
        self.assertEqual(report["errors"]["database_fingerprint"], 1)
        self.assertTrue((self.corpus / "audit-samples.json").is_file())

    def test_malformed_schema_and_mislabeled_split_are_reported(self):
        self.write([self.row()])
        path = self.corpus / "train.jsonl"
        path.write_text(json.dumps(self.row(split="dev")) + "\n" + json.dumps({"gold": None}) + "\n")
        report, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(report["errors"]["split_label"], 1)
        self.assertEqual(report["errors"]["row_schema"], 1)

    def test_manifest_protection_is_automatic_and_additional_files_are_unioned(self):
        rows = [self.row(0, context="甲"), self.row(1, context="乙")]
        self.write(rows)
        original, additional = self.root / "original.json", self.root / "additional.json"
        original.write_text(json.dumps(rows[:1]))
        additional.write_text(json.dumps(rows[1:]))
        manifest_path = self.corpus / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["protection"] = {str(original): corpus_audit.file_metadata(original)["sha256"]}
        manifest_path.write_text(json.dumps(manifest))
        automatic, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(automatic["errors"]["protected_overlap"], 1)
        combined, _ = corpus_audit.audit(self.corpus, self.database, [additional])
        self.assertEqual(combined["errors"]["protected_overlap"], 2)
        self.assertEqual(len(combined["protection"]), 2)
        original.write_text("[]")
        changed, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(changed["errors"]["protection_fingerprint"], 1)
        original.unlink()
        missing, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(missing["errors"]["protection_fingerprint"], 1)
        self.assertEqual(missing["errors"]["protection_file"], 1)

    def test_yijing_symbol_is_not_a_han_character_even_if_in_dictionary(self):
        symbol = "\u4dc0"
        with contextlib.closing(sqlite3.connect(self.database)) as db:
            db.execute("INSERT INTO entry VALUES (?, 'aa')", (symbol,))
            db.commit()
        self.write([self.row(gold=[symbol], input="aa")])
        report, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(report["errors"]["gold_shape"], 1)
        self.assertEqual(report["errors"]["illegal_reading_key"], 1)

    def test_content_cannot_cross_splits_under_different_document_ids(self):
        self.write([self.row(0, context="甲", content="same-document-content"),
                    self.row(1, split="test", context="乙", content="same-document-content"),
                    self.row(2, context="丙", content="")])
        report, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(report["errors"]["content_cross_split"], 1)
        self.assertEqual(report["errors"]["content_schema"], 1)
        self.assertNotIn("document_cross_split", report["errors"])
        self.assertNotIn("group_cross_split", report["errors"])


if __name__ == "__main__":
    unittest.main()
