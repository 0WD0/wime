"""Regression checks for leakage guards and pinned corpus build inputs."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from corpus_integrity import UniqueRows, checked_row, empty_output, reading_assets, sample_key, verify_sources
from replay import Protection, fingerprint


class Integrity(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.unique = UniqueRows(self.root / "identities.sqlite3")
        self.addCleanup(self.unique.close)

    def protection(self, rows):
        path = self.root / "heldout.json"
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return Protection([path])

    @staticmethod
    def row(identity="first", context="这是一段此前输入的中文上下文", gold="需要保护的测试句子", code="ceui"):
        return dict(id=identity, document=identity, context=context, gold=[gold], input=code)

    def check(self, row, protection=None, blank_share=0):
        return checked_row(row, protection or Protection([]), self.unique, random.Random(1), blank_share)

    def test_heldout_context_is_checked_before_dropout(self):
        row = self.row()
        protected = self.protection([row])
        # A different document ID and context dropout cannot bypass the exact
        # held-out context/target guard.
        row["document"] = "different-document"
        result, rejection = self.check(row, protected, blank_share=1)
        self.assertIsNone(result)
        self.assertEqual(rejection, "protected")
        self.assertTrue(row["context"])

    def test_dropout_cannot_create_a_heldout_contextless_example(self):
        protected = self.protection([self.row(identity="heldout", context="")])
        result, rejection = self.check(self.row(), protected, blank_share=1)
        self.assertIsNone(result)
        self.assertEqual(rejection, "protected")

    def test_punctuation_normalized_samples_are_deduplicated_across_splits(self):
        first = self.row(context="此前，输入的文字。")
        first["split"] = "train"
        second = self.row(identity="second", context="此前输入的文字")
        second["split"] = "test"
        self.assertEqual(sample_key(first), sample_key(second))
        self.assertIsNone(self.check(first)[1])
        result, rejection = self.check(second)
        self.assertIsNone(result)
        self.assertEqual(rejection, "duplicate_sample")

    def test_reused_id_for_different_sample_fails(self):
        self.assertIsNone(self.check(self.row())[1])
        with self.assertRaisesRegex(ValueError, "duplicate row id"):
            self.check(self.row(context="完全不同的上下文"))

    def test_unique_rows_persists_committed_ids_and_separates_key_types(self):
        self.assertTrue(self.unique.add("id", "one"))
        self.assertFalse(self.unique.add("id", "one"))
        self.assertTrue(self.unique.add("sample", "one"))
        self.unique.commit()
        reopened = UniqueRows(self.root / "identities.sqlite3")
        self.addCleanup(reopened.close)
        self.assertFalse(reopened.add("id", "one"))
        self.assertTrue(reopened.add("id", "two"))

    def test_reading_assets_inside_cache_keep_model_files_but_skip_nested_caches(self):
        root = self.root / ".cache" / "g2pw"
        model = root / "model/G2PWModel"
        tokenizer = root / "bert-base-chinese"
        model.mkdir(parents=True)
        tokenizer.mkdir(parents=True)
        (model / "g2pw.onnx").write_bytes(b"model")
        (tokenizer / "vocab.txt").write_text("vocabulary")
        for parent, cache in ((model, "__pycache__"), (tokenizer, ".cache")):
            (parent / cache).mkdir()
            (parent / cache / "ignored").write_text("cache")
        assets = reading_assets(root)
        self.assertEqual(set(assets), {"model/G2PWModel/g2pw.onnx", "bert-base-chinese/vocab.txt"})
        self.assertEqual(assets["model/G2PWModel/g2pw.onnx"]["sha256"], fingerprint(model / "g2pw.onnx"))

    def test_explicit_tokenizer_assets_are_required_and_fingerprinted(self):
        root = self.root / "model-root"
        model = root / "model/G2PWModel"
        model.mkdir(parents=True)
        (model / "g2pW.onnx").write_bytes(b"model")
        tokenizer = self.root / "separate-tokenizer"
        tokenizer.mkdir()
        with self.assertRaisesRegex(ValueError, "no reading assets"):
            reading_assets(root, tokenizer)
        (tokenizer / "vocab.txt").write_text("vocabulary")
        assets = reading_assets(root, tokenizer)
        self.assertEqual(set(assets), {"model/G2PWModel/g2pW.onnx", "tokenizer/vocab.txt"})
        self.assertEqual(assets["tokenizer/vocab.txt"]["sha256"], fingerprint(tokenizer / "vocab.txt"))

    def test_verify_sources_rejects_changed_or_missing_input(self):
        raw = self.root / "raw"
        source = raw / "wildchat/data/source.parquet"
        source.parent.mkdir(parents=True)
        source.write_text("original\n")
        manifest = self.root / "sources.json"
        manifest.write_text(json.dumps({"sources": {"wildchat": {"files": {
            "wildchat/data/source.parquet": fingerprint(source)}}}}))
        result = verify_sources(raw, manifest, ["wildchat"])
        self.assertEqual(result["files"]["wildchat/data/source.parquet"], fingerprint(source))
        source.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "changed corpus source"):
            verify_sources(raw, manifest, ["wildchat"])
        source.unlink()
        with self.assertRaisesRegex(ValueError, "missing=.*source.parquet"):
            verify_sources(raw, manifest, ["wildchat"])

    def test_verify_sources_rejects_unpinned_extra_matching_file(self):
        raw = self.root / "raw"
        source = raw / "wildchat/data/source.parquet"
        source.parent.mkdir(parents=True)
        source.write_text("original\n")
        manifest = self.root / "sources.json"
        manifest.write_text(json.dumps({"sources": {"wildchat": {"files": {
            "wildchat/data/source.parquet": fingerprint(source)}}}}))
        (source.parent / "extra.parquet").write_text("unpinned\n")
        with self.assertRaisesRegex(ValueError, "unpinned=.*extra.parquet"):
            verify_sources(raw, manifest, ["wildchat"])

    def test_empty_output_does_not_overwrite_existing_build(self):
        output = self.root / "new-output"
        empty_output(output)
        self.assertTrue(output.is_dir())
        empty_output(output)
        marker = output / "train.jsonl"
        marker.write_text("retained\n")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            empty_output(output)
        self.assertEqual(marker.read_text(), "retained\n")


if __name__ == "__main__":
    unittest.main()
