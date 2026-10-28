"""Corpus audit invariants on tiny, completed synthetic datasets."""
from collections import Counter
import contextlib
import hashlib
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
        self.model_asset = self.root / "model.onnx"
        self.model_asset.write_bytes(b"synthetic model asset")
        self.database = self.root / "dictionary.sqlite3"
        self.syllables = json.loads((NEURAL / "context-pinyin.json").read_text())["syllables"]
        with contextlib.closing(sqlite3.connect(self.database)) as db:
            db.execute("CREATE TABLE entry (word TEXT, code TEXT)")
            db.executemany("INSERT INTO entry VALUES (?, ?)", [
                ("你", self.syllables["ni"]), ("好", self.syllables["hao"]),
                ("我", self.syllables["wo"]), ("们", self.syllables["men"]),
                ("行", self.syllables["xing"]), ("行", self.syllables["hang"])])
            db.commit()

    def row(self, index=0, split="train", **updates):
        row = dict(id=f"id-{index}", document=f"doc-{index}", group=f"group-{index}",
                   source="synthetic", split=split, context=f"语境{index}",
                   input=self.syllables["ni"] + self.syllables["hao"], gold=["你好"])
        row.update(updates)
        return row

    def write(self, rows, tone_policy=None, annotation_window=None):
        counts = Counter(f"{r['source']}/{r['split']}" for r in rows)
        counts["blank_context"] = sum(not r["context"] for r in rows)
        for split in corpus_audit.SPLITS:
            (self.corpus / f"{split}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r["split"] == split))
        manifest = dict(format="wime-corpus-v2", stats=dict(counts), protection={},
                        provenance=dict(database=corpus_audit.file_metadata(self.database),
                                        reading_assets={"model.onnx": corpus_audit.file_metadata(self.model_asset)}),
                        outputs={s: corpus_audit.file_metadata(self.corpus / f"{s}.jsonl") for s in corpus_audit.SPLITS})
        if tone_policy is not None:
            manifest["tone_policy"] = tone_policy
        if annotation_window is not None:
            manifest["annotation_window"] = annotation_window
        (self.corpus / "manifest.json").write_text(json.dumps(manifest))

    def toned_row(self):
        return self.row(gold=["我们好"], input="".join(self.syllables[p] for p in ("wo", "men", "hao")),
                        pinyin=["wo", "men", None], tones=[3, 5, None],
                        tone_sources=["mandarin_frontend", "mandarin_frontend", "unresolved"],
                        reading_rules=[[], [], []],
                        reading_evidence=dict(frontend_before_sandhi=["wo3", "men5", None],
                                              final=["wo3", "men5", None]))

    def test_contextual_tones_report_neutral_and_unknown_separately(self):
        policy = dict(name="gpt-sovits-mandarin-v1", neutral=5, unknown=None)
        self.write([self.toned_row()], tone_policy=policy)
        report, samples = corpus_audit.audit(self.corpus, self.database)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["tone_policy"], policy)
        self.assertEqual(report["tones"]["synthetic"], dict(total=3, known=2, unknown=1, neutral=1,
                                                              sources=dict(mandarin_frontend=2, unresolved=1)))
        self.assertEqual(samples["sources"]["synthetic"][0]["tones"], [3, 5, None])

    def test_invalid_tone_alignment_keys_and_evidence_are_rejected(self):
        changes = [
            {"pinyin": ["ni", "men", None]},
            {"tones": [2, 5, None]},  # The frontend's final first reading is wo3.
            {"tones": [True, 5, None]},
            {"tones": [3, 0, None]},  # Neutral is 5; 0 is not unknown.
            {"pinyin": [None, "men", None]},
            {"tone_sources": ["mandarin_frontend"]},
            {"reading_rules": [[], []]},
            {"reading_rules": ["contextual_tone_sandhi", [], []]},
            {"reading_evidence": {"frontend_before_sandhi": ["wǒ3", "men5", None], "final": ["wo3", "men5", None]}},
            {"reading_evidence": {"frontend_before_sandhi": ["wo3"], "final": ["wo3", "men5", None]}},
            {"reading_evidence": {"frontend_before_sandhi": ["wo3", "men5", "hao3"], "final": ["wo3", "men5", "hao3"]}},
        ]
        for change in changes:
            with self.subTest(change=change):
                row = self.toned_row()
                row.update(change)
                self.write([row], tone_policy={"name": "gpt-sovits-mandarin-v1"})
                report, _ = corpus_audit.audit(self.corpus, self.database)
                self.assertEqual(report["errors"].get("tone_schema"), 1, report)

    def test_tone_policy_requires_annotations_and_checks_manifest_tone_totals(self):
        self.write([self.row()], tone_policy={"name": "gpt-sovits-mandarin-v1"})
        report, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(report["errors"].get("tone_schema"), 1)
        self.write([self.toned_row()], tone_policy={"name": "gpt-sovits-mandarin-v1"})
        path = self.corpus / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["stats"].update({"tone/known": 3, "tone/unknown": 0, "tone/neutral": 1})
        path.write_text(json.dumps(manifest))
        report, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(report["errors"].get("manifest_tone_count"), 2)

    def test_selected_tone_matches_final_stage_not_before_sandhi(self):
        row = self.row(pinyin=["ni", "hao"], tones=[2, 3],
                       tone_sources=["contextual_tone_sandhi", "mandarin_frontend"],
                       reading_rules=[["contextual_tone_sandhi"], []],
                       reading_evidence=dict(frontend_before_sandhi=["ni3", "hao3"], final=["ni2", "hao3"]))
        self.write([row], tone_policy={"name": "gpt-sovits-mandarin-v1"})
        valid, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertTrue(valid["ok"], valid["errors"])
        row["tones"][0] = 3
        self.write([row], tone_policy={"name": "gpt-sovits-mandarin-v1"})
        invalid, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(invalid["errors"].get("tone_schema"), 1)

    def test_complete_tone_policy_rejects_unknown_labels_and_invalid_policy(self):
        policy = dict(name="gpt-sovits-mandarin-v1", require_complete=True)
        row = self.toned_row()
        self.write([row], tone_policy=policy)
        invalid, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(invalid["errors"].get("tone_schema"), 1)
        row["pinyin"][2], row["tones"][2], row["tone_sources"][2] = "hao", 3, "mandarin_frontend"
        for values in row["reading_evidence"].values():
            values[2] = "hao3"
        self.write([row], tone_policy=policy)
        valid, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertTrue(valid["ok"], valid["errors"])
        self.write([row], tone_policy=dict(policy, require_complete="true"))
        invalid_policy, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(invalid_policy["errors"].get("tone_policy"), 1)

    def test_cleaned_document_is_the_exact_source_of_gold_and_context(self):
        cleaned = self.root / "cleaned"
        cleaned.mkdir()
        document = dict(document="doc-0", group="group-0", source="synthetic", category="web", split="train",
                        text="语境你好", raw_content="raw-digest")
        document["content"] = hashlib.sha256(document["text"].encode()).hexdigest()
        source_path = cleaned / "synthetic.jsonl"
        source_path.write_text(json.dumps(document, ensure_ascii=False) + "\n")
        document_db = cleaned / "documents.sqlite3"
        with contextlib.closing(sqlite3.connect(document_db)) as db:
            db.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, group_id TEXT, source TEXT, category TEXT, "
                       "split TEXT, content TEXT, raw_content TEXT, text TEXT)")
            db.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       tuple(document[key] for key in ("document", "group", "source", "category", "split", "content", "raw_content", "text")))
            db.commit()
        clean_manifest = cleaned / "manifest.json"
        clean_manifest.write_text(json.dumps(dict(format="wime-cleaned-corpus-v1",
                                                outputs={"synthetic": corpus_audit.file_metadata(source_path)},
                                                documents=corpus_audit.file_metadata(document_db))))
        pinned = corpus_audit.file_metadata(clean_manifest)
        row = self.row(context="语境", category="web", content=document["content"],
                       offset=2, reading_window=[0, 4], context_dropped=False)

        def write_and_audit(value):
            self.write([value])
            path = self.corpus / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["provenance"]["cleaned_corpus"] = pinned
            path.write_text(json.dumps(manifest))
            return corpus_audit.audit(self.corpus, self.database)[0]

        valid = write_and_audit(row)
        self.assertTrue(valid["ok"], valid["error_examples"])
        self.assertEqual(valid["cleaned_corpus"], pinned)
        dropped = write_and_audit(dict(row, context="", context_dropped=True))
        self.assertTrue(dropped["ok"], dropped["errors"])
        for changes in (dict(context="别处"), dict(context=""),
                        dict(gold=["好好"], input=self.syllables["hao"] * 2)):
            with self.subTest(changes=changes):
                invalid = write_and_audit(dict(row, **changes))
                self.assertEqual(invalid["errors"].get("cleaned_text"), 1, invalid)
        clean_manifest.write_text(clean_manifest.read_text() + " ")
        changed = write_and_audit(row)
        self.assertEqual(changed["errors"].get("cleaned_text"), 1, changed)

    def test_annotation_window_policy_and_offsets_are_audited(self):
        policy = dict(core=4, left_halo=2, right_halo=2, max_characters=8)
        self.write([self.row(offset=10, reading_window=[8, 14])], annotation_window=policy)
        valid, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertTrue(valid["ok"], valid["errors"])
        self.assertEqual(valid["annotation_window"], policy)
        for changes in (dict(offset=0, reading_window=[0, 9]),
                        dict(offset=6, reading_window=[0, 7]),
                        dict(offset=True, reading_window=[0, 8]),
                        dict(offset=1, reading_window=[0, "8"])):
            with self.subTest(changes=changes):
                self.write([self.row(**changes)], annotation_window=policy)
                invalid, _ = corpus_audit.audit(self.corpus, self.database)
                self.assertEqual(invalid["errors"].get("annotation_window"), 1)

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

    def test_reading_assets_must_be_pinned_present_and_unchanged(self):
        self.write([self.row()])
        manifest_path = self.corpus / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        asset = manifest["provenance"]["reading_assets"]["model.onnx"]
        manifest["provenance"]["reading_assets"] = {}
        manifest_path.write_text(json.dumps(manifest))
        empty, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(empty["errors"]["reading_assets_unpinned"], 1)
        manifest["provenance"]["reading_assets"] = {"model.onnx": {"path": str(self.model_asset)}}
        manifest_path.write_text(json.dumps(manifest))
        unpinned, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(unpinned["errors"]["reading_asset_unpinned"], 1)
        manifest["provenance"]["reading_assets"] = {"model.onnx": asset}
        manifest_path.write_text(json.dumps(manifest))
        self.model_asset.write_bytes(b"tampered model")
        changed, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(changed["errors"]["reading_asset_changed"], 1)
        self.model_asset.unlink()
        missing, _ = corpus_audit.audit(self.corpus, self.database)
        self.assertEqual(missing["errors"]["reading_asset_missing"], 1)


if __name__ == "__main__":
    unittest.main()
