"""Corpus compaction preserves labels and provenance, including no-op corrections."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

NEURAL = Path(__file__).resolve().parents[1] / "neural"
sys.path.insert(0, str(NEURAL))
from corpus_integrity import file_metadata
from corpus_tones import READING_SCHEMA, SYLLABLES, compact_tones

spec = importlib.util.spec_from_file_location("compact_corpus", NEURAL / "compact-corpus.py")
compactor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compactor)


class CompactCorpus(unittest.TestCase):
    def row(self):
        return dict(id="sample-id", document="doc-id", context="他说", gold=["你好"],
                    input=SYLLABLES["ni"] + SYLLABLES["hao"], pinyin=["ni", "hao"], tones=[2, 3],
                    tone_sources=["contextual_tone_sandhi", "phrase_correction"],
                    reading_rules=[["contextual_tone_sandhi"], ["phrase_correction:你好"]],
                    reading_evidence=dict(frontend_before_sandhi=["ni3", "hao3"], final=["ni2", "hao3"]))

    def test_reconstruction_detects_lost_labels_and_rule_evidence(self):
        row = self.row()
        compact = compact_tones(row)
        self.assertEqual(compactor.verify_row(row, compact)["id"], row["id"])
        changed = copy.deepcopy(compact)
        changed["context"] = "错误上下文"
        with self.assertRaisesRegex(ValueError, "text or final labels"):
            compactor.verify_row(row, changed)
        changed = copy.deepcopy(compact)
        changed["reading_notes"].pop()
        with self.assertRaisesRegex(ValueError, "lost reading provenance"):
            compactor.verify_row(row, changed)

    def test_new_version_pins_parent_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, out = root / "dense", root / "sparse"
            source.mkdir()
            for split in compactor.SPLITS:
                (source / f"{split}.jsonl").write_text(json.dumps(self.row(), ensure_ascii=False) + "\n")
            parent = dict(format="wime-corpus-v2", arguments=dict(out=str(source)),
                          outputs={split: file_metadata(source / f"{split}.jsonl") for split in compactor.SPLITS})
            (source / "manifest.json").write_text(json.dumps(parent))
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            manifest = compactor.compact_corpus(source, out)
            self.assertEqual(manifest["reading_schema"], READING_SCHEMA)
            self.assertEqual(manifest["compaction"]["source_manifest"], file_metadata(source / "manifest.json"))
            for split in compactor.SPLITS:
                self.assertEqual(json.loads((out / f"{split}.jsonl").read_text()), compact_tones(self.row()))
                self.assertEqual(manifest["compaction"]["verified"][split]["rows"], 1)
            self.assertEqual(before, {path.name: path.read_bytes() for path in source.iterdir()})
            (source / "train.jsonl").write_text(json.dumps(dict(self.row(), context="变动")) + "\n")
            with self.assertRaisesRegex(ValueError, "source fingerprint"):
                compactor.compact_corpus(source, root / "invalid")
            self.assertFalse((root / "invalid/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
