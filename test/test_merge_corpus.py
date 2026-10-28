"""Merge real on-disk corpus parts and reject incompatible or altered inputs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from corpus_integrity import file_metadata
from corpus_tones import SYLLABLES

spec = importlib.util.spec_from_file_location(
    "merge_corpus", Path(__file__).resolve().parents[1] / "neural/merge-corpus.py")
merger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merger)


class MergeCorpus(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        clean_manifest = self.root / "cleaned-manifest.json"
        clean_manifest.write_text('{"format":"wime-cleaned-corpus-v1"}')
        self.cleaned_metadata = file_metadata(clean_manifest)

    @staticmethod
    def row(identity, document, content, context, source="first", split="train", group=None):
        return dict(id=identity, document=document, group=group or document, content=content,
                    source=source, split=split, context=context, input=SYLLABLES["ni"] + SYLLABLES["hao"], gold=["你好"],
                    category="web", reading="corpus-readings-v1", pinyin=["ni", "hao"], tones=[2, 3],
                    offset=0, reading_window=[0, 24], context_dropped=False,
                    tone_sources=["contextual_tone_sandhi", "mandarin_frontend"],
                    reading_rules=[["contextual_tone_sandhi"], []],
                    reading_evidence=dict(frontend_before_sandhi=["ni3", "hao3"], final=["ni2", "hao3"]))

    def part(self, name, rows, annotation_window=None, cleaned_corpus=None, **argument_overrides):
        directory = self.root / name
        directory.mkdir()
        outputs = {}
        for split in merger.SPLITS:
            path = directory / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                    for row in rows if row["split"] == split), encoding="utf-8")
            outputs[split] = file_metadata(path)
        asset = dict(path="/assets/common", sha256="asset-digest", bytes=10)
        manifest = dict(format="wime-corpus-v2", stats=dict(rows=len(rows)), readings={}, tone_policy=merger.TONE_POLICY,
                        annotation_window=annotation_window or dict(max_characters=256),
                        arguments=dict(sources=name, out=str(directory), max_chars="100", seed="fixed",
                                       density="0.12", blank_share="0.15"),
                        sampling=dict(max_context=64, max_span=24), protection={"heldout.json": "heldout-digest"},
                        providers=["CUDAExecutionProvider"], packages={"g2pw": "0.1.1"},
                        provenance=dict(database=asset, code={"prepare-corpus.py": asset},
                                        cleaned_corpus=cleaned_corpus or self.cleaned_metadata,
                                        reading_assets={"g2pw.onnx": asset}, manifest_sha256="source-manifest",
                                        files={f"{name}/raw": f"{name}-raw-digest"}), outputs=outputs)
        manifest["arguments"].update(argument_overrides)
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory

    def merge(self, parts):
        with contextlib.redirect_stdout(io.StringIO()):
            return merger.merge(self.root / "merged", parts)

    def test_merge_keeps_first_content_owner_and_deduplicates_across_splits(self):
        first = self.part("first", [
            self.row("a1", "a", "content-a", "此前，输入的文字。"),
            self.row("a2", "a", "content-a", "同一个文档中另一个位置"),
            self.row("dev", "dev", "content-dev", "验证上下文", split="dev"),
        ])
        second = self.part("second", [
            self.row("copy", "copy", "content-a", "不同位置也必须去除", source="second"),
            self.row("sample", "b", "content-b", "此前输入的文字", source="second", split="test"),
            self.row("kept", "c", "content-c", "", source="second", split="test"),
        ], max_chars="200")
        result = self.merge([first, second])
        output = self.root / "merged"
        ids = {split: [json.loads(line)["id"] for line in (output / f"{split}.jsonl").read_text().splitlines()]
               for split in merger.SPLITS}
        self.assertEqual(ids, dict(train=["a1", "a2"], dev=["dev"], test=["kept"]))
        self.assertEqual(result["stats"]["second/duplicate_document_rows"], 1)
        self.assertEqual(result["stats"]["second/duplicate_sample"], 1)
        self.assertEqual(result["stats"]["blank_context"], 1)
        self.assertEqual(result["stats"]["first/documents"], 2)
        self.assertEqual(result["stats"]["first/train"], 2)
        self.assertEqual(len(result["parts"]), 2)
        self.assertEqual(set(result["provenance"]["files"]), {"first/raw", "second/raw"})
        self.assertTrue((output / "manifest.json").is_file())
        self.assertFalse(list(output.glob("*.partial")))
        self.assertEqual(result["outputs"]["test"], file_metadata(output / "test.jsonl"))

    def test_incompatible_seed_or_code_is_rejected_before_output(self):
        first = self.part("first", [])
        second = self.part("second", [], seed="different")
        with self.assertRaisesRegex(ValueError, "incompatible.*arguments"):
            self.merge([first, second])
        self.assertFalse((self.root / "merged").exists())
        path = second / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["arguments"]["seed"] = "fixed"
        manifest["provenance"]["code"]["prepare-corpus.py"]["sha256"] = "changed-code"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "incompatible.*provenance"):
            self.merge([first, second])

    def test_mutated_part_output_is_rejected(self):
        part = self.part("first", [])
        (part / "train.jsonl").write_text("modified after completion\n")
        with self.assertRaisesRegex(ValueError, "changed corpus part output"):
            self.merge([part])
        self.assertFalse((self.root / "merged").exists())

    def test_group_crossing_splits_fails_even_for_duplicate_content(self):
        first = self.part("first", [self.row("a", "a", "same", "上下文", group="conversation")])
        second = self.part("second", [self.row("b", "b", "same", "上下文", source="second",
                                              split="test", group="conversation")])
        with self.assertRaisesRegex(ValueError, "group crosses splits"):
            self.merge([first, second])
        self.assertFalse((self.root / "merged/manifest.json").exists())

    def test_duplicate_ids_fail_instead_of_silently_deduplicating(self):
        first = self.part("first", [self.row("same-id", "a", "a", "上下文")])
        second = self.part("second", [self.row("same-id", "b", "b", "上下文", source="second")])
        with self.assertRaisesRegex(ValueError, "duplicate row id"):
            self.merge([first, second])
        self.assertFalse((self.root / "merged/manifest.json").exists())

    def test_tone_policy_is_preserved_and_counts_exclude_deduplicated_rows(self):
        neutral = self.row("a", "a", "a", "左边的上下文")
        neutral.update(gold=["我们"], input=SYLLABLES["wo"] + SYLLABLES["men"],
                       pinyin=["wo", "men"], tones=[3, 5], tone_sources=["mandarin_frontend", "mandarin_frontend"],
                       reading_rules=[[], []],
                       reading_evidence=dict(frontend_before_sandhi=["wo3", "men5"], final=["wo3", "men5"]))
        greeting = self.row("b", "b", "b", "另一段上下文", source="second", split="test")
        duplicate = dict(neutral, id="copy", document="copy", group="copy", content="copy", source="second")
        first = self.part("first", [neutral])
        second = self.part("second", [duplicate, greeting])
        result = self.merge([first, second])
        self.assertEqual(result["tone_policy"], merger.TONE_POLICY)
        self.assertEqual({key: value for key, value in result["stats"].items() if key.startswith("tone/")},
                         {"tone/total": 4, "tone/neutral": 1})
        self.assertEqual(result["stats"]["second/duplicate_sample"], 1)
        retained = json.loads((self.root / "merged/test.jsonl").read_text())
        self.assertEqual(retained["tones"], [2, 3])
        self.assertEqual(retained["reading_evidence"], greeting["reading_evidence"])

    def test_other_annotation_schemes_are_rejected(self):
        first = self.part("first", [])
        path = first / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["tone_policy"] = {"name": "citation-v1"}
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "unsupported tone_policy"):
            self.merge([first])

    def test_fabricated_sandhi_is_rejected_before_merge(self):
        row = self.row("one", "doc", "content", "语境")
        row["reading_evidence"]["frontend_before_sandhi"][0] = "ni2"
        invalid = self.part("invalid", [row])
        with self.assertRaisesRegex(ValueError, "actual corrections and tone changes"):
            self.merge([invalid])
        self.assertFalse((self.root / "merged/manifest.json").exists())

    def test_parts_must_use_the_same_cleaned_manifest_content(self):
        first_manifest = self.root / "cleaned-first.json"
        second_manifest = self.root / "cleaned-second.json"
        first_manifest.write_text('{"format":"wime-cleaned-corpus-v1"}')
        second_manifest.write_text(first_manifest.read_text())
        first = self.part("first", [], cleaned_corpus=file_metadata(first_manifest))
        second = self.part("second", [], cleaned_corpus=file_metadata(second_manifest))
        merged = self.merge([first, second])
        self.assertEqual(merged["provenance"]["cleaned_corpus"], file_metadata(first_manifest))
        second_manifest.write_text('{"format":"different-cleaned-content"}')
        changed = self.part("changed", [], cleaned_corpus=file_metadata(second_manifest))
        with self.assertRaisesRegex(ValueError, "incompatible.*cleaned_corpus"):
            self.merge([first, changed])

    def test_parts_require_cleaned_provenance_and_annotation_windows(self):
        for field in ("cleaned_corpus", "annotation_window"):
            with self.subTest(field=field):
                part = self.part(field, [])
                path = part / "manifest.json"
                manifest = json.loads(path.read_text())
                (manifest["provenance"] if field == "cleaned_corpus" else manifest).pop(field)
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, field):
                    self.merge([part])
                self.assertFalse((self.root / "merged").exists())

    def test_annotation_window_is_preserved_in_merged_output(self):
        policy = dict(core=4, left_halo=2, right_halo=2, max_characters=8)
        row = dict(self.row("a", "a", "a", "上下文"), offset=10, reading_window=[8, 14])
        first = self.part("first", [row], annotation_window=policy)
        second = self.part("second", [], annotation_window=policy)
        result = self.merge([first, second])
        self.assertEqual(result["annotation_window"], policy)
        retained = json.loads((self.root / "merged/train.jsonl").read_text())
        self.assertEqual((retained["offset"], retained["reading_window"]), (10, [8, 14]))

    def test_different_annotation_windows_cannot_be_merged(self):
        first = self.part("first", [], annotation_window={"max_characters": 8})
        second = self.part("second", [], annotation_window={"max_characters": 16})
        with self.assertRaisesRegex(ValueError, "incompatible.*annotation_window"):
            self.merge([first, second])
        self.assertFalse((self.root / "merged").exists())

    def test_sample_outside_reading_window_is_rejected(self):
        row = dict(self.row("a", "a", "a", "上下文"), offset=6, reading_window=[0, 7])
        part = self.part("first", [row], annotation_window={"max_characters": 8})
        with self.assertRaisesRegex(ValueError, "sample interval must be contained"):
            self.merge([part])
        self.assertFalse((self.root / "merged/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
