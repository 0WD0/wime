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

    @staticmethod
    def row(identity, document, content, context, source="first", split="train", group=None):
        return dict(id=identity, document=document, group=group or document, content=content,
                    source=source, split=split, context=context, input=SYLLABLES["ni"] + SYLLABLES["hao"], gold=["你好"],
                    category="web", reading="corpus-readings-v1")

    def part(self, name, rows, tone_policy=None, **argument_overrides):
        directory = self.root / name
        directory.mkdir()
        outputs = {}
        for split in merger.SPLITS:
            path = directory / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                    for row in rows if row["split"] == split), encoding="utf-8")
            outputs[split] = file_metadata(path)
        asset = dict(path="/assets/common", sha256="asset-digest", bytes=10)
        manifest = dict(format="wime-corpus-v2", stats=dict(rows=len(rows)), readings={},
                        arguments=dict(sources=name, out=str(directory), max_chars="100", seed="fixed",
                                       density="0.12", blank_share="0.15"),
                        sampling=dict(max_context=64, max_span=24), protection={"heldout.json": "heldout-digest"},
                        providers=["CUDAExecutionProvider"], packages={"g2pw": "0.1.1"},
                        provenance=dict(database=asset, code={"prepare-corpus.py": asset},
                                        reading_assets={"g2pw.onnx": asset}, manifest_sha256="source-manifest",
                                        files={f"{name}/raw": f"{name}-raw-digest"}), outputs=outputs)
        manifest["arguments"].update(argument_overrides)
        if tone_policy is not None:
            manifest["tone_policy"] = tone_policy
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
        policy = dict(name="gpt-sovits-mandarin-v1", neutral=5, unknown=None)
        neutral = self.row("a", "a", "a", "左边的上下文")
        neutral.update(gold=["我们"], input=SYLLABLES["wo"] + SYLLABLES["men"],
                       pinyin=["wo", "men"], tones=[3, 5], tone_sources=["mandarin_frontend", "mandarin_frontend"],
                       reading_rules=[[], []],
                       reading_evidence=dict(frontend_before_sandhi=["wo3", "men5"], final=["wo3", "men5"]))
        unknown = self.row("b", "b", "b", "另一段上下文", source="second", split="test")
        unknown.update(pinyin=["ni", None], tones=[3, None], tone_sources=["mandarin_frontend", "unresolved"],
                       reading_rules=[[], []],
                       reading_evidence=dict(frontend_before_sandhi=["ni3", None], final=["ni3", None]))
        duplicate = dict(neutral, id="copy", document="copy", group="copy", content="copy", source="second")
        first = self.part("first", [neutral], tone_policy=policy)
        second = self.part("second", [duplicate, unknown], tone_policy=policy)
        result = self.merge([first, second])
        self.assertEqual(result["tone_policy"], policy)
        self.assertEqual({key: value for key, value in result["stats"].items() if key.startswith("tone/")},
                         {"tone/known": 3, "tone/unknown": 1, "tone/neutral": 1})
        self.assertEqual(result["stats"]["second/duplicate_sample"], 1)
        retained = json.loads((self.root / "merged/test.jsonl").read_text())
        self.assertEqual(retained["tones"], [3, None])
        self.assertEqual(retained["reading_evidence"], unknown["reading_evidence"])

    def test_tone_policy_mismatch_is_rejected(self):
        first = self.part("first", [])
        second = self.part("second", [], tone_policy={"name": "gpt-sovits-mandarin-v1"})
        with self.assertRaisesRegex(ValueError, "incompatible.*tone_policy"):
            self.merge([first, second])
        path = first / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["tone_policy"] = {"name": "citation-v1"}
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "incompatible.*tone_policy"):
            self.merge([first, second])

    def test_tone_policy_requires_valid_aligned_annotations(self):
        row = self.row("a", "a", "a", "上下文")
        row.update(pinyin=["ni", "hao"], tones=[3, 3], tone_sources=["contextual_tone_sandhi", "mandarin_frontend"],
                   reading_rules=[["contextual_tone_sandhi"], []],
                   reading_evidence=dict(frontend_before_sandhi=["ni3", "hao3"], final=["ni2", "hao3"]))
        part = self.part("first", [row], tone_policy={"name": "gpt-sovits-mandarin-v1"})
        with self.assertRaisesRegex(ValueError, "does not match the final frontend reading"):
            self.merge([part])
        self.assertFalse((self.root / "merged/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
