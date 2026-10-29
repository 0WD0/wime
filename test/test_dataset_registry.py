"""Dataset identities survive relocation and reject changed or misused artifacts."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from dataset_registry import Registry, DATASET_PYTHON, fingerprint
from replay import Protection


class DatasetRegistry(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "datasets.json"
        self.catalog.write_text(json.dumps(dict(version=1, datasets={})))
        self.data = self.root / "storage" / "example"
        self.data.mkdir(parents=True)
        self.rows = [dict(context="上文", input="jayi", gold=["甲乙"])]
        for name in ("train", "dev", "test"):
            (self.data / (name + ".jsonl")).write_text(json.dumps(self.rows[0], ensure_ascii=False) + "\n")
        (self.data / "manifest.json").write_text('{}\n')
        (self.data / "cases.json").write_text(json.dumps(self.rows))
        self.registry = Registry(self.catalog, roots={"corpus": self.data.parent})

    def corpus(self):
        return dict(id="text-v1", kind="corpus", status="active", annotation="contextual-v1", parents=[],
                    location=dict(root="corpus", path="example"),
                    files={"manifest":dict(path="manifest.json"),
                           **{split:dict(path=split + ".jsonl", format="jsonl") for split in ("train", "dev", "test")}})

    def evaluation(self):
        return dict(id="eval-v1", kind="evaluation", status="regression", annotation="contextual-v1", parents=[],
                    location=dict(root="corpus", path="example"),
                    files=dict(cases=dict(path="cases.json", format="json-array")))

    def test_registration_pins_rows_hashes_and_version_ids(self):
        self.registry.register(self.corpus())
        record = self.registry.entry("text-v1")
        self.assertEqual(record["files"]["train"]["rows"], 1)
        self.assertEqual(self.registry.verify("text-v1"), 4)
        with self.assertRaisesRegex(ValueError, "new version ID"):
            self.registry.register(self.corpus())
        (self.data / "train.jsonl").write_text('{}\n')
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.registry.training("text-v1")

    def test_relocation_keeps_identity_and_training_does_not_read_test(self):
        self.registry.register(self.corpus())
        first = self.registry.training("text-v1")[2]
        moved = self.root / "elsewhere"
        shutil.move(self.data.parent, moved)
        (moved / "example/test.jsonl").unlink()
        relocated = Registry(self.catalog, roots={"corpus": moved})
        directory, _, second = relocated.training("text-v1")
        self.assertEqual(directory, moved / "example")
        self.assertEqual(first, second)

    def test_training_protects_registered_cases_and_rejects_evaluation_input(self):
        self.registry.register(self.evaluation())
        spec = self.corpus()
        spec["exclude"] = [dict(dataset="eval-v1", files=["cases"])]
        self.registry.register(spec)
        directory, excluded, snapshot = self.registry.training("text-v1")
        self.assertEqual(excluded, [self.data / "cases.json"])
        self.assertEqual(snapshot["exclude"][0]["id"], "eval-v1")
        with self.assertRaisesRegex(ValueError, "not a training corpus"):
            self.registry.training("eval-v1")
        with self.assertRaisesRegex(ValueError, "not an evaluation suite"):
            self.registry.evaluation("text-v1")
        (self.data / "cases.json").write_text('[]\n')
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.registry.training("text-v1")

    def test_failed_registration_leaves_catalog_unchanged(self):
        before = self.catalog.read_bytes()
        (self.data / "train.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            self.registry.register(self.corpus())
        self.assertEqual(self.catalog.read_bytes(), before)
        self.assertEqual(self.registry.datasets, {})

    def test_new_heldout_group_protects_existing_training_recipes(self):
        self.registry.register(self.corpus())
        heldout = dict(self.rows[0], document="first-turn", group="conversation")
        (self.data / "cases.json").write_text(json.dumps([heldout]))
        self.registry.register(dict(self.evaluation(), exclude_from_training=True))
        _, excluded, snapshot = self.registry.training("text-v1")
        self.assertEqual(snapshot["exclude"][0]["id"], "eval-v1")
        guard = Protection(excluded)
        other_turn = dict(context="全新的左文", input="abcd", gold=["另句"],
                          document="second-turn", group="conversation")
        self.assertTrue(guard.contains(other_turn))
        self.assertFalse(guard.contains(dict(other_turn, group="unrelated")))

    @unittest.skipUnless(DATASET_PYTHON.is_file(), "dataset runtime required")
    def test_explicit_dvc_stage_rejects_changed_and_missing_content(self):
        def dvc(*args):
            subprocess.run([str(DATASET_PYTHON), "-m", "dvc", *args], cwd=self.root,
                           check=True, capture_output=True)
        dvc("init", "--no-scm")
        dvc("add", "storage/example")
        metadata = self.data.with_suffix(".dvc")
        entry = dict(storage=dict(type="dvc", metadata=str(metadata.relative_to(self.root)),
                                  sha256=fingerprint(metadata)))
        registry = Registry(self.catalog, roots={"repo": self.root})
        registry._verify_dvc(entry)
        (self.data / "train.jsonl").write_text('changed\n')
        with self.assertRaisesRegex(ValueError, "checkout or repair"):
            registry._verify_dvc(entry)
        (self.data / "train.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "checkout or repair"):
            registry._verify_dvc(entry)


if __name__ == "__main__":
    unittest.main()
