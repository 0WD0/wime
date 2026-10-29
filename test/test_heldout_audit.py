import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import hashlib
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
spec = importlib.util.spec_from_file_location("audit_heldout", Path(__file__).resolve().parents[1] / "neural/audit-heldout.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HeldoutAudit(unittest.TestCase):
    def test_dvc_training_file_is_verified_without_a_leaf_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            path.write_text(json.dumps(dict(context="甲乙丙丁", gold=["戊己庚辛"])) + "\n")
            registry = Mock()
            registry.datasets = {"version": dict(kind="corpus", storage=dict(type="dvc"),
                                                  files=dict(train=dict(path="train.jsonl")))}
            registry.path.return_value = path
            registry.snapshot.return_value = dict(verified=True)
            report = module.audit([dict(id="case", document="doc", group="group",
                                        text="甲乙丙丁戊己庚辛")], registry)
            registry.verify.assert_called_once_with("version", ["train"])
            self.assertEqual(report["hits"]["case"]["datasets"], ["version"])
            registry.verify.side_effect = ValueError("changed DVC artifact")
            with self.assertRaisesRegex(ValueError, "changed DVC"):
                module.audit([], registry)

    def test_overlapping_matches_cross_the_committed_boundary(self):
        detector = module.Overlap([
            dict(id="a", document="one", group="thread", text="甲乙丙丁戊己庚辛"),
            dict(id="b", document="two", group="other", text="乙丙丁戊己庚辛壬")])
        row = dict(context="甲乙丙丁", gold=["戊己庚辛壬"])
        self.assertEqual(set(detector.scan(row)), {"a", "b"})
        self.assertEqual(set(detector.scan(dict(context="", gold=["无关文字"], group="thread"))), {"a"})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            for escaped in (False, True):
                path.write_text(json.dumps(row, ensure_ascii=escaped) + "\n")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(list(detector.archived_rows(path, digest)), [row])


if __name__ == "__main__":
    unittest.main()
