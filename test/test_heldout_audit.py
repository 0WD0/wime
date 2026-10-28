import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
spec = importlib.util.spec_from_file_location("audit_heldout", Path(__file__).resolve().parents[1] / "neural/audit-heldout.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HeldoutAudit(unittest.TestCase):
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
