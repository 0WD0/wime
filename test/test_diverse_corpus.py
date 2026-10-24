"""Corpus supervision and held-out context exclusion contracts."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

NEURAL = Path(__file__).resolve().parents[1] / "neural"
sys.path.insert(0, str(NEURAL))
spec = importlib.util.spec_from_file_location("prepare_diverse", NEURAL / "prepare-diverse.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class DiverseCorpus(unittest.TestCase):
    def test_cloze_labels_restore_original_order_without_candidate_leakage(self):
        text, spans = prepare.restore_idioms(dict(content="先#idiom#，再#idiom#。",
                                                 groundTruth=["脚踏实地", "持之以恒"],
                                                 candidates=[["错误答案"]], realCount=2))
        self.assertEqual(text, "先脚踏实地，再持之以恒。")
        self.assertEqual([text[a:b] for a,b in spans], ["脚踏实地", "持之以恒"])
        self.assertEqual(text[:spans[1][0]], "先脚踏实地，再")
        with self.assertRaises(ValueError):
            prepare.restore_idioms(dict(content="#idiom#", groundTruth=[], realCount=0))

    def test_protection_excludes_context_pairs_not_all_uses_of_a_word(self):
        row = dict(context="我们为这条固定评测专门写了一段足够长的自然上下文", gold=["持之以恒"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps([row], ensure_ascii=False))
            protection = prepare.Protection([path])
        self.assertTrue(protection.contains(row))
        self.assertTrue(protection.overlaps("标题。" + row["context"] + "，持之以恒。"))
        self.assertFalse(protection.contains(dict(context="独立来源的另一个句子", gold=["持之以恒"])))
        self.assertFalse(protection.overlaps("一个独立来源谈到学习要持之以恒。"))


if __name__ == "__main__":
    unittest.main()
