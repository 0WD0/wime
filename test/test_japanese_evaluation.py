import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("prepare_eval", Path(__file__).resolve().parents[1] / "neural/prepare-japanese-evaluation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvaluationCorpus(unittest.TestCase):
    def test_overlap_checks_context_and_segment_boundary(self):
        case = dict(id="c", context="", right_context="", answers=["これは隠れた評価文章です。"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"train.jsonl"
            path.write_text(json.dumps(dict(context="これは隠れた", text="評価文章です。", source="a", doc="1")) + "\n")
            result = module.overlap([case], [path])
            hit = result["cases"]["c"][str(path.resolve())]
            self.assertEqual(hit["full_target_rows"], 1)
            self.assertEqual(hit["full_surrounding_rows"], 1)

    def test_short_target_is_not_evidence_but_surrounding_is(self):
        case = dict(id="c", context="川の上にある", right_context="を渡って隣の村に向かう。", answers=["橋"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"train.jsonl"
            path.write_text(json.dumps(dict(text="川の上にある橋を渡って隣の村に向かう。")) + "\n")
            hit = module.overlap([case], [path])["cases"]["c"][str(path.resolve())]
            self.assertEqual(hit["full_target_rows"], 0)
            self.assertEqual(hit["full_surrounding_rows"], 1)

    def pool(self):
        texts = ["昨日は雪山を登りました。", "明日は海辺で泳ぎます。", "青い小鳥が空を飛んだ。", "赤い花が庭に咲いた。"]
        pool = [dict(id=str(i), kana="かな"+str(i), answers=[t], context="", right_context="", source="reserve") for i,t in enumerate(texts)]
        reviews = [dict(id=r["id"], status="approved", original_kana=r["kana"], original_answers=r["answers"]) for r in pool]
        return pool, reviews, dict(cases={r["id"]:{} for r in pool})

    def test_only_reviewed_disjoint_rows_can_be_sealed(self):
        pool, reviews, audit = self.pool()
        cases, info = module.assemble(pool, reviews, [], audit, [], 2)
        self.assertEqual([c["split"] for c in cases], ["dev", "sealed", "dev", "sealed"])
        audit["cases"][pool[0]["id"]] = {"training": {"shared8_rows":1}}
        with self.assertRaisesRegex(ValueError, "need 4"):
            module.assemble(pool, reviews, [], audit, [], 2)

    def test_changed_source_invalidates_review(self):
        pool, reviews, audit = self.pool()
        pool[0]["kana"] += "変更"
        with self.assertRaisesRegex(ValueError, "review no longer matches"):
            module.assemble(pool, reviews, [], audit, [], 1)

    def test_controlled_groups_cannot_cross_splits(self):
        pool, reviews, audit = self.pool()
        controlled = [dict(id="new1", split="dev", topic_group="shared"), dict(id="new2", split="sealed", topic_group="shared")]
        with self.assertRaisesRegex(ValueError, "topic group crosses"):
            module.assemble(pool, reviews, controlled, audit, [], 1)

    def test_future_training_filter_also_protects_context(self):
        spec = importlib.util.spec_from_file_location("prepare_train", Path(__file__).resolve().parents[1] / "neural/prepare-japanese.py")
        prep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prep)
        clean = prep.evaluation_filter(["この文章は評価専用です。", "数字１２３４５６７８"])
        self.assertFalse(clean("評価専用です。", "この文章は"))
        self.assertFalse(clean("別の続き。", "この文章は評価専用です。"))
        self.assertFalse(clean("数字12345678"))
        self.assertTrue(clean("関係のない学習用の文章。"))


if __name__ == "__main__":
    unittest.main()
