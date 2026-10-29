"""Paired trajectory metrics must not reward extra boundaries or omit failures."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from trajectory_metrics import summarize


def state(prediction, candidates=None, error=None):
    return dict(prediction=prediction, candidates=candidates or [prediction], milliseconds=10.0, error=error)


def row(identity, gold, full, boundaries, source="chat"):
    return dict(id=identity, document=identity, group=identity, source=source, gold=[gold], full=state(full),
                prefixes=[dict(end=end, reason="reviewed phrase boundary", prefix=state(prefix), continuation=state(suffix))
                          for end, prefix, suffix in boundaries])


class TrajectoryMetrics(unittest.TestCase):
    def test_sentence_weighting_and_same_suffix_pairing(self):
        # Three boundaries in A cannot outweigh B's single boundary.
        a = row("a", "甲乙丙丁戊己", "错乙丙丁戊己",
                [(2, "甲乙", "丙丁戊己"), (3, "甲乙丙", "丁戊己"), (4, "甲乙丙丁", "戊己")])
        b = row("b", "一二三四五六", "一二三四五六", [(2, "错误", "错误错误")], "wiki")
        result = summarize([a, b])["overall"]
        self.assertEqual(result["prefix"]["exact_correct"], 3)
        self.assertEqual(result["prefix"]["exact_accuracy"], .5)
        self.assertEqual(result["continuation"]["exact_accuracy"], .5)
        self.assertEqual(result["continuation_pair"]["uncommitted_suffix_accuracy"], 1)
        self.assertEqual(result["continuation_pair"]["delta_accuracy"], -.5)
        self.assertEqual(result["continuation_pair"]["counts"], dict(helped=0, harmed=1, both_correct=3, both_wrong=0))
        self.assertEqual(result["continuation_pair"]["rates"]["harmed"], .5)
        self.assertEqual(result["prefix_recovery"]["counts"]["regressed"], 3)
        self.assertEqual(result["prefix_recovery"]["counts"]["recovered"], 1)
        subset = result["correct_prefix_only"]
        self.assertEqual((subset["sentences"], subset["comparisons"]), (1, 3))
        self.assertEqual(subset["committed_suffix_accuracy"], 1)

    def test_unsupported_predictions_are_failures_without_dropping_coverage(self):
        failed = row("long", "甲" * 25, "甲" * 25, [(10, "甲" * 10, "甲" * 15)])
        failed["full"]["error"] = "unsupported_window"
        failed["prefixes"][0]["continuation"]["error"] = "decoder_error"
        no_prefix = row("short", "乙" * 6, "乙" * 6, [])
        failed["test_partition"] = "natural-long"
        no_prefix["test_partition"] = "authored-technical"
        failed["unit_kind"] = "utterance"
        no_prefix["unit_kind"] = "title"
        result = summarize([failed, no_prefix])
        overall = result["overall"]
        self.assertEqual(overall["coverage"]["sentences"], 2)
        self.assertEqual(overall["coverage"]["sentences_without_prefixes"], 1)
        self.assertEqual(overall["coverage"]["errors"], 2)
        self.assertEqual(overall["whole"]["exact_accuracy"], .5)
        self.assertEqual(overall["whole"]["top5_accuracy"], .5)
        self.assertEqual(overall["whole"]["character_accuracy"], .5)
        self.assertEqual(overall["whole"]["error_counts"], {"unsupported_window": 1})
        self.assertEqual(overall["continuation_pair"]["counts"]["both_wrong"], 1)
        self.assertEqual(result["by_length"]["25+"]["whole"]["exact_accuracy"], 0)
        self.assertEqual(result["by_length"]["6-12"]["whole"]["exact_accuracy"], 1)
        authored = result["by_partition"]["authored-technical"]
        self.assertEqual(authored["whole"]["exact_accuracy"], 1)
        self.assertIsNone(authored["prefix"]["exact_accuracy"])
        self.assertEqual(authored["coverage"]["requests"], 1)
        self.assertEqual(result["by_unit_kind"]["utterance"]["whole"]["exact_accuracy"], 0)
        self.assertEqual(result["by_unit_kind"]["title"]["whole"]["exact_accuracy"], 1)
        self.assertIsNone(result["by_unit_kind"]["title"]["prefix"]["exact_accuracy"])

    def test_top5_character_alignment_and_bootstrap_are_explicit_and_deterministic(self):
        a = row("a", "甲乙丙丁戊己", "甲乙丙丁戊错", [(3, "甲乙丙", "丁戊己")])
        a["full"]["candidates"] = ["甲乙丙丁戊错", "甲乙丙丁戊己"]
        b = row("b", "一二三四五六", "一二三四五六七", [(3, "一二三", "四五六")])
        result = summarize([a, b])
        self.assertEqual(result, summarize([a, b]))
        self.assertEqual(result["overall"]["whole"]["top5_accuracy"], .5)
        self.assertAlmostEqual(result["overall"]["whole"]["character_accuracy"], (5/6 + 6/7)/2)
        self.assertEqual(result["overall"]["continuation_pair"]["delta_95_ci"], [1, 1])
        self.assertIsNone(summarize([])["overall"]["continuation_pair"]["delta_95_ci"])

    def test_duplicate_sentences_or_boundaries_cannot_inflate_scores(self):
        a = row("a", "甲乙丙丁戊己", "甲乙丙丁戊己", [(2, "甲乙", "丙丁戊己")])
        with self.assertRaisesRegex(ValueError, "each sentence once"):
            summarize([a, deepcopy(a)])
        a["prefixes"].append(deepcopy(a["prefixes"][0]))
        with self.assertRaisesRegex(ValueError, "distinct positions"):
            summarize([a])


if __name__ == "__main__":
    unittest.main()
