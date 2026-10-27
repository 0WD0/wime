"""Right-aware ranking must isolate suffix likelihood and retain blind phases."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
spec = importlib.util.spec_from_file_location(
    "experiment_japanese_right_ranking",
    Path(__file__).resolve().parents[1] / "neural/experiment-japanese-right-ranking.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
previous = experiment.previous


def case(identifier="1", split="dev", right="を学ぶ。", answers=None):
    c = dict(id=identifier, split=split, context="私は", kana="にほんご", right_context=right,
             topic_group=split + identifier, domain="controlled", case_type="disambiguation")
    if answers is not None:
        c["answers"] = answers
    return c


def prediction(c, z=(-1., -2.), q=(-2., -.1), suffix=(0., -2.)):
    p = dict(generated=["日本語", "二本語"], cands=["日本語", "二本語"],
             z=list(z), q=list(q), r=list(suffix) if c["right_context"] else [0., 0.])
    return dict(input=previous.public_input(c), pools={name: dict(p) for name in experiment.CONDITIONS},
                q_union=sorted(p["cands"]), q_union_ms=4., q_right_union_ms=1. if c["right_context"] else 0.)


class SuffixScoring(unittest.TestCase):
    def fake(self):
        calls = []
        encoded = []
        def tokenize(text, **kw):
            encoded.append((text, kw))
            return {"input_ids": [ord(c) for c in text]}
        def tails(context, prefixes, suffixes):
            calls.append((context, prefixes, suffixes))
            return [-float(len(p)) for p in prefixes]
        return SimpleNamespace(table=SimpleNamespace(tokenizer=tokenize),
                               lm_context=lambda c: [10, 20], lm_tails=tails), calls, encoded

    def test_suffix_only_common_tokens_and_independent_right_character_budget(self):
        decoder, calls, encoded = self.fake()
        values, count = experiment.suffix_score(decoder, "前" * 80, ["紙を", "髪の毛を"], "後" * 50)
        self.assertEqual(values, [-2., -4.])
        self.assertEqual(count, 40)
        context, prefixes, tails = calls[0]
        self.assertEqual(context, [10, 20])
        self.assertEqual(prefixes, [[ord(c) for c in t] for t in ["紙を", "髪の毛を"]])
        self.assertEqual(tails, [[ord("後")] * 40] * 2)
        self.assertEqual(encoded[-1], ("後" * 40, {"add_special_tokens": False}))

    def test_empty_right_or_pool_never_calls_model(self):
        decoder, calls, encoded = self.fake()
        self.assertEqual(experiment.suffix_score(decoder, "", ["日本語"], ""), ([0.], 0))
        self.assertEqual(experiment.suffix_score(decoder, "", [], "後"), ([], 0))
        self.assertFalse(calls)
        self.assertFalse(encoded)

    def test_actual_lm_tails_starts_at_last_candidate_token_and_excludes_candidate_score(self):
        import torch
        vocabulary = torch.arange(10, dtype=torch.float32)
        def body(input_ids):
            # Predictive logits depend on the last input token; arbitrary but
            # exact enough to expose one-position and prefix-boundary mistakes.
            hidden = -(vocabulary[None, None, :] - input_ids[:, :, None].float()) ** 2
            return SimpleNamespace(last_hidden_state=hidden)
        tokenizer = lambda text, **kw: {"input_ids": {"a": [3], "bb": [4, 5], "suffix": [6, 7]}[text]}
        model = SimpleNamespace(model=body, lm_head=lambda hidden: hidden)
        scorer = SimpleNamespace(device="cpu", pad=0, model=model)
        decoder = previous.Decoder(SimpleNamespace(tokenizer=tokenizer), scorer, None)
        decoder.lm_context = lambda context: [1, 2]
        values, count = experiment.suffix_score(decoder, "left", ["a", "bb"], "suffix")
        logp = lambda last, nxt: float((-(vocabulary-last)**2).log_softmax(-1)[nxt])
        self.assertEqual(count, 2)
        self.assertAlmostEqual(values[0], logp(3, 6) + logp(6, 7), places=5)
        self.assertAlmostEqual(values[1], logp(5, 6) + logp(6, 7), places=5)

    def test_cached_q_preserved_and_union_scored_once_without_labels(self):
        decoder, calls, encoded = self.fake()
        c = case(answers=["秘密の正解"])
        original = prediction(c)
        before = json.dumps(original, ensure_ascii=False)
        with patch.object(previous, "timed", side_effect=lambda f: (f(), 1.)):
            row = experiment.inspect(c, decoder, original)
        self.assertNotIn("秘密", json.dumps(row, ensure_ascii=False))
        self.assertEqual(json.dumps(original, ensure_ascii=False), before)
        self.assertEqual(len(calls), 1)
        self.assertEqual(row["pool_source"], "cached")
        for condition in experiment.CONDITIONS:
            self.assertEqual(row["pools"][condition]["q"], original["pools"][condition]["q"])
        experiment.align_rows([c], [row])

    def test_absent_right_reuses_cached_scores_without_model_or_timer(self):
        decoder, calls, encoded = self.fake()
        c = case(right="")
        with patch.object(previous, "timed") as timed:
            row = experiment.inspect(c, decoder, prediction(c))
        timed.assert_not_called()
        self.assertFalse(calls)
        self.assertEqual(row["q_right_union_ms"], 0.)
        for setting in experiment.GRID:
            self.assertEqual(experiment.ranking(c, row["pools"]["right"], setting),
                             previous.ranking(row["pools"]["right"], previous.BASELINE))


class SelectionAndAlignment(unittest.TestCase):
    def test_selection_can_distinguish_suffix_from_gating_and_uses_dev_only(self):
        # First case demands suffix help; second demands keeping q. No gate
        # alone gets both correct, but a positive suffix weight can do so.
        a, b = case("1", answers=["日本語"]), case("2", answers=["二本語"])
        rows = [prediction(a, z=(-2., -1.), q=(0., 0.), suffix=(0., -10.)),
                prediction(b, z=(-1., -2.), q=(-3., 0.), suffix=(0., 0.))]
        chosen = experiment.select_setting([a, b], rows)
        self.assertGreater(chosen["right_weight"], 0)
        self.assertEqual(chosen["selected_gate"]["top1"], 1)
        self.assertEqual(chosen["selected_suffix"]["top1"], 2)
        a["split"] = "sealed"
        with self.assertRaisesRegex(ValueError, "only dev"):
            experiment.select_setting([a, b], rows)

    def test_ties_prefer_baseline_and_smallest_suffix_in_suffix_family(self):
        c = case(answers=["日本語"])
        chosen = experiment.select_setting([c], [prediction(c, z=(0., -100.), q=(0., 0.), suffix=(0., 0.))])
        self.assertEqual((chosen["q_weight"], chosen["right_weight"]), experiment.BASELINE)
        self.assertEqual(chosen["selected_suffix"]["right_weight"], .175)
        self.assertEqual(chosen["selected_suffix"]["q_weight"], .7)

    def test_pool_alignment_rejects_changed_shared_scores_or_invalid_suffix(self):
        c = case()
        row = prediction(c)
        row["pools"]["right"] = dict(row["pools"]["right"], r=[0., 99.])
        with self.assertRaisesRegex(ValueError, "identical Qwen scores"):
            experiment.align_rows([c], [row])
        row = prediction(c)
        row["pools"]["right"] = dict(row["pools"]["right"], r=[float("nan"), 0.])
        with self.assertRaisesRegex(ValueError, "non-finite suffix"):
            experiment.align_rows([c], [row])

    def test_absent_right_rejects_divergent_cached_candidate_pools(self):
        c = case(right="")
        row = prediction(c)
        row["pools"]["right"] = dict(row["pools"]["right"], z=[0., -10.])
        with self.assertRaisesRegex(ValueError, "identical baseline pools"):
            experiment.align_rows([c], [row])

    def test_missing_or_changed_input_rejected_before_report(self):
        c = case()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            experiment.align_rows([c], [])
        row = prediction(c)
        row["input"] = dict(row["input"], right_context="different")
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            experiment.align_rows([c], [row])

    def test_suffix_changes_do_not_affect_oracle_or_candidate_pool(self):
        c = case(answers=["日本語"])
        r = prediction(c)
        baseline = experiment.metrics([c], [r], "right", experiment.BASELINE)
        suffix = experiment.metrics([c], [r], "right", (.7, .7))
        self.assertEqual((baseline["valid_pool"], suffix["valid_pool"]), (1, 1))
        self.assertEqual((baseline["top1"], suffix["top1"]), (0, 1))

    def test_cluster_contrast_keeps_paired_sentences_together(self):
        cases = [case(str(i), answers=["日本語"]) for i in range(2)]
        for c in cases:
            c["topic_group"] = "shared-ambiguity"
        rows = [prediction(c) for c in cases]
        comparison = experiment.compare(cases, rows, "right", (.7, .7), "right", experiment.BASELINE)
        self.assertEqual((comparison["wins"], comparison["losses"], comparison["topic_groups"]), (2, 0, 1))
        self.assertEqual(comparison["cluster_bootstrap95_delta_pp"], [100., 100.])

    def test_report_keeps_labels_closed_until_all_blind_rows_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev, sealed = case(answers=["日本語"]), case("2", split="sealed")
            (root / "dev.json").write_text(json.dumps([dev]))
            (root / "inputs.json").write_text(json.dumps([sealed]))
            (root / "labels.json").write_text("sealed labels must stay unopened")
            manifest = dict(files={"source": "fixed"})
            (root / "dev.rows.jsonl").write_text(json.dumps(prediction(dev)) + "\n")
            selected = experiment.select_setting([dev], [prediction(dev)])
            selection = dict(manifest=manifest, dev_rows_sha256=previous.digest(root / "dev.rows.jsonl"), selected=selected)
            (root / "selection.json").write_text(json.dumps(selection))
            (root / "sealed.run.json").write_text(json.dumps(dict(selection_sha256=previous.digest(root / "selection.json"))))
            argv = ["experiment", "report", "--dev-cases", str(root / "dev.json"),
                    "--sealed-inputs", str(root / "inputs.json"), "--sealed-labels", str(root / "labels.json"),
                    "--evaluation-manifest", str(root / "manifest.json"), "--out", str(root),
                    "--table", str(root), "--rescorer", str(root), "--zenz", str(root)]
            with patch.object(sys, "argv", argv), patch.object(experiment, "fingerprint", return_value=manifest), \
                    patch.object(previous, "read_labels") as labels:
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    experiment.main()
                labels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
