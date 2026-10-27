"""Research controls must isolate context/ranking and keep sealed labels blind."""
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
    "experiment_japanese", Path(__file__).resolve().parents[1] / "neural/experiment-japanese.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def case(split="dev", identifier="1", answers=None, right=""):
    c = dict(id=identifier, split=split, context="", kana="にほんご", right_context=right,
             topic_group=split + identifier, domain="natural")
    if answers is not None:
        c["answers"] = answers
    return c


def prediction(c, z=(-1., -2.), q=(-2., -.1)):
    p = dict(generated=["日本語", "二本語"], cands=["日本語", "二本語"], z=list(z), q=list(q))
    return dict(input=experiment.public_input(c), pools={name: dict(p) for name in experiment.CONDITIONS})


class Prompts(unittest.TestCase):
    def test_empty_right_preserves_production_prompt_exactly(self):
        zenz = experiment.RightZenz.__new__(experiment.RightZenz)
        zenz.encode = lambda s: s
        for context in ("", "前" * 55):
            self.assertEqual(zenz.prompt(context, "にほんご"),
                             experiment.Zenz.prompt(zenz, context, "にほんご"))

    def test_right_uses_native_order_and_independent_character_budgets(self):
        zenz = experiment.RightZenz.__new__(experiment.RightZenz)
        zenz.encode = lambda s: s
        zenz.right_context = "後" * 50
        self.assertEqual(zenz.prompt("前" * 50, "にほんご"),
                         "\uee02" + "前" * 40 + "\uee07" + "後" * 40 + "\uee00ニホンゴ\uee01")


class Pools(unittest.TestCase):
    def fake(self):
        class FakeZenz:
            right_context = ""

            def __init__(self):
                self.generated, self.scored = [], []

            def generate(self, context, kana):
                self.generated.append(self.right_context)
                return ["日本語", "二本語", "不正"] if self.right_context else ["日本語", "日本語", "不正"]

            def score(self, context, kana, texts):
                self.scored.append((self.right_context, list(texts)))
                return [-float(len(t)) - bool(self.right_context) for t in texts]

        class FakeDecoder:
            def __init__(self):
                self.zenz = FakeZenz()
                self.table = SimpleNamespace(spells=lambda k, t: t != "不正",
                                             tokenizer=lambda t: {"input_ids": [t]})
                self.q_calls = []

            def lm_context(self, context):
                return "production-prefix"

            def lm_tails(self, context, prefixes, tails):
                self.q_calls.append((context, prefixes, tails))
                return [-float(i) for i in range(len(tails))]
        return FakeDecoder()

    def test_same_union_q_scores_and_same_pool_right_rescore(self):
        decoder = self.fake()
        with patch.object(experiment, "timed", side_effect=lambda f: (f(), 1.)):
            r = experiment.inspect(case(answers=["秘密の正解"], right="を学ぶ。"), decoder)
        self.assertNotIn("秘密", json.dumps(r, ensure_ascii=False))
        self.assertTrue(r["right_applied"])
        self.assertEqual(len(decoder.q_calls), 1)
        self.assertEqual(decoder.q_calls[0][0], "production-prefix")
        none, right, same = (r["pools"][k] for k in experiment.CONDITIONS)
        self.assertEqual(none["cands"], ["日本語"])
        self.assertEqual(right["cands"], ["日本語", "二本語"])
        self.assertEqual(none["q"][0], right["q"][0])
        self.assertEqual(same["cands"], none["cands"])
        self.assertEqual(same["q"], none["q"])
        self.assertNotEqual(same["z"], none["z"])
        self.assertEqual(decoder.zenz.scored[-1], ("を学ぶ。", ["日本語"]))
        self.assertEqual(decoder.zenz.right_context, "")

    def test_absent_right_reuses_predictions_without_another_model_call(self):
        decoder = self.fake()
        with patch.object(experiment, "timed", side_effect=lambda f: (f(), 1.)):
            r = experiment.inspect(case(), decoder)
        self.assertFalse(r["right_applied"])
        self.assertEqual(len(decoder.zenz.generated), 1)
        self.assertEqual(len(decoder.zenz.scored), 1)
        self.assertEqual(r["pools"]["right"]["reused_from"], "none")


class Separation(unittest.TestCase):
    def test_sealed_inputs_reject_answer_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps([case("sealed", answers=["日本語"])]))
            with self.assertRaisesRegex(ValueError, "must not contain labels"):
                experiment.read_cases(path, "sealed", labels=False)

    def test_partition_split_disallows_shared_topic_group(self):
        a, b = case(), case("sealed", "2")
        b["topic_group"] = a["topic_group"]
        with self.assertRaisesRegex(ValueError, "topic groups overlap"):
            experiment.check_partitions([a], [b])

    def test_selection_only_accepts_dev_and_ties_prefer_existing_baseline(self):
        c = case(answers=["日本語"])
        r = prediction(c, z=(-1., -3.), q=(0., 0.))
        setting = experiment.select_setting([c], [r])
        self.assertEqual((setting["weight"], setting["length_power"]), experiment.BASELINE)
        c["split"] = "sealed"
        with self.assertRaisesRegex(ValueError, "only dev"):
            experiment.select_setting([c], [r])

    def test_dev_can_select_different_weight_without_accessing_sealed_labels(self):
        c = case(answers=["日本語"])
        result = experiment.select_setting([c], [prediction(c)])
        self.assertEqual(result["weight"], .35)
        self.assertEqual(result["length_power"], 0)

    def test_sealed_requires_freeze_and_unchanged_inputs_and_dev_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = dict(files={"model": "original"})
            with self.assertRaisesRegex(ValueError, "freeze dev"):
                experiment.require_frozen(out, manifest)
            rows = out / "dev.rows.jsonl"
            rows.write_text("{}\n")
            selection = dict(manifest=manifest, dev_rows_sha256=experiment.digest(rows))
            (out / "selection.json").write_text(json.dumps(selection))
            self.assertEqual(experiment.require_frozen(out, manifest), selection)
            with self.assertRaisesRegex(ValueError, "changed after freeze"):
                experiment.require_frozen(out, dict(files={"model": "changed"}))
            rows.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "dev predictions changed"):
                experiment.require_frozen(out, manifest)

    def test_resume_only_discards_unterminated_last_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            row = prediction(case())
            line = json.dumps(row) + "\n"
            path.write_text(line + '{"unfinished":')
            with self.assertRaisesRegex(ValueError, "interrupted"):
                experiment.read_rows(path)
            self.assertEqual(experiment.read_rows(path, repair=True), [row])
            self.assertEqual(path.read_text(), line)
            path.write_text(line * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                experiment.read_rows(path, repair=True)

    def test_report_does_not_open_labels_until_blind_predictions_are_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev, sealed = case(answers=["日本語"]), case("sealed", "2")
            (root / "dev.json").write_text(json.dumps([dev]))
            (root / "sealed.json").write_text(json.dumps([sealed]))
            # An intentionally unparsable label file must not be reached yet.
            (root / "labels.json").write_text("this file stays sealed")
            manifest = dict(files={"input": "fixed"})
            dev_rows = root / "dev.rows.jsonl"
            dev_rows.write_text(json.dumps(prediction(dev)) + "\n")
            selection = dict(manifest=manifest, dev_rows_sha256=experiment.digest(dev_rows),
                             selected=dict(weight=.7, length_power=0))
            (root / "selection.json").write_text(json.dumps(selection))
            (root / "sealed.run.json").write_text(json.dumps(
                dict(selection_sha256=experiment.digest(root / "selection.json"))))
            argv = ["experiment", "report", "--dev-cases", str(root / "dev.json"),
                    "--evaluation-manifest", str(root / "manifest.json"),
                    "--sealed-inputs", str(root / "sealed.json"), "--sealed-labels", str(root / "labels.json"),
                    "--table", str(root), "--zenz", str(root), "--rescorer", str(root), "--out", str(root)]
            with patch.object(sys, "argv", argv), patch.object(experiment, "fingerprint", return_value=manifest), \
                    patch.object(experiment, "read_labels") as labels:
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    experiment.main()
                labels.assert_not_called()


class Measurement(unittest.TestCase):
    def test_pure_context_contrast_keeps_ranking_fixed_and_clusters_pairs(self):
        cases = [case(identifier=str(i), answers=["日本語"], right="を学ぶ。") for i in range(2)]
        for c in cases:
            c["topic_group"] = "paired-topic"
        rows = [prediction(c) for c in cases]
        for r in rows:
            r["pools"]["right"] = dict(r["pools"]["right"], z=[0., -20.])
        result = experiment.compare(cases, rows, "right", experiment.BASELINE, "none", experiment.BASELINE)
        self.assertEqual((result["wins"], result["losses"], result["topic_groups"]), (2, 0, 1))
        self.assertEqual(result["cluster_bootstrap95_delta_pp"], [100., 100.])

    def test_changed_data_is_rejected_before_first_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dev.json"
            data.write_text("[]")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(files={"dev.json": experiment.digest(data)})))
            data.write_text("[{}]")
            args = SimpleNamespace(evaluation_manifest=manifest, dev_cases=data,
                                   sealed_inputs=data, sealed_labels=data)
            with self.assertRaisesRegex(ValueError, "changed after sealing"):
                experiment.fingerprint(args)

    def test_literal_fallback_accuracy_does_not_inflate_candidate_oracle(self):
        c = case(answers=["にほんご"])
        pool = dict(generated=["不正"], cands=[], z=[], q=[])
        row = dict(input=experiment.public_input(c), pools={n: pool for n in experiment.CONDITIONS})
        result = experiment.metrics([c], [row], "none", experiment.BASELINE)
        self.assertEqual((result["top1"], result["top5"], result["valid_pool"], result["no_valid_pool"]),
                         (1, 1, 0, 1))

    def test_right_present_and_domain_groups_are_reported_separately(self):
        a, b = case(answers=["日本語"]), case(identifier="2", answers=["日本語"], right="を学ぶ。")
        b["domain"] = "controlled"
        summary = experiment.summary([a, b], [prediction(a), prediction(b)], dict(weight=.7, length_power=0))
        self.assertEqual(summary["all"]["none"]["baseline"]["total"], 2)
        self.assertEqual(summary["right_present"]["right"]["baseline"]["total"], 1)
        self.assertEqual(summary["domain/controlled"]["right"]["baseline"]["total"], 1)

    def test_label_identity_and_prediction_completeness_are_checked(self):
        a, b = case(), case(identifier="2")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            experiment.align_rows([a, b], [prediction(a)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps({"wrong": ["日本語"]}))
            with self.assertRaisesRegex(ValueError, "differ"):
                experiment.read_labels(path, [a])


if __name__ == "__main__":
    unittest.main()
