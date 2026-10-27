"""Balanced ranking retains candidate plausibility and sealed evaluation controls."""

import copy
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
    "balanced_ranking",
    Path(__file__).resolve().parents[1]
    / "neural/experiment-japanese-balanced-ranking.py",
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
previous = experiment.previous


def case(identifier="1", split="dev", right="を学ぶ。", answers=None):
    c = dict(
        id=identifier,
        split=split,
        context="私は",
        kana="にほんご",
        right_context=right,
        topic_group=split + identifier,
        domain="natural",
        case_type="full",
    )
    if answers is not None:
        c["answers"] = answers
    return c


def row(c, z=(0.0, -1.0), q=(0.0, 0.0), r=(0.0, 2.0)):
    pool = dict(
        generated=["日本語", "二本語"],
        cands=["日本語", "二本語"],
        z=list(z),
        q=list(q),
        r=list(r) if c["right_context"] else [0.0, 0.0],
    )
    return dict(
        input=previous.public_input(c),
        pools={name: copy.deepcopy(pool) for name in experiment.CONDITIONS},
        q_union=sorted(pool["cands"]),
        q_union_ms=4.0,
        q_right_union_ms=1.0 if c["right_context"] else 0.0,
    )


class GuardedRanking(unittest.TestCase):
    def test_guard_rejects_candidate_with_worse_q_without_supporting_suffix(self):
        c = case(right="よう掲示されている。")
        r = row(c, z=(0.0, 2.0), q=(0.0, -3.0), r=(0.0, -0.1))
        # Dropping candidate q selects the second spelling; keeping plausibility
        # and requiring actual suffix evidence leaves the original first.
        self.assertEqual(
            experiment.decision(c, r, "right", experiment.PREVIOUS)["candidates"][0],
            "二本語",
        )
        result = experiment.decision(c, r, "right", experiment.GUARDED[0])
        self.assertEqual(result["candidates"][0], "日本語")
        self.assertEqual(set(result["candidates"]), set(r["pools"]["right"]["cands"]))

    def test_large_candidate_deficit_can_be_overcome_by_real_right_evidence(self):
        c = case(right="300人まで入れる。")
        r = row(c, z=(0.0, -2.0), q=(0.0, -6.0), r=(0.0, 12.0))
        config = experiment.setting("guarded", suffix=1.05, evidence_ratio=1.0)
        result = experiment.decision(c, r, "right", config)
        self.assertEqual(result["candidates"][0], "二本語")
        self.assertEqual(result["reason"], "supported_replacement")
        self.assertEqual(result["raw_q_deficit"], 6.0)
        config["evidence_ratio"] = 2.0
        result = experiment.decision(c, r, "right", config)
        self.assertEqual(result["candidates"][0], "日本語")
        self.assertEqual(result["reason"], "guard_rejected_additive_top")

    def test_guard_can_choose_second_supported_candidate_without_deleting_others(self):
        c = case()
        r = row(c, z=(0.0, 1.3), q=(0.0, -2.0), r=(0.0, 1.0))
        # Raw top wins largely from z, not sufficient right gain to justify q loss.
        result = experiment.decision(c, r, "right", experiment.GUARDED[0])
        self.assertEqual(result["candidates"], ["日本語", "二本語"])
        self.assertEqual(result["reason"], "guard_rejected_additive_top")

    def test_short_right_uses_original_pool_not_merely_old_weights_on_right_pool(self):
        c = case(right="。次")
        r = row(c)
        r["pools"]["right"]["z"] = [-10.0, 0.0]
        result = experiment.decision(c, r, "right", experiment.GUARDED[0])
        self.assertEqual(
            result["candidates"],
            previous.ranking(r["pools"]["none"], previous.BASELINE),
        )
        self.assertNotEqual(
            result["candidates"],
            previous.ranking(r["pools"]["right"], previous.BASELINE),
        )
        self.assertEqual(result["effective_pool"], "none")
        self.assertEqual(result["reason"], "short_right_original_pool")
        ordinary = experiment.decision(c, r, "right", experiment.JOINT)
        self.assertEqual(ordinary["effective_pool"], "right")

    def test_lexical_count_excludes_punctuation_spaces_and_honors_character_budget(
        self,
    ):
        self.assertEqual(experiment.lexical_chars(" 、。！？\n\t"), 0)
        self.assertEqual(experiment.lexical_chars("。a１日"), 3)
        self.assertEqual(experiment.lexical_chars("。" * 40 + "日本語"), 0)

    def test_empty_right_is_exact_original_order_for_all_settings(self):
        c = case(right="")
        r = row(c, z=(0.0, 2.0), q=(0.0, -3.0))
        expected = previous.ranking(r["pools"]["none"], previous.BASELINE)
        for config in experiment.GRID + [experiment.PREVIOUS, experiment.JOINT]:
            for condition in experiment.CONDITIONS:
                result = experiment.decision(c, r, condition, config)
                self.assertEqual(result["candidates"], expected)
                self.assertEqual(result["effective_pool"], "none")

    def test_oracle_reports_actual_fallback_pool(self):
        c = case(right="。", answers=["二本語"])
        r = row(c)
        r["pools"]["none"] = dict(
            generated=["日本語"], cands=["日本語"], z=[0.0], q=[0.0], r=[0.0]
        )
        metrics = experiment.metrics([c], [r], "right", experiment.GUARDED[0])
        self.assertEqual(metrics["valid_pool"], 0)
        self.assertEqual(metrics["effective_pools"], {"none": 1})


class FreshVariantSharing(unittest.TestCase):
    def paired(self):
        a, b, c = (
            case("a", right="を学ぶ。"),
            case("b", right="を学"),
            case("c", right=""),
        )
        for item in (a, b, c):
            item["topic_group"] = "same-natural-source"
        return a, b, c

    def test_batch_drift_is_removed_and_suffix_scores_stay_context_specific(self):
        a, b, _ = self.paired()
        shared = {}
        first = row(a, q=(-1.0, -2.0), r=(-3.0, -4.0))
        second = row(b, q=(-1.01, -1.98), r=(-5.0, -6.0))
        # A candidate seen only in the later right-conditioned pool receives
        # its first score there; existing candidates retain the first batch.
        second["pools"]["right"] = dict(
            generated=["日本語", "二本語", "日本後"],
            cands=["日本語", "二本語", "日本後"],
            z=[0.0, -1.0, -2.0],
            q=[-1.01, -1.98, -3.0],
            r=[-5.0, -6.0, -7.0],
        )
        second["q_union"] = sorted(["日本語", "二本語", "日本後"])
        experiment.share_fresh_q(a, first, shared)
        experiment.share_fresh_q(b, second, shared)
        self.assertEqual(second["pools"]["none"]["q"], [-1.0, -2.0])
        self.assertEqual(second["pools"]["right"]["q"], [-1.0, -2.0, -3.0])
        self.assertEqual(second["pools"]["right"]["r"], [-5.0, -6.0, -7.0])
        self.assertEqual(second["shared_q"]["reused_candidates"], 2)
        self.assertEqual(second["shared_q"]["new_candidates"], 1)
        self.assertGreater(second["shared_q"]["max_abs_q_adjustment"], 0.019)
        replay = {}
        experiment.share_fresh_q(a, first, replay, replay=True)
        experiment.share_fresh_q(b, second, replay, replay=True)
        self.assertEqual(shared, replay)

    def test_fresh_variants_must_preserve_original_candidate_pool_and_zenz_scores(self):
        a, b, _ = self.paired()
        shared = {}
        experiment.share_fresh_q(a, row(a), shared)
        changed = row(b)
        changed["pools"]["none"]["z"][0] = 0.01
        with self.assertRaisesRegex(ValueError, "original none pool"):
            experiment.share_fresh_q(b, changed, shared)

    def test_complete_rows_are_revalidated_before_selection_or_unsealing(self):
        a, b, _ = self.paired()
        state = {}
        first, second = row(a), row(b, q=(0.01, 0.01))
        experiment.share_fresh_q(a, first, state)
        experiment.share_fresh_q(b, second, state)
        experiment.verify_shared_rows([a, b], [first, second])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            experiment.verify_shared_rows([a, b], [first])
        second["shared_q"]["none_reference_id"] = "wrong-source"
        with self.assertRaisesRegex(ValueError, "shared-q provenance"):
            experiment.verify_shared_rows([a, b], [first, second])

    def test_resume_reconstructs_first_seen_q_without_rewriting_saved_row(self):
        a, b, c = self.paired()
        first = row(a, q=(-1.0, -2.0))
        experiment.share_fresh_q(a, first, {})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            path = out / "dev.rows.jsonl"
            first_line = json.dumps(first) + "\n"
            path.write_text(first_line)
            args = SimpleNamespace(
                out=out,
                dev_pools=None,
                sealed_pools=None,
                rescorer=out,
                table=out,
                zenz=out,
            )
            scorer = SimpleNamespace(tokenizer=object())

            def inspect(item, decoder):
                return row(item, q=(-0.99, -2.01))

            with (
                patch.object(previous, "Rescorer", return_value=scorer),
                patch.object(previous, "Table"),
                patch.object(previous, "Decoder"),
                patch.object(previous, "RightZenz"),
                patch.object(experiment.right, "suffix_score"),
                patch.object(
                    experiment.right, "inspect", side_effect=inspect
                ) as inference,
            ):
                experiment.score_cases(args, [a, b, c], "dev")
            self.assertEqual(inference.call_count, 2)
            self.assertTrue(path.read_text().startswith(first_line))
            saved = previous.read_rows(path)
            for item in saved:
                self.assertEqual(item["pools"]["none"]["q"], [-1.0, -2.0])
                self.assertEqual(item["shared_q"]["none_reference_id"], "a")
            with patch.object(previous, "Rescorer") as model:
                experiment.score_cases(args, [a, b, c], "dev")
            model.assert_not_called()
            saved[1]["shared_q"]["q_origins"][0] = "changed-origin"
            path.write_text("".join(json.dumps(item) + "\n" for item in saved))
            with self.assertRaisesRegex(ValueError, "shared-q provenance"):
                experiment.score_cases(args, [a, b, c], "dev")


class DevelopmentControls(unittest.TestCase):
    def test_safety_constraint_blocks_higher_accuracy_setting_that_breaks_reviewed_case(
        self,
    ):
        cases = [
            case(str(i), answers=["日本語"] if i == 0 else ["二本語"]) for i in range(3)
        ]
        rows = [row(c) for c in cases]
        chosen = experiment.select_setting(cases, rows, {"0": ["日本語"]})
        self.assertEqual(chosen["config"], experiment.BASELINE)
        self.assertEqual(chosen["safety"]["regressions"], 0)
        self.assertGreater(
            chosen["best_unconstrained"]["top1"], chosen["selected_performance"]["top1"]
        )
        self.assertTrue(any(not s["eligible"] for s in chosen["dev_sweep"]))

    def test_source_macro_metric_does_not_count_three_variants_as_three_sources(self):
        a = case("a", answers=["日本語"])
        bs = [case("b" + str(i), answers=["二本語"]) for i in range(3)]
        for b in bs:
            b["topic_group"] = "source-b"
        cases = [a] + bs
        result = experiment.performance(
            cases, [row(c) for c in cases], experiment.BASELINE, {}
        )
        self.assertEqual(result["top1"], 1)
        self.assertEqual(result["topic_groups"], 2)
        self.assertEqual(result["macro_accuracy"], 0.5)
        self.assertEqual(result["macro_fraction"], [1, 2])

    def test_no_dev_selection_from_sealed_labels(self):
        c = case(split="sealed", answers=["日本語"])
        with self.assertRaisesRegex(ValueError, "only dev"):
            experiment.select_setting([c], [row(c)])

    def test_safety_file_rejects_unknown_or_duplicate_ids(self):
        c = case(answers=["日本語"])
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "safety.json"
            p.write_text(json.dumps([dict(id="sealed-only", answers=["日本語"])]))
            with self.assertRaisesRegex(ValueError, "development IDs"):
                experiment.read_safety(p, [c])
            p.write_text(json.dumps([dict(id="1", answers=["日本語"])] * 2))
            with self.assertRaisesRegex(ValueError, "unique"):
                experiment.read_safety(p, [c])

    def test_cached_r_and_q_reused_without_model_calls_or_label_leakage(self):
        c = case(answers=["秘密の正解"])
        original = row(c)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cached = out / "cached.jsonl"
            cached.write_text(json.dumps(original) + "\n")
            args = SimpleNamespace(out=out, dev_pools=cached, sealed_pools=None)
            with patch.object(previous, "Rescorer") as model:
                experiment.score_cases(args, [c], "dev")
            model.assert_not_called()
            data = (out / "dev.rows.jsonl").read_text()
            self.assertNotIn("秘密", data)
            saved = json.loads(data)
            self.assertEqual(saved["pools"], original["pools"])
            self.assertEqual(saved["new_suffix_inference_ms"], 0.0)

    def test_cache_unknown_input_cannot_silently_enter_requested_partition(self):
        c = case(answers=["日本語"])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cached = out / "cached.jsonl"
            cached.write_text(json.dumps(row(case("other"))) + "\n")
            args = SimpleNamespace(out=out, dev_pools=cached, sealed_pools=None)
            with self.assertRaisesRegex(ValueError, "absent from requested"):
                experiment.score_cases(args, [c], "dev")

    def test_partial_cache_only_generates_missing_rows(self):
        a, b = case("a", answers=["日本語"]), case("b", answers=["日本語"])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cached = out / "cached.jsonl"
            cached.write_text(json.dumps(row(a)) + "\n")
            args = SimpleNamespace(
                out=out,
                dev_pools=cached,
                sealed_pools=None,
                rescorer=out,
                table=out,
                zenz=out,
            )
            fake_scorer = SimpleNamespace(tokenizer=object())
            with (
                patch.object(previous, "Rescorer", return_value=fake_scorer),
                patch.object(previous, "Table"),
                patch.object(previous, "Decoder"),
                patch.object(previous, "RightZenz"),
                patch.object(experiment.right, "suffix_score"),
                patch.object(
                    experiment.right, "inspect", side_effect=lambda c, d: row(c)
                ) as inspect,
            ):
                experiment.score_cases(args, [a, b], "dev")
            self.assertEqual(inspect.call_count, 1)
            self.assertEqual(inspect.call_args.args[0]["id"], "b")
            result = previous.read_rows(out / "dev.rows.jsonl")
            experiment.right.align_rows([a, b], result)
            self.assertEqual(result[0]["pools"], row(a)["pools"])

    def test_cached_model_or_scoring_mismatch_rejected_before_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            table = out / "table.json"
            table.write_text("{}")
            model = out / "model"
            model.mkdir()
            config = model / "config.json"
            config.write_text("{}")
            cache = out / "cache.jsonl"
            cache.write_text("")
            files = [
                table,
                config,
                experiment._RIGHT_PATH,
                experiment._RIGHT_PATH.with_name("experiment-japanese.py"),
                experiment._RIGHT_PATH.with_name("japanese.py"),
                experiment._RIGHT_PATH.with_name("rescorer.py"),
            ]
            manifest = {"files": {str(p.resolve()): previous.digest(p) for p in files}}
            prior = out / "prior.json"
            prior.write_text(json.dumps(manifest))
            args = SimpleNamespace(
                dev_pools=cache,
                sealed_pools=None,
                pool_manifest=prior,
                safety_cases=None,
                table=table,
                zenz=model,
                rescorer=model,
            )
            with patch.object(
                previous, "fingerprint", return_value=copy.deepcopy(manifest)
            ):
                experiment.fingerprint(args)
            damaged = copy.deepcopy(manifest)
            damaged["files"][str(config.resolve())] = "different-model"
            prior.write_text(json.dumps(damaged))
            with patch.object(
                previous, "fingerprint", return_value=copy.deepcopy(manifest)
            ):
                with self.assertRaisesRegex(
                    ValueError, "cached model/scoring dependency differs"
                ):
                    experiment.fingerprint(args)

    def test_report_requires_all_predictions_before_reading_sealed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev, sealed = case(answers=["日本語"]), case("2", split="sealed")
            (root / "dev.json").write_text(json.dumps([dev]))
            (root / "inputs.json").write_text(json.dumps([sealed]))
            (root / "labels.json").write_text("sealed labels must stay closed")
            manifest = dict(files={"source": "fixed"})
            (root / "dev.rows.jsonl").write_text(json.dumps(row(dev)) + "\n")
            selected = experiment.select_setting([dev], [row(dev)])
            selection = dict(
                manifest=manifest,
                dev_rows_sha256=previous.digest(root / "dev.rows.jsonl"),
                selected=selected,
            )
            (root / "selection.json").write_text(json.dumps(selection))
            (root / "sealed.run.json").write_text(
                json.dumps(
                    dict(selection_sha256=previous.digest(root / "selection.json"))
                )
            )
            argv = [
                "experiment",
                "report",
                "--dev-cases",
                str(root / "dev.json"),
                "--sealed-inputs",
                str(root / "inputs.json"),
                "--sealed-labels",
                str(root / "labels.json"),
                "--evaluation-manifest",
                str(root / "manifest.json"),
                "--out",
                str(root),
                "--table",
                str(root),
                "--rescorer",
                str(root),
                "--zenz",
                str(root),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(experiment, "fingerprint", return_value=manifest),
                patch.object(previous, "read_labels") as labels,
            ):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    experiment.main()
                labels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
