"""Context ablation keeps examples paired and selects settings only on dev."""
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location(
    "evaluate_context", Path(__file__).resolve().parents[1] / "neural/evaluate-japanese-context.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


class History(unittest.TestCase):
    def test_character_budget_preserves_the_strongest_retrieval(self):
        spec = importlib.util.spec_from_file_location(
            "prepare_context", Path(evaluate.__file__).with_name("prepare-japanese-context.py"))
        prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        history = [dict(text=c * 160) for c in "ABC"]
        chosen = prepare.selection(history, [0, 1, 2], {0: 8, 1: 7, 2: 6})
        self.assertEqual(len(chosen["context"]), 200)
        self.assertTrue(chosen["context"].endswith("A" * 160 + "\n"))
        self.assertNotIn("C", chosen["context"])

    def test_unknown_authors_are_not_grouped_as_one_person(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sources").mkdir()
            (root / "work/data").mkdir(parents=True)
            cases = [dict(id="1", input="い", gold="い"), dict(id="2", input="う", gold="う")]
            (root / "work/data/test.json").write_text(json.dumps(cases))
            (root / "work/data/dev.json").write_text("[]")
            entries = [("1", "い", "author", "2025-01-02"), ("2", "う", "\\N", "2025-01-02"),
                       ("3", "あ", "author", "2020-01-01"), ("4", "え", "\\N", "2020-01-01")]
            (root / "sources/jpn_sentences_detailed.tsv").write_text("".join(
                f"{i}\tjpn\t{text}\t{author}\t{date}\t{date}\n" for i, text, author, date in entries))
            (root / "sources/jpn_transcriptions.tsv").write_text("".join(
                f"{i}\t0\tHrkt\t1\t{text}\n" for i, text, _, _ in entries))
            output = root / "contexts.json"
            subprocess.run([sys.executable, str(Path(evaluate.__file__).with_name("prepare-japanese-context.py")),
                            "--root", str(root), "--out", str(output)], check=True, capture_output=True)
            rows = json.loads(output.read_text())["cases"]
            self.assertEqual(rows[0]["contexts"]["retrieved"]["context"], "あ\n")
            self.assertEqual(rows[1]["eligible_history_count"], 0)
            self.assertEqual(rows[1]["contexts"]["retrieved"]["context"], "")


def row(split, answer="A", none=(0., 1.), history=(2., 0.)):
    return dict(split=split, cands=["A", "B"], answers=[answer], z=[0., 0.],
                q=dict(none=list(none), random=list(history), retrieved=list(history)))


class Pairing(unittest.TestCase):
    def test_duplicate_kana_keeps_both_answers(self):
        contexts = dict(cases=[dict(split="test", case_index=i, id=str(i), kana="かな") for i in range(2)])
        pools = [dict(kana="かな", cands=["仮名", "加奈"], z=[0., 0.], answers=[answer])
                 for answer in ("仮名", "加奈")]
        paired = list(evaluate.paired_cases(contexts, pools, "test"))
        self.assertEqual([p["answers"] for _, p in paired], [["仮名"], ["加奈"]])
        contexts["cases"][1]["case_index"] = 0
        with self.assertRaisesRegex(ValueError, "alignment"):
            list(evaluate.paired_cases(contexts, pools, "test"))

    def test_count_and_kana_mismatches_fail_before_scoring(self):
        contexts = dict(cases=[dict(split="test", case_index=0, id="1", kana="かな")])
        with self.assertRaisesRegex(ValueError, "pool rows"):
            list(evaluate.paired_cases(contexts, [], "test"))
        with self.assertRaisesRegex(ValueError, "alignment"):
            list(evaluate.paired_cases(contexts, [dict(kana="かに")], "test"))

    def test_legacy_kana_id_is_accepted_but_other_mismatches_fail(self):
        contexts = dict(cases=[dict(split="test", case_index=0, id="123", kana="かな")])
        pool = dict(id="かな", kana="かな", cands=["仮名"], z=[0.], answers=["仮名"])
        self.assertEqual(len(list(evaluate.paired_cases(contexts, [pool], "test"))), 1)
        pool["id"] = "unrelated-id"
        with self.assertRaisesRegex(ValueError, "case IDs differ"):
            list(evaluate.paired_cases(contexts, [pool], "test"))


class Selection(unittest.TestCase):
    def test_test_answers_cannot_change_selected_strength(self):
        report = evaluate.summarize([row("dev"), row("test", "B")])
        # .25 still favors B; .5 flips to A and is the first tied optimum on dev.
        self.assertEqual(report["modes"]["retrieved"]["strength"], .5)
        test = report["modes"]["retrieved"]["test"]
        self.assertEqual((test["wins_nfkc"], test["losses_nfkc"]), (0, 1))
        changed = evaluate.summarize([row("dev"), row("test", "A")])
        self.assertEqual(changed["modes"]["retrieved"]["strength"], .5)

    def test_no_gain_tie_uses_no_history(self):
        report = evaluate.summarize([row("dev", "B"), row("test")])
        self.assertEqual(report["modes"]["retrieved"]["strength"], 0)

    def test_exact_nfkc_and_oracle_are_distinct(self):
        r = dict(split="test", cands=["ABC", "別"], answers=["ＡＢＣ"], z=[0., 0.],
                 q=dict(none=[1., 0.], random=[1., 0.], retrieved=[1., 0.]))
        m = evaluate.metrics([r])
        self.assertEqual((m["exact"], m["nfkc"], m["oracle_exact"], m["oracle_nfkc"]), (0, 1, 0, 1))


class Progress(unittest.TestCase):
    def test_resume_checks_metadata_and_truncates_only_partial_last_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            metadata = dict(version=1, inputs={"pool": "original"})
            self.assertEqual(evaluate.resume(path, metadata), {})
            record = dict(type="result", split="dev", case_index=0, id="one")
            with path.open("a") as f:
                f.write(json.dumps(record) + "\n" + '{"partial":')
            self.assertEqual(evaluate.resume(path, metadata), {("dev", 0): record})
            self.assertTrue(path.read_text().endswith("\n"))
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                evaluate.resume(path, dict(version=1, inputs={"pool": "changed"}))


class Scoring(unittest.TestCase):
    def test_selected_hidden_scores_match_full_logits_with_padding_and_chunks(self):
        import torch

        class Tokenizer:
            pad_token_id = 0

            def __call__(self, text, **kwargs):
                return {"input_ids": ["pqabc".index(c) + 1 for c in text]}

        class Model:
            def __init__(self):
                self.model = self
                self.embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4).sin()
                self.head = torch.arange(32, dtype=torch.float32).reshape(4, 8).cos()

            def __call__(self, input_ids, **kwargs):
                return SimpleNamespace(last_hidden_state=self.embedding[input_ids].cumsum(1))

            def lm_head(self, hidden):
                return hidden @ self.head

        scorer = evaluate.Scorer.__new__(evaluate.Scorer)
        scorer.torch, scorer.model, scorer.tokenizer = torch, Model(), Tokenizer()
        scorer.device, scorer.batch_size, scorer.head_chunk = "cpu", 2, 2
        texts = ["a", "abc", "bc"]
        expected = []
        for text in texts:
            sequence = scorer.tokenizer("pq" + text)["input_ids"]
            hidden = scorer.model(torch.tensor([sequence])).last_hidden_state
            logits = scorer.model.lm_head(hidden)[0].log_softmax(-1)
            expected.append(sum(float(logits[1 + i, token]) for i, token in enumerate(sequence[2:])))
        torch.testing.assert_close(torch.tensor(scorer.score("pq", texts)), torch.tensor(expected))

    def test_context_preserves_history_before_original(self):
        self.assertEqual(evaluate.score_context("履歴\n", "本文"), "履歴\n本文")
        self.assertTrue(evaluate.score_context("", "").startswith(evaluate.NEUTRAL))


if __name__ == "__main__":
    unittest.main()
