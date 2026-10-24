"""Worker contract tests run without a GPU or model artifact."""
import importlib.util
from pathlib import Path
import sys
import unittest

directory = Path(__file__).resolve().parents[1] / "neural"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("gpu_worker", directory / "gpu-worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerContract(unittest.TestCase):
    def request(self, **changes):
        return dict(dict(id=1, context="", syllables=2, candidates=[], operation="verify",
                         input="bmyi", draft="编译"), **changes)

    def test_malformed_envelopes_are_rejected(self):
        for request in [None, [], {}, self.request(id=True), self.request(syllables=-1),
                        self.request(context=[]), self.request(candidates=[{}]),
                        self.request(context_anchor=-1), self.request(context_anchor=9),
                        self.request(context_anchor=True),
                        self.request(prior_confidence=0), self.request(prior_confidence=True),
                        self.request(prior_confidence=float("nan")),
                        self.request(context_reading=dict(text="便宜", input="bm")),
                        self.request(context_reading=dict(text="便宜", input="bmyi"))]:
            with self.assertRaises(ValueError):
                worker.validate(request)

    def test_wrong_operation_or_coverage_never_calls_the_model(self):
        for changes in [dict(operation="rerank"), dict(syllables=0),
                        dict(draft="编"), dict(input="bmA")]:
            result = worker.respond(None, self.request(**changes))
            self.assertIsNone(result["promoted"])
            self.assertNotIn("score", result)

    def test_invalid_keyboard_or_window_retains_the_traditional_draft(self):
        class OutOfRange:
            def evaluate(self, *_):
                raise ValueError("unencodable composition")
        result = worker.respond(OutOfRange(), self.request())
        self.assertNotIn("score", result)
        self.assertEqual(result["prefixes"], [])

    def test_seed_uses_joint_scores_and_original_complete_candidate_indices(self):
        class Evaluator:
            def evaluate(self, context, code, texts):
                self.texts = texts
                return [-1, -1.1], ["first", "second"]

            def proposals(self, draft, distribution, margin):
                self.proposed = (draft, distribution)
                return []

        evaluator = Evaluator()
        request = self.request(seed_candidates=True, prior_weight=.1, candidates=[
            dict(text="编译", end=2, dictionary=True, score=-10),
            dict(text="编", end=1, dictionary=True, score=100),
            dict(text="便宜", end=2, dictionary=True, score=-1)])
        response = worker.respond(evaluator, request)
        self.assertEqual(evaluator.texts, ["编译", "便宜"])
        self.assertEqual(response["verified_index"], 2)
        self.assertEqual(response["score"], -1.1)
        self.assertEqual(response["evaluated"], 2)
        self.assertEqual(evaluator.proposed, ("便宜", "second"))
        for change in (dict(prior_weight=float("nan")), dict(prior_weight=-1),
                       dict(seed_candidates=1), dict(candidates=[dict(request["candidates"][0], score=None)])):
            with self.assertRaises(ValueError):
                worker.validate(dict(request, **change))

    def test_confident_seed_reports_the_weight_used_for_later_rounds(self):
        class Evaluator:
            def evaluate(self, context, code, texts):
                return [-.05, -.2], [None, None]

            def proposals(self, draft, distribution, margin):
                return []

        request = self.request(seed_candidates=True, prior_weight=.1, candidates=[
            dict(text="编译", end=2, dictionary=True, score=-10),
            dict(text="便宜", end=2, dictionary=True, score=-1)])
        default = worker.respond(Evaluator(), request)
        self.assertEqual(default["verified_index"], 1)
        self.assertEqual(default["effective_prior_weight"], .1)
        confident = worker.respond(Evaluator(), dict(request, prior_confidence=.5))
        self.assertEqual(confident["verified_index"], 0)
        self.assertEqual(confident["effective_prior_weight"], 0)


if __name__ == "__main__":
    unittest.main()
