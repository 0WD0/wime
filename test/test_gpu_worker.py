"""Worker contract tests run without a GPU or model artifact."""
import importlib.util
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

directory = Path(__file__).resolve().parents[1] / "neural"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("gpu_worker", directory / "gpu-worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerContract(unittest.TestCase):
    def request(self, **changes):
        return dict(dict(id=1, context="", syllables=2, candidates=[], operation="verify",
                         input="bmyi", draft="编译"), **changes)

    def test_default_and_explicit_qwen_share_text_search(self):
        for flags in ([], ["--backend", "qwen-hf"]):
            args = worker.parse_args(["model", "--rescorer", "lm", "--database", "words"] + flags)
            self.assertEqual(args.backend, "qwen-hf")
            table, evaluator, rescorer = object(), object(), object()
            model = SimpleNamespace(readings=mock.Mock(return_value=table))
            text = SimpleNamespace(TextDecoder=mock.Mock())
            with mock.patch.dict(sys.modules, decoding_model=model, text_decoder=text):
                result = worker.make_decoder(args, evaluator, rescorer)
            text.TextDecoder.assert_called_once_with(evaluator, rescorer, table)
            self.assertIs(result, text.TextDecoder.return_value)
        self.assertIsNone(worker.make_decoder(worker.parse_args(["model"]), evaluator, None))

    def test_invalid_backend_fails_before_loading_models(self):
        base = ["model", "--rescorer", "lm", "--database", "words"]
        cases = [["--backend", "legacy"], ["--decoder-kind", "legacy"],
                 ["--backend", "minicpm-native"], ["--native-model", "model.gguf"],
                 ["--backend", "minicpm-native", "--native-model", "model.gguf"],
                 ["--backend", "minicpm-native", "--native-model", "model.gguf",
                  "--native-library", "bridge.so", "--japanese-model", "kana"]]
        for flags in cases:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    worker.parse_args(base + flags)
                self.assertEqual(raised.exception.code, 2)

    def test_native_selection_loads_only_tokenizer_and_native_backend(self):
        args = worker.parse_args(["model", "--rescorer", "tokenizer", "--database", "words",
                                  "--backend", "minicpm-native", "--native-library", "bridge.so",
                                  "--native-model", "model.gguf"])
        native = SimpleNamespace(NativeRescorer=mock.Mock())
        model = SimpleNamespace(readings=mock.Mock(return_value={}))
        text = SimpleNamespace(TextDecoder=mock.Mock())
        evaluator = object()
        with mock.patch.dict(sys.modules, native_rescorer=native, decoding_model=model, text_decoder=text), \
                mock.patch("rescorer.Rescorer") as hf:
            rescorer = worker.make_rescorer(args)
            worker.make_decoder(args, evaluator, rescorer)
        native.NativeRescorer.assert_called_once_with("tokenizer", 1.5, "bridge.so", "model.gguf")
        hf.assert_not_called()
        self.assertIs(text.TextDecoder.call_args.kwargs["lm_backend"], rescorer.backend)

    def test_production_worker_rejects_experimental_search_overrides(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            worker.parse_args(["model", "--text-final-mode", "fused"])

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
        self.assertEqual(response["candidate_scores"], [[0, -1], [2, -1.1]])
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

    def test_rescorer_joins_seed_selection_but_not_the_confidence_gate(self):
        class Evaluator:
            def evaluate(self, context, code, texts):
                return [-.05, -.2], ["first", "second"]

            def proposals(self, draft, distribution, margin):
                self.proposed = distribution
                return ["便利", "便"]

        class Rescorer:
            def combine(self, context, texts, scores):
                self.seen = (context, texts)
                return [s + r for s, r in zip(scores, [-2, 0])]

            def proposals(self, context, draft):
                return ["变", "便"]

        evaluator, rescorer = Evaluator(), Rescorer()
        request = self.request(context="上文", seed_candidates=True, prior_weight=.1,
                               prior_confidence=.5, candidates=[
                                   dict(text="编译", end=2, dictionary=True, score=-1),
                                   dict(text="便宜", end=2, dictionary=True, score=-1)])
        response = worker.respond(evaluator, request, rescorer)
        self.assertEqual(rescorer.seen, ("上文", ["编译", "便宜"]))
        self.assertEqual(response["effective_prior_weight"], 0)
        self.assertEqual(response["verified_index"], 1)
        self.assertEqual(response["score"], -.2)
        self.assertEqual(evaluator.proposed, "second")
        self.assertEqual(response["prefixes"], ["变", "便利", "便"])

    def test_learned_reading_bonus_chooses_the_seed_without_changing_its_score(self):
        class Evaluator:
            def evaluate(self, context, code, texts):
                return [-1, -3], [None, None]

            def proposals(self, draft, distribution, margin):
                return []

        request = self.request(seed_candidates=True, prior_weight=.1, candidates=[
            dict(text="编译", end=2, dictionary=True, score=-1),
            dict(text="便宜", end=2, dictionary=True, score=-1, bonus=10.0)])
        response = worker.respond(Evaluator(), request)
        self.assertEqual(response["verified_index"], 1)
        self.assertEqual(response["score"], -3)
        for bonus in (float("nan"), "10", None):
            with self.assertRaises(ValueError):
                worker.validate(dict(request, candidates=[dict(request["candidates"][1], bonus=bonus)]))

    def test_decode_passes_pins_and_learned_readings_and_needs_a_decoder(self):
        class Decoder:
            def decode(self, *args):
                self.args = args
                return [("便宜", -.5)]

        request = self.request(operation="decode", input="bmy", syllables=2, draft="",
                               context_anchor=2, pins=[[0, "便"]], learned=[[0, "便宜"]], memory_weight=10.0)
        decoder = Decoder()
        self.assertEqual(worker.respond(None, request, None, decoder)["candidates"], [["便宜", -.5]])
        self.assertEqual(decoder.args, ("", "bmy", 2, None, [(0, "便")], [(0, "便宜")], 10.0, []))
        worker.respond(None, dict(request, aux=[[1, "kp"]]), None, decoder)
        self.assertEqual(decoder.args[-1], [(1, "kp")])
        self.assertEqual(worker.respond(None, request)["candidates"], [])
        for change in (dict(input="bmA"), dict(pins=[[0, "便宜"]]), dict(pins=[["0", "便"]]),
                       dict(learned=[[0, ""]]), dict(learned=[0]), dict(memory_weight=-1),
                       dict(aux=[[0, "K"]]), dict(aux=[[0, ""]]), dict(aux=[[-1, "k"]]),
                       dict(memory_weight=float("inf"))):
            with self.assertRaises(ValueError):
                worker.validate(dict(request, **change))

    def test_japanese_passes_kana_and_pending_starts_and_needs_a_decoder(self):
        class Japanese:
            def decode(self, *args):
                self.args = args
                return [("日本語", -1.0)]

        request = dict(id=1, operation="japanese", context="前", kana="にほん", pending=["ご"],
                       syllables=3, candidates=[])
        japanese = Japanese()
        self.assertEqual(worker.respond(None, request, japanese=japanese)["candidates"], [["日本語", -1.0]])
        self.assertEqual(japanese.args, ("前", "にほん", ["ご"], None, False, False))
        worker.respond(None, dict(request, shift="110", explicit=True, pending_shift=True), japanese=japanese)
        self.assertEqual(japanese.args[3:], ("110", True, True))
        self.assertEqual(worker.respond(None, request)["candidates"], [])
        for change in (dict(kana=None), dict(pending="ご"), dict(pending=["ごご"]), dict(pending=[1]),
                       dict(shift="11"), dict(shift="1x0"), dict(explicit=1), dict(pending_shift="t")):
            with self.assertRaises(ValueError):
                worker.validate(dict(request, **change))


if __name__ == "__main__":
    unittest.main()
