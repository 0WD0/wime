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
                        self.request(context=[]), self.request(candidates=[{}])]:
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


if __name__ == "__main__":
    unittest.main()
