"""Worker contract tests run without a GPU or model artifact."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

directory = Path(__file__).resolve().parents[1] / "neural"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("gpu_worker", directory / "gpu-worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerContract(unittest.TestCase):
    def request(self, **changes):
        return dict(dict(id=1, context="", syllables=2, operation="decode", input="bmyi"), **changes)

    def test_malformed_envelopes_are_rejected(self):
        for request in [None, [], {}, self.request(id=True), self.request(syllables=-1),
                        self.request(context=[]), self.request(operation="verify"),
                        self.request(context_anchor=-1), self.request(context_anchor=9),
                        self.request(context_anchor=True),
                        self.request(input=None), self.request(input=[]),
                        self.request(context_reading=dict(text="便宜", input="bm")),
                        self.request(context_reading=dict(text="便宜", input="bmyi"))]:
            with self.assertRaises(ValueError):
                worker.validate(request)
    def test_decode_tones_are_syllable_tone_pairs(self):
        decode = self.request(operation="decode", input="bmyi", tones=[[0, 1], [1, 5]])
        worker.validate(decode)
        for tones in ([[0, 0]], [[0, 6]], [[0, "1"]], [[-1, 1]], [[0]], "1"):
            with self.subTest(tones=tones), self.assertRaises(ValueError):
                worker.validate(dict(decode, tones=tones))

    def test_decode_passes_pins_and_learned_readings_and_needs_a_decoder(self):
        class Decoder:
            def decode(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
                return [("便宜", -.5)]

        request = self.request(operation="decode", input="bmy", syllables=2,
                               pins=[[0, "便"]], learned=[[0, "便宜"]], memory_weight=10.0,
                               literals=[[1, "GPU "]], tones=[[1, 2]])
        decoder = Decoder()
        self.assertEqual(worker.respond(SimpleNamespace(model=SimpleNamespace(cfg=SimpleNamespace(context=128))), request, decoder)["candidates"], [["便宜", -.5]])
        self.assertEqual(decoder.args, ("", "bmy"))
        self.assertEqual(decoder.kwargs, dict(pins=[(0,"便")], learned=[(0,"便宜")], weight=10., aux=[],
                                              literals=[(1, "GPU ")], tones=[(1, 2)]))
        worker.respond(SimpleNamespace(model=SimpleNamespace(cfg=SimpleNamespace(context=128))), dict(request, aux=[[1, "kp"]]), decoder)
        self.assertEqual(decoder.kwargs["aux"], [(1,"kp")])
        self.assertEqual(worker.respond(None, request)["candidates"], [])
        for change in (dict(input="bmA"), dict(pins=[[0, "便宜"]]), dict(pins=[["0", "便"]]),
                       dict(learned=[[0, ""]]), dict(learned=[0]), dict(memory_weight=-1),
                       dict(aux=[[0, "K"]]), dict(aux=[[0, ""]]), dict(aux=[[-1, "k"]]),
                       dict(memory_weight=float("inf")), dict(literals=[[0, "a"]]),
                       dict(literals=[[1, ""]]), dict(literals=[[1, "\ue2e1"]])):
            with self.assertRaises(ValueError):
                worker.validate(dict(request, **change))
    def test_japanese_passes_kana_and_pending_starts_and_needs_a_decoder(self):
        class Japanese:
            def decode(self, *args):
                self.args = args
                return [("日本語", -1.0)]

        request = dict(id=1, operation="japanese", context="前", kana="にほん", pending=["ご"],
                       syllables=3)
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
