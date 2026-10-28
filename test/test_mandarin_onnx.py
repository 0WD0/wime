"""Character alignment and output cardinality in the imported Mandarin frontend."""
import importlib
import importlib.machinery
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


def load_frontend():
    source = Path(__file__).resolve().parents[3] / "third_party/gpt-sovits/GPT_SoVITS/text"
    namespace = "_wime_mandarin_onnx_tests"
    for name, directory in ((namespace, source), (namespace + ".g2pw", source / "g2pw"),
                            (namespace + ".zh_normalization", source / "zh_normalization")):
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = package
    return (importlib.import_module(namespace + ".g2pw.onnx_api"),
            importlib.import_module(namespace + ".g2pw.dataset"))


class CharacterTokenizer:
    def tokenize(self, word):
        return list(word)

    def convert_tokens_to_ids(self, tokens):
        return [ord(token) if len(token) == 1 else 101 if token == "[CLS]" else 102 for token in tokens]


class MandarinOnnx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.api, cls.dataset = load_frontend()
            import numpy as np
            cls.np = np
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))

    def fallback_converter(self):
        converter = object.__new__(self.api._G2PWBaseOnnxConverter)
        converter.polyphonic_chars_new = set()
        converter.monophonic_chars_dict = {}
        converter.char_bopomofo_dict = {}
        converter.polyphonic_context_chars = 16
        return converter

    def model_input(self, texts, positions):
        return self.dataset.prepare_onnx_input(
            CharacterTokenizer(), ["yin2", "hang2"], {"银": [0], "行": [1]}, ["银", "行"],
            texts, positions, use_mask=True)

    def test_actual_pypinyin_preserves_latin_emoji_and_unknown_han_positions(self):
        converter = self.fallback_converter()
        sentence = "EMİN GÜN SİRER的和面😀\u4dbf"
        *_, results = converter._prepare_data([sentence])
        self.assertEqual(len(results[0]), len(sentence))
        self.assertEqual(results[0][:13], list(sentence[:13]))
        at = sentence.index("和")
        self.assertEqual(results[0][at:at + 2], ["huo2", "mian4"])
        # pypinyin applies its neutral-tone suffix even to an unknown Han
        # placeholder; it must still occupy exactly one character position.
        self.assertEqual(results[0][-2:], ["😀", "\u4dbf5"])

    def test_pure_chinese_contextual_fallback_is_unchanged(self):
        *_, results = self.fallback_converter()._prepare_data(["和工作", "和面", "和稀泥", "咋舌"])
        self.assertEqual(results, [["he2", "gong1", "zuo4"], ["huo2", "mian4"],
                                   ["huo4", "xi1", "ni2"], ["ze2", "she2"]])

    def test_invalid_fallback_cardinality_and_nested_shape_raise(self):
        for result in ([["huo2"]], [["huo2"], []], [["huo2"], [["mian4"]]]):
            with self.subTest(result=result), patch.object(self.api, "pinyin", return_value=result):
                with self.assertRaisesRegex(ValueError, "one reading per sentence character"):
                    self.fallback_converter()._prepare_data(["和面"])

    def test_expanding_lowercase_is_model_only_and_keeps_query_positions(self):
        texts = ["İ银行", "İ银行"]
        actual = self.model_input(texts, [1, 2])
        expected = self.model_input([" 银行", " 银行"], [1, 2])
        for name in actual:
            self.np.testing.assert_array_equal(actual[name], expected[name])
        self.assertEqual(actual["char_ids"].tolist(), [0, 1])
        self.assertEqual(actual["position_ids"].tolist(), [1, 2])
        self.assertEqual(texts, ["İ银行", "İ银行"])

    def test_invalid_query_metadata_and_tokenization_raise(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            self.model_input(["银行"], [])
        with self.assertRaisesRegex(ValueError, "character indices"):
            self.model_input(["银行"], [2])
        with self.assertRaisesRegex(ValueError, "no valid token"):
            self.model_input([" 银行"], [0])
        with patch.object(self.dataset, "tokenize_and_map", side_effect=RuntimeError("broken tokenizer")):
            with self.assertRaisesRegex(ValueError, "cannot tokenize"):
                self.model_input(["银行"], [0])

    def test_model_probability_shape_must_match_queries_and_labels(self):
        inputs = self.model_input(["银行", "银行"], [0, 1])
        for shape in ((1, 2), (2, 1), (2,), (2, 2, 1)):
            session = types.SimpleNamespace(run=lambda *args, shape=shape: [self.np.zeros(shape)])
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "probability shape"):
                self.api.predict(session, inputs, ["yin2", "hang2"])
        session = types.SimpleNamespace(run=lambda *args: [self.np.array([[0.9, 0.1], [0.2, 0.8]])])
        self.assertEqual(self.api.predict(session, inputs, ["yin2", "hang2"]),
                         (["yin2", "hang2"], [0.9, 0.8]))

    def test_truncated_prediction_list_cannot_silently_leave_partial_results(self):
        converter = self.fallback_converter()
        converter.enable_opencc = False
        converter.enable_sentence_dedup = False
        converter.config = types.SimpleNamespace(use_mask=False, use_char_phoneme=False)
        converter.tokenizer = CharacterTokenizer()
        converter.labels = ["yin2"]
        converter.char2phonemes, converter.chars = {}, []
        converter.char2id, converter.char_phoneme_masks = {}, {}
        converter._predict = lambda **kwargs: ([], [])
        with patch.object(converter, "_prepare_data", return_value=(["银"], [0], [0], [0], [[None]])), \
             patch.object(self.api, "prepare_onnx_input", return_value={"char_ids": self.np.array([0])}):
            with self.assertRaisesRegex(ValueError, "one result per requested character"):
                converter(["银"])

    def test_deduplicated_prediction_count_is_checked_before_zip(self):
        converter = self.fallback_converter()
        converter._predict = lambda **kwargs: (["yin2"], [1.0])
        inputs = self.model_input(["银行", "银行"], [0, 1])
        with self.assertRaisesRegex(ValueError, "group's requested characters"):
            converter._predict_with_sentence_dedup(inputs, ["银行", "银行"])


if __name__ == "__main__":
    unittest.main()
