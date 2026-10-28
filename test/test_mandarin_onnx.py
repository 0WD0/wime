"""Character alignment and output cardinality in the imported Mandarin frontend."""
from collections import Counter
import gc
import importlib
import importlib.machinery
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import weakref


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
            cls.utils = importlib.import_module(cls.api.__package__ + ".utils")
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

    def bert_tokenizer(self):
        # Exercise real BERT normalization without network or model assets.
        from transformers import BertTokenizer
        jamo = "래눈하임ᅳᆯᅱᆼᄌ곡"
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "银", "行", "n", "abc", "##def"]
        vocab += sorted(set(jamo) | {"##" + char for char in jamo})
        return BertTokenizer(vocab={token: index for index, token in enumerate(vocab)})

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

    def test_actual_bert_expansion_keeps_original_corpus_character_spans(self):
        tokenizer = self.bert_tokenizer()
        self.assertEqual(tokenizer.tokenize("래"), ["ᄅ", "##ᅢ"])
        self.assertEqual(tokenizer.tokenize("눈"), ["ᄂ", "##ᅮ", "##ᆫ"])
        # First failures in the forum and WildChat pilot, plus Han queries
        # following the expanded characters (which must never shift).
        for text in ("@赵恩来♂n래", "评价一下임을 위한 행진곡", "눈_눈银行"):
            with self.subTest(text=text):
                tokens, text2token, token2text = self.utils.tokenize_and_map(tokenizer, text)
                self.assertEqual(len(text2token), len(text))
                for index, char in enumerate(text):
                    if char == " ":
                        self.assertIsNone(text2token[index])
                        continue
                    source_span = (index, index + 1)
                    expected = tokenizer.tokenize(char) or ["[UNK]"]
                    self.assertEqual([token for token, span in zip(tokens, token2text) if span == source_span], expected)
                    self.assertEqual(token2text[text2token[index]], source_span)
                self.assertTrue(all(0 <= start < end <= len(text) for start, end in token2text))

    def test_ascii_wordpiece_alignment_is_unchanged_and_malformed_offsets_raise(self):
        tokens, text2token, token2text = self.utils.tokenize_and_map(self.bert_tokenizer(), "abcdef银行")
        self.assertEqual(tokens, ["abc", "##def", "银", "行"])
        self.assertEqual(text2token, [0, 0, 0, 1, 1, 1, 2, 3])
        self.assertEqual(token2text, [(0, 3), (3, 6), (6, 7), (7, 8)])
        for tokens in (["ab"], ["abcd"]):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, "ASCII wordpiece offsets"):
                self.utils.tokenize_and_map(types.SimpleNamespace(tokenize=lambda word: tokens), "abc")

    def test_word_cache_reuses_results_but_isolates_and_releases_tokenizers(self):
        class CountingTokenizer:
            def __init__(self, pieces):
                self.pieces, self.calls = pieces, Counter()

            def tokenize(self, word):
                self.calls[word] += 1
                return self.pieces.get(word, ["[UNK]"])

        first = CountingTokenizer({"银": ["银"], "行": ["行"], "abc": ["abc"]})
        second = CountingTokenizer({"银": ["银"], "行": ["行"], "abc": ["a", "##bc"]})
        text = "银行 abc missing 银行 abc"
        expected = self.utils.tokenize_and_map(first, text)
        self.assertEqual(expected[0], ["银", "行", "abc", "[UNK]", "银", "行", "abc"])
        self.assertEqual(self.utils.tokenize_and_map(first, text), expected)
        self.assertEqual(first.calls, Counter({"银": 1, "行": 1, "abc": 1, "missing": 1}))
        self.assertEqual(first._g2pw_word_tokenize("abc"), ("abc",))
        self.assertEqual(first._g2pw_word_tokenize.cache_info().maxsize, 65536)
        self.assertEqual(self.utils.tokenize_and_map(second, text)[0],
                         ["银", "行", "a", "##bc", "[UNK]", "银", "行", "a", "##bc"])
        self.assertEqual(second.calls, first.calls)
        reference = weakref.ref(first)
        del first
        gc.collect()
        self.assertIsNone(reference())

    def test_han_model_queries_after_expansion_select_the_original_han_tokens(self):
        tokenizer = self.bert_tokenizer()
        sentence = "눈_눈银行"
        positions = [sentence.index(char) for char in "银行"]
        inputs = self.dataset.prepare_onnx_input(
            tokenizer, ["yin2", "hang2"], {"银": [0], "行": [1]}, ["银", "行"],
            [sentence] * 2, positions, use_mask=True)
        self.assertEqual(inputs["char_ids"].tolist(), [0, 1])
        self.assertEqual([row[pos] for row, pos in zip(inputs["input_ids"], inputs["position_ids"])],
                         tokenizer.convert_tokens_to_ids(["银", "行"]))

    def test_truncation_inside_expanded_character_retains_valid_source_mapping(self):
        text = "한은행行"  # A three-token source character at the retained left edge.
        tokenizer = self.bert_tokenizer()
        tokens, text2token, token2text = self.utils.tokenize_and_map(tokenizer, text)
        query = text.index("行")
        cropped, position, pieces, forward, backward = self.dataset._truncate(
            5, text, query, tokens, text2token, token2text)
        self.assertEqual(pieces[:2], ["##ᅢ", "##ᆼ"])
        self.assertEqual(cropped[position], "行")
        self.assertEqual(pieces[forward[position]], "行")
        self.assertTrue(all(value is None or 0 <= value < len(pieces) for value in forward))
        for index, value in enumerate(forward):
            if value is not None:
                self.assertLessEqual(backward[value][0], index)
                self.assertGreater(backward[value][1], index)

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

    def test_dedup_shares_only_identical_actual_query_windows(self):
        converter = self.fallback_converter()
        observed_rows = []

        def predict(model_input):
            observed_rows.append(len(model_input["input_ids"]))
            return ["yin2"] * len(model_input["char_ids"]), [1.0] * len(model_input["char_ids"])

        converter._predict = predict
        for max_len, expected_rows in ((4, 2), (512, 1)):
            with self.subTest(max_len=max_len):
                text = "银行行银"
                inputs = self.dataset.prepare_onnx_input(
                    CharacterTokenizer(), ["yin2", "hang2"], {"银": [0], "行": [1]}, ["银", "行"],
                    [text] * 2, [0, 3], use_mask=True, max_len=max_len)
                converter._predict_with_sentence_dedup(inputs, [text] * 2)
                self.assertEqual(observed_rows[-1], expected_rows)


if __name__ == "__main__":
    unittest.main()
