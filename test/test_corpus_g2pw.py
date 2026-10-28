"""Bounded tokenization caching without loading the g2pW model."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
try:
    from corpus_g2pw import CachedTokenizer, CorpusG2PWConverter, G2PWConverter
except ImportError:
    CachedTokenizer = None


class FakeTokenizer:
    cls_token = "[CLS]"

    def __init__(self):
        self.calls = []

    def tokenize(self, text, *args, **kwargs):
        self.calls.append((text, args, kwargs))
        return [str(text)] + list(args) + list(kwargs.values())

    def convert_tokens_to_ids(self, tokens):
        return [len(token) for token in tokens]


@unittest.skipIf(CachedTokenizer is None, "g2pw is not installed")
class TokenizationCache(unittest.TestCase):
    def test_cached_results_preserve_tokens_and_do_not_share_mutable_lists(self):
        original = FakeTokenizer()
        cached = CachedTokenizer(original)
        first = cached.tokenize("你好")
        first.append("corruption")
        self.assertEqual(cached.tokenize("你好"), ["你好"])
        self.assertEqual(len(original.calls), 1)
        self.assertEqual(cached.cls_token, "[CLS]")
        self.assertEqual(cached.convert_tokens_to_ids(["你好", "word"]), [2, 4])

    def test_lru_evicts_the_least_recent_entry(self):
        original = FakeTokenizer()
        cached = CachedTokenizer(original, maxsize=2)
        for text in ("甲", "乙", "甲", "丙", "乙"):
            cached.tokenize(text)
        self.assertEqual([call[0] for call in original.calls], ["甲", "乙", "丙", "乙"])
        self.assertEqual(cached._tokens.cache_info().currsize, 2)

    def test_options_and_nonstring_inputs_bypass_cache(self):
        original = FakeTokenizer()
        cached = CachedTokenizer(original)
        for _ in range(2):
            self.assertEqual(cached.tokenize("a", suffix="option"), ["a", "option"])
            self.assertEqual(cached.tokenize("a", "positional"), ["a", "positional"])
            self.assertEqual(cached.tokenize(["a"]), ["['a']"])
        self.assertEqual(len(original.calls), 6)
        self.assertEqual(cached._tokens.cache_info().currsize, 0)

    def test_instances_do_not_reuse_another_tokenizers_cache(self):
        first, second = FakeTokenizer(), FakeTokenizer()
        CachedTokenizer(first).tokenize("甲")
        CachedTokenizer(second).tokenize("甲")
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)

    def test_converter_installs_cache_without_changing_constructor_arguments(self):
        original = FakeTokenizer()

        def initialize(converter, *args, **kwargs):
            self.assertEqual(args, ("model-directory",))
            self.assertEqual(kwargs, {"style": "pinyin", "use_cuda": True})
            converter.tokenizer = original

        with patch.object(G2PWConverter, "__init__", initialize):
            converter = CorpusG2PWConverter("model-directory", style="pinyin", use_cuda=True)
        self.assertIsInstance(converter.tokenizer, CachedTokenizer)
        self.assertIs(converter.tokenizer.tokenizer, original)


if __name__ == "__main__":
    unittest.main()
