"""Fixed contextual Mandarin tone golds over supplied model readings, no GPU."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import jieba_fast.posseg as posseg

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "wime_test_tone_sandhi", ROOT / "third_party/gpt-sovits/GPT_SoVITS/text/tone_sandhi.py")
sandhi_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sandhi_module)


class MandarinSandhi(unittest.TestCase):
    def setUp(self):
        self.sandhi = sandhi_module.ToneSandhi()

    def surface(self, text, readings):
        segments = [(word, pos) for word, pos in posseg.cut(text)]
        with patch.object(sandhi_module, "lazy_pinyin", side_effect=AssertionError("supplied readings must be used")):
            merged = self.sandhi.pre_merge_for_modify(segments, readings=readings)
            self.assertEqual("".join(word for word, _ in merged), text)
            result, offset = [], 0
            for word, pos in merged:
                result.extend(self.sandhi.modified_tone(word, pos, readings[offset:offset + len(word)]))
                offset += len(word)
        self.assertEqual(len(result), len(text))
        return result

    def test_fixed_contextual_readings_with_real_jieba_boundaries(self):
        cases = [
            ("你好啊", ["ni3", "hao3", "a1"], ["ni2", "hao3", "a5"]),
            ("第一天", ["di4", "yi1", "tian1"], ["di4", "yi1", "tian1"]),
            ("是不是", ["shi4", "bu4", "shi4"], ["shi4", "bu5", "shi4"]),
            ("一个", ["yi1", "ge4"], ["yi2", "ge5"]),
            ("一个", ["yi1", "ge5"], ["yi2", "ge5"]),
            ("大地", ["da4", "di4"], ["da4", "di4"]),
            ("目的", ["mu4", "di4"], ["mu4", "di4"]),
            ("得到", ["de2", "dao4"], ["de2", "dao4"]),
            ("取得", ["qu3", "de2"], ["qu3", "de2"]),
            ("银行行长", ["yin2", "hang2", "hang2", "zhang3"], ["yin2", "hang2", "hang2", "zhang3"]),
            ("妈妈", ["ma1", "ma1"], ["ma1", "ma5"]),
            ("试试", ["shi4", "shi4"], ["shi4", "shi5"]),
            ("看看书", ["kan4", "kan4", "shu1"], ["kan4", "kan5", "shu1"]),
            ("不可解", ["bu4", "ke3", "jie3"], ["bu4", "ke2", "jie3"]),
            ("不可以", ["bu4", "ke3", "yi3"], ["bu4", "ke2", "yi3"]),
            ("不可能", ["bu4", "ke3", "neng2"], ["bu4", "ke3", "neng2"]),
            ("不可见", ["bu4", "ke3", "jian4"], ["bu4", "ke3", "jian4"]),
            ("不理解", ["bu4", "li3", "jie3"], ["bu4", "li2", "jie3"]),
            ("自己与身边", ["zi4", "ji3", "yu3", "shen1", "bian1"], ["zi4", "ji2", "yu3", "shen1", "bian1"]),
            ("这件事做得很好", ["zhe4", "jian4", "shi4", "zuo4", "de2", "hen3", "hao3"],
             ["zhe4", "jian4", "shi4", "zuo4", "de5", "hen2", "hao3"]),
        ]
        for text, readings, expected in cases:
            with self.subTest(text=text, readings=readings):
                self.assertEqual(self.surface(text, readings), expected)

    def test_particle_guard_also_accepts_upstream_final_only_readings(self):
        for word, pos, finals, expected in [
            ("大地", "n", ["a4", "i4"], ["a4", "i4"]),
            ("目的", "n", ["u4", "i4"], ["u4", "i4"]),
            ("取得", "v", ["v3", "e2"], ["v3", "e2"]),
            ("得", "ud", ["e2"], ["e5"]),
            ("地", "uv", ["e4"], ["e5"]),
        ]:
            with self.subTest(word=word):
                self.assertEqual(self.sandhi.modified_tone(word, pos, finals), expected)

    def test_supplied_readings_control_both_three_tone_merges(self):
        with patch.object(sandhi_module, "lazy_pinyin", side_effect=AssertionError("must not re-read")):
            # A model-selected second tone must not be replaced by dictionary ni3.
            self.assertEqual(self.sandhi.pre_merge_for_modify([("你", "r"), ("好", "a")], ["ni2", "hao3"]),
                             [["你", "r"], ["好", "a"]])
            # Only the word boundary, not the entire left word, has third tones.
            self.assertEqual(self.sandhi.pre_merge_for_modify([("没有", "v"), ("好", "a")],
                                                              ["mei2", "you3", "hao3"]),
                             [["没有好", "v"]])

    def test_invalid_supplied_readings_raise_without_fallback(self):
        for readings in (["ni3"], ["ni", "hao3"], ["ni0", "hao3"], [None, "hao3"]):
            with self.subTest(readings=readings), self.assertRaises(ValueError):
                self.sandhi.pre_merge_for_modify([("你好", "l")], readings=readings)
        with patch.object(self.sandhi, "_merge_yi", side_effect=ValueError("bad supplied path")):
            with self.assertRaisesRegex(ValueError, "bad supplied path"):
                self.sandhi.pre_merge_for_modify([("你好", "l")], readings=["ni3", "hao3"])

    def test_upstream_default_still_uses_pypinyin(self):
        with patch.object(sandhi_module, "lazy_pinyin", wraps=sandhi_module.lazy_pinyin) as original:
            result = self.sandhi.pre_merge_for_modify([("你", "r"), ("好", "a")])
        self.assertTrue(original.called)
        self.assertEqual(result, [["你好", "r"]])


if __name__ == "__main__":
    unittest.main()
