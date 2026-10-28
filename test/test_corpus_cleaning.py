"""Fixed real examples and preservation checks for pre-annotation cleaning."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from corpus_cleaning import clean_document


class Simplifier:
    def convert(self, text):
        return text.translate(str.maketrans({"體": "体", "國": "国"}))


class Cleaning(unittest.TestCase):
    def clean(self, text, source="forum"):
        return clean_document(source, text, Simplifier())

    def test_url_removal_keeps_boundaries_and_adjacent_chinese(self):
        for url in ("https://example.com/path?x=1&y=2", "http://127.0.0.1:8000/x", "www.example.org"):
            with self.subTest(url=url):
                text, operations, reason = self.clean("前" + url + "后面是中文句子。")
                self.assertEqual(text, "前 后面是中文句子。")
                self.assertIn("urls_to_spaces", operations)
                self.assertIsNone(reason)

    def test_existing_base_normalization_and_auditable_operations(self):
        text, operations, reason = self.clean(" \r\n中國\x00\t　體育 \n")
        self.assertEqual(text, "中国 体育")
        self.assertEqual(operations, ["carriage_returns_removed", "ascii_controls_removed",
                                     "horizontal_whitespace_collapsed", "traditional_to_simplified", "outer_whitespace_trimmed"])
        self.assertIsNone(reason)
        self.assertEqual(self.clean(text), (text, [], None))
        self.assertEqual(self.clean("甲\r乙\r\n丙\x00丁"),
                         ("甲\n乙\n丙 丁", ["carriage_returns_removed", "ascii_controls_removed"], None))

    def test_source_limited_entities_and_html_breaks(self):
        for source in ("forum", "zhihu", "fineweb"):
            self.assertEqual(self.clean('形容她用&quot;优秀&quot;<br/>下一行', source)[0], '形容她用"优秀"\n下一行')
        code = '<p class="chat-entry-content">{{ entry.content }}</p>\n上面的&gt;&gt;是什么？<br/>'
        self.assertEqual(self.clean(code, "wildchat"), (code, [], None))
        self.assertEqual(self.clean("&amp;quot;")[0], "&quot;")

    def test_platform_links_preserve_visible_text_and_other_markup(self):
        text = "前<a href='' namecard='*' target='_blank'>@张梦倩</a><br>后"
        cleaned, operations, reason = self.clean(text)
        self.assertEqual(cleaned, "前@张梦倩\n后")
        self.assertIn("platform_link_wrappers_removed", operations)
        self.assertIsNone(reason)
        self.assertEqual(self.clean("兜兜<a usercard='42'>@朋友[大笑]")[0], "兜兜@朋友[大笑]")
        normal = '<a href="local">普通链接</a><span>条件</span> x < y'
        self.assertEqual(self.clean(normal), (normal, [], None))
        quoted = '<a title=" namecard=example">字面属性示例</a>'
        self.assertEqual(self.clean(quoted), (quoted, [], None))

    def test_source_specific_paragraph_markers(self):
        self.assertEqual(self.clean("第一段$P$第二段", "fineweb")[0], "第一段\n第二段")
        self.assertEqual(self.clean("变量$P$", "wildchat"), ("变量$P$", [], None))
        self.assertEqual(self.clean("thumb|right|图注", "zhwiki"), ("thumb|right|图注", [], None))

    def test_invisible_boundaries_do_not_damage_emoji(self):
        text, operations, reason = self.clean("前\u200b后\ufeff文🧘\u200d♀️")
        self.assertEqual(text, "前 后 文🧘\u200d♀️")
        self.assertEqual(operations, ["invisible_separators_to_spaces"])
        self.assertIsNone(reason)

    def test_unrecoverable_unicode_is_quarantined_not_guessed(self):
        for text, kind in (("网址不对，�ż�是什么", "replacement_character"),
                           ("卧槽贵圈真乱\ue413", "private_use_character"),
                           ("前\U000f0001后", "private_use_character")):
            with self.subTest(kind=kind):
                self.assertEqual(self.clean(text), (text, [], "invalid_unicode:" + kind))
        self.assertEqual(self.clean("前\ud800后"), ("", ["surrogate_text_not_serialized"], "invalid_unicode:surrogate_character"))
        self.assertEqual(self.clean("表情&#xE413;")[2], "invalid_unicode:private_use_character")
        self.assertEqual(self.clean("坏字&#xFFFD;")[2], "invalid_unicode:replacement_character")

    def test_conservative_repetition_quarantine(self):
        for text, reason in (("烫" * 123, "repeated_mojibake:tang"),
                             ("锟斤拷" * 4, "repeated_mojibake:kunjinkao"),
                             ("哈" * 40, "excessive_repetition:han_character"),
                             ("让你说实话" * 16 + "【但是不论哪里都有人参赢家】", "excessive_repetition:short_phrase")):
            with self.subTest(reason=reason):
                self.assertEqual(self.clean(text)[2], reason)
        for text in ("哈" * 8, "[大笑]" * 4, "锟斤拷是什么意思", "前言" + "-" * 100 + "后记",
                     "".join(f"第{index}段介绍编码现象和正常的完整文章。" for index in range(10)) + "烫" * 16):
            with self.subTest(text=text[:20]):
                self.assertIsNone(self.clean(text)[2])

    def test_long_foreign_and_mixed_documents_are_retained(self):
        for text in ("This is a normal long passage. " * 100 + "中文解释。",
                     "正常长文章，有完整句子和语义。" * 1000,
                     "音乐作品 The Dark Side of the Moon 值得欣赏。",
                     "if (a < b) { return x; }\n这段代码是什么意思？"):
            with self.subTest(text=text[:20]):
                cleaned, operations, reason = self.clean(text, "wildchat")
                self.assertEqual(cleaned, text)
                self.assertEqual(operations, [])
                self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
