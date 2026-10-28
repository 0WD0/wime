"""Source identity, streaming and word-boundary checks without a g2pW model."""
import bz2
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
spec = importlib.util.spec_from_file_location(
    "prepare_corpus", Path(__file__).resolve().parents[1] / "neural/prepare-corpus.py")
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


class Sources(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def parquet(self, relative, rows):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)

    def gz(self, relative, rows):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_wildchat_turns_share_split_group_but_have_distinct_documents(self):
        self.parquet("wildchat/data/0.parquet", [
            dict(conversation_hash="conversation", language="Chinese", conversation=[
                dict(role="user", content="我想了解一个问题"),
                dict(role="assistant", content="请说"),
                dict(role="user", content="我想了解另一个问题"),
                dict(role="user", content="我想了解一个问题")]),
            dict(conversation_hash="english", language="English", conversation=[
                dict(role="user", content="hello")]),
            dict(conversation_hash="another", language="Chinese", conversation=[
                dict(role="user", content="这是一段不同的文字")])])
        with patch.object(corpus, "PARQUET_BATCH_SIZE", 1):
            rows = list(corpus.wildchat(self.root))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][0], rows[1][0])
        self.assertNotEqual(rows[0][1], rows[1][1])
        self.assertEqual(rows[1][1], "wildchat:conversation:turn:2")
        self.assertNotEqual(rows[0][0], rows[2][0])

    def test_fineweb_row_ids_survive_batch_boundaries_and_copies_share_group(self):
        self.parquet("fineweb-edu/4_5/000000.parquet", [
            {"text": text} for text in ("一", "二", "三", "四", "五")])
        self.parquet("fineweb-edu/4_5/000001.parquet", [{"text": "三"}])
        with patch.object(corpus, "PARQUET_BATCH_SIZE", 2):
            rows = list(corpus.fineweb(self.root))
        self.assertEqual([row[1] for row in rows[:5]],
                         [f"fineweb:000000:{i}" for i in range(5)])
        self.assertEqual(rows[2][0], rows[5][0])
        self.assertNotEqual(rows[2][1], rows[5][1])

    def test_zhihu_groups_answers_by_question_across_files(self):
        def answer(index, answer_id):
            return {"id": index, "问": "同一个问题", "答": f"第{index}个回答", "元数据": {
                "扩展字段": json.dumps({"question_id": 17.0, "answer_id": answer_id})}}
        self.gz("mnbvc/qa/20230196/zhihu/0.jsonl.gz", [answer(0, 101.0)])
        self.gz("mnbvc/qa/20230196/zhihu/1.jsonl.gz", [answer(1, 102)])
        rows = list(corpus.zhihu(self.root))
        self.assertEqual(len(rows), 3)
        self.assertEqual({r[0] for r in rows}, {"zhihu:question:17"})
        self.assertEqual(len({r[1] for r in rows}), 3)
        self.assertEqual(sum(r[3] == "同一个问题" for r in rows), 1)

    def test_forum_replies_have_distinct_documents_with_one_thread_group(self):
        thread = {"ID": 123, "主题": "主题", "回复": [{"回复": "同意"}, {"回复": "同意"}]}
        self.gz("mnbvc/forum/20230253/example/0.jsonl.gz", [thread])
        self.gz("mnbvc/forum/20230253/example/1.jsonl.gz", [thread])
        rows = list(corpus.forum(self.root))
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({r[0] for r in rows}), 1)
        self.assertEqual(len({r[1] for r in rows[:3]}), 3)
        # A duplicated thread in another shard retains its identity.
        self.assertEqual(rows[:3], rows[3:])

    def test_forum_zip_directory_entries_are_skipped(self):
        path = self.root / "mnbvc/forum/20260107/example/0.zip"
        path.parent.mkdir(parents=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data/", "")
            archive.writestr("data/0.jsonl", json.dumps({"ID": 1, "主题": "主题", "回复": []}) + "\n")
        self.assertEqual(len(list(corpus.forum(self.root))), 1)

    def test_forum_snapshot_versions_share_group_without_document_collisions(self):
        # The raw PKU archive contains adjacent rows with ID 15680 whose
        # second topic adds a timestamp. Replies may also change in snapshots.
        snapshots = [
            {"ID": 15680, "主题": "同一个主题的内容", "回复": [{"回复": "原始回复"}]},
            {"ID": 15680, "主题": "2014-10-25 23:38:24\n同一个主题的内容", "回复": [{"回复": "修改后的回复"}]},
        ]
        self.gz("mnbvc/forum/20230253/example/0.jsonl.gz", snapshots)
        rows = list(corpus.forum(self.root))
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row[0] for row in rows}), 1)
        self.assertEqual(len({row[1] for row in rows}), 4)
        self.assertEqual({corpus.split_of(row[0]) for row in rows}, {corpus.split_of(rows[0][0])})

    def test_wiki_streams_multiple_pages_and_skips_redirects_and_other_namespaces(self):
        pages = [
            '<page><title>甲</title><ns>0</ns><revision><text>第一篇文章。[[File:example.png|right|thumb|200px|图片说明]]</text></revision></page>',
            '<page><title>重定向</title><ns>0</ns><redirect title="甲"/><revision><text>#REDIRECT [[甲]]</text></revision></page>',
            '<page><title>讨论</title><ns>1</ns><revision><text>讨论文章。</text></revision></page>',
            '<page><title>乙</title><ns>0</ns><revision><text>第二篇[[文章]]。</text></revision></page>',
        ]
        path = self.root / "zhwiki/zhwiki-20261001-pages-articles.xml.bz2"
        path.parent.mkdir(parents=True)
        with bz2.open(path, "wt", encoding="utf-8") as stream:
            stream.write('<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">' + "".join(pages) + "</mediawiki>")
        rows = list(corpus.zhwiki(self.root))
        self.assertEqual([r[3] for r in rows], ["第一篇文章。", "第二篇文章。"])
        self.assertEqual([r[0] for r in rows], ["zhwiki:甲", "zhwiki:乙"])
        self.assertTrue(all(group == document for group, document, _, _ in rows))

    def test_wiki_plant_layout_and_unmatched_styles_preserve_link_labels(self):
        # The real 植物学 source closes a three-quote opener with five quotes.
        source = """通过纸色谱法看到的不同种类。
<div style="position:relative">
<div style="top:245px">[[叶黄素类|<span style="color:#e2c000;">'''叶黄素'''</span>]]</div>
<div style="top:272px">[[叶绿素|<span style="color:#00bb90;">'''叶绿素a'''''</span>]]</div>
<div style="top:300px">[[叶绿素b|<span style="color:#00bb34;">'''叶绿素b'''''</span>]]</div>
</div>
植物细胞内有[[叶绿体]]。"""
        rendered = corpus.render_wiki(source)
        self.assertEqual(rendered.splitlines(), ["通过纸色谱法看到的不同种类。", "叶黄素", "叶绿素a",
                                                 "叶绿素b", "植物细胞内有叶绿体。"])
        self.assertNotIn("[[", rendered)
        self.assertNotIn("style=", rendered)

    def test_wiki_psychology_references_do_not_swallow_following_prose(self):
        # A citation's unmatched italic delimiter used to expose later <ref>.
        source = ("在中世纪，已经开始进行[[精神科]]治疗"
                  "<ref>Ibrahim B. Syed, Journal of the Islamic Medical Association''.</ref>"
                  "和[[实验心理学|实验]]研究<ref name=Khaleefa/>"
                  "<ref>这段是引用资料，不是正文。</ref>。\n后续[[科学]]研究。")
        before = corpus.SOURCE_TRANSFORMS["zhwiki/references_removed"]
        self.assertEqual(corpus.render_wiki(source), "在中世纪，已经开始进行精神科治疗和实验研究。\n后续科学研究。")
        self.assertEqual(corpus.SOURCE_TRANSFORMS["zhwiki/references_removed"] - before, 3)

    def test_wiki_history_table_keeps_cell_and_paragraph_boundaries(self):
        source = """<table><tr align="center">
<td rowspan="3" style="background-color: rgb(153, 255, 153);">[[北魏]]<br>[[386年|386]] - [[534年|534]]</td>
<td style="background-color:#eee">[[534年]] - 北魏分裂为[[西魏]]、[[东魏]]。</td>
</tr><tr><td>[[梁 (南朝)|南朝梁]]</td><td>[[548年]] - 梁将军侯景叛乱。</td></tr></table>"""
        rendered = corpus.render_wiki(source)
        self.assertEqual(rendered.splitlines(), ["北魏 386 - 534", "534年 - 北魏分裂为西魏、东魏。",
                                                 "南朝梁", "548年 - 梁将军侯景叛乱。"])
        self.assertNotRegex(rendered, r"</?(?:table|tr|td)\b")

    def test_wiki_xml_nested_code_nowiki_keeps_visible_div_example(self):
        example = '<div class="inner greeting-box">Welcome!</div>'
        source = f'例如，<code><nowiki>{example}</nowiki></code>中，属性 "class" 表示两个 CSS 类名。'
        self.assertEqual(corpus.render_wiki(source), f'例如，{example}中，属性 "class" 表示两个 CSS 类名。')

    def test_wiki_literal_regions_bypass_media_layout_styles_and_variant_rendering(self):
        literal = '<div>[[File:x.png|thumb]] \'\'code\'\' -{zh-hans:字面;}- &amp;amp; &copyfr=x &notit;</div>'
        visible = literal.replace("&amp;amp;", "&amp;")
        for tag in ("nowiki", "code", "pre", "syntaxhighlight", "source"):
            with self.subTest(tag=tag):
                source = f'正文。\n<{tag} lang="example">{literal}</{tag}>\n后文[[普通链接|显示文字]]。'
                self.assertEqual(corpus.render_wiki(source), f"正文。\n{visible}\n后文显示文字。")

    def test_wiki_quote_aware_tags_do_not_consume_ordinary_comparison_or_literal_angle_text(self):
        source = '正文<div title="a > b">第一段</div><p>第二段，a < b 且 b > c。</p>'
        self.assertEqual(corpus.render_wiki(source), "正文\n第一段\n第二段，a < b 且 b > c。")

    def test_wiki_self_closing_nowiki_cannot_protect_later_article_prose(self):
        # HTML and 统一资源标识符 used <nowiki/> to stop adjacent link syntax;
        # treating it as an opener swallowed thousands of characters of prose.
        for empty in ("<nowiki/>", "<nowiki />", "<NOWIKI />"):
            with self.subTest(empty=empty):
                source = (f"[[HTML]]{empty}发布。\n历史[[版本]]"
                          "<ref>这是历史引用</ref>仍在维护。"
                          "[[File:HTML5-logo.svg|缩略图|右|200px|图片]]\n"
                          "示例<nowiki><div>代码</div></nowiki>结束。")
                self.assertEqual(corpus.render_wiki(source),
                                 "HTML发布。\n历史版本仍在维护。\n示例<div>代码</div>结束。")
        literal = "示例<nowiki/>[[File:x.svg|200px|文本]]<div>代码</div>"
        self.assertEqual(corpus.render_wiki(f"<pre>{literal}</pre>"), literal)

    def test_wiki_known_layout_tags_recover_bad_quotes_without_consuming_prose(self):
        damaged = ['<div class="thumbcaption" sty-size: 88%;">',
                   '<div class="collapsible-title" style="background-color: #D4ECF7;style="height: 12em;">']
        for tag in damaged:
            with self.subTest(tag=tag):
                self.assertEqual(corpus.render_wiki(f"前文{tag}正文</div>后文。"), "前文\n正文\n后文。")
                self.assertEqual(corpus.render_wiki(f"示例<code>{tag}正文</div></code>。"), f"示例{tag}正文</div>。")
        self.assertEqual(corpus.render_wiki("前文</ref>正文。</ref>后文。"), "前文正文。后文。")

    def test_wiki_break_inside_table_cell_preserves_later_column_syntax(self):
        # Actual 巴黎 and 島嶼 tables place || after a cell's <br>/<small>.
        source = ('{| class="wikitable"\n'
                  '| 巴黎市區<br /><small>（75省）</small>|| align=right | 2,211,297 '
                  '|| align=right | 20169人/平方公里\n'
                  '|-\n| 本州岛<br>日本海 || 亞洲 || align=right|225,800\n|}')
        rendered = corpus.render_wiki(source)
        self.assertNotIn("align=", rendered)
        self.assertNotIn("||", rendered)
        for value in ("巴黎市區", "（75省）", "2,211,297", "20169人/平方公里", "本州岛", "日本海", "亞洲", "225,800"):
            self.assertIn(value, rendered)

    def test_wiki_comments_in_attributes_and_broken_reference_delimiters(self):
        source = ('<div style="position:relative; <!--border:1px solid;--> width:100px;">'
                  '植物色素</div>正文<ref group="cctld>{{cite web|title=参考文献}}</ref>后文。\n'
                  '{{citation|title=参考书籍}}&lt;/ref>\n'
                  '引用标记&lt;/ref&gt;应保留。')
        self.assertEqual(corpus.render_wiki(source), "植物色素\n正文后文。\n引用标记</ref>应保留。")
        self.assertEqual(corpus.render_wiki('正文<ref name="坏/>保留正文<ref>引用内容</ref>后文。'),
                         "正文保留正文后文。")
        literal = '<!--示例-->{{citation|title=书籍}}&lt;/ref>'
        self.assertEqual(corpus.render_wiki(f"<nowiki>{literal}</nowiki>"), '<!--示例-->{{citation|title=书籍}}</ref>')

    def test_wiki_recovers_misplaced_chinese_variant_link_but_preserves_literal_cdata(self):
        broken = ("-[[{zh-hans:安东尼·德·圣-埃克苏佩里;zh-hk:安東尼·德·聖-艾修伯里;"
                  "zh-tw:安托萬·德·聖-埃克絮佩里;}]]-")
        source = f"1943年由{broken}寫的《[[小王子]]》。"
        before = corpus.SOURCE_TRANSFORMS["zhwiki/malformed_link_variants_recovered"]
        self.assertEqual(corpus.render_wiki(source), "1943年由安东尼·德·圣-埃克苏佩里寫的《小王子》。")
        self.assertEqual(corpus.SOURCE_TRANSFORMS["zhwiki/malformed_link_variants_recovered"] - before, 1)
        literal = f"<![CDATA[{broken}]]>"
        self.assertEqual(corpus.render_wiki(f"代码<nowiki>{literal}</nowiki>。"), f"代码{literal}。")
        corpus.render_wiki("普通-[[{en:English;fr:Français;}]]-文本。")
        self.assertEqual(corpus.SOURCE_TRANSFORMS["zhwiki/malformed_link_variants_recovered"] - before, 1)

    def test_clean_stage_materializes_only_supported_text_without_annotation(self):
        from corpus_cleaned import load_cleaned, iter_cleaned
        relative = "mnbvc/forum/20230253/example/0.jsonl.gz"
        self.gz(relative, [
            {"ID": 1, "主题": "正常中文&amp;英语context", "回复": []},
            {"ID": 2, "主题": "坏字\ufffd必须隔离", "回复": []},
            {"ID": 3, "主题": "正常中文&amp;英语context", "回复": []},
        ])
        manifest_path = self.root / "sources.json"
        manifest_path.write_text(json.dumps({"sources": {"forum": {"files": {
            relative: corpus.file_metadata(self.root / relative)["sha256"]}}}}))
        database = self.root / "dictionary.db"
        database.write_bytes(b"dictionary fingerprint only; cleaning never loads a model")
        args = SimpleNamespace(raw=self.root, out=self.root / "cleaned", source_manifest=manifest_path,
                               database=database, exclude_cases=[], max_chars=0)
        with patch.object(corpus, "Reader", side_effect=AssertionError("clean stage must not annotate")):
            corpus.clean_corpus(args, ["forum"])
        manifest = load_cleaned(args.out, ["forum"])
        docs = list(iter_cleaned(args.out, "forum"))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["text"], "正常中文&英语context")
        self.assertNotEqual(docs[0]["raw_content"], docs[0]["content"])
        self.assertEqual(manifest["stats"]["forum/rejected/invalid_unicode:replacement_character"], 1)
        self.assertEqual(manifest["stats"]["forum/rejected/duplicate_document"], 1)


class WordSpans(unittest.TestCase):
    def test_length_cap_never_cuts_a_dictionary_word(self):
        class Lexicon:
            def segment(self, text):
                return [(i, i + 7, text[i:i + 7]) for i in range(0, len(text), 7)]

        class LongSpans:
            def __init__(self):
                self.density = False

            def random(self):
                self.density = not self.density
                return 0 if self.density else 0.99

            def randint(self, low, high):
                return 24 if high == 24 else 0

        spans = list(corpus.spans("甲" * 70, ["jx"] * 70, Lexicon(), LongSpans(), 1))
        self.assertEqual(spans, [(0, 21), (21, 42), (42, 63), (63, 70)])


class AnnotationWindows(unittest.TestCase):
    class EveryCharacter:
        def random(self):
            return 0

        def randint(self, low, high):
            return low

    def test_short_sentences_keep_original_input_and_document_offsets(self):
        self.assertEqual(list(corpus.annotation_windows("你好。下一句！")), [
            corpus.AnnotationWindow("你好。", 0, 0, 3),
            corpus.AnnotationWindow("下一句！", 3, 0, 4)])

    def test_halos_are_bounded_and_every_document_character_has_one_core(self):
        document = "前句。\n" + "啊" * 127 + "你好" + "啊" * 140 + "。\n尾句！"
        owners = []
        for window in corpus.annotation_windows(document):
            self.assertLessEqual(len(window.text), 256)
            self.assertLessEqual(window.core_end - window.core_start, 128)
            self.assertLessEqual(window.core_start, 64)
            self.assertLessEqual(len(window.text) - window.core_end, 64)
            self.assertEqual(window.text, document[window.offset:window.offset + len(window.text)])
            owners.extend(range(window.offset + window.core_start, window.offset + window.core_end))
        self.assertEqual(owners, list(range(len(document))))

    def annotated_rows(self, document):
        from corpus_readings import Reader, key
        from mandarin_frontend import MandarinFrontend
        readings = {"啊": "a1", "你": "ni3", "好": "hao3", "一": "yi1", "个": "ge4",
                    "不": "bu4", "是": "shi4", "㐂": None}

        class CharacterLexicon:
            legal = {char: {key(reading[:-1])} for char, reading in readings.items() if reading}

            def segment(self, text):
                return [(i, i + 1, None) for i in range(len(text))]

        lexicon = CharacterLexicon()
        frontend = MandarinFrontend(
            converter=lambda sentences: [[readings.get(char, char) for char in sentence] for sentence in sentences],
            corrections={})
        windows = list(corpus.annotation_windows(document))
        annotation = Reader(lexicon, frontend).readings([window.text for window in windows])
        return [row for window, reading in zip(windows, annotation)
                for row in corpus.annotated_spans(document, window, reading, lexicon, self.EveryCharacter(), 1)]

    def test_contextual_tones_survive_core_boundary_and_single_character_sampling(self):
        from corpus_tones import validate_tones
        for phrase, final, before in (("你好", "ni2", "ni3"), ("一个", "yi2", "yi1"), ("不是", "bu2", "bu4")):
            with self.subTest(phrase=phrase):
                document = "啊" * 127 + phrase + "啊" * 140
                rows = self.annotated_rows(document)
                self.assertEqual([row["offset"] for row in rows], list(range(len(document))))
                first, second = rows[127], rows[128]
                self.assertEqual(first["gold"], [phrase[0]])
                self.assertEqual(first["reading_notes"], [dict(index=0, before_tone=int(before[-1]),
                                                             rules=["contextual_tone_sandhi"])])
                self.assertEqual(first["tones"], [int(final[-1])])
                self.assertEqual(first["pinyin"], [final[:-1]])
                self.assertNotIn("reading_evidence", first)
                self.assertNotIn("tone_sources", first)
                self.assertEqual(first["reading_window"], [0, 192])
                self.assertEqual(second["reading_window"], [64, 269])
                self.assertEqual(second["context"], document[64:128])
                for row in rows:
                    validate_tones(row)
                    self.assertEqual(row["gold"][0], document[row["offset"]:row["offset"] + 1])
                    self.assertLessEqual(row["reading_window"][0], row["offset"])
                    self.assertLess(row["offset"], row["reading_window"][1])

    def test_unknown_characters_never_become_emitted_spans(self):
        rows = self.annotated_rows("你㐂好")
        self.assertEqual([row["offset"] for row in rows], [0, 2])
        self.assertTrue(all(all(tone is not None for tone in row["tones"]) for row in rows))


if __name__ == "__main__":
    unittest.main()
