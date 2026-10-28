"""Source identity, streaming and word-boundary checks without a g2pW model."""
import bz2
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
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
            '<page><title>甲</title><ns>0</ns><revision><text>第一篇文章。</text></revision></page>',
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
        self.assertTrue(all(group == document for group, document, _, _ in rows))


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


if __name__ == "__main__":
    unittest.main()
