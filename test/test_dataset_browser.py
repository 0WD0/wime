"""Pagination, version-scoped reviews and lossless training reads."""
import hashlib
import json
import sqlite3
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from dataset_browser import Browser
from dataset_registry import Registry, fingerprint
from dataset_store import export
from replay import rows, split_path


class DatasetBrowser(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.samples = [dict(id=str(i), source=source, context="他说", gold=[text], input=code,
                             pinyin=pinyin, tones=tones, **extra)
                        for i, (source, text, code, pinyin, tones, extra) in enumerate([
                            ("chat", "你好", "nihk", ["ni", "hao"], [2, 3],
                             dict(reading_notes=[dict(index=0, before_tone=3, rules=["contextual_tone_sandhi"])])),
                            ("web", "银行", "ybhh", ["yin", "hang"], [2, 2], {}),
                            ("chat", "吗", "ma", ["ma"], [5], {})])]
        for split in ("train", "dev", "test"):
            (self.data / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.samples))
        (self.data / "manifest.json").write_text('{}')
        catalog = self.root / "catalog.json"
        catalog.write_text('{"version":1,"datasets":{}}')
        self.registry = Registry(catalog, roots={"corpus":self.root})
        self.registry.register(dict(id="example",kind="corpus",status="active",annotation="contextual",parents=[],
                                    location=dict(root="corpus",path="data"),
                                    files=dict(manifest=dict(path="manifest.json"),
                                               **{s:dict(path=s+'.jsonl',format="jsonl") for s in ("train","dev","test")})))
        self.browser = Browser(self.registry, self.root / "cache", self.root / "reviews.sqlite3")
        self.addCleanup(self.browser.db.close)
        self.addCleanup(self.browser.reviews.close)

    def request(self, **extra):
        return dict(dataset="example",split="dev",**extra)

    def test_experiment_link_resolves_exact_sample_and_rejects_another_release(self):
        expected = json.loads(json.dumps(self.registry.entry('example')))
        request = self.request(id='1', expected=expected)
        detail = self.browser.locate(request)
        self.assertEqual(detail['row']['gold'], ['银行'])
        self.assertEqual(detail['snapshot'], self.browser.snapshot('example'))
        self.registry.datasets['example']['files']['dev']['sha256'] = 'another-version'
        with self.assertRaisesRegex(ValueError, '数据版本'):
            self.browser.locate(request)

    def test_cursor_filtering_and_literal_search(self):
        first = self.browser.page(self.request(limit=2))
        second = self.browser.page(self.request(limit=2,after=first['next']))
        self.assertEqual([r['_row'] for r in first['rows']+second['rows']],[1,2,3])
        self.assertTrue(first['has_next']); self.assertFalse(second['has_next'])
        self.assertEqual(self.browser.page(self.request(source="web"))['total'],1)
        self.assertEqual(self.browser.page(self.request(tone=5))['rows'][0]['gold'],['吗'])
        self.assertEqual(self.browser.page(self.request(query="' OR true --"))['total'],0)
        self.assertEqual(self.browser.page(self.request(query="银行"))['total'],1)

    def test_reviews_are_separate_and_undo_rejects_intervening_edits(self):
        before = fingerprint(self.data/'dev.jsonl')
        snapshot = self.browser.snapshot('example')
        request = self.request(row=1,snapshot=snapshot,status='problem',note='上屏边界不自然')
        event = self.browser.review(request)['event']
        self.assertEqual(self.browser.detail(self.request(row=1))['row']['review']['note'],request['note'])
        self.registry.datasets['example']['description'] = 'Updated quality assessment'
        self.assertEqual(self.browser.snapshot('example'), snapshot)
        self.browser.undo(event)
        self.assertEqual(self.browser.review_value('example',snapshot,'dev',1)['status'],'pending')
        event = self.browser.review(request)['event']
        self.browser.review(dict(request,note='另一次审查'))
        with self.assertRaisesRegex(ValueError,'已被修改'):
            self.browser.undo(event)
        with self.assertRaisesRegex(ValueError,'版本已经变化'):
            self.browser.review(dict(request,snapshot='wrong'))
        self.assertEqual(fingerprint(self.data/'dev.jsonl'),before)

    def test_parquet_shards_preserve_missing_notes_and_training_order(self):
        out = self.root/'release'
        result = export(self.registry,'example',out)
        self.assertEqual(result['outputs']['train']['rows'],3)
        self.assertEqual(list(rows(split_path(out,'train'))),self.samples)
        self.assertNotIn('_row',next(rows(out/'train')))
        self.assertNotIn('reading_notes',list(rows(out/'train'))[1])

    def test_legacy_dataset_without_source_can_be_browsed(self):
        for row in self.samples: row.pop('source')
        path=self.data/'dev.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in self.samples))
        self.registry.datasets['example']['files']['dev']['sha256']=fingerprint(path)
        page=self.browser.page(self.request())
        self.assertEqual(page['total'],3)
        self.assertEqual(page['sources'],[('',3)])

    def test_authored_source_detail_survives_without_the_cleaned_database(self):
        text = "报错之后。请检查文件权限。"
        target = "请检查文件权限"
        row = dict(id="authored", source="authored-technical", document="scenario", context="报错之后。",
                   gold=[target], input="ab"*len(target), offset=5, reading_window=[0,len(text)], boundaries=[])
        (self.data / "dev.jsonl").write_text(json.dumps(row,ensure_ascii=False)+'\n')
        (self.data / "documents.jsonl").write_text(json.dumps(dict(id="scenario",text=text,
            content=hashlib.sha256(text.encode()).hexdigest(),origin="authored scenario",source_document="scenario-1")))
        entry = self.registry.datasets['example']
        entry['files']['dev']['sha256'] = fingerprint(self.data / 'dev.jsonl')
        entry['files']['documents'] = dict(path='documents.jsonl',format='jsonl',
                                         sha256=fingerprint(self.data/'documents.jsonl'))
        detail = self.browser.detail(self.request(row=1))
        self.assertEqual(detail['annotation_text'],text)
        self.assertEqual(detail['sentence_text'],target+'。')
        self.assertEqual(detail['source_document'],'scenario-1')
        self.assertEqual(detail['row']['review']['status'],'pending')

    def test_registry_source_snapshot_resolves_original_text_and_detects_drift(self):
        source=self.root/'cleaned';source.mkdir()
        text='前文。你好。'
        (source/'manifest.json').write_text('{}')
        with sqlite3.connect(source/'documents.sqlite3') as db:
            db.execute('CREATE TABLE documents(id TEXT,text TEXT,content TEXT)')
            db.execute('INSERT INTO documents VALUES(?,?,?)',('doc',text,hashlib.sha256(text.encode()).hexdigest()))
        (source/'chat.jsonl').write_text(json.dumps(dict(document='doc',source_document='original-1',cleaning=[]))+'\n')
        self.registry.register(dict(id='cleaned',kind='cleaned',status='active',annotation='none',
            location=dict(root='corpus',path='cleaned'),files=dict(manifest=dict(path='manifest.json'))))
        (self.data/'manifest.json').write_text(json.dumps(dict(selection=dict(
            source=self.registry.snapshot('cleaned',['manifest'])))))
        row=dict(self.samples[0],document='doc',context='前文。',offset=3,reading_window=[0,len(text)])
        (self.data/'dev.jsonl').write_text(json.dumps(row)+'\n')
        for name in ('manifest','dev'):
            self.registry.datasets['example']['files'][name]['sha256']=fingerprint(self.registry.path('example',name))
        detail=self.browser.detail(self.request(row=1))
        self.assertEqual(detail['source_document'],'original-1')
        self.assertEqual(detail['annotation_text'],text)
        (source/'manifest.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'清洗语料版本'):
            self.browser.detail(self.request(row=1))

    def test_indexed_local_documents_preserve_source_detail(self):
        text='前文。你好。'
        path=self.data/'documents.sqlite3'
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE documents(id TEXT PRIMARY KEY,text TEXT,content TEXT,source_document TEXT,origin TEXT,cleaning TEXT)')
            db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?)',('doc',text,hashlib.sha256(text.encode()).hexdigest(),
                'library/example.po','pinned-source','["rendered prose"]'))
        row=dict(self.samples[0],document='doc',context='前文。',offset=3,reading_window=[0,len(text)])
        (self.data/'dev.jsonl').write_text(json.dumps(row)+'\n')
        entry=self.registry.datasets['example']
        entry['files']['dev']['sha256']=fingerprint(self.data/'dev.jsonl')
        entry['files']['documents']=dict(path='documents.sqlite3',sha256=fingerprint(path))
        detail=self.browser.detail(self.request(row=1))
        self.assertEqual(detail['annotation_text'],text)
        self.assertEqual(detail['source_document'],'library/example.po')
        self.assertEqual(detail['cleaning'],['rendered prose'])
        self.assertEqual(detail['source_origin'],'pinned-source')


if __name__ == '__main__':
    unittest.main()
