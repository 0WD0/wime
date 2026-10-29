"""Training boundary rules should reject the fragments found during source review."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'neural'))
from sentence_sampling import boundary,legacy_span,source_units,retain_group


class SentenceSampling(unittest.TestCase):
    def test_units_preserve_source_and_do_not_cut_to_length(self):
        text='  系统出现异常以后，程序会保留原始错误信息。'
        units=list(source_units(text))
        self.assertEqual([r['text'] for r in units],['系统出现异常以后','程序会保留原始错误信息'])
        for row in units:
            self.assertEqual(text[row['offset']:row['offset']+len(row['text'])+1],row['text']+row['terminal'])
            self.assertEqual(row['unit_type'],'clause')
        self.assertEqual(list(source_units('这是 HTTP 请求需要传输数据。')),[])
        self.assertEqual(list(source_units('系统会保留原始错误信息'*8+'。')),[])
        long='系统出现异常以后程序会保留原始错误信息并且将详细的处理结果展示给用户'
        self.assertEqual(list(source_units(long+'。'))[0]['text'],long)

    def test_group_reservoir_keeps_whole_only_rows_and_long_band(self):
        rows=[dict(id=str(i),group='one',gold=['系统出现异常以后' if i<5 else '系统会保留原始错误信息'*3],boundaries=[])
              for i in range(10)]
        a,b={},{}
        for row in rows:retain_group(a,row,2)
        for row in reversed(rows):retain_group(b,row,2)
        self.assertEqual(a,b)
        self.assertEqual([[r['id'] for r in bucket] for bucket in a.values()],[['0','1'],['5','6']])

    def test_explicit_condition_and_topic(self):
        self.assertEqual(boundary('如果系统失败就重新启动', [('如果','c'),('系统','n'),('失败','v'),('就','d'),('重新','d'),('启动','v')])[0]['end'],6)
        self.assertEqual(boundary('这个软件需要更新配置', [('这个','r'),('软件','n'),('需要','v'),('更新','v'),('配置','n')])[0]['end'],4)

    def test_rejected_request_stub_and_relative_clause(self):
        self.assertEqual(boundary('那我们该怎么办呢',[('那','r'),('我们','r'),('该','r'),('怎么办','v'),('呢','y')]),[])
        self.assertEqual(boundary('光学工程以后的发展如何',[('光学','n'),('工程','n'),('以后','f'),('的','uj'),('发展','vn'),('如何','r')]),[])
        self.assertEqual(boundary('如果是的话那就是我了',[('如果','c'),('是','v'),('的话','u'),('那','r'),('就','d'),('是','v'),('我','r'),('了','ul')]),[])
        self.assertEqual(boundary('去专卖店换应该很贵吧',[('去','v'),('专卖店','n'),('换','n'),('应该','v'),('很','d'),('贵','a'),('吧','y')]),[])

    def test_span_control_preserves_word_boundaries(self):
        spans=[(0,2,'这个'),(2,4,'软件'),(4,6,'需要'),(6,8,'更新')]
        for i in range(100):
            start,end=legacy_span('这个软件需要更新',spans,str(i))
            self.assertIn(start,(0,2,4,6));self.assertIn(end,(2,4,6,8));self.assertLess(start,end)


if __name__=='__main__':unittest.main()
