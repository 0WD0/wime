"""Training boundary rules should reject the fragments found during source review."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'neural'))
from sentence_sampling import boundary_candidates,legacy_span,source_units,retain_group,technical_line_fragment
from commit_pause import PAUSE_THRESHOLD,choose


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

    def test_document_soft_wrap_and_temporal_modifier_are_not_boundaries(self):
        text='说明 Kubernetes\n的日志如何工作以及怎样实现。'
        self.assertTrue(technical_line_fragment(text,text.index('的日志')))
        text='上一步执行完成。\n现在可以重新启动服务。'
        self.assertFalse(technical_line_fragment(text,text.index('现在')))
        self.assertEqual(boundary_candidates('并且不会执行之前提及的换行符转写',
            [('并且','c'),('不会','v'),('执行','v'),('之前','f'),('提及','v'),('的','uj'),('换行符','n'),('转写','v')]),[])

    def test_condition_clause_ends_before_its_main_clause(self):
        self.assertEqual(boundary_candidates('如果系统失败就重新启动',
            [('如果','c'),('系统','n'),('失败','v'),('就','d'),('重新','d'),('启动','v')]),[('condition-clause',6)])
        # 没通过 is a verb even when jieba tags 通过 as a preposition.
        self.assertIn(('condition-clause',10),boundary_candidates('如果第一次申请没通过还能再申请吗',
            [('如果','c'),('第一次','m'),('申请','v'),('没','v'),('通过','p'),('还','d'),('能','v'),('再','d'),('申请','v'),('吗','y')]))
        self.assertEqual(boundary_candidates('这个软件需要更新配置',
            [('这个','r'),('软件','n'),('需要','v'),('更新','v'),('配置','n')]),[('topic-predicate',4)])

    def test_rejected_request_stub_and_relative_clause(self):
        self.assertEqual(boundary_candidates('那我们该怎么办呢',[('那','r'),('我们','r'),('该','r'),('怎么办','v'),('呢','y')]),[])
        self.assertEqual(boundary_candidates('光学工程以后的发展如何',[('光学','n'),('工程','n'),('以后','f'),('的','uj'),('发展','vn'),('如何','r')]),[])
        self.assertEqual(boundary_candidates('如果是的话那就是我了',[('如果','c'),('是','v'),('的话','u'),('那','r'),('就','d'),('是','v'),('我','r'),('了','ul')]),[])
        self.assertEqual(boundary_candidates('去专卖店换应该很贵吧',[('去','v'),('专卖店','n'),('换','n'),('应该','v'),('很','d'),('贵','a'),('吧','y')]),[])

    def test_open_frames_objects_and_modifiers_are_not_cut(self):
        # An open preposition, a verb awaiting its object clause, a modifier of a later noun.
        self.assertEqual(boundary_candidates('以便向开发人员提供状态',
            [('以便','c'),('向','p'),('开发','vn'),('人员','n'),('提供','v'),('状态','n')]),[])
        self.assertEqual(boundary_candidates('男生认定一个女生不可能接受他',
            [('男生','n'),('认定','v'),('一个','m'),('女生','n'),('不','d'),('可能','v'),('接受','v'),('他','r')]),[])
        self.assertEqual(boundary_candidates('刚在战争遭到打击的军队撤退',
            [('刚','d'),('在','p'),('战争','n'),('遭到','v'),('打击','vn'),('的','uj'),('军队','n'),('撤退','v')]),[])
        # 通｜选课: a split beside a one-character content token needs the dictionary's agreement.
        segments=[('和','c'),('专业课','n'),('一起','d'),('上','v'),('的','uj'),('通','v'),('选课','n'),('是不是','v'),('很','d'),('虐','a')]
        text=''.join(w for w,_ in segments)
        self.assertNotIn(('topic-predicate',9),boundary_candidates(text,segments,{3,5,7,8,11,14,16}))

    def test_the_most_pause_like_candidate_above_threshold_wins(self):
        rows=[dict(context='',gold=['有梦想的人才会设闹钟'],boundary_candidates=[('topic-predicate',3),('topic-predicate',5)]),
              dict(context='',gold=['这个软件需要更新配置'],boundary_candidates=[('topic-predicate',4)])]
        scores={'有梦想':.01,'有梦想的人':.3,'这个软件':PAUSE_THRESHOLD/2}
        choose(rows,lambda contexts,prefixes:[scores[p] for p in prefixes])
        self.assertEqual([b['end'] for b in rows[0]['boundaries']],[5])
        self.assertEqual(rows[1]['boundaries'],[])

    def test_span_control_preserves_word_boundaries(self):
        spans=[(0,2,'这个'),(2,4,'软件'),(4,6,'需要'),(6,8,'更新')]
        for i in range(100):
            start,end=legacy_span('这个软件需要更新',spans,str(i))
            self.assertIn(start,(0,2,4,6));self.assertIn(end,(2,4,6,8));self.assertLess(start,end)


if __name__=='__main__':unittest.main()
