"""Committing correct text only fixes output; conditioning stays identical."""
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'neural'))
from composition import composition_prompt, prompt, expand, decode, validate_committed
from conditional import composition_span, encode_example


class CompositionState(unittest.TestCase):
    def test_commit_preserves_entire_history_keys_and_output_prefix(self):
        history='前文'*32; text='甲乙丙丁'*8; code='aabbccdd'*8
        for end in (1,2,17,30):
            fixed=dict(text=text[:end],input=code[:2*end])
            self.assertEqual(composition_prompt(history,code[2*end:],128,fixed),
                             prompt(history,code,128)+text[:end])
        # Unknown history has no invented pinyin and no fixed output.
        self.assertEqual(composition_prompt('银行行长', 'aa'),prompt('银行行长','aa'))

    def test_supervised_suffix_of_a_commit_is_identical(self):
        row=dict(context='前文'*32,input='aabbccdd',gold=['甲乙丙丁'])
        whole=composition_prompt(row['context'],row['input'])
        vocab=['<pad>','<unk>','<bos>']+sorted(set(whole+'甲乙丙丁\ue2e2'))
        index={c:i for i,c in enumerate(vocab)}
        suffix=composition_span(row,2,4)
        x,y=encode_example(row,index,128);sx,sy=encode_example(suffix,index,128)
        self.assertEqual(x,sx)
        self.assertEqual([v for v in sy if v],[index[c] for c in '丙丁\ue2e2'])
        self.assertEqual(y[-3:],sy[-3:])

    def test_multiple_commits_keep_original_readings_and_no_future_text(self):
        row=dict(context='允许',input='bmyix;ui',gold=['便宜行事'])
        first=composition_span(row,1,4);second=composition_span(first,1,3)
        self.assertEqual(second['context'],'允许')
        self.assertEqual(second['committed'],dict(text='便宜',input='bmyi'))
        self.assertEqual(second['input'],'x;ui')
        early=composition_span(row,0,2)
        self.assertNotIn('committed',early)
        self.assertEqual(early['input'],'bmyi')
        self.assertNotIn('行事',composition_prompt(early['context'],early['input']))

    def test_window_rebase_moves_the_whole_confirmed_segment(self):
        fixed=dict(text='甲乙丙丁',input='aabbccdd')
        self.assertEqual(expand('历史','eeff',fixed,16),('历史','aabbccddeeff','甲乙丙丁'))
        self.assertEqual(expand('历史','eeffg',fixed,16),('历史甲乙丙丁','eeffg',''))
        with self.assertRaises(ValueError):composition_prompt('','aa'*7,16)

    def test_decoder_pins_prefix_and_shifts_pending_constraints(self):
        engine=Mock();engine.decode.return_value=[('便宜行事',2.),('便宜形式',1.)]
        result=decode(engine,'允许','x;ui',dict(text='便宜',input='bmyi'),
                      pins=[(0,'行')],aux=[(1,'a')],learned=[(0,'行事')],weight=3.,
                      literals=[(1,'AB')],tones=[(1,4)])
        self.assertEqual(result,[('行事',4.),('形式',2.)])
        self.assertEqual(2 * (result[0][1] - result[1][1]), 4 * (2. - 1.))
        engine.decode.assert_called_once_with('允许','bmyix;ui',
            pins=[(0,'便'),(1,'宜'),(2,'行')],aux=[(3,'a')],learned=[(2,'行事')],weight=3.,
            literals=[(3,'AB')],tones=[(3,4)])

    def test_malformed_commits_do_not_become_fixed_output(self):
        for value in ([],dict(text='便宜',input='bm'),dict(text='\ue2e1',input='aa')):
            with self.assertRaises(ValueError):validate_committed(value)


if __name__=='__main__':unittest.main()
