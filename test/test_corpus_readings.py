"""Final frontend readings and wime lattice reachability; no model required."""
import sqlite3
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'neural'))
from corpus_readings import HAN, Lexicon, Reader, key, numbered_reading


def lexicon(entries):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'db.sqlite3'
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE entry (code TEXT, word TEXT, aux TEXT, weight INTEGER)')
            db.executemany("INSERT INTO entry VALUES (?, ?, '', ?)", entries)
        return Lexicon(path)


class FakeFrontend:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def readings(self, sentences):
        self.calls.append(list(sentences))
        return [dict(tone_sources=['mandarin_frontend'] * len(values), reading_rules=[[] for _ in values],
                     reading_evidence={'frontend_before_sandhi': list(values), 'final': list(values)})
                for values in self.rows]


class Readings(unittest.TestCase):
    def test_single_final_reading_drives_keys_syllables_and_tones(self):
        lex = lexicon([(key('chang'), '长', 99), (key('zhang'), '长', 9), (key('da'), '大', 9),
                       (key('chang') + key('da'), '长大', 99999)])
        frontend = FakeFrontend([['zhang3', 'da4']])
        reader = Reader(lex, frontend)
        row = reader.readings(['长大'])[0]
        self.assertEqual(row['keys'], [key('zhang'), key('da')])
        self.assertEqual(row['pinyin'], ['zhang', 'da'])
        self.assertEqual(row['tones'], [3, 4])
        self.assertEqual(frontend.calls, [['长大']])
        self.assertEqual(reader.keys(['长大']), [row['keys']])

    def test_unreachable_reading_is_not_replaced_by_dictionary_vote(self):
        lex = lexicon([(key('chang'), '长', 9)])
        row = Reader(lex, FakeFrontend([['zhang3']])).readings(['长'])[0]
        self.assertEqual(row['keys'], [None])
        self.assertEqual(row['pinyin'], [None])
        self.assertEqual(row['tones'], [None])
        self.assertEqual(row['tone_sources'], ['unreachable'])
        self.assertEqual(row['reading_evidence']['final'], ['zhang3'])

    def test_merged_key_retains_the_final_syllable(self):
        self.assertEqual(key('en'), key('n'))
        lex = lexicon([(key('en'), '嗯', 9)])
        row = Reader(lex, FakeFrontend([['n2']])).readings(['嗯'])[0]
        self.assertEqual(row['keys'], [key('en')])
        self.assertEqual(row['pinyin'], ['n'])
        self.assertEqual(row['tones'], [2])

    def test_neutral_unknown_and_non_han_are_distinct(self):
        lex = lexicon([(key('de'), '的', 9), (key('xi'), '㐂', 9)])
        row = Reader(lex, FakeFrontend([['de5', None, None, None]])).readings(['的㐂䷀。'])[0]
        self.assertEqual(row['tones'], [5, None, None, None])
        self.assertEqual(row['tone_sources'], ['mandarin_frontend', 'unresolved', 'non_han', 'non_han'])
        self.assertIsNone(HAN.search('䷀䷁'))

    def test_stage_evidence_and_rules_are_preserved(self):
        lex = lexicon([(key('ni'), '你', 9), (key('hao'), '好', 9)])
        frontend = FakeFrontend([])
        frontend.readings = lambda sentences: [dict(tone_sources=['contextual_tone_sandhi', 'mandarin_frontend'],
            reading_rules=[['contextual_tone_sandhi'], []],
            reading_evidence={'frontend_before_sandhi': ['ni3', 'hao3'], 'final': ['ni2', 'hao3']})]
        row = Reader(lex, frontend).readings(['你好'])[0]
        self.assertEqual(row['tones'], [2, 3])
        self.assertEqual(row['reading_rules'][0], ['contextual_tone_sandhi'])
        self.assertEqual(row['reading_evidence']['frontend_before_sandhi'][0], 'ni3')

    def test_invalid_alignment_and_evidence_fail_loudly(self):
        lex = lexicon([(key('ni'), '你', 9)])
        for values in ([], [[None, None]], [['ni']], [['Ni3']], [['ni0']]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Reader(lex, FakeFrontend(values)).readings(['你'])
        with self.assertRaises(ValueError):
            Reader(lex, FakeFrontend([['ni3']])).readings(['A'])

    def test_numbered_reading_parser_does_not_invent_neutral_tones(self):
        self.assertEqual(numbered_reading('lü4'), 'lv4')
        self.assertEqual(numbered_reading('nu:e4'), 'nve4')
        for value in (None, '', 'de', 'de0', 'de6', '㐂', 'ma55'):
            self.assertIsNone(numbered_reading(value))


if __name__ == '__main__':
    unittest.main()
