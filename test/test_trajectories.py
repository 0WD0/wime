"""Input trajectories must not leak future labels into earlier predictions."""
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from trajectories import states, validate

spec = importlib.util.spec_from_file_location("prepare_trajectories", Path(__file__).resolve().parents[1] / "neural/prepare-trajectories.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


class Trajectories(unittest.TestCase):
    def test_source_sentences_preserve_context_and_do_not_join_punctuation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text('{}')
            with sqlite3.connect(root / "documents.sqlite3") as db:
                db.execute("CREATE TABLE documents (id TEXT, group_id TEXT, source TEXT, text TEXT, split TEXT)")
                db.executemany("INSERT INTO documents VALUES (?,?,?,?,?)", [
                    ('one','a','chat','之前。我们明天一起出发。','dev'),
                    ('two','b','chat','我们明天，一起出发。','dev'),
                    ('three','c','chat','我们明天一起出发 。','dev'),
                    ('four','d','chat','我们明天一起出发。','train')])
            out = root / "candidates.json"
            preparation.candidates(root,out)
            rows = json.loads(out.read_text())
            self.assertEqual(len(rows),1)
            self.assertEqual((rows[0]['text'],rows[0]['context'],rows[0]['offset']),('我们明天一起出发','之前。',3))
            report = json.loads(out.with_suffix('.manifest.json').read_text())
            self.assertEqual(report['counts']['excluded_terminal_spacing'],1)

    def test_paired_inputs_reveal_only_the_explicitly_committed_prefix(self):
        row = dict(context="刚才聊到", input="abcd" * 4, gold=["我们明天一起出发"],
                   boundaries=[dict(end=4, reason="完整的主语和时间短语")])
        before = copy.deepcopy(row)
        full, pairs = states(row)
        self.assertEqual(full, dict(context=row["context"],input=row["input"]))
        self.assertEqual(pairs[0]["prefix"],dict(context=row["context"],input=row["input"][:8]))
        suffix = pairs[0]["continuation"]
        self.assertEqual(suffix["context"], "刚才聊到我们明天")
        self.assertEqual(suffix["input"], row["input"][8:])
        self.assertEqual(suffix["known"],dict(text="我们明天",input=row["input"][:8]))
        self.assertEqual(row,before)
        for invalid in ([dict(end=8,reason="空后缀")], [dict(end=4,reason="")],
                        [dict(end=4,reason="重复")]*2):
            with self.assertRaises(ValueError):
                validate(dict(row,boundaries=invalid))

    def test_whole_sentence_does_not_require_an_artificial_commit(self):
        row = dict(context="", input="abcd" * 4, gold=["请给我一个商业画布"], boundaries=[])
        # Input must still align even when no meaningful internal commit exists.
        row["input"] = "ab" * len(row["gold"][0])
        full, pairs = states(row)
        self.assertEqual(full["input"], row["input"])
        self.assertEqual(pairs, [])


if __name__ == "__main__":
    unittest.main()
