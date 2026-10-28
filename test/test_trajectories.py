"""Input trajectories must not leak future labels into earlier predictions."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from trajectories import states, validate


class Trajectories(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
