import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"neural"))
from regression import requests, typed_keys


class Regression(unittest.TestCase):
    def test_auxiliary_keys_keep_their_syllable_and_half_syllable(self):
        self.assertEqual(typed_keys("vfABdkQv"),("vfdkv",[(0,"ab"),(1,"q")]))
        with self.assertRaises(ValueError):typed_keys("Avf")

    def test_archived_commits_recover_only_visible_prefix_readings(self):
        whole=dict(id="one/whole",parent="one",variant="whole",context="",input="vfdkvfqd",gold=["真刀真枪"])
        suffix=dict(id="one/suffix",parent="one",variant="suffix",context="真刀",input="vfqd",gold=["真枪"])
        full,part=list(requests([whole,suffix]))
        self.assertIsNone(full["committed"])
        self.assertEqual(part["committed"],dict(text="真刀",input="vfdk"))
        explicit=dict(id="explicit",context="这是",committed_text="真刀",committed_input="vfdk",input="vfqd",gold=["真枪"])
        request=next(requests([explicit]))
        self.assertEqual(request["context"],"这是")
        self.assertEqual(request["committed"],part["committed"])
        mismatched=dict(suffix,context="不是确认的前缀")
        self.assertIsNone(list(requests([whole,mismatched]))[-1]["committed"])

    def test_archive_adapter_recovers_history_before_truncation(self):
        whole = dict(id="whole", parent="pair", variant="whole", context="史"*64,
                     input="aa"*22, gold=["甲"*22])
        suffix = dict(id="suffix", parent="pair", context="史"*44+"甲"*20,
                      input="aaaa", gold=["甲甲"])
        state = list(requests([whole, suffix]))[-1]
        self.assertEqual(state["context"], whole["context"])
        self.assertEqual(state["committed"], dict(text="甲"*20, input="aa"*20))


if __name__ == '__main__':unittest.main()
