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
        self.assertIsNone(full["known"])
        self.assertEqual(part["known"],dict(text="真刀",input="vfdk"))
        explicit=dict(id="explicit",context="这是",committed_text="真刀",committed_input="vfdk",input="vfqd",gold=["真枪"])
        request=next(requests([explicit]))
        self.assertEqual(request["context"],"这是真刀")
        self.assertEqual(request["known"],part["known"])
        mismatched=dict(suffix,context="不是确认的前缀")
        self.assertIsNone(list(requests([whole,mismatched]))[-1]["known"])


if __name__ == '__main__':unittest.main()
