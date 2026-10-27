"""The ablation must change candidate generation/scoring, not input fidelity."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from test_japanese import FakeZenz, Toy

spec = importlib.util.spec_from_file_location(
    "ablation", Path(__file__).resolve().parents[1] / "neural/ablate-japanese.py")
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


class Generated(unittest.TestCase):
    def decoder(self):
        return ablation.GeneratedDecoder(Toy().table, None, None, zenz=FakeZenz())

    def test_generated_candidates_must_cover_all_typed_kana(self):
        with patch.object(ablation, "timed", side_effect=lambda f: (f(), 0.0)):
            self.assertEqual(self.decoder().decode("", "にほんご"), [("日本後", 0.0)])

    def test_shift_can_reject_every_candidate_without_leaking_unchecked_text(self):
        with patch.object(ablation, "timed", side_effect=lambda f: (f(), 0.0)):
            self.assertEqual(self.decoder().decode("", "にほんご", shift="1110", explicit=True), [])
        self.assertEqual(ablation.displayed("にほんご", []), ["にほんご", "ニホンゴ"])

    def test_pending_keys_do_not_invent_or_drop_a_provisional_syllable(self):
        decoder = self.decoder()
        self.assertEqual(decoder.decode("", "にほん", pending=["ご", "が"]), [])
        self.assertEqual(decoder.zenz.calls, 0)

    def test_qwen_reranking_uses_the_same_candidates_and_fixed_weight(self):
        details = dict(accepted=["日本後", "日本語"], z=[-2.0, -3.0], q=[-4.0, -1.0])
        self.assertEqual(ablation.GeneratedDecoder.ranked(details, False)[0][0], "日本後")
        ranked = ablation.GeneratedDecoder.ranked(details, True)
        self.assertEqual(ranked[0], ("日本語", -3.7))
        self.assertEqual({t for t, _ in ranked}, set(details["accepted"]))


if __name__ == "__main__":
    unittest.main()
