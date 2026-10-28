"""The one Mandarin pipeline's stage ordering, alignment and safe assets."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from mandarin_frontend import MandarinFrontend, imported_modules, load_corrections, validate_literal_config


class IdentityTone:
    def pre_merge_for_modify(self, segments, readings):
        self.group_input = list(readings)
        return segments

    def modified_tone(self, word, pos, readings):
        self.tone_input = list(readings)
        return readings


class Frontend(unittest.TestCase):
    def test_phrase_correction_precedes_tone_rules(self):
        tone = IdentityTone()
        frontend = MandarinFrontend(converter=lambda texts: [["le4", "se4"]],
            segmenter=lambda text: [(text, "n")], tone_modifier=tone, corrections={"垃圾": ["la1", "ji1"]})
        row = frontend.readings(["垃圾"])[0]
        self.assertEqual(tone.group_input, ["la1", "ji1"])
        self.assertEqual(tone.tone_input, ["la1", "ji1"])
        self.assertEqual(row["reading_evidence"], {"frontend_before_sandhi": ["la1", "ji1"], "final": ["la1", "ji1"]})
        self.assertEqual(row["reading_rules"], [["phrase_correction:垃圾"], ["phrase_correction:垃圾"]])

    def test_full_word_correction_overrides_character_correction(self):
        frontend = MandarinFrontend(converter=lambda texts: [["ze2", "she2"], ["ze2", "xie3"]],
            segmenter=lambda text: [(text, "v")], tone_modifier=IdentityTone(),
            corrections={"咋": ["za3"], "咋舌": ["ze2", "she2"]})
        rows = frontend.readings(["咋舌", "咋写"])
        self.assertEqual(rows[0]["reading_evidence"]["final"], ["ze2", "she2"])
        self.assertEqual(rows[1]["reading_evidence"]["final"], ["za3", "xie3"])
        self.assertEqual(rows[1]["reading_rules"][0], ["character_correction:咋"])

    def test_unknown_and_non_han_keep_original_positions(self):
        sentence = "A你䷀㐂好。"
        calls = []

        def converter(sentences):
            calls.extend(sentences)
            return [["A", "ni3", "䷀", "㐂", "hao3", "。"]]

        frontend = MandarinFrontend(converter=converter, segmenter=lambda text: [(text, "n")],
                                    tone_modifier=IdentityTone(), corrections={})
        row = frontend.readings([sentence])[0]
        self.assertEqual(calls, [sentence])
        self.assertEqual(row["reading_evidence"]["final"], [None, "ni3", None, None, "hao3", None])
        self.assertEqual(row["tone_sources"], ["non_han", "mandarin_frontend", "non_han", "unresolved", "mandarin_frontend", "non_han"])

    def test_bad_converter_and_segmentation_fail(self):
        for response in ([], [["ni3", "hao3"]], [None]):
            frontend = MandarinFrontend(converter=lambda texts: response, segmenter=lambda text: [(text, "n")],
                                        tone_modifier=IdentityTone(), corrections={})
            with self.subTest(response=response), self.assertRaises(ValueError):
                frontend.readings(["你"])
        frontend = MandarinFrontend(converter=lambda texts: [["ni3"]], segmenter=lambda text: [("好", "n")],
                                    tone_modifier=IdentityTone(), corrections={})
        with self.assertRaisesRegex(ValueError, "segmentation changed"):
            frontend.readings(["你"])

    def test_upstream_phrase_dictionary_is_parsed_without_eval_or_pickle(self):
        corrections = load_corrections()
        self.assertGreater(len(corrections), 40000)
        self.assertEqual(corrections["和面"], ("huo2", "mian4"))
        self.assertEqual(corrections["咋舌"], ("ze2", "she2"))
        self.assertEqual(corrections["好处"], ("hao3", "chu4"))

    def test_reviewed_phrase_survives_tone_driven_merging_in_context(self):
        frontend = MandarinFrontend(converter=lambda sentences: [
            ["hao3", "chu3"], ["hao3", "chu3", "you3", "hen3", "duo1"]])
        for row in frontend.readings(["好处", "好处有很多"]):
            self.assertEqual(row["reading_evidence"]["frontend_before_sandhi"][:2], ["hao3", "chu4"])
            self.assertEqual(row["reading_evidence"]["final"][:2], ["hao3", "chu4"])
            self.assertIn("phrase_correction:好处", row["reading_rules"][1])

    def test_literal_config_validation_rejects_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.py"
            path.write_text("use_mask = True\nparam = {'x': 2}\n")
            self.assertEqual(validate_literal_config(path), {"use_mask": True, "param": {"x": 2}})
            for content in ("import os\n", "value = print('executed')\n", "__builtins__ = {}\n"):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    validate_literal_config(path)

    def test_imported_tone_rules_receive_one_provided_reading_stream(self):
        _, module = imported_modules()
        originals = [["ni3", "hao3"], ["yi1", "ge4"], ["bu4", "shi4"]]
        frontend = MandarinFrontend(converter=lambda sentences: originals, tone_modifier=module.ToneSandhi(), corrections={})
        rows = frontend.readings(["你好", "一个", "不是"])
        self.assertEqual([row["reading_evidence"]["final"] for row in rows],
                         [["ni2", "hao3"], ["yi2", "ge5"], ["bu2", "shi4"]])
        self.assertEqual(originals, [["ni3", "hao3"], ["yi1", "ge4"], ["bu4", "shi4"]])
        self.assertIn("contextual_tone_sandhi", rows[0]["reading_rules"][0])
        self.assertNotIn("text.g2pw.g2pw", sys.modules)
        self.assertNotIn("_wime_gpt_sovits_text.g2pw.g2pw", sys.modules)


if __name__ == "__main__":
    unittest.main()
