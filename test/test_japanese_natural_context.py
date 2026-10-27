"""Natural context extraction must preserve source readings and split isolation."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "natural_context", Path(__file__).resolve().parents[1]/"neural/prepare-japanese-natural-context.py")
natural = importlib.util.module_from_spec(spec)
spec.loader.exec_module(natural)


class NaturalContext(unittest.TestCase):
    def test_annotation_units_preserve_supplied_reading_and_offsets(self):
        surface, reading, units = natural.units("[昨日|さくじつ]は[図書館|としょかん]へ。")
        self.assertEqual(surface, "昨日は図書館へ。")
        self.assertEqual(reading, "さくじつはとしょかんへ。")
        self.assertEqual(units[0]["surface_end"], 2)
        self.assertEqual(units[0]["kana_end"], 4)
        with self.assertRaisesRegex(ValueError, "unread kanji"):
            natural.units("今日は[晴|は]れ。")

    def test_span_intersects_word_and_annotation_boundaries(self):
        text, reading, units = natural.units("[明日|あした]は[図書館|としょかん]で[本|ほん]を[読|よ]みます。")
        words = [("明日","名詞"),("は","助詞"),("図書館","名詞"),("で","助詞"),
                 ("本","名詞"),("を","助詞"),("読み","動詞"),("ます","助動詞"),("。","補助記号")]
        row = natural.choose_span("1", text, reading, units, words)
        self.assertIsNotNone(row)
        start,end = row["source_span"]
        ks,ke = row["source_kana_span"]
        self.assertEqual(row["context"]+row["answers"][0]+row["right_context"], text)
        self.assertEqual(row["kana"], reading[ks:ke])
        self.assertTrue(start in {0,3,7,9})
        self.assertFalse(end in {1,4,5,10})

    def test_queue_excludes_corpus_and_cross_source_overlap(self):
        pool = dict(cases=[dict(id="1",source_sentence="abcdefghij"),
                           dict(id="2",source_sentence="abcdefghzz"),
                           dict(id="3",source_sentence="mnopqrstuv"),
                           dict(id="4",source_sentence="0123456789")])
        report = dict(cases={"1":{},"2":{},"3":{"corpus":{"shared8_rows":1}},"4":{}})
        selected = natural.select_queue(pool,report,2)
        self.assertEqual([r["id"] for r in selected["cases"]],["1","4"])
        self.assertEqual(selected["excluded"],dict(queue_shared8=1,corpus_shared8=1))
        with self.assertRaisesRegex(ValueError,"need 3"):
            natural.select_queue(pool,report,3)

    def test_variants_keep_source_group_and_actual_short_suffix(self):
        source = dict(id="natural-source:1",source_id="1",context="昨日は",kana="としょかんで",answers=["図書館で"],right_context="本を読んだ。")
        review = dict(id=source["id"],original_kana=source["kana"],original_answers=source["answers"],status="approved")
        cases = natural.variants(source,"sealed",review)
        self.assertEqual([c["right_context"] for c in cases],["本を読んだ。","本を",""])
        self.assertEqual([c["case_type"] for c in cases],["natural_full","natural_short","natural_no_right"])
        self.assertEqual(len({c["topic_group"] for c in cases}),1)
        self.assertTrue(all(c["split"]=="sealed" for c in cases))
        review["original_kana"]="としょかん"
        with self.assertRaisesRegex(ValueError,"no longer matches"):
            natural.variants(source,"sealed",review)


if __name__ == "__main__":
    unittest.main()
