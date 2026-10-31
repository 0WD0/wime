"""Training/inference contract tests; no corpus or model download required."""
import sys
from pathlib import Path
import unittest
import random
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from conditional import composition_span, encode_example, input_tokens, prompt, INPUT, OUTPUT, EOS
from conditional import CharLM, Config, grow_depth
from conditional import augment_composition
from composition import composition_prompt


class Conditioning(unittest.TestCase):
    def test_ordinary_training_matches_committed_inference_without_scoring_prefix(self):
        row = dict(context="允许", input="bmyix;ui", gold=["便宜行事"])
        suffix = composition_span(row, 2, 4)
        expected = composition_prompt(suffix["context"], suffix["input"], 128,
                                   dict(text="便宜", input="bmyi"))
        index = {c:i+3 for i,c in enumerate(sorted(set(expected + "行事" + EOS)))}
        x, y = encode_example(suffix, index, 128)
        self.assertEqual(x[:1+len(expected)], [2] + [index[c] for c in expected])
        self.assertEqual([value for value in y if value], [index[c] for c in "行事"+EOS])
        self.assertNotIn("行事", expected)

    def test_single_spans_keep_polyphonic_reading_and_short_rows_are_valid(self):
        row = dict(context="允许", input="bmyix;ui", gold=["便宜行事"],
                   pinyin="bian yi xing shi")
        for seed in range(20):
            sample = augment_composition(row, random.Random(seed), single_share=1)
            start = len(sample.get("committed", {}).get("text", ""))
            self.assertEqual(sample["input"], row["input"][2 * start:2 * start + 2])
            self.assertEqual(sample["gold"], [row["gold"][0][start]])
        single = dict(context="前文", input="qd", gold=["枪"])
        self.assertEqual(augment_composition(single, random.Random(1), span_share=1), single)

    def test_depth_growth_preserves_logits_and_can_learn(self):
        torch.manual_seed(7)
        model = CharLM(Config(vocab=32, n_layer=2, n_head=2, n_embd=8,
                              context=8, dropout=0)).eval()
        inputs = torch.randint(32, (2, 8))
        expected = model(inputs)[0].detach()
        grown = grow_depth(model, 6)
        torch.testing.assert_close(grown(inputs)[0], expected, rtol=0, atol=0)
        self.assertIs(grown.head.weight, grown.tok.weight)
        self.assertGreater(sum(p.numel() for p in grown.parameters()),
                           sum(p.numel() for p in model.parameters()))
        grown(inputs, inputs)[1].backward()
        self.assertGreater(grown.blocks[1].out.weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            grow_depth(model, 1)

    def test_composition_span_keeps_reading_and_only_committed_context(self):
        row = dict(context="这个", input="bud;yij;heb;le", gold=["补丁已经合并了"],
                   pinyin="bu ding yi jing he bing le", document="training-document")
        span = composition_span(row, 2, 6)
        self.assertEqual(span["context"], "这个")
        self.assertEqual(span["committed"], dict(text="补丁", input="bud;"))
        self.assertEqual(span["input"], "yij;heb;")
        self.assertEqual(span["gold"], ["已经合并"])
        self.assertEqual(span["pinyin"], "yi jing he bing")
        self.assertEqual(span["document"], row["document"])
        self.assertEqual(row["gold"], ["补丁已经合并了"])
        with self.assertRaises(ValueError):
            composition_span(dict(row, input="bu"), 2, 6)

    def test_span_keeps_contextual_tones_and_evidence_aligned(self):
        row = dict(context="觉得", input="hfhkba", gold=["很好吧"], offset=17,
                   pinyin=["hen", "hao", "ba"], tones=[2, 3, 5],
                   tone_sources=["sandhi", "frontend", "particle"], reading_rules=[["3-3"], [], ["neutral"]],
                   reading_evidence={"frontend_before_sandhi": ["hen3", "hao3", "ba1"],
                                     "final": ["hen2", "hao3", "ba5"]})
        span = composition_span(row, 1, 3)
        self.assertEqual(span["context"], "觉得")
        self.assertEqual(span["offset"], 18)
        for field in ("pinyin", "tones", "tone_sources", "reading_rules"):
            self.assertEqual(span[field], row[field][1:3])
        self.assertEqual(span["reading_evidence"]["final"], ["hao3", "ba5"])
        self.assertEqual(row["tones"], [2, 3, 5])

    def test_input_never_collapses_distinct_keystrokes(self):
        self.assertNotEqual(input_tokens("bmyi"), input_tokens("pmyi"))
        self.assertEqual(len(input_tokens("m;tm")), 2)
        for code in ("bmA", "b", "AA"):
            with self.assertRaises(ValueError):
                input_tokens(code)

    def test_sparse_reading_notes_follow_partial_commits(self):
        row = dict(context="觉得", input="hfhkba", gold=["很好吧"], offset=17,
                   pinyin=["hen", "hao", "ba"], tones=[2, 3, 5],
                   reading_notes=[dict(index=0, before_tone=3, rules=["contextual_tone_sandhi"]),
                                  dict(index=2, rules=["character_correction:吧"])])
        span = composition_span(row, 1, 3)
        self.assertEqual(span["reading_notes"], [dict(index=1, rules=["character_correction:吧"])])
        self.assertEqual(span["pinyin"], ["hao", "ba"])
        self.assertEqual(span["tones"], [3, 5])
        self.assertEqual(span["offset"], 18)
        self.assertNotIn("reading_notes", composition_span(row, 1, 2))
        self.assertEqual(row["reading_notes"][1]["index"], 2)

    def test_loss_only_predicts_output_and_terminal(self):
        row = dict(context="上文", input="aabb", gold=["甲乙"])
        chars = set(prompt(row["context"], row["input"]) + "甲乙" + EOS)
        index = {ch: i + 3 for i, ch in enumerate(sorted(chars))}
        x, y = encode_example(row, index, 128)
        self.assertEqual(len(x), len(y))
        self.assertEqual([t for t in y if t], [index[c] for c in "甲乙" + EOS])
        self.assertEqual(x[len(y) - 3], index[OUTPUT])

    def test_prompt_fits_without_truncating_conditioning(self):
        value = prompt("上" * 200 + INPUT + OUTPUT, "aa" * 32)
        self.assertEqual(value.count(INPUT), 1)
        self.assertEqual(value.count(OUTPUT), 1)
        self.assertLessEqual(1 + len(value) + 32 + 1, 128)
        with self.assertRaises(ValueError):
            prompt("", "aa" * 63)


if __name__ == "__main__":
    unittest.main()
