"""Training/inference contract tests; no corpus or model download required."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from conditional import encode_example, input_tokens, prompt, INPUT, OUTPUT, EOS


class Conditioning(unittest.TestCase):
    def test_input_never_collapses_distinct_keystrokes(self):
        self.assertNotEqual(input_tokens("bmyi"), input_tokens("pmyi"))
        self.assertEqual(len(input_tokens("m;tm")), 2)
        for code in ("bmA", "b", "AA"):
            with self.assertRaises(ValueError):
                input_tokens(code)

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
