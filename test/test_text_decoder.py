"""Text-constrained decoding with a tiny real byte grammar and toy LM rows."""
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from text_decoder import TextDecoder  # noqa: E402


VOCAB = ["<s>", "\n", "车", "站", "展", "车站", "车展", "韩", "钢", "刚",
         "<0xE9>", "<0x82>", "<0xAF>", "韩钢", "好", "坏"]
READINGS = {"邯": {"hj": ["ge"]}, "韩": {"hj": ["uw"]}, "好": {"hk": ["nz"]},
            "坏": {"hy": ["tb"]}, "钢": {"gh": ["jg"]}, "刚": {"gh": ["kd"]},
            "站": {"vj": ["lv"]}, "展": {"vj": ["ui"]}}


class Tokenizer:
    """Serializable byte-fallback tokenizer with deterministic greedy merges."""

    def __init__(self, bos=True):
        self.bos = bos
        self.all_special_ids = [0]
        definition = dict(model=dict(type="BPE", vocab={text: i for i, text in enumerate(VOCAB)}),
                          decoder=dict(type="Sequence", decoders=[dict(type="ByteFallback"), dict(type="Fuse")]))
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps(definition))
        self.raw = [None if i == 0 else bytes([int(text[3:5], 16)]) if text.startswith("<0x")
                    else text.encode() for i, text in enumerate(VOCAB)]
        self.greedy = sorted(((raw, i) for i, raw in enumerate(self.raw) if raw),
                             key=lambda item: (-len(item[0]), item[1]))

    def __len__(self):
        return len(VOCAB)

    def __call__(self, text, add_special_tokens=True):
        remaining = text.encode()
        ids = [0] if self.bos and add_special_tokens else []
        while remaining:
            token = next(((raw, i) for raw, i in self.greedy if remaining.startswith(raw)), None)
            if token is None:
                raise ValueError("text is outside the toy tokenizer")
            raw, index = token
            ids.append(index)
            remaining = remaining[len(raw):]
        return dict(input_ids=ids)


class Toy(TextDecoder):
    def __init__(self, *, bos=True, small=True, lm_scores=None, conditional_scores=None, **kwargs):
        self.tokenizer = Tokenizer(bos=bos)
        evaluator = SimpleNamespace(device="cpu") if small else None
        rescorer = SimpleNamespace(tokenizer=self.tokenizer, weight=1.5, pad=0)
        self.lm_scores = dict(lm_scores or {})
        self.conditional_scores = dict(conditional_scores or {})
        self.next_calls, self.tail_calls, self.conditional_texts = [], [], []
        super().__init__(evaluator, rescorer, READINGS, **kwargs)

    def prompts(self, context, code, anchor, known):
        return [code], self.lm_ids(context)

    def window(self):
        return 128

    def _row(self, context, sequence):
        # Both model APIs use the same normalized rows, so search paths and
        # canonical paths are probabilities from one consistent toy model.
        return torch.tensor([self.lm_scores.get(token, -1.0) for token in VOCAB]).log_softmax(-1)

    def lm_next(self, context, sequences):
        self.next_calls.extend((tuple(context), tuple(seq)) for seq in sequences)
        return torch.stack([self._row(context, seq) for seq in sequences])

    def lm_tails(self, context, prefixes, tails):
        self.tail_calls.append((tuple(context), tuple(tails)))
        result = []
        for prefix, tail in zip(prefixes, tails):
            sequence, score = tuple(prefix), 0.0
            for token in tail:
                score += float(self._row(context, sequence)[token])
                sequence += (token,)
            result.append(score)
        return result

    def model_score(self, prompt, texts):
        self.conditional_texts.extend(texts)
        return [sum(self.conditional_scores.get(char, 0.0) for char in text) for text in texts]


class TextDecoding(unittest.TestCase):
    def test_split_byte_character_survives_absent_or_very_negative_small_model(self):
        for small in (False, True):
            decoder = Toy(small=small, conditional_scores={"邯": -1_000_000},
                          guide_weight=100, beam=16)
            candidates = dict(decoder.decode("", "hj"))
            self.assertEqual(set(candidates), {"邯", "韩"})
            self.assertEqual(decoder.decode("", "hj", pins=[(0, "邯")])[0][0], "邯")
            self.assertEqual(decoder.last_audit["conditional_active"], small)
            # The tokenizer has no whole 邯 token: inference traverses three
            # separate bytes, which would be replacement characters alone.
            self.assertNotIn("邯", VOCAB)
            self.assertTrue(any(VOCAB[seq[-1]] == "<0x82>" for _, seq in decoder.next_calls if seq))

    def test_healing_admits_cross_boundary_token_and_strips_exact_literal(self):
        decoder = Toy(small=False, lm_scores={"车站": -0.1, "车展": -2,
                                             "站": -6, "展": -0.2, "车": -0.2})
        candidates = decoder.decode("车", "vj")
        self.assertEqual(candidates[0][0], "站")
        self.assertEqual({text for text, _ in candidates}, {"站", "展"})
        self.assertEqual(decoder.last_audit["actual_rewind"], 1)
        self.assertEqual(decoder.last_audit["fixed_prefix_bytes"], len("车".encode()))
        self.assertEqual(decoder.last_audit["final_scoring"], "joint_canonical")
        self.assertEqual(decoder.last_components[0]["text"] in {"站", "展"}, True)
        # The joint canonical path starts from BOS, then takes 车站 directly.
        self.assertTrue(any((VOCAB.index("车站"),) in tails for _, tails in decoder.tail_calls))

    def test_no_bos_single_token_context_retains_original_causal_prefix(self):
        decoder = Toy(bos=False, small=False, lm_scores={"车站": -0.1, "车展": -2,
                                                       "站": -6, "展": -0.2})
        candidates = decoder.decode("车", "vj")
        self.assertEqual(candidates[0][0], "展")
        self.assertEqual(decoder.last_audit["actual_rewind"], 0)
        self.assertEqual(decoder.last_audit["fixed_prefix_bytes"], 0)
        self.assertEqual(decoder.last_audit["final_scoring"], "no_common_causal_prefix")
        self.assertTrue(all(context == (VOCAB.index("车"),) for context, _ in decoder.next_calls))

    def test_pins_aux_full_input_and_lone_half_syllable(self):
        decoder = Toy()
        self.assertEqual({text for text, _ in decoder.decode("", "h")}, {"邯", "韩", "好", "坏"})
        self.assertFalse(decoder.last_audit["conditional_active"])
        self.assertEqual(decoder.decode("", "hjgh", pins=[(0, "邯")], aux=[(1, "jg")])[0][0], "邯钢")
        self.assertEqual(decoder.decode("", "hjg", pins=[(0, "邯")], aux=[(1, "jg")])[0][0], "邯钢")
        self.assertEqual(decoder.decode("", "h", aux=[(0, "u")])[0][0], "韩")
        self.assertEqual(decoder.decode("", "hk", pins=[(0, "韩")]), [])
        self.assertEqual(decoder.decode("", "hj", aux=[(0, "x")]), [])
        self.assertEqual(decoder.decode("", ""), [])

    def test_typing_and_constraint_changes_reuse_only_valid_model_prefixes(self):
        decoder = Toy(beam=16)
        decoder.decode("", "hj")
        decoder.decode("", "hjgh")
        before = len(decoder.next_calls)
        decoder.decode("", "hjgh", pins=[(0, "邯")])
        self.assertEqual(len(decoder.next_calls), before)
        decoder.decode("", "hjgh", aux=[(1, "jg")])
        self.assertEqual(len(decoder.next_calls), before)
        decoder.decode("", "hj")
        self.assertEqual(len(decoder.next_calls), before)
        self.assertTrue(all(count == 1 for count in Counter(decoder.next_calls).values()))
        decoder.decode("车", "hj")
        self.assertGreater(len(decoder.next_calls), before)

    def test_canonical_final_score_can_replace_a_better_noncanonical_search_path(self):
        decoder = Toy(small=False, guide_weight=0, lm_scores={"韩钢": -20, "韩": -0.1, "钢": -0.1})
        candidates = decoder.decode("", "hjgh", pins=[(1, "钢")])
        # Search has a strong 韩 + 钢 path, but canonical 韩钢 is a single
        # low-probability token. 邯 + 钢 wins canonical ranking.
        self.assertEqual(candidates[0][0], "邯钢")
        self.assertEqual(decoder.last_audit["final_scoring"], "joint_canonical")
        component = {row["text"]: row for row in decoder.last_components}
        self.assertAlmostEqual(component["韩钢"]["lm"],
                               float(decoder._row([], ())[VOCAB.index("韩钢")]) / 2)
        self.assertGreater(component["邯钢"]["lm"], component["韩钢"]["lm"])

    def test_guide_score_is_not_added_to_final_lm_ranking(self):
        settings = dict(conditional_scores={"邯": -1000}, lm_scores={"韩": -10}, beam=16,
                        guide_weight=20)
        decoder = Toy(**settings)
        self.assertEqual(decoder.decode("", "hj")[0][0], "邯")
        self.assertIn("邯", decoder.conditional_texts)
        control = Toy(final_mode="fused", **settings)
        self.assertEqual(control.decode("", "hj")[0][0], "韩")

    def test_terminal_beam_counts_distinct_texts_not_token_segmentations(self):
        decoder = Toy(small=False, beam=2, guide_weight=0,
                      lm_scores={"韩钢": -0.1, "韩": -0.1, "钢": -0.1})
        # 韩钢 and 韩 + 钢 both beat the three-byte 邯 path. They must use one
        # terminal surface slot, leaving the other slot for 邯钢.
        self.assertEqual({text for text, _ in decoder.decode("", "hjgh", pins=[(1, "钢")])},
                         {"韩钢", "邯钢"})

    def test_learned_bonus_is_counted_once_and_cannot_override_pin(self):
        decoder = Toy(small=False, lm_scores={"韩": -0.1})
        self.assertEqual(decoder.decode("", "hj")[0][0], "韩")
        self.assertEqual(decoder.decode("", "hj", learned=[(0, "邯")], weight=20)[0][0], "邯")
        self.assertEqual(decoder.decode("", "hj", pins=[(0, "韩")],
                                        learned=[(0, "邯")], weight=20)[0][0], "韩")
        decoder.decode("", "hjgh", learned=[(0, "韩钢"), (1, "钢")], weight=4)
        component = next(row for row in decoder.last_components if row["text"] == "韩钢")
        self.assertEqual(component["bonus"], 4)

    def test_stale_constraints_and_unsupported_context_fall_back(self):
        decoder = Toy()
        for context, code, kwargs in [("", "hj", dict(pins=[(3, "邯")])),
                                      ("", "h1", {}), ("unknown", "hj", {})]:
            self.assertEqual(decoder.decode(context, code, **kwargs), [])
            self.assertEqual(decoder.last_audit["fallback"], "incompatible_context_or_constraints")
        self.assertEqual(decoder.last_components, [])


if __name__ == "__main__":
    unittest.main()
