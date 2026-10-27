"""Dictionary legality over raw UTF-8 bytes, without a model or tokenizer."""
import gc
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from byte_constraints import PinyinConstraint, ReadingIndex, TokenTrie  # noqa: E402


READINGS = {
    "邯": {"hj": ["ge"]}, "韩": {"hj": ["uw"]}, "好": {"hk": ["nz"]},
    "钢": {"gh": ["jg"]}, "刚": {"gh": ["kd"]}, "站": {"vj": ["lv"]},
    "展": {"vj": ["ui"]}, "长": {"ih": ["pn"], "vh": ["pn"]},
    "龘": {"da": ["ll"]}, "𠀀": {"he": ["ab"]},
}


def accepts(constraint, text):
    state = constraint.step(constraint.initial_state, text.encode("utf-8"))
    return state is not None and constraint.is_final(state)


class ByteConstraints(unittest.TestCase):
    def test_rare_character_split_inside_utf8_and_token_crossing_characters(self):
        constraint = PinyinConstraint(READINGS, "hjgh", pins=[(0, "邯")])
        raw = "邯".encode()
        trie = TokenTrie([raw[:2], raw[2:], "钢".encode(), "邯钢".encode(), "韩".encode(), None, b""])
        initial = dict(trie.allowed(constraint, constraint.initial_state))
        self.assertEqual(set(initial), {0, 3})
        self.assertEqual(constraint.completed_characters(initial[0]), 0)
        self.assertEqual(constraint.completed_characters(initial[3]), 2)
        self.assertTrue(constraint.is_final(initial[3]))
        after_char = dict(trie.allowed(constraint, initial[0]))[1]
        self.assertEqual(constraint.completed_characters(after_char), 1)
        final = dict(trie.allowed(constraint, after_char))[2]
        self.assertTrue(constraint.is_final(final))
        self.assertEqual(trie.allowed(constraint, final), [])

    def test_healed_literal_and_composition_can_share_one_token(self):
        constraint = PinyinConstraint(READINGS, "vj", fixed_prefix="车".encode())
        trie = TokenTrie(["车站".encode(), "车展".encode(), "站".encode(), "车".encode()])
        choices = dict(trie.allowed(constraint, constraint.initial_state))
        self.assertEqual(set(choices), {0, 1, 3})
        self.assertTrue(constraint.is_final(choices[0]))
        self.assertEqual(constraint.completed_characters(choices[0]), 1)
        self.assertEqual(constraint.completed_characters(choices[3]), 0)
        self.assertEqual(set(dict(trie.allowed(constraint, choices[3]))), {2})

    def test_literal_is_exact_arbitrary_unicode_and_can_begin_with_partial_bytes(self):
        literal = "Hi 🐉，\n车".encode()
        constraint = PinyinConstraint(READINGS, "vj", fixed_prefix=literal)
        self.assertTrue(accepts(constraint, "Hi 🐉，\n车站"))
        self.assertFalse(accepts(constraint, "Hi 🐉，\n汽站"))
        self.assertFalse(accepts(constraint, "站"))
        partial = PinyinConstraint(READINGS, "vj", fixed_prefix="车".encode()[1:])
        end = partial.step(partial.initial_state, "车".encode()[1:] + "站".encode())
        self.assertTrue(partial.is_final(end))

    def test_half_syllable_including_one_key_and_supplementary_unicode(self):
        single = PinyinConstraint(READINGS, "h")
        self.assertTrue(all(accepts(single, text) for text in ("邯", "韩", "好", "𠀀")))
        self.assertFalse(accepts(single, "钢"))
        self.assertFalse(accepts(single, "邯钢"))
        compound = PinyinConstraint(READINGS, "hjg")
        self.assertTrue(accepts(compound, "邯刚"))
        self.assertTrue(accepts(compound, "韩钢"))
        self.assertFalse(accepts(compound, "韩好"))

    def test_pins_and_aux_intersect_with_same_reading(self):
        constraint = PinyinConstraint(READINGS, "hjgh", pins=[(0, "韩")], aux=[(0, "u"), (1, "jg")])
        self.assertTrue(accepts(constraint, "韩钢"))
        self.assertFalse(accepts(constraint, "邯钢"))
        self.assertFalse(accepts(constraint, "韩刚"))
        half = PinyinConstraint(READINGS, "h", aux=[(0, "n")])
        self.assertTrue(accepts(half, "好"))
        self.assertFalse(accepts(half, "韩"))
        different_groups = {"长": {"ih": ["aa"], "vh": ["bb"]}}
        self.assertFalse(accepts(PinyinConstraint(different_groups, "i", aux=[(0, "b")]), "长"))

    def test_unknown_or_conflicting_constraints_have_no_live_prefixes(self):
        trie = TokenTrie(["车".encode(), "邯".encode(), b"\xe9"])
        cases = [PinyinConstraint(READINGS, "xx", fixed_prefix="车".encode()),
                 PinyinConstraint(READINGS, "hjxx"),
                 PinyinConstraint(READINGS, "hj", pins=[(0, "邯"), (0, "韩")]),
                 PinyinConstraint(READINGS, "hj", pins=[(0, "好")]),
                 PinyinConstraint(READINGS, "hj", aux=[(0, "u"), (0, "g")])]
        for constraint in cases:
            self.assertFalse(constraint.is_final(constraint.initial_state))
            self.assertEqual(trie.allowed(constraint, constraint.initial_state), [])
        compatible = PinyinConstraint(READINGS, "hj", aux=[(0, "u"), (0, "uw")])
        self.assertTrue(accepts(compatible, "韩"))

    def test_empty_code_accepts_only_its_literal_prefix(self):
        empty = PinyinConstraint(READINGS, "")
        self.assertTrue(empty.is_final(empty.initial_state))
        self.assertEqual(empty.progress(empty.initial_state), (1, 0))
        self.assertFalse(empty.is_final(False))
        self.assertFalse(accepts(empty, "邯"))
        literal = PinyinConstraint(READINGS, "", fixed_prefix=b"hello")
        self.assertTrue(accepts(literal, "hello"))
        self.assertFalse(accepts(literal, "hello邯"))

    def test_progress_increases_on_every_edge_with_mixed_utf8_lengths(self):
        for prefix in (b"", "🐉车".encode(), "车".encode()[1:]):
            constraint = PinyinConstraint(READINGS, "hjh", fixed_prefix=prefix)
            initial = (0, 0) if prefix else (1, 0)
            self.assertEqual(constraint.progress(constraint.initial_state), initial)
            for state, edges in enumerate(constraint._edges):
                for following in edges.values():
                    self.assertGreater(constraint.progress(following), constraint.progress(state))
            for text in ("邯韩", "韩𠀀"):
                state = constraint.step(constraint.initial_state, prefix)
                self.assertEqual(constraint.progress(state), (1, 0))
                for position, char in enumerate(text):
                    raw = char.encode()
                    for depth, byte in enumerate(raw, 1):
                        state = constraint.step(state, bytes([byte]))
                        expected = (position + 1, depth) if depth < len(raw) else (position + 2, 0)
                        self.assertEqual(constraint.progress(state), expected)
                self.assertTrue(constraint.is_final(state))
                self.assertEqual(constraint.progress(state), (constraint.syllables + 1, 0))

    def test_trie_matches_brute_force_and_preserves_duplicate_token_ids(self):
        tokens = [None, b"", b"\xe9", "邯".encode(), "邯".encode(), "韩钢".encode(),
                  "邯刚".encode()[:4], "𠀀".encode(), b"illegal", "好钢".encode()]
        trie = TokenTrie(tokens)
        constraint = PinyinConstraint(ReadingIndex(READINGS), "hjg")
        frontier = {constraint.initial_state}
        visited = set()
        while frontier:
            state = frontier.pop()
            visited.add(state)
            expected = [(i, following) for i, raw in enumerate(tokens)
                        if raw and (following := constraint.step(state, raw)) is not None]
            self.assertEqual(trie.allowed(constraint, state), expected)
            frontier.update(following for _, following in expected if following not in visited)
        self.assertGreater(len(visited), 1)
        self.assertEqual(len(trie._cache), 1)
        del constraint
        gc.collect()
        self.assertEqual(len(trie._cache), 0)

    def test_invalid_keys_positions_and_token_types_are_rejected(self):
        for code in ("Hj", "h1", None):
            with self.assertRaises(ValueError):
                PinyinConstraint(READINGS, code)
        for kwargs in (dict(pins=[(1, "韩")]), dict(pins=[(True, "韩")]),
                       dict(pins=[(0, "韩钢")]), dict(pins=[(0, "\ud800")]), dict(aux=[(0, "U")]),
                       dict(aux=[(0, "")]), dict(fixed_prefix="车")):
            with self.assertRaises(ValueError):
                PinyinConstraint(READINGS, "hj", **kwargs)
        with self.assertRaises(ValueError):
            TokenTrie(["邯"])
        constraint = PinyinConstraint(READINGS, "hj")
        for state in (-1, True, 100000):
            with self.assertRaises(ValueError):
                constraint.step(state, b"")


if __name__ == "__main__":
    unittest.main()
