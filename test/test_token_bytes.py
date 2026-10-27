"""Byte vocabulary and context healing tests without model dependencies."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))
from token_bytes import heal_context, raw_token_bytes  # noqa: E402


def byte_level(text):
    # An independent forward construction of the documented GPT-2 alphabet.
    direct = list(range(ord("!"), ord("~") + 1))
    direct += list(range(ord("¡"), ord("¬") + 1))
    direct += list(range(ord("®"), ord("ÿ") + 1))
    chars = direct[:]
    missing = [byte for byte in range(256) if byte not in direct]
    mapping = dict(zip(direct + missing, chars + list(range(256, 256 + len(missing)))))
    return "".join(chr(mapping[byte]) for byte in text)


class Backend:
    def __init__(self, definition):
        self.definition = definition

    def to_str(self):
        return json.dumps(self.definition)


class Tokenizer:
    def __init__(self, tokens, decoder, encodings=None, bos=False, added=()):
        self.backend_tokenizer = Backend(dict(
            decoder=decoder, model=dict(type="BPE", vocab={t: i for i, t in enumerate(tokens)}),
            added_tokens=list(added)))
        self.encodings = encodings or {}
        self.all_special_ids = [0] if bos else []
        self.bos = bos
        self.size = max(len(tokens), max((a["id"] + 1 for a in added), default=0))

    def __len__(self):
        return self.size

    def __call__(self, text, add_special_tokens=True):
        return dict(input_ids=([0] if self.bos and add_special_tokens else []) + self.encodings[text])


class TokenBytes(unittest.TestCase):
    def test_byte_level_preserves_partial_utf8_and_added_tokens(self):
        raw = "邯".encode()
        tokenizer = Tokenizer([byte_level(raw[:2]), byte_level(raw[2:]), byte_level("车站".encode())],
                              dict(type="ByteLevel"), added=[
                                  dict(id=3, content="<special>", special=True),
                                  dict(id=4, content="<ordinary>", special=False)])
        vocabulary = raw_token_bytes(tokenizer)
        self.assertEqual(vocabulary[:3], [raw[:2], raw[2:], "车站".encode()])
        self.assertEqual(vocabulary[3:], [None, b"<ordinary>"])
        with self.assertRaises(UnicodeDecodeError):
            vocabulary[0].decode()

    def test_byte_fallback_preserves_bytes_without_unicode_replacement(self):
        tokenizer = Tokenizer(["<s>", "<0xE9>", "<0x82>", "<0xAF>", "车站", ""],
                              dict(type="Sequence", decoders=[dict(type="ByteFallback"), dict(type="Fuse")]), bos=True)
        vocabulary = raw_token_bytes(tokenizer)
        self.assertEqual(vocabulary, [None, b"\xe9", b"\x82", b"\xaf", "车站".encode(), None])
        self.assertEqual(b"".join(vocabulary[1:4]), "邯".encode())

    def test_healing_bos_can_rewind_all_text_exactly(self):
        tokenizer = Tokenizer(["<s>", byte_level("车".encode()), byte_level("站".encode())],
                              dict(type="ByteLevel"), {"车": [1]}, bos=True)
        healed = heal_context(tokenizer, raw_token_bytes(tokenizer), "车", rewind_tokens=2)
        self.assertEqual(healed.prefix_ids, (0,))
        self.assertEqual(healed.fixed_prefix, "车".encode())
        self.assertEqual(healed.original_context, "车".encode())
        self.assertEqual(healed.actual_rewind, 1)

    def test_healing_without_bos_retains_causal_prefix_and_can_split_utf8(self):
        raw = "邯".encode()
        tokenizer = Tokenizer([byte_level(raw[:2]), byte_level(raw[2:]), byte_level("车".encode())],
                              dict(type="ByteLevel"), {"邯车": [0, 1, 2]})
        vocabulary = raw_token_bytes(tokenizer)
        healed = heal_context(tokenizer, vocabulary, "邯车", rewind_tokens=2)
        self.assertEqual(healed.prefix_ids, (0,))
        self.assertEqual(healed.fixed_prefix, raw[2:] + "车".encode())
        self.assertEqual(vocabulary[0] + healed.fixed_prefix, healed.original_context)
        self.assertEqual(healed.actual_rewind, 2)

    def test_single_token_without_bos_does_not_invent_context(self):
        tokenizer = Tokenizer([byte_level("车".encode()), byte_level(b"\n")],
                              dict(type="ByteLevel"), {"车": [0], "\n": [1]})
        vocabulary = raw_token_bytes(tokenizer)
        for context, expected in [("车", 0), ("", 1)]:
            healed = heal_context(tokenizer, vocabulary, context)
            self.assertEqual(healed.prefix_ids, (expected,))
            self.assertEqual(healed.fixed_prefix, b"")
            self.assertEqual(healed.actual_rewind, 0)

    def test_empty_context_keeps_synthetic_newline_even_with_bos(self):
        tokenizer = Tokenizer(["<s>", byte_level(b"\n")], dict(type="ByteLevel"),
                              {"\n": [1]}, bos=True)
        healed = heal_context(tokenizer, raw_token_bytes(tokenizer), "")
        self.assertEqual(healed.prefix_ids, (0, 1))
        self.assertEqual(healed.fixed_prefix, b"")
        self.assertEqual(healed.original_context, b"\n")
        self.assertEqual(healed.actual_rewind, 0)

    def test_normalized_context_and_unknown_decoder_fail_explicitly(self):
        tokenizer = Tokenizer([byte_level("é".encode())], dict(type="ByteLevel"), {"e\u0301": [0]})
        with self.assertRaisesRegex(ValueError, "normalization"):
            heal_context(tokenizer, raw_token_bytes(tokenizer), "e\u0301")
        tokenizer.backend_tokenizer.definition["decoder"] = dict(type="Sequence", decoders=[
            dict(type="ByteFallback"), dict(type="Strip", content=" ", start=1, stop=0)])
        with self.assertRaisesRegex(ValueError, "unsupported tokenizer decoder"):
            raw_token_bytes(tokenizer)
        # Fusing first would turn adjacent <0x..> spellings into a single
        # string, preventing ByteFallback from recognizing individual bytes.
        tokenizer.backend_tokenizer.definition["decoder"] = dict(type="Sequence", decoders=[
            dict(type="Fuse"), dict(type="ByteFallback")])
        with self.assertRaisesRegex(ValueError, "unsupported tokenizer decoder"):
            raw_token_bytes(tokenizer)

    def test_special_token_inside_context_is_not_silently_removed(self):
        tokenizer = Tokenizer(["<s>", byte_level("车".encode())], dict(type="ByteLevel"),
                              {"车<s>": [1, 0]}, bos=True)
        with self.assertRaisesRegex(ValueError, "special tokens"):
            heal_context(tokenizer, raw_token_bytes(tokenizer), "车<s>")


if __name__ == "__main__":
    unittest.main()
