"""Raw token bytes and exact-text context rewind for constrained decoding.

The byte map is derived from the tokenizer backend vocabulary and decoder,
never from decoding individual tokens into replacement Unicode characters.
Only byte-level BPE and byte-fallback decoders (optionally followed by Fuse)
are supported. Context normalization must preserve the original UTF-8 bytes;
unsupported or lossy contexts fail explicitly rather than changing text that
has already been committed.
"""
from dataclasses import dataclass
import json
import re


def _byte_level_inverse():
    # GPT-2 ByteLevel's reversible byte-to-Unicode alphabet. Bytes that would
    # be whitespace/control characters occupy code points above Latin-1.
    direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes, extra = dict(zip(direct, direct)), 0
    for byte in range(256):
        if byte not in codes:
            codes[byte] = 256 + extra
            extra += 1
    return {chr(code): byte for byte, code in codes.items()}


_BYTE_LEVEL = _byte_level_inverse()
_BYTE_FALLBACK = re.compile(r"<0x([0-9A-Fa-f]{2})>\Z")


def _decoder_kind(decoder):
    if not isinstance(decoder, dict):
        raise ValueError("tokenizer must expose a supported byte decoder")
    kind = decoder.get("type")
    if kind == "Sequence":
        children = decoder.get("decoders", [])
        # Fuse only concatenates already decoded pieces. Strip/Metaspace and
        # replacement decoders can make bytes depend on their position.
        kinds = [child.get("type") for child in children]
        if kinds in (["ByteFallback"], ["ByteFallback", "Fuse"]):
            return "ByteFallback"
    if kind in ("ByteLevel", "ByteFallback"):
        return kind
    raise ValueError(f"unsupported tokenizer decoder: {kind}")


def raw_token_bytes(tokenizer):
    """Return raw bytes for every vocabulary ID; specials/empty IDs are None.

    Some added tokens are ordinary text rather than special symbols; they
    pass through the backend decoder just like vocabulary tokens. ByteLevel
    preserves non-alphabet Unicode in those token strings as UTF-8, matching
    its decoder's fallback behavior.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("a fast tokenizer with backend serialization is required")
    definition = json.loads(backend.to_str())
    kind = _decoder_kind(definition.get("decoder"))
    model = definition.get("model", {})
    vocabulary = model.get("vocab")
    if isinstance(vocabulary, dict):
        tokens = {index: text for text, index in vocabulary.items()}
    elif model.get("type") == "Unigram" and isinstance(vocabulary, list):
        tokens = {index: record[0] for index, record in enumerate(vocabulary)}
    else:
        raise ValueError("unsupported tokenizer vocabulary serialization")
    special = set(getattr(tokenizer, "all_special_ids", ()))
    for item in definition.get("added_tokens", ()):
        tokens[item["id"]] = item["content"]
        if item.get("special"):
            special.add(item["id"])
    if any(type(index) is not int or index < 0 for index in tokens):
        raise ValueError("invalid tokenizer vocabulary ID")
    size = max(len(tokenizer), max(tokens, default=-1) + 1)
    result = [None] * size
    for index, text in tokens.items():
        if index in special or not text:
            continue
        if kind == "ByteLevel":
            raw = b"".join(bytes([_BYTE_LEVEL[char]]) if char in _BYTE_LEVEL
                           else char.encode("utf-8") for char in text)
        else:
            match = _BYTE_FALLBACK.fullmatch(text)
            raw = bytes([int(match[1], 16)]) if match else text.encode("utf-8")
        result[index] = raw or None
    return result


@dataclass(frozen=True)
class HealingContext:
    prefix_ids: tuple[int, ...]
    fixed_prefix: bytes
    original_context: bytes
    actual_rewind: int


def heal_context(tokenizer, token_bytes, context, rewind_tokens=2):
    """Rewind content tokens while retaining a nonempty causal-model prefix.

    The effective context is the last 64 characters, or a newline for empty
    context as in the existing decoder. This synthetic newline is never
    rewound. With BOS, every real content token may be rewound; without BOS,
    at least one content token remains. The returned
    fixed bytes may begin in the middle of a UTF-8 character, which the byte
    constraint automaton must preserve. A bounded rewind is not a guarantee
    that every possible cross-boundary BPE merge has been exposed.
    """
    if not isinstance(context, str):
        raise ValueError("context must be a string")
    if type(rewind_tokens) is not int or rewind_tokens < 0:
        raise ValueError("rewind_tokens must be a nonnegative integer")
    effective = context[-64:] or "\n"
    original = effective.encode("utf-8")
    ids = tuple(tokenizer(effective, add_special_tokens=True)["input_ids"])
    if not ids or any(type(token) is not int or not 0 <= token < len(token_bytes) for token in ids):
        raise ValueError("context has no usable token sequence")
    first_content = 0
    while first_content < len(ids) and token_bytes[ids[first_content]] is None:
        first_content += 1
    if first_content == len(ids) or any(token_bytes[token] is None for token in ids[first_content:]):
        raise ValueError("context contains unsupported embedded or trailing special tokens")
    pieces = [token_bytes[token] for token in ids[first_content:]]
    if b"".join(pieces) != original:
        raise ValueError("tokenizer normalization does not preserve committed context bytes")
    # A tokenizer-added leading BOS is a valid causal prefix on its own.
    minimum = first_content if first_content else 1
    cut = max(minimum, len(ids) - (rewind_tokens if context else 0))
    prefix_ids = ids[:cut]
    fixed = b"".join(token_bytes[token] for token in ids[cut:])
    retained = b"".join(token_bytes[token] for token in ids[first_content:cut])
    if not prefix_ids or retained + fixed != original:
        raise ValueError("context rewind failed exact-byte restoration")
    return HealingContext(prefix_ids, fixed, original, len(ids) - cut)
