"""One input state: original history, confirmed text/keys, pending keys.

History has no fabricated keystrokes. A commit fixes output without moving
that output or its input keys across the model's delimiters. If the combined
composition exceeds the window, rebase the entire confirmed segment into
history; never retain an arbitrary two-character tail.
"""
KEYS = "abcdefghijklmnopqrstuvwxyz;"
CODE_KEYS = KEYS + ",./"
INPUT, OUTPUT, EOS = "\ue2e0", "\ue2e1", "\ue2e2"
ENCODING = "wime-composition-v2"


def input_tokens(code):
    if len(code) % 2 or any(c not in KEYS for c in code):
        raise ValueError("expected complete Microsoft double-pinyin syllables")
    return "".join(chr(0xE000 + KEYS.index(a) * 27 + KEYS.index(b))
                   for a, b in zip(code[::2], code[1::2]))


def validate_committed(committed):
    if committed is None:
        return
    if not isinstance(committed, dict):
        raise ValueError("committed state must be an object")
    text, code = committed.get("text"), committed.get("input")
    if (not isinstance(text, str) or not text or not all("㐀" <= c <= "鿿" for c in text)
            or not isinstance(code, str) or len(code) != 2 * len(text)):
        raise ValueError("committed text requires aligned actual keys")
    if any(c not in CODE_KEYS for c in code):
        raise ValueError("committed input contains invalid code keys")


def expand(context, code, committed=None, window=128):
    """Return history, complete composition keys, and its fixed text prefix.

    Pending keys may end in a half syllable for search. Its eventual output
    character counts toward the window even before its second key arrives.
    """
    validate_committed(committed)
    if committed is None:
        return context, code, ""
    text, keys = committed["text"], committed["input"]
    if len(text) + (len(code) + 1) // 2 > (window - 4) // 2:
        # Each model keeps its own suffix of the history.
        return context + text, code, ""
    return context, keys + code, text


def prompt(context, code, window=128):
    syllables = input_tokens(code)
    room = min(64, window - 2 * len(syllables) - 4)
    if room < 0:
        raise ValueError("composition exceeds model window")
    context = "".join(c for c in context if not 0xE000 <= ord(c) <= 0xF8FF)
    return (context[-room:] if room else "") + INPUT + syllables + OUTPUT


def composition_prompt(context, code, window=128, committed=None):
    history, keys, fixed = expand(context, code, committed, window)
    return prompt(history, keys, window) + fixed


def decode(decoder, context, code, committed=None, *, window=128,
           pins=(), learned=(), weight=0.0, aux=(), literals=(), tones=()):
    """Decode under fixed confirmed output, returning only uncommitted text.

    Existing search sees the original full input with confirmed characters
    pinned. Its LM token paths and transducer input remain comparable to
    whole-input search. Rescale scores to pending characters so the editor's
    length-scaled score differences retain the same confidence temperature.
    Common confirmed-prefix contributions remain; these are ranking scores,
    not absolute suffix log probabilities.
    """
    history, keys, fixed = expand(context, code, committed, window)
    offset = len(fixed)
    result = decoder.decode(history, keys,
        pins=list(enumerate(fixed)) + [(p + offset, c) for p, c in pins],
        learned=[(p + offset, text) for p, text in learned], weight=weight,
        aux=[(p + offset, typed) for p, typed in aux],
        literals=[(p + offset, text) for p, text in literals],
        tones=[(p + offset, tone) for p, tone in tones])
    return [(text[offset:], score * len(text) / (len(text) - offset))
            for text, score in result if text.startswith(fixed) and len(text) > offset]
