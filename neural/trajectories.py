"""Paired input states derived from one sentence and reviewed commit boundaries."""


def validate(row):
    text = row["gold"][0]
    if len(row["gold"]) != 1 or len(row["input"]) != 2 * len(text):
        raise ValueError("trajectory requires one sentence aligned with complete syllables")
    ends = [boundary["end"] for boundary in row["boundaries"]]
    if ends != sorted(set(ends)):
        raise ValueError("trajectory requires unique ordered commit boundaries")
    for boundary in row["boundaries"]:
        if not 2 <= boundary["end"] <= len(text) - 2 or not boundary["reason"]:
            raise ValueError("commit boundary needs a reason and at least two characters on either side")


def states(row):
    """No future text enters prefix prediction; only an explicit commit reveals it."""
    validate(row)
    text, code, context = row["gold"][0], row["input"], row["context"]
    full = dict(context=context, input=code)
    pairs = []
    for boundary in row["boundaries"]:
        end = boundary["end"]
        prefix = dict(context=context, input=code[:2 * end])
        continuation = dict(context=(context + text[:end])[-64:], input=code[2 * end:],
                            known=dict(text=text[:end], input=code[:2 * end]))
        pairs.append(dict(**boundary, prefix=prefix, continuation=continuation))
    return full, pairs
