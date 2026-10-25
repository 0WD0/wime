"""Differentiable deployed candidate scores, with no loss on fixed context."""
import torch
import torch.nn.functional as F

from inference import anchored_prompt
from conditional import composition_span


def ranking_composition(row, rng, whole_share=0.0):
    """Commit a prefix and preserve the original corpus row's endpoint.

    Further cutting off its end can turn a valid alternate continuation into
    a false negative. Original corpus windows may themselves be truncated;
    this prevents adding another hidden suffix during augmentation.
    """
    size = len(row["gold"][0])
    if whole_share and rng.random() < whole_share:
        length = min(size, 16)
    else:
        choice = rng.random()
        length = 1 if choice < .2 else rng.randint(2, min(5, size)) if choice < .65 and size >= 2 else min(size, 10)
    start = size - length
    result = composition_span(row, start, size)
    result["composition"] = "whole" if start == 0 else "suffix"
    if start:
        # The committed prefix has real keys; do not re-infer a polyphone.
        result["known_reading"] = dict(text=row["gold"][0][:start][-16:],
                                       input=row["input"][:2*start][-32:])
    if result["context"] and rng.random() < .1:
        result = dict(result, context=(result["context"] + rng.choice((" ", "\n", "\n\n")))[-64:])
    return result


def spelling_equivalent(a, b):
    """Do not force a preference between interchangeable interjection forms."""
    table = str.maketrans({"喔":"哦", "噢":"哦", "哎":"唉"})
    return a.translate(table) == b.translate(table)


def encode_candidate(row, text, index, window, anchor=2):
    if len(text) * 2 != len(row["input"]):
        raise ValueError("ranking requires aligned complete candidates")
    prefix = anchored_prompt(row["context"], row["input"], window, anchor,
                             row.get("known_reading"))
    pre = [2] + [index.get(c, 1) for c in prefix]
    target = [index.get(c, 1) for c in text]
    if 1 in target or len(pre) + len(target) > window:
        raise ValueError("unknown or oversized ranking target")
    # Match ConditionalEvaluator exactly: average NEW characters, no EOS.
    return pre + target[:-1], [0] * (len(pre)-1) + target


def candidate_scores(logits, labels):
    mask = labels != 0
    token_scores = logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_scores * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def preference_loss(scores, priors, weight=.1, margin=.2, temperature=.1):
    """First member of each pair is the observed correct output."""
    joint = scores.reshape(-1, 2) + weight * priors.reshape(-1, 2)
    return (F.softplus((margin + joint[:, 1] - joint[:, 0]) / temperature) * temperature).mean()
