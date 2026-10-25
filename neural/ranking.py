"""Differentiable deployed candidate scores, with no loss on fixed context."""
import torch
import torch.nn.functional as F

from inference import anchored_prompt
from conditional import composition_span


def ranking_composition(row, rng):
    """Commit a prefix and keep the entire remaining suffix as the label.

    Cutting off the end can turn a perfectly valid alternate continuation
    into a false negative (e.g. an unfinished word versus a complete one).
    """
    size = len(row["gold"][0])
    choice = rng.random()
    length = 1 if choice < .2 else rng.randint(2, min(5, size)) if choice < .65 and size >= 2 else min(size, 10)
    result = composition_span(row, size-length, size)
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
