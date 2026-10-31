"""Differentiable deployed candidate scores, with no loss on fixed context."""
from collections import Counter, defaultdict
import hashlib
import json

import torch
import torch.nn.functional as F

from composition import composition_prompt
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
    if result["context"] and rng.random() < .1:
        result = dict(result, context=(result["context"] + rng.choice((" ", "\n", "\n\n")))[-64:])
    return result


def spelling_equivalent(a, b):
    """Do not force a preference between interchangeable interjection forms."""
    table = str.maketrans({"喔":"哦", "噢":"哦", "哎":"唉"})
    return a.translate(table) == b.translate(table)


def encode_candidate(row, text, index, window):
    if "known_reading" in row:
        raise ValueError("obsolete ranking state: re-mine pairs with wime-composition-v2")
    if len(text) * 2 != len(row["input"]):
        raise ValueError("ranking requires aligned complete candidates")
    prefix = composition_prompt(row["context"], row["input"], window, row.get("committed"))
    pre = [2] + [index.get(c, 1) for c in prefix]
    target = [index.get(c, 1) for c in text]
    if 1 in target or len(pre) + len(target) > window:
        raise ValueError("unknown or oversized ranking target")
    # As the 25M verify scorer did: average NEW characters, no EOS.
    return pre + target[:-1], [0] * (len(pre)-1) + target


def candidate_scores(logits, labels):
    mask = labels != 0
    token_scores = logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_scores * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def preference_loss(scores, priors, weight=.1, margin=.2, temperature=.1):
    """First member of each pair is the observed correct output."""
    joint = scores.reshape(-1, 2) + weight * priors.reshape(-1, 2)
    return (F.softplus((margin + joint[:, 1] - joint[:, 0]) / temperature) * temperature).mean()


def curate_pairs(pairs, protected):
    """Merge training-only pairs without inventing intent for bare homophones.

    Observed text without any effective context is not a preference label.
    CE still learns those corpus frequencies. Contradictory observed positives
    and duplicates are removed across mining rounds, retaining exact keys.
    """
    counts, positives = Counter(), defaultdict(set)
    eligible = []
    for row in pairs:
        if row.get("split") != "train":
            raise ValueError("ranking curation accepts training data only")
        if len(row["input"]) != 2 * len(row["gold"][0]) or len(row["negative"]) != len(row["gold"][0]):
            raise ValueError("unaligned candidate pair")
        if protected.contains(row):
            counts["protected"] += 1
            continue
        if not row["context"].strip() and not row.get("committed"):
            counts["ambiguous_blank_context"] += 1
            continue
        slot = json.dumps([row["context"], row["input"], row.get("committed")],
                          ensure_ascii=False, sort_keys=True)
        positives[slot].update(row["gold"])
        eligible.append((slot, row))
    result, seen = [], set()
    for slot, row in eligible:
        if any(spelling_equivalent(row["negative"], g) for g in positives[slot]):
            counts["alternate_observed_positive"] += 1
            continue
        key = json.dumps([slot, row["gold"], row["negative"]], ensure_ascii=False)
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        result.append(dict(row, id="rank-" + hashlib.sha256(key.encode()).hexdigest()[:24],
                           source_id=row.get("source_id", row["id"])))
        counts["kept"] += 1
        counts["origin/" + row.get("mining_origin", "unknown")] += 1
        counts["composition/" + row.get("composition", "legacy")] += 1
    return result, dict(counts)
