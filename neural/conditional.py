#!/usr/bin/env python3
"""Pinyin-conditioned experiment shared by preparation, training and evaluation.

One PUA token represents one *typed* Microsoft double-pinyin syllable. It
preserves keyboard ambiguities; no gold pronunciation is used at inference.
Auxiliary codes remain hard constraints in wime's dictionary search.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/chinese-ime-lm/neural"))
from model import CharLM, Config  # noqa: E402

KEYS = "abcdefghijklmnopqrstuvwxyz;"
INPUT, OUTPUT, EOS = "\ue2e0", "\ue2e1", "\ue2e2"
FORMAT = "wime-ms-pinyin-v1"
EXTRA = [chr(0xE000 + i) for i in range(27 * 27)] + [INPUT, OUTPUT, EOS]
EXTRA += list("，。！？、；：‘’“”（）—…\n ") + list(map(chr, range(33, 127)))


def input_tokens(code):
    if len(code) % 2 or any(c not in KEYS for c in code):
        raise ValueError("expected complete Microsoft double-pinyin syllables")
    return "".join(chr(0xE000 + KEYS.index(a) * 27 + KEYS.index(b))
                   for a, b in zip(code[::2], code[1::2]))


def prompt(context, code, window=128):
    syllables = input_tokens(code)
    # BOS, input/output delimiters, all output characters and EOS must fit.
    room = min(64, window - 2 * len(syllables) - 4)
    if room < 0:
        raise ValueError("composition exceeds model window")
    # PUA delimiters in arbitrary committed text must not become instructions.
    context = "".join(c for c in context if not 0xE000 <= ord(c) <= 0xF8FF)
    return (context[-room:] if room else "") + INPUT + syllables + OUTPUT


def load_model(path, extend=False):
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
        vocab = json.loads(metadata["vocab"])
        config = Config(**json.loads(metadata["config"]))
        weights = {}
        for key in f.keys():
            if key.endswith(".scale"):
                continue
            value = f.get_tensor(key).float()
            if key + ".scale" in f.keys():
                value *= f.get_tensor(key + ".scale")[:, None]
            weights[key] = value
    if extend:
        old = len(vocab)
        vocab += [t for t in EXTRA if t not in vocab]
        extra = torch.empty(len(vocab) - old, config.n_embd).normal_(std=0.02)
        weights["tok.weight"] = torch.cat([weights["tok.weight"], extra])
        config.vocab = len(vocab)
    weights["head.weight"] = weights["tok.weight"]
    model = CharLM(config)
    model.load_state_dict(weights)
    return model, vocab, metadata


def encode_example(row, index, window):
    prefix = [2] + [index.get(c, 1) for c in prompt(row["context"], row["input"], window)]
    target = [index.get(c, 1) for c in row["gold"][0] + EOS]
    tokens = prefix + target
    return tokens[:-1], [0] * (len(prefix) - 1) + target


def composition_span(row, start, end):
    """Simulate committing a prefix, then typing a shorter part of this row.

    Slice the original reading, not a new isolated-character pronunciation.
    Only already committed characters become context; the suffix stays hidden.
    """
    target = row["gold"][0]
    if not 0 <= start < end <= len(target) or len(row["input"]) != 2 * len(target):
        raise ValueError("composition span must align with complete input syllables")
    result = dict(row, context=(row["context"] + target[:start])[-64:],
                  input=row["input"][2 * start:2 * end], gold=[target[start:end]])
    if "pinyin" in row:
        result["pinyin"] = " ".join(row["pinyin"].split()[start:end])
    return result
