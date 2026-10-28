#!/usr/bin/env python3
"""Pinyin-conditioned experiment shared by preparation, training and evaluation.

One PUA token represents one *typed* Microsoft double-pinyin syllable. It
preserves keyboard ambiguities; no gold pronunciation is used at inference.
Auxiliary codes remain hard constraints in wime's dictionary search.
"""
import dataclasses
from functools import lru_cache
import re
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
# The immutable runtime ships the imported model implementation beside this
# file. Source-tree training uses the same implementation from the monorepo.
if not Path(__file__).with_name("model.py").is_file():
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


@lru_cache(maxsize=4096)
def context_reading(text):
    """Infer only the visible committed suffix; never read candidate answers."""
    from pypinyin import lazy_pinyin
    mapping = reading_map()
    codes = [mapping.get(py) for py in lazy_pinyin(text)]
    return "".join(codes) if all(codes) else None


@lru_cache(maxsize=1)
def reading_map():
    return json.loads(Path(__file__).with_name("context-pinyin.json").read_text())["syllables"]


def anchored_prompt(context, code, window, limit=0, known_reading=None):
    """Keep committed characters inside the decoder; return only a prompt.

    Actual committed keystrokes take precedence over inferred pronunciation.
    The current composition's keys are always appended verbatim. The caller
    scores only NEW characters, and never proposes edits to this fixed prefix.
    """
    if not limit:
        return prompt(context, code, window)
    match = re.search(r"[\u4e00-\u9fff]{1,16}$", context)
    if not match:
        return prompt(context, code, window)
    text = match[0]
    size = min(limit, len(text), max(0, (window - 4) // 2 - len(code) // 2))
    if not size:
        return prompt(context, code, window)
    anchor = text[-size:]
    known_size = (min(size, len(known_reading["text"]))
                  if known_reading and context.endswith(known_reading["text"]) else 0)
    if known_size == size:
        reading = known_reading["input"][-2 * size:]
    else:
        reading = context_reading(text)
        if reading is None:
            return prompt(context, code, window)
        reading = reading[-2 * size:]
        if known_size:
            reading = reading[:-2 * known_size] + known_reading["input"][-2 * known_size:]
    return prompt(context[:-size], reading + code, window) + anchor



def load_model(path, extend=False, extra_tokens=()):
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
    if extend or extra_tokens:
        old = len(vocab)
        additions = dict.fromkeys((EXTRA if extend else []) + sorted(extra_tokens))
        known = set(vocab)
        vocab += [t for t in additions if t not in known]
        extra = torch.empty(len(vocab) - old, config.n_embd).normal_(std=0.02)
        weights["tok.weight"] = torch.cat([weights["tok.weight"], extra])
        config.vocab = len(vocab)
    weights["head.weight"] = weights["tok.weight"]
    model = CharLM(config)
    model.load_state_dict(weights)
    return model, vocab, metadata


def grow_depth(model, layers):
    """Interleave identity residual blocks, preserving the starting function.

    Existing blocks retain their order and weights. New attention and MLP
    output projections start at zero; their internal weights remain random so
    the projections can learn immediately without duplicating block symmetry.
    """
    old = model.cfg.n_layer
    if layers < old:
        raise ValueError("depth growth cannot remove pretrained blocks")
    if layers == old:
        return model
    grown = CharLM(dataclasses.replace(model.cfg, n_layer=layers))
    grown.tok.load_state_dict(model.tok.state_dict())
    grown.pos.load_state_dict(model.pos.state_dict())
    grown.ln_f.load_state_dict(model.ln_f.state_dict())
    retained = {i * layers // old: block for i, block in enumerate(model.blocks)}
    with torch.no_grad():
        for i, block in enumerate(grown.blocks):
            if i in retained:
                grown.blocks[i] = retained[i]
            else:
                for projection in (block.proj, block.out):
                    projection.weight.zero_()
                    projection.bias.zero_()
    grown.train(model.training)
    return grown


def encode_example(row, index, window, context_anchor=0):
    text = anchored_prompt(row["context"], row["input"], window, context_anchor, row.get("known_reading"))
    prefix = [2] + [index.get(c, 1) for c in text]
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
    if start:
        result["known_reading"] = dict(text=target[:start][-16:], input=row["input"][:2*start][-32:])
    # Derived states must not carry source-relative proposal locations.
    result.pop("boundaries", None)
    result.pop("legacy_span", None)
    if "pinyin" in row:
        reading = row["pinyin"]
        result["pinyin"] = (" ".join(reading.split()[start:end])
                            if isinstance(reading, str) else reading[start:end])
    for field in ("tones", "tone_sources", "reading_rules"):
        if field in row:
            result[field] = row[field][start:end]
    if "reading_evidence" in row:
        result["reading_evidence"] = {key: value[start:end]
                                      for key, value in row["reading_evidence"].items()}
    if "reading_notes" in row:
        notes = [dict(note, index=note["index"] - start) for note in row["reading_notes"]
                 if start <= note["index"] < end]
        result.pop("reading_notes")
        if notes:
            result["reading_notes"] = notes
    if "offset" in row:
        result["offset"] += start
    return result


def augment_composition(row, rng, span_share=0.0, single_share=0.0, whitespace_share=0.0):
    """Sample disjoint single/short/full modes without changing the reading."""
    target = row["gold"][0]
    choice = rng.random()
    if choice < single_share:
        start = rng.randrange(len(target))
        row = composition_span(row, start, start + 1)
    elif choice < single_share + span_share and len(target) >= 2:
        length = rng.randint(2, min(5, len(target)))
        start = rng.randint(0, len(target) - length)
        row = composition_span(row, start, start + length)
    if row["context"] and rng.random() < whitespace_share:
        row = dict(row, context=(row["context"] + rng.choice((" ", "\n", "\n\n")))[-64:])
    return row
