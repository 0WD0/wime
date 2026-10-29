#!/usr/bin/env python3
"""Pinyin-conditioned experiment shared by preparation, training and evaluation.

One PUA token represents one *typed* Microsoft double-pinyin syllable. It
preserves keyboard ambiguities; no gold pronunciation is used at inference.
Auxiliary codes remain hard constraints in wime's dictionary search.
"""
import dataclasses
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

from composition import KEYS, INPUT, OUTPUT, EOS, ENCODING, input_tokens, prompt, composition_prompt

FORMAT = "wime-ms-pinyin-v1"
EXTRA = [chr(0xE000 + i) for i in range(27 * 27)] + [INPUT, OUTPUT, EOS]
EXTRA += list("，。！？、；：‘’“”（）—…\n ") + list(map(chr, range(33, 127)))


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


def encode_example(row, index, window):
    text = composition_prompt(row["context"], row["input"], window, row.get("committed"))
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
    result = dict(row, context=row["context"],
                  input=row["input"][2 * start:2 * end], gold=[target[start:end]])
    if start:
        previous = row.get("committed") or dict(text="", input="")
        result["committed"] = dict(text=previous["text"] + target[:start],
                                   input=previous["input"] + row["input"][:2*start])
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
