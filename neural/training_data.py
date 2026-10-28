"""File-backed conditional examples; only the current batch moves to the GPU."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from conditional import augment_composition, encode_example
from replay import fingerprint, rows


def training_characters(directories):
    """Extend target coverage from training text only, never dev/test labels."""
    characters = set()
    for directory in directories:
        for row in rows(directory / "train.jsonl"):
            characters.update(row["gold"][0])
    return sorted(characters)


class PackedExamples:
    def __init__(self, directory):
        self.info = json.loads((directory / "metadata.json").read_text())
        self.data = np.memmap(directory / "tokens.bin", mode="r", dtype="<i4",
                              shape=(self.info["rows"], 2, self.info["window"]))
        self.categories = np.memmap(directory / "categories.bin", mode="r", dtype="u1")

    def __len__(self):
        return len(self.data)

    def batch(self, indices, device):
        # Advanced indexing materializes just this batch, not the mapped corpus.
        data = torch.from_numpy(self.data[indices]).long()
        length = int((data[:, 0] != 0).sum(1).max())
        data = data[:, :, :length].to(device)
        return data[:, 0], data[:, 1]


def pack_examples(path, cache, vocab, window, protection, *, seed=0, span_share=0,
                  single_share=0, whitespace_share=0, limit=0):
    split = path.stem
    recipe = dict(source_sha256=fingerprint(path), vocab=vocab, window=window,
                  protection=protection.files, seed=seed, span_share=span_share,
                  single_share=single_share, whitespace_share=whitespace_share, limit=limit,
                  split=split, code={name: fingerprint(Path(__file__).with_name(name))
                                    for name in ("training_data.py", "conditional.py", "replay.py")})
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    directory = cache / key
    if (directory / "metadata.json").exists():
        return PackedExamples(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index, rng = {c: i for i, c in enumerate(vocab)}, random.Random(seed)
    excluded, lengths = Counter(), Counter()
    count = 0
    buffer = np.zeros((4096, 2, window), dtype="<i4")
    categories = np.zeros(4096, dtype="u1")
    with (directory / "tokens.part").open("wb") as tokens, (directory / "categories.part").open("wb") as cats:
        for row in rows(path):
            if row.get("split", split) != split:
                raise ValueError(f"{path}: row belongs to another split")
            if split == "train":
                if protection.contains(row):
                    excluded["protected_source"] += 1
                    continue
                row = augment_composition(row, rng, span_share, single_share, whitespace_share)
                if protection.contains(row):
                    excluded["protected_pair"] += 1
                    continue
            try:
                x, y = encode_example(row, index, window)
            except ValueError:
                excluded["window"] += 1
                continue
            if 1 in y:
                excluded["unknown_target"] += 1
                continue
            slot = count % len(buffer)
            buffer[slot] = 0
            buffer[slot, 0, :len(x)], buffer[slot, 1, :len(y)] = x, y
            categories[slot] = {"chat": 1, "technical": 2}.get(row.get("category", "chat"), 0)
            count += 1
            lengths[str(len(row["gold"][0]))] += 1
            if count % len(buffer) == 0:
                buffer.tofile(tokens)
                categories.tofile(cats)
            if limit and split == "train" and count >= limit:
                break
        tail = count % len(buffer)
        buffer[:tail].tofile(tokens)
        categories[:tail].tofile(cats)
    if not count:
        raise ValueError(f"{path}: no encodable examples")
    for name in ("tokens", "categories"):
        os.replace(directory / (name + ".part"), directory / (name + ".bin"))
    info = dict(recipe=recipe, rows=count, window=window, exclusions=dict(excluded), lengths=dict(lengths))
    (directory / "metadata.part").write_text(json.dumps(info, ensure_ascii=False) + "\n")
    os.replace(directory / "metadata.part", directory / "metadata.json")
    return PackedExamples(directory)


def supervised_logits(model, inputs, labels):
    """Project supervised positions only; context still participates in attention."""
    hidden = model.drop(model.tok(inputs) + model.pos(torch.arange(inputs.shape[1], device=inputs.device)))
    for block in model.blocks:
        hidden = block(hidden)
    mask = labels != 0
    return model.head(model.ln_f(hidden[mask])), labels[mask]
