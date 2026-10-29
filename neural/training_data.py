"""File-backed conditional examples; only the current batch moves to the GPU."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from conditional import augment_composition, composition_span, encode_example
from trajectories import validate
from replay import fingerprint, rows, split_path


def training_characters(directories, protection=None):
    """Extend target coverage from training text only, never dev/test labels."""
    characters = set()
    for directory in directories:
        for row in rows(split_path(directory, "train")):
            if protection is not None and protection.contains(row):
                continue
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


class PackedTrajectories:
    """One sampling identity per sentence; four disposable encoded views."""
    def __init__(self, directory, sampling):
        self.info = json.loads((directory / "metadata.json").read_text())
        self.data = np.memmap(directory / "tokens.bin", mode="r", dtype="<i4",
                             shape=(self.info["rows"], 4, 2, self.info["window"]))
        self.probability = np.memmap(directory / "probability.bin", mode="r", dtype="<f4")
        self.categories = np.memmap(directory / "categories.bin", mode="r", dtype="u1")
        self.sampling = sampling

    def __len__(self):
        return len(self.data)

    def batch(self, indices, device):
        # A/B also draw this variate: all arms see identical sentence indices,
        # despite selecting different views and consuming different token counts.
        draw = torch.rand(len(indices)).numpy()
        probability = self.probability[indices]
        choices = 1 + (draw >= 1-2*probability) + (draw >= 1-probability).astype(np.int64)
        if self.sampling == "legacy-spans":
            choices[:] = 0
        data = torch.from_numpy(self.data[indices, choices]).long()
        length = int((data[:,0] != 0).sum(1).max())
        data = data[:,:,:length].to(device)
        return data[:,0], data[:,1]


def sentence_views(row):
    validate(row)
    start, end = row["legacy_span"]
    legacy = composition_span(row, start, end)
    if not row["boundaries"]:
        return [legacy, row, row, row]
    if len(row["boundaries"]) != 1:
        raise ValueError("training pilot expects one reviewed boundary per sentence")
    end = row["boundaries"][0]["end"]
    return [legacy, row, composition_span(row,0,end), composition_span(row,end,len(row["gold"][0]))]


def pack_trajectories(path, cache, vocab, window, protection, context_anchor, sampling):
    recipe = dict(source_sha256=fingerprint(path),vocab=vocab,window=window,protection=protection.files,
                  context_anchor=context_anchor,format="paired-sentence-cache-v1",
                  code={name:fingerprint(Path(__file__).with_name(name)) for name in
                        ("training_data.py","conditional.py","trajectories.py","replay.py")})
    directory = cache / hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    if (directory/"metadata.json").exists():
        return PackedTrajectories(directory,sampling)
    directory.mkdir(parents=True,exist_ok=True)
    index = {c:i for i,c in enumerate(vocab)}
    excluded, lengths, sources, paired = Counter(), Counter(), Counter(), Counter()
    retained, identities = [], []
    with (directory/"tokens.part").open("wb") as stream:
        for row in rows(path):
            if row['split'] != path.stem:
                raise ValueError("sentence split changed")
            views = sentence_views(row)
            if any(protection.contains(view) for view in views):
                excluded['protected_source_or_state'] += 1
                continue
            encoded = [encode_example(view,index,window,context_anchor) for view in views]
            if any(1 in y for x,y in encoded):
                excluded['unknown_target'] += 1
                continue
            packed = np.zeros((4,2,window),dtype='<i4')
            for i,(x,y) in enumerate(encoded):
                packed[i,0,:len(x)],packed[i,1,:len(y)] = x,y
            packed.tofile(stream)
            retained.append((row['source'],bool(row['boundaries'])))
            identities.append(row['id'])
            sources[row['source']] += 1
            paired[row['source']] += bool(row['boundaries'])
            lengths[str(len(row['gold'][0]))] += 1
    if not retained:
        raise ValueError("no encodable sentences")
    # Uniform sentence/group sampling. Paired sentences supply enough prefix
    # and continuation requests for 60/20/20 per source without duplicating data.
    probability = {source:min(.5,.2*count/paired[source]) if paired[source] else 0
                   for source,count in sources.items()}
    np.array([probability[source] if has_boundary else 0 for source,has_boundary in retained],dtype='<f4').tofile(directory/'probability.bin')
    np.ones(len(retained),dtype='u1').tofile(directory/'categories.bin')
    os.replace(directory/'tokens.part',directory/'tokens.bin')
    info = dict(recipe=recipe,rows=len(retained),window=window,exclusions=dict(excluded),lengths=dict(lengths),
                source_counts=dict(sources),paired_counts=dict(paired),
                sentence_ids_sha256=hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
                expected_state_shares={source:dict(whole=1-2*probability[source]*paired[source]/count,
                    prefix=probability[source]*paired[source]/count,continuation=probability[source]*paired[source]/count)
                    for source,count in sources.items()})
    (directory/'metadata.json').write_text(json.dumps(info,ensure_ascii=False)+'\n')
    return PackedTrajectories(directory,sampling)


def pack_examples(path, cache, vocab, window, protection, *, seed=0, span_share=0,
                  single_share=0, whitespace_share=0, limit=0, context_anchor=0):
    split = path.stem
    recipe = dict(source_sha256=fingerprint(path), vocab=vocab, window=window,
                  protection=protection.files, seed=seed, span_share=span_share,
                  single_share=single_share, whitespace_share=whitespace_share, limit=limit,
                  context_anchor=context_anchor,
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
                x, y = encode_example(row, index, window, context_anchor)
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
