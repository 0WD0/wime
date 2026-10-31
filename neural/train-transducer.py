#!/usr/bin/env python3
"""Train the aligned syllable transducer on a registered sentence corpus.

States follow train-conditional.py's trajectory sampling: whole sentences,
prefixes ending at the reviewed boundary, and the continuation after it is
committed (its characters fixed, its syllables still encoded). Tones: a
composition is toneless with probability --toneless; otherwise each syllable
keeps a tone with probability --tone-keep, the citation tone (sandhi undone)
or, with probability --spoken, the spoken one. Validation reports per-character
loss on the corpus dev split, toneless and fully toned, by stratum.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from dataset_registry import Registry
from replay import Protection, fingerprint, rows, split_path
from training_data import TrajectorySampler, sentence_views, training_characters
import syllables
import transducer

STATES = ("whole", "prefix", "continuation")


class Corpus:
    """Sentences as flat int arrays; a state picks the composition at draw time."""

    def __init__(self, path, index, protection):
        self.context, self.text, self.keys, self.citation, self.spoken = [], [], [], [], []
        boundary, categories, self.excluded = [], [], Counter()
        for row in rows(path):
            views = sentence_views(row)
            # The corpus protects its own dev split; only training rows are filtered.
            if protection and any(protection.contains(view) for view in views):
                self.excluded["protected_source_or_state"] += 1
                continue
            text = row["gold"][0]
            if any(c not in index for c in text):
                self.excluded["unknown_target"] += 1
                continue
            features = syllables.typed(row["input"])
            self.context.append([index.get(c, 1) for c in row["context"][-transducer.CONTEXT:]])
            self.text.append([index[c] for c in text])
            self.keys.append([f[:2] for f in features])
            self.citation.append(syllables.citation_tones(row))
            self.spoken.append(list(row["tones"]))
            boundary.append(row["boundaries"][0]["end"] if row["boundaries"] else 0)
            categories.append(2 if row.get("category") == "technical" else 1)
        self.boundary = np.array(boundary)
        self.categories = np.array(categories, dtype="u1")
        self.probability = (self.boundary > 0).astype("f4")

    def __len__(self):
        return len(self.text)

    def state(self, i, state):
        """(context, composition text, committed length, keys, citation, spoken) of a state."""
        end, n = self.boundary[i], len(self.text[i])
        stop, committed = (n, 0) if state == 1 else (end, 0) if state == 2 else (n, end)
        return (self.context[i], self.text[i][:stop], committed, self.keys[i][:stop],
                self.citation[i][:stop], self.spoken[i][:stop])


def tones_for(citation, spoken, committed, policy, rng):
    """Tone indices per syllable under POLICY: None, "all" or (toneless, keep, spoken)."""
    if policy is None:
        return [0] * len(citation)
    if policy == "all":
        return [0 if j < committed else t for j, t in enumerate(citation)]
    toneless, keep, spoken_share = policy
    if rng.random() < toneless:
        return [0] * len(citation)
    return [0 if j < committed or rng.random() >= keep else (s if rng.random() < spoken_share else c)
            for j, (c, s) in enumerate(zip(citation, spoken))]


def collate(corpus, picks, index, policy, rng, device, marker):
    """Tensors for (sentence, state) PICKS: ids, encoder features, label positions."""
    output = [index[transducer.OUTPUT]] * marker
    sequences, features, targets = [], [], []
    for i, state in picks:
        context, text, committed, keys, citation, spoken = corpus.state(i, state)
        room = max(0, min(transducer.CONTEXT, 127 - marker - len(text)))
        context = context[len(context) - room:] if room else []
        sequences.append([2] + context + output + text)
        tones = tones_for(citation, spoken, committed, policy, rng)
        features.append([(a, b, t) for (a, b), t in zip(keys, tones)])
        offset = len(context) + marker
        targets.append([(offset + j, j, text[j]) for j in range(committed, len(text))])
    width, syllable_width = max(map(len, sequences)), max(map(len, features))
    ids = torch.zeros(len(picks), width, dtype=torch.long)
    feats = torch.zeros(len(picks), syllable_width, 3, dtype=torch.long)
    padding = torch.ones(len(picks), syllable_width, dtype=torch.bool)
    rows_, positions, slots, labels = [], [], [], []
    for b, (sequence, feature, target) in enumerate(zip(sequences, features, targets)):
        ids[b, :len(sequence)] = torch.tensor(sequence)
        feats[b, :len(feature)] = torch.tensor(feature)
        padding[b, :len(feature)] = False
        for position, slot, label in target:
            rows_.append(b), positions.append(position), slots.append(slot), labels.append(label)
    move = lambda *xs: [torch.tensor(x, device=device) if isinstance(x, list) else x.to(device) for x in xs]
    return move(ids, feats, padding, rows_, positions, slots, labels)


def loss(model, batch):
    ids, feats, padding, rows_, positions, slots, labels = batch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        states = model.states(ids)[rows_, positions]
        encoded = model.encoder(feats, padding)[rows_, slots]
        logits = model.logits(states, encoded)
    return F.cross_entropy(logits.float(), labels, reduction="sum"), len(labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="conditional checkpoint for the character network")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--predictor-rate", type=float, default=5e-5)
    p.add_argument("--rate", type=float, default=5e-4, help="encoder and joint learning rate")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--state-shares", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    p.add_argument("--technical-share", type=float, default=0.1)
    p.add_argument("--toneless", type=float, default=0.3)
    p.add_argument("--tone-keep", type=float, default=0.5)
    p.add_argument("--spoken", type=float, default=0.15)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--no-marker", action="store_true", help="run the context straight into the composition")
    p.add_argument("--encoder-layers", type=int, default=4)
    p.add_argument("--encoder-width", type=int, default=256)
    p.add_argument("--encoder-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261031)
    args = p.parse_args()
    if (args.out / "model.safetensors").exists():
        raise ValueError("output already holds a trained model; choose a new directory")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    directory, excluded, snapshot = Registry().training(args.dataset)
    protection = Protection(excluded)
    additions = training_characters([directory], protection)
    encoder = transducer.EncoderConfig(n_layer=args.encoder_layers, n_head=args.encoder_heads,
                                       n_embd=args.encoder_width, marker=not args.no_marker)
    model, vocab = transducer.from_conditional(str(args.base), additions, encoder)
    marker = encoder.marker
    model.cuda()
    index = {c: i for i, c in enumerate(vocab)}
    started = time.monotonic()
    train = Corpus(split_path(directory, "train"), index, protection)
    dev = Corpus(split_path(directory, "dev"), index, None)
    print(json.dumps(dict(train=len(train), dev=len(dev), excluded=dict(train.excluded),
                          seconds=time.monotonic() - started)), flush=True)
    sampler = TrajectorySampler(train, args.state_shares, args.technical_share)
    dev_sampler = TrajectorySampler(dev, args.state_shares, args.technical_share)
    new = [p for name, p in model.named_parameters() if not name.startswith("predictor.")]
    optimizer = torch.optim.AdamW([dict(params=model.predictor.parameters(), lr=args.predictor_rate),
                                   dict(params=new, lr=args.rate)], weight_decay=0.01)
    rates = [args.predictor_rate, args.rate]
    policy = (args.toneless, args.tone_keep, args.spoken)

    def validate(step):
        model.eval()
        report = {}
        with torch.no_grad():
            for tones in (None, "all"):
                weighted = 0.
                for name, state, weight, pool in dev_sampler.strata():
                    total, count = 0., 0
                    for chunk in pool.split(128):
                        value, n = loss(model, collate(dev, [(i, state) for i in chunk.tolist()], index, tones, rng, "cuda", marker))
                        total, count = total + value.item(), count + n
                    report[f"{'toneless' if tones is None else 'toned'}/{name}/{STATES[state - 1]}"] = total / count
                    weighted += weight * total / count
                report["toneless" if tones is None else "toned"] = weighted
        print(json.dumps(dict(step=step, validation=report)), flush=True)
        return report

    history = [validate(0)]
    seen = Counter()
    for step in range(1, args.steps + 1):
        model.train()
        scale = min(1, step / args.warmup) * (0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        for group, rate in zip(optimizer.param_groups, rates):
            group["lr"] = rate * scale
        indices, states = sampler.draw(args.batch)
        picks = list(zip(indices.tolist(), states.tolist()))
        seen.update(STATES[s - 1] for _, s in picks)
        value, n = loss(model, collate(train, picks, index, policy, rng, "cuda", marker))
        optimizer.zero_grad(set_to_none=True)
        (value / n).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(json.dumps(dict(step=step, loss=value.item() / n, seconds=time.monotonic() - started)), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            history.append(validate(step))
            transducer.save(model, vocab, args.out / f"step-{step}.safetensors", training_steps=step)
    transducer.save(model, vocab, args.out / "model.safetensors", training_steps=args.steps)
    run = dict(vars(args), base=str(args.base), out=str(args.out), base_sha256=fingerprint(args.base), dataset_snapshot=snapshot,
               train_rows=len(train), dev_rows=len(dev), excluded=dict(train.excluded), seen_states=dict(seen),
               parameters=sum(p.numel() for p in model.parameters()),
               new_parameters=sum(p.numel() for p in new), validation=history, seconds=time.monotonic() - started,
               code_sha256={name: fingerprint(Path(__file__).with_name(name))
                            for name in ("train-transducer.py", "transducer.py", "syllables.py", "training_data.py")})
    (args.out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
