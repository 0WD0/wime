#!/usr/bin/env python3
"""Fine-tune a released character LM on output-only conditional loss."""
import argparse
from collections import Counter
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from conditional import grow_depth, load_model, FORMAT
from ranking import encode_candidate, candidate_scores, preference_loss
from replay import Protection
from training_data import pack_examples, supervised_logits, training_characters


def reference_kl(logits, reference_logits, labels):
    """Preserve the reference distribution only at supervised output positions."""
    mask = labels != 0
    return F.kl_div(logits[mask].float().log_softmax(-1),
                    reference_logits[mask].float().softmax(-1), reduction="batchmean")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--encoded-cache", type=Path,
                   default=Path.home() / ".cache/wime-conditional/encoded",
                   help="shared file-backed token cache; GPU memory is independent of corpus size")
    p.add_argument("--extend-vocabulary", action="store_true",
                   help="add characters found in training targets, without consulting held-out labels")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--extra-data", type=Path)
    p.add_argument("--extra-share", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--span-share", type=float, default=0.0,
                   help="fraction of training rows replaced by 2-5 character composition spans")
    p.add_argument("--single-share", type=float, default=0.0)
    p.add_argument("--whitespace-share", type=float, default=0.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--reference-model", help="fixed older checkpoint for primary-domain retention")
    p.add_argument("--reference-weight", type=float, default=0.0,
                   help="output-token KL weight on primary replay batches only; extra data uses CE")
    p.add_argument("--exclude-cases", type=Path, action="append", default=[],
                   help="exclude exact normalized context/answer pairs after span augmentation")
    p.add_argument("--layers", type=int, help="grow depth with initially identity residual blocks")
    p.add_argument("--ranking-data", type=Path, help="training-only mined same-key pairs")
    p.add_argument("--ranking-weight", type=float, default=0)
    p.add_argument("--ranking-batch", type=int, default=32)
    p.add_argument("--ranking-prior-weight", type=float, default=.1,
                   help="weight of each pair's fixed priors in the trained joint score, e.g. 1.5 for "
                        "per-character general-LM scores fused by the text decoder")
    p.add_argument("--primary-technical-share", type=float,
                   help="preserve technical sampling share when expanding chat replay")
    p.add_argument("--save-every", type=int, default=0,
                   help="retain intermediate checkpoints for development top-1 selection")
    args = p.parse_args()
    if (args.steps < 1 or args.batch < 1 or not 0 <= args.extra_share <= 1
            or not 0 <= args.span_share <= 1 or not 0 <= args.single_share <= 1
            or args.span_share + args.single_share > 1
            or not 0 <= args.whitespace_share <= 1 or args.learning_rate <= 0
            or args.reference_weight < 0 or (args.reference_weight and not args.reference_model)
            or args.ranking_weight < 0 or args.ranking_batch < 1 or args.save_every < 0
            or not math.isfinite(args.ranking_prior_weight) or args.ranking_prior_weight < 0
            or (args.primary_technical_share is not None and not 0 < args.primary_technical_share < 1)
            or (args.ranking_weight and not args.ranking_data)):
        p.error("positive steps/batch/rate required; shares in [0, 1], span + single <= 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    additions = training_characters([d for d in (args.data, args.extra_data) if d]) if args.extend_vocabulary else []
    model, vocab, metadata = load_model(args.base, extend=True, extra_tokens=additions)
    model.cfg.dropout = 0.0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    original_layers = model.cfg.n_layer
    if args.layers is not None:
        if args.layers < original_layers:
            p.error("--layers must not remove pretrained blocks")
        model = grow_depth(model, args.layers)
    index = {c: i for i, c in enumerate(vocab)}
    window = model.cfg.context
    args.out.mkdir(parents=True, exist_ok=True)
    exclusions = Counter()
    compositions = Counter()
    protection = Protection(args.exclude_cases)
    def pack(split, directory=args.data):
        data = pack_examples(directory / (split + ".jsonl"), args.encoded_cache, vocab, window,
                             protection, seed=args.seed, span_share=args.span_share,
                             single_share=args.single_share, whitespace_share=args.whitespace_share,
                             limit=args.limit)
        exclusions.update({f"{directory}/{split}/{key}": value
                           for key, value in data.info["exclusions"].items()})
        if split == "train":
            compositions.update({f"{directory}/length_{key}": value
                                 for key, value in data.info["lengths"].items()})
        print(json.dumps(dict(packed=str(directory / split), rows=len(data),
                              exclusions=data.info["exclusions"])), flush=True)
        return data

    train, dev = pack("train"), pack("dev")
    extra_train = pack("train", args.extra_data) if args.extra_data else None
    extra_dev = pack("dev", args.extra_data) if args.extra_data else None
    primary_groups = None
    if args.primary_technical_share is not None:
        primary_groups = {name: torch.tensor(np.flatnonzero(train.categories == category), device="cuda")
                          for name, category in (("chat", 1), ("technical", 2))}
        if any(len(group) == 0 for group in primary_groups.values()) or sum(map(len, primary_groups.values())) != len(train):
            raise ValueError("explicit primary shares require chat and technical categories only")
    ranked, ranked_priors = None, None
    if args.ranking_data:
        pairs, priors = [], []
        with args.ranking_data.open() as source:
            for line in source:
                row = json.loads(line)
                if row["split"] != "train" or protection.contains(row):
                    raise ValueError("held-out example in ranking data")
                if row["negative"] in row["gold"]:
                    raise ValueError("ranking negative is an accepted answer")
                try:
                    encoded = [encode_candidate(row, text, index, window)
                               for text in (row["gold"][0], row["negative"])]
                except ValueError:
                    exclusions["ranking/invalid"] += 1
                    continue
                pairs.append([(x + [0]*(window-len(x)), y + [0]*(window-len(y))) for x, y in encoded])
                priors.append(row["priors"])
        if not pairs:
            raise ValueError("no encodable ranking pairs")
        ranked = torch.tensor(np.array(pairs, dtype=np.int64), device="cuda")
        ranked_priors = torch.tensor(priors, device="cuda")
    model = model.cuda()
    reference = None
    if args.reference_weight:
        reference, reference_vocab, reference_metadata = load_model(args.reference_model)
        if reference_vocab != vocab or reference_metadata.get("conditional") != FORMAT:
            raise ValueError("reference token semantics must match the conditional model")
        if reference.cfg.context < window:
            raise ValueError("reference window is too short")
        reference = reference.cuda().eval().requires_grad_(False)
    # Architecture initialization consumes random numbers. Reset the sampling
    # stream so same-seed capacity controls receive the same training batches.
    torch.cuda.manual_seed_all(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1,
                                 fused=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    settings = {**vars(args), "data": str(args.data), "out": str(args.out),
                "encoded_cache": str(args.encoded_cache),
                "ranking_data": str(args.ranking_data) if args.ranking_data else None,
                "ranking_pairs": len(ranked) if ranked is not None else 0,
                "exclude_cases": {str(p): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                                  for p in args.exclude_cases},
                "extra_data": str(args.extra_data) if args.extra_data else None,
                "base_sha256": hashlib.file_digest(open(args.base, "rb"), "sha256").hexdigest(),
                "training_examples": len(train), "validation_examples": len(dev), "data_manifest": manifest,
                "extra_training_examples": len(extra_train) if extra_train is not None else 0,
                "extra_validation_examples": len(extra_dev) if extra_dev is not None else 0,
                "exclusions": dict(exclusions), "format": FORMAT, "config": dataclasses.asdict(model.cfg)}
    settings["compositions"] = dict(compositions)
    if args.reference_model:
        with open(args.reference_model, "rb") as source:
            settings["reference_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    settings["parameters"] = sum(p.numel() for p in model.parameters())
    settings["depth_growth"] = dict(original_layers=original_layers,
                                  retained_positions=[i * model.cfg.n_layer // original_layers
                                                      for i in range(original_layers)],
                                  initialization="zero residual output projections")
    if args.extra_data:
        settings["extra_manifest"] = json.loads((args.extra_data / "manifest.json").read_text())
    if args.ranking_data:
        settings["ranking_sha256"] = hashlib.file_digest(args.ranking_data.open("rb"), "sha256").hexdigest()
        settings["ranking_objective"] = dict(context_anchor=2, prior_weight=args.ranking_prior_weight, margin=.2,
                                             temperature=.1, score="mean new-character log probability; no EOS")
    (args.out / "run.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    start, best = time.monotonic(), float("inf")

    seen = {id(data):torch.zeros(len(data), dtype=torch.bool, device="cuda")
            for data in (train, extra_train) if data is not None}

    def batch(data, n):
        if data is train and primary_groups is not None:
            technical = round(n * args.primary_technical_share)
            indices = torch.cat([group[torch.randint(len(group), (count,), device="cuda")]
                                 for group, count in ((primary_groups["chat"], n-technical),
                                                      (primary_groups["technical"], technical))])
        else:
            indices = torch.randint(len(data), (n,), device="cuda")
        if id(data) in seen:
            seen[id(data)][indices] = True
        return data.batch(indices.cpu().numpy(), "cuda")

    def loss(x, y, preserve=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, labels = supervised_logits(model, x, y)
            value = F.cross_entropy(logits.float(), labels)
            if preserve and reference is not None:
                with torch.no_grad():
                    reference_logits, _ = supervised_logits(reference, x, y)
                value = value + args.reference_weight * reference_kl(logits, reference_logits, labels)
            return value

    def validate(step):
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(42)
            primary = sum(loss(*batch(dev, 64)).item() for _ in range(16)) / 16
            extra = sum(loss(*batch(extra_dev, 64)).item() for _ in range(16)) / 16 if extra_dev is not None else None
        combined = (1 - args.extra_share) * primary + args.extra_share * extra if extra is not None else primary
        print(json.dumps(dict(step=step, validation_loss=combined,
                              primary_validation=primary, extra_validation=extra)), flush=True)
        return combined

    validate(0)

    for step in range(1, args.steps + 1):
        model.train()
        lr = args.learning_rate * min(1, step / 100) * (0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        # Deterministic batch-level domain mixture. Do not physically duplicate
        # technical rows, and retain the actual sampling share in run.json.
        domain = extra_train if extra_train is not None and np.random.random() < args.extra_share else train
        value = loss(*batch(domain, args.batch), preserve=domain is train)
        rank_value = None
        if ranked is not None:
            # Draw in the CE control too, preserving identical CE batch RNG.
            choices = torch.randint(len(ranked), (args.ranking_batch,), device="cuda")
            if args.ranking_weight:
                pair_batch = ranked[choices].flatten(0, 1)
                length = int((pair_batch[:, 0] != 0).sum(1).max())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(pair_batch[:, 0, :length])
                    scores = candidate_scores(logits, pair_batch[:, 1, :length])
                    rank_value = preference_loss(scores, ranked_priors[choices], args.ranking_prior_weight)
                value = value + args.ranking_weight * rank_value
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == 1:
            print(json.dumps(dict(step=step, loss=value.item(),
                                  ranking_loss=rank_value.item() if rank_value is not None else None,
                                  seconds=time.monotonic() - start)), flush=True)
        if step % 500 == 0 or step == args.steps or (args.save_every and step % args.save_every == 0):
            # Fixed validation batches; held-out tests only filter leakage.
            validation = validate(step)
            improved = validation < best
            best = min(best, validation)
            checkpoint = args.save_every and (step % args.save_every == 0 or step == args.steps)
            if improved or checkpoint:
                tensors = {k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items()
                           if k != "head.weight"}
                info = dict(metadata, config=json.dumps(dataclasses.asdict(model.cfg)),
                            vocab=json.dumps(vocab, ensure_ascii=False), precision="f16", conditional=FORMAT,
                            validation_loss=str(validation), training_steps=str(step),
                            ranking_weight=str(args.ranking_weight),
                            single_character_training=str(any(key.endswith("/length_1") and count
                                                              for key, count in compositions.items())).lower(),
                            reference_retention=str(args.reference_weight),
                            attribution=metadata.get("attribution", "") + " Continued training; corpus manifests in run.json."
                            + (" Additional corpus attribution and licenses: "
                               + json.dumps(settings["extra_manifest"], ensure_ascii=False)
                               if args.extra_data else ""))
                if improved:
                    temporary = args.out / "model.safetensors.part"
                    save_file(tensors, str(temporary), metadata=info)
                    os.replace(temporary, args.out / "model.safetensors")
                if checkpoint:
                    target = args.out / f"step-{step}.safetensors"
                    temporary = target.with_suffix(".part")
                    save_file(tensors, str(temporary), metadata=info)
                    os.replace(temporary, target)
    settings["seen_training_rows"] = {"primary":int(seen[id(train)].sum()),
                                      "extra":int(seen[id(extra_train)].sum()) if extra_train is not None else 0}
    if primary_groups is not None:
        settings["seen_primary_categories"] = {name:int(seen[id(train)][indices].sum())
                                                for name, indices in primary_groups.items()}
    (args.out / "run.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(best_validation_loss=best, seconds=time.monotonic() - start)), flush=True)


if __name__ == "__main__":
    main()
