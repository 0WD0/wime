#!/usr/bin/env python3
"""Fine-tune a released character LM on output-only conditional loss."""
import argparse
from collections import Counter
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from conditional import composition_span, encode_example, load_model, FORMAT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--extra-data", type=Path)
    p.add_argument("--extra-share", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--span-share", type=float, default=0.0,
                   help="fraction of training rows replaced by 2-5 character composition spans")
    args = p.parse_args()
    if (args.steps < 1 or args.batch < 1 or not 0 <= args.extra_share <= 1
            or not 0 <= args.span_share <= 1):
        p.error("steps/batch must be positive; extra-share and span-share must be in [0, 1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, vocab, metadata = load_model(args.base, extend=True)
    model.cfg.dropout = 0.0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    index = {c: i for i, c in enumerate(vocab)}
    window = model.cfg.context
    args.out.mkdir(parents=True, exist_ok=True)
    exclusions = Counter()
    compositions = Counter()

    def pack(split, directory=args.data):
        rows = []
        span_rng = random.Random(args.seed)
        with (directory / (split + ".jsonl")).open() as f:
            for line in f:
                row = json.loads(line)
                # Documents retain their existing split. Common short words can
                # occur in different splits; no held-out document is trained on.
                # Full dev examples remain unchanged for checkpoint selection.
                if split == "train" and span_rng.random() < args.span_share:
                    length = span_rng.randint(2, min(5, len(row["gold"][0])))
                    start = span_rng.randint(0, len(row["gold"][0]) - length)
                    row = composition_span(row, start, start + length)
                    compositions[str(directory) + "/short_span"] += 1
                try:
                    x, y = encode_example(row, index, window)
                except ValueError:
                    exclusions[str(directory) + "/" + split + "/window"] += 1
                    continue
                if 1 in y:  # record exclusions rather than silently train unknown targets
                    exclusions[str(directory) + "/" + split + "/unknown_target"] += 1
                    continue
                rows.append((x + [0] * (window - len(x)), y + [0] * (window - len(y))))
                if args.limit and split == "train" and len(rows) >= args.limit:
                    break
        if not rows:
            raise ValueError(f"{directory}/{split}: no encodable examples")
        data = torch.tensor(np.array(rows, dtype=np.int64))
        return data.to("cuda")

    train, dev = pack("train"), pack("dev")
    extra_train = pack("train", args.extra_data) if args.extra_data else None
    extra_dev = pack("dev", args.extra_data) if args.extra_data else None
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1,
                                 fused=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    settings = {**vars(args), "data": str(args.data), "out": str(args.out),
                "extra_data": str(args.extra_data) if args.extra_data else None,
                "base_sha256": hashlib.file_digest(open(args.base, "rb"), "sha256").hexdigest(),
                "training_examples": len(train), "validation_examples": len(dev), "data_manifest": manifest,
                "exclusions": dict(exclusions), "format": FORMAT, "config": dataclasses.asdict(model.cfg)}
    settings["compositions"] = dict(compositions)
    if args.extra_data:
        settings["extra_manifest"] = json.loads((args.extra_data / "manifest.json").read_text())
    (args.out / "run.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    start, best = time.monotonic(), float("inf")

    def batch(data, n):
        b = data[torch.randint(len(data), (n,), device="cuda")]
        # Right padding is ignored; truncate to the longest example in this batch.
        length = int((b[:, 0] != 0).sum(1).max())
        return b[:, 0, :length], b[:, 1, :length]

    def loss(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            return F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), ignore_index=0)

    for step in range(1, args.steps + 1):
        model.train()
        lr = 2e-4 * min(1, step / 100) * (0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        # Deterministic batch-level domain mixture. Do not physically duplicate
        # technical rows, and retain the actual sampling share in run.json.
        domain = extra_train if extra_train is not None and np.random.random() < args.extra_share else train
        value = loss(*batch(domain, args.batch))
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == 1:
            print(json.dumps(dict(step=step, loss=value.item(), seconds=time.monotonic() - start)), flush=True)
        if step % 500 == 0 or step == args.steps:
            model.eval()
            # Fixed validation batches; model selection never reads test.jsonl.
            with torch.no_grad(), torch.random.fork_rng(devices=[0]):
                torch.manual_seed(42)
                validation = sum(loss(*batch(dev, 64)).item() for _ in range(16)) / 16
                if extra_dev is not None:
                    technical_validation = sum(loss(*batch(extra_dev, 64)).item() for _ in range(16)) / 16
                    validation = (1 - args.extra_share) * validation + args.extra_share * technical_validation
            print(json.dumps(dict(step=step, validation_loss=validation)), flush=True)
            if validation < best:
                best = validation
                tensors = {k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items()
                           if k != "head.weight"}
                info = dict(metadata, config=json.dumps(dataclasses.asdict(model.cfg)),
                            vocab=json.dumps(vocab, ensure_ascii=False), precision="f16", conditional=FORMAT,
                            validation_loss=str(best), training_steps=str(step),
                            attribution=metadata.get("attribution", "") + " Fine-tuned on LCCC (MIT), manifest in run.json."
                            + (" The Kubernetes Authors, Chinese documentation (CC-BY-4.0), "
                               + settings["extra_manifest"]["source"] + ", revision "
                               + settings["extra_manifest"]["revision"] + "; modified into pinyin/text training pairs."
                               if args.extra_data else ""))
                save_file(tensors, str(args.out / "model.safetensors"), metadata=info)
    print(json.dumps(dict(best_validation_loss=best, seconds=time.monotonic() - start)), flush=True)


if __name__ == "__main__":
    main()
