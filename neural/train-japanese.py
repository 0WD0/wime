#!/usr/bin/env python3
"""Train a kana-conditioned Japanese character LM from scratch.

Row: <bos> context <input> kana <output> text <eos>; loss on text and <eos>
only, as the Chinese conditional model. Sources are sampled by weight, so a
large Wikipedia does not drown the conversational Tatoeba rows.
"""
import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/chinese-ime-lm/neural"))
from model import CharLM, Config, RESERVED  # noqa: E402

from japanese import EOS, FORMAT, INPUT, OUTPUT  # noqa: E402

WEIGHTS = dict(tatoeba=0.2, aozora=0.3, wikipedia=0.4, wikinews=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--vocab", type=int, default=12288)
    ap.add_argument("--cap", type=int, default=3_500_000, help="rows kept per source, sampled uniformly")
    args = ap.parse_args()
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    rows = collections.defaultdict(list)
    counts = collections.Counter()
    total = json.loads((args.data / "counts.json").read_text())
    keep = {s: min(1.0, args.cap / total[s + "/train"]) for s in WEIGHTS}
    sampler = random.Random(3)
    with open(args.data / "train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if sampler.random() >= keep[r["source"]]:
                continue
            rows[r["source"]].append(r)
            for c in r["text"] + r["kana"]:
                counts[c] += 1
    vocab = RESERVED + [INPUT, OUTPUT, EOS] + [c for c, _ in counts.most_common(args.vocab - 6)]
    index = {c: i for i, c in enumerate(vocab)}
    print("rows", {k: len(v) for k, v in rows.items()}, "vocab", len(vocab),
          "coverage %.5f" % (sum(counts[c] for c in vocab[6:]) / sum(counts.values())), flush=True)

    lut = np.ones(0x110000, dtype=np.int32)
    for c, i in index.items():
        if len(c) == 1:
            lut[ord(c)] = i
    bos = "\x02"
    lut[ord(bos)] = 2

    def pack(items, keep_context):
        """One code-point lookup over all rows' concatenated strings."""
        parts, offsets, starts = [], [0], []
        for r in items:
            body = INPUT + r["kana"] + OUTPUT + r["text"] + EOS
            room = args.window - 1 - len(body)
            if room < 0:
                continue
            context = r["context"][-room:] if keep_context and room else ""
            parts.append(bos + context + body)
            offsets.append(offsets[-1] + len(parts[-1]))
            starts.append(1 + len(context) + len(r["kana"]) + 2)
        codes = np.frombuffer("".join(parts).encode("utf-32-le"), dtype=np.uint32)
        return lut[codes], np.array(offsets, dtype=np.int64), np.array(starts, dtype=np.int32)

    packed = {}
    for source, items in rows.items():
        # A third of rows without context, so short or fresh compositions are covered too.
        random.Random(2).shuffle(items)
        cut = len(items) // 3
        a, b = pack(items[:cut], False), pack(items[cut:], True)
        offsets = np.concatenate([a[1], b[1][1:] + len(a[0])])
        packed[source] = (np.concatenate([a[0], b[0]]), offsets, np.concatenate([a[2], b[2]]))
        print(source, "packed", len(offsets) - 1, flush=True)
    del rows
    valid = [json.loads(l) for l in open(args.data / "valid.jsonl")][:20000]
    vx = pack(valid, True)

    def batch(source_arrays, n, generator):
        flat, offsets, starts = source_arrays
        pick = generator.integers(0, len(offsets) - 1, n)
        x = np.zeros((n, args.window), dtype=np.int64); y = np.zeros((n, args.window), dtype=np.int64)
        for k, i in enumerate(pick):
            t = flat[offsets[i]:offsets[i + 1]]
            x[k, :len(t) - 1] = t[:-1]
            s = starts[i]
            y[k, s - 1:len(t) - 1] = t[s:]
        return torch.from_numpy(x).cuda(non_blocking=True), torch.from_numpy(y).cuda(non_blocking=True)

    sources = list(packed)
    weights = np.array([WEIGHTS[s] for s in sources]); weights /= weights.sum()
    cfg = Config(vocab=len(vocab), n_layer=8, n_head=8, n_embd=448, context=args.window, dropout=0.0)
    model = CharLM(cfg).cuda()
    print("parameters %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    args.out.mkdir(parents=True, exist_ok=True)

    def evaluate():
        model.eval()
        losses, generator = [], np.random.default_rng(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(20):
                x, y = batch(vx, args.batch, generator)
                losses.append(float(model(x, y)[1]))
        model.train()
        return sum(losses) / len(losses)

    def save(name):
        tensors = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items() if k != "head.weight"}
        meta = dict(vocab=json.dumps(vocab, ensure_ascii=False), config=json.dumps(cfg.__dict__),
                    conditional=FORMAT)
        save_file(tensors, str(args.out / f"{name}.safetensors"), metadata=meta)

    start = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1, step / 1000) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in optimizer.param_groups:
            g["lr"] = lr
        counts_per = rng.multinomial(args.batch, weights)
        xs, ys = zip(*[batch(packed[s], int(n), rng) for s, n in zip(sources, counts_per) if n])
        x, y = torch.cat(xs), torch.cat(ys)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)[1]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.3f} lr {lr:.2e} {time.time() - start:.0f}s", flush=True)
        if step % 5000 == 0 or step == args.steps:
            print(f"step {step} valid {evaluate():.4f}", flush=True)
            save(f"step-{step}")


if __name__ == "__main__":
    main()
