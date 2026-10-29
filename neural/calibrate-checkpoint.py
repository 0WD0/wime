#!/usr/bin/env python3
"""Dev-only weight retention and grammar calibration, with frozen real pools.

The new checkpoint descends from the old one. Linear interpolation retains a
fraction of that update without adding a second model to runtime inference.
Choose on equal-weight old/new-domain accuracy, requiring no old-dev loss
relative to the current alpha=1, prior=.1 profile. Never read frozen tests.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from conditional import ROOT
from composition import ENCODING
from inference import ConditionalEvaluator


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def interpolate(old_path, new_path, alpha, output):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    with safe_open(old_path, framework="pt") as old, safe_open(new_path, framework="pt") as new:
        for key in ("config", "vocab", "conditional"):
            if old.metadata()[key] != new.metadata()[key]:
                raise ValueError("checkpoint architectures, vocabularies or conditioning differ")
        if set(old.keys()) != set(new.keys()) or any(k.endswith(".scale") for k in old.keys()):
            raise ValueError("expected identical, unquantized checkpoints")
        tensors = {}
        for key in old.keys():
            a, b = old.get_tensor(key), new.get_tensor(key)
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError("incompatible tensors")
            tensors[key] = torch.lerp(a.float(), b.float(), alpha).to(b.dtype).contiguous()
        metadata = dict(new.metadata(), retained_update=str(alpha),
                        parent_checkpoints=json.dumps({str(p):digest(p)
                                                       for p in (old_path, new_path)}))
        save_file(tensors, str(output), metadata=metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("old-model", "new-model", "old-dev", "new-dev", "database", "grammar-dir", "grammar-module", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [dict(x, domain="old") for x in json.loads(args.old_dev.read_text())]
    new = [json.loads(l) for l in args.new_dev.read_text().splitlines()]
    for category in sorted({x["category"] for x in new}):
        rows += [dict(x, domain="new") for x in sorted((x for x in new if x["category"] == category),
                 key=lambda x:hashlib.sha256(x["id"].encode()).digest())[:128]]
    # Select the candidate grid before evaluating; ties favor smaller changes.
    alphas, priors = (0.5, 0.75, 1.0), (0.0, 0.025, 0.05, 0.1, 0.2)
    grid = {(a, w):Counter() for a in alphas for w in priors}
    models = {}
    for alpha in alphas:
        path = args.new_model if alpha == 1 else args.out / f"retain-{alpha}.safetensors"
        if alpha != 1:
            interpolate(str(args.old_model), str(args.new_model), alpha, path)
        models[alpha] = ConditionalEvaluator(str(path))
    env = dict(os.environ, WIME_DATABASE=str(args.database), WIME_GRAMMAR=str(args.grammar_dir),
               WIME_GRAMMAR_MODULE=str(args.grammar_module))
    server = subprocess.Popen(["emacs", "-Q", "--batch", "-L", str(ROOT / "emacs-packages/wime"),
                               "--eval", "(setq load-prefer-newer t)", "-l",
                               str(ROOT / "emacs-packages/wime/test/search-conditional.el")],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    results = []
    try:
        for row in rows:
            server.stdin.write(json.dumps(dict(input=row["input"], context=row["context"], prefix="", width=16)) + "\n")
            server.stdin.flush()
            answer = json.loads(server.stdout.readline())
            n = answer["syllables"]
            records = [c for c in answer["candidates"] if c["end"] == n]
            if not records:
                raise ValueError("dev example has no complete candidate")
            pool = [dict(text=c["text"], prior=c["score"] / n, neural={}) for c in records]
            for alpha, model in models.items():
                scores, _ = model.evaluate(answer["context"], answer["input"], [c["text"] for c in records])
                for c, s in zip(pool, scores):
                    c["neural"][str(alpha)] = s
                for weight in priors:
                    best = max(range(len(scores)), key=lambda j:scores[j] + weight * pool[j]["prior"])
                    good = pool[best]["text"] in row["gold"]
                    counter = grid[alpha, weight]
                    counter.update(total=1, correct=good)
                    for group in (row["domain"], row["domain"] + "/" + row["category"]):
                        counter[group + "/total"] += 1
                        counter[group + "/correct"] += good
            results.append(dict(row, pool=pool))
            if len(results) % 100 == 0:
                print(json.dumps(dict(cases=len(results))), flush=True)
    finally:
        server.stdin.close()
        server.wait(timeout=15)
    baseline = grid[1.0, .1]["old/correct"]
    eligible = [k for k, c in grid.items() if c["old/correct"] >= baseline]
    objective = lambda c: sum(c[d+"/correct"] / c[d+"/total"] for d in ("old", "new")) / 2
    selected = max(eligible, key=lambda k:(objective(grid[k]), k[0], -abs(k[1] - .1)))
    report = dict(selected=dict(retained_update=selected[0], prior_weight=selected[1]),
                  selection="mean of old/new domain accuracy; old-dev accuracy >= current; N-best only",
                  grid=[dict(retained_update=a, prior_weight=w, objective=objective(c), counts=c)
                        for (a, w), c in grid.items()],
                  hashes={str(f):digest(f)
                          for f in (args.old_model, args.new_model, args.old_dev, args.new_dev)})
    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.out / "cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
