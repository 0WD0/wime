#!/usr/bin/env python3
"""Measure a conditional checkpoint's standalone key-to-text accuracy and latency."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from conditional import load_model
from conditional_decode import ConditionalDecoder
from replay import fingerprint
from dataset_registry import Registry, CATALOG


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        n = len(row["gold"][0])
        size = "1" if n == 1 else "2-3" if n <= 3 else "4-6" if n <= 6 else "7+"
        # Legacy technical rows use `source` for an individual filename.
        # Categories remain comparable across documents in both corpus formats.
        for key in ("all", "category/" + row.get("category", row.get("source", "unknown")),
                    "length/" + size, "context/" + ("present" if row["context"] else "blank")):
            groups[key].append(row)
    return {key: dict(cases=len(group), top1=sum(r["correct"] for r in group),
                      top5=sum(r["top5"] for r in group), reachable=sum(r["reachable"] for r in group),
                      character_accuracy=sum(r["correct_characters"] for r in group) /
                                         sum(len(r["gold"][0]) for r in group),
                      median_ms=float(np.median([r["milliseconds"] for r in group])),
                      p95_ms=float(np.percentile([r["milliseconds"] for r in group], 95)))
            for key, group in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    cases = parser.add_mutually_exclusive_group(required=True)
    cases.add_argument("--cases", type=Path, help="unregistered JSON case array")
    cases.add_argument("--dataset", help="versioned evaluation-suite ID")
    parser.add_argument("--registry", type=Path, default=CATALOG)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dataset = None
    if args.dataset:
        args.cases, dataset = Registry(args.registry).evaluation(args.dataset)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, vocab, metadata = load_model(args.model)
    decoder = ConditionalDecoder.from_database(model.to(args.device), vocab, args.database, args.beam)
    cases = json.loads(args.cases.read_text())
    for row in cases[:5]:
        decoder.decode(row["context"], row["input"], row.get("committed"))
    legal = {code: set(ids.tolist()) for code, ids in decoder.legal.items()}
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".jsonl").open("w") as stream:
        for row in cases:
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            candidates = decoder.decode(row["context"], row["input"], row.get("committed"))
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            milliseconds = 1000 * (time.perf_counter() - start)
            texts = [text for text, score in candidates]
            first = texts[0] if texts else ""
            reachable = any(len(gold) * 2 == len(row["input"]) and all(
                decoder.index.get(char) in legal.get(row["input"][2*i:2*i+2], set())
                for i, char in enumerate(gold)) for gold in row["gold"])
            result = dict(row, candidates=candidates, milliseconds=milliseconds, correct=first in row["gold"],
                          top5=any(t in row["gold"] for t in texts[:5]), reachable=reachable,
                          correct_characters=max(sum(a == b for a, b in zip(first, gold)) for gold in row["gold"]))
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            if len(results) % 100 == 0:
                print(json.dumps(dict(completed=len(results), total=len(cases))), flush=True)
    report = dict(model=str(args.model), model_sha256=fingerprint(args.model),
                  dataset=dataset,
                  cases_sha256=fingerprint(args.cases), database_sha256=fingerprint(args.database),
                  code_sha256={name: fingerprint(Path(__file__).with_name(name))
                               for name in ("evaluate-direct.py", "conditional_decode.py", "conditional.py")},
                  parameters=sum(p.numel() for p in model.parameters()), vocabulary=len(vocab),
                  beam=args.beam, precision="float32, TF32 disabled", training_steps=metadata.get("training_steps"),
                  scores="sum full-vocabulary character log probabilities; no EOS or priors",
                  metrics=summarize(results))
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["metrics"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
