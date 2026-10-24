#!/usr/bin/env python3
"""Select a traditional-score weight from development-only search traces.

The weight changes only final selection among verified drafts, so it does not
require replaying search. Each category receives equal weight, regardless of
its size. Do not supply test or handcrafted diagnostic results for calibration.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(rows, weight):
    groups = defaultdict(list)
    for row in rows:
        trace = row["trace"]
        prediction = (max(trace, key=lambda t: t["neural"] + weight * t["prior"])["text"]
                      if trace else row["beam"])
        groups[row["category"]].append(prediction in row["gold"])
    categories = {k: dict(correct=sum(v), total=len(v), accuracy=sum(v) / len(v))
                  for k, v in groups.items()}
    return dict(weight=weight, categories=categories,
                macro_accuracy=sum(g["accuracy"] for g in categories.values()) / len(categories))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev-results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", type=float, nargs="+", default=[0, 0.025, 0.05, 0.1, 0.2, 0.5, 1])
    args = p.parse_args()
    rows = json.loads(args.dev_results.read_text())
    if not rows or any(not r.get("category") for r in rows) or min(args.weights) < 0:
        p.error("nonempty categorized development results and nonnegative weights required")
    grid = [summarize(rows, w) for w in sorted(set(args.weights))]
    result = dict(development_results=str(args.dev_results),
                  selected=max(grid, key=lambda g: g["macro_accuracy"]), grid=grid)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
