#!/usr/bin/env python3
"""Select committed-prefix length on dev rows, scoring the same real pool."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

from conditional import ROOT
from inference import ConditionalEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--grammar-dir", required=True)
    parser.add_argument("--grammar-module", required=True)
    args = parser.parse_args()
    rows = []
    for directory, count in ((args.data, 128), (args.replay, 192)):
        source = [json.loads(l) for l in (directory / "dev.jsonl").read_text().splitlines()]
        for category in sorted({r["category"] for r in source}):
            rows += sorted((r for r in source if r["category"] == category),
                           key=lambda r:hashlib.sha256(r["id"].encode()).digest())[:count]
    env = dict(os.environ, WIME_DATABASE=args.database, WIME_GRAMMAR=args.grammar_dir,
               WIME_GRAMMAR_MODULE=args.grammar_module)
    server = subprocess.Popen(["emacs", "-Q", "--batch", "-L", str(ROOT / "emacs-packages/wime"),
                               "--eval", "(setq load-prefer-newer t)", "-l",
                               str(ROOT / "emacs-packages/wime/test/search-conditional.el")],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    evaluator = ConditionalEvaluator(args.model)
    stats, results = {n:Counter() for n in (0, 2, 4, 8)}, []
    try:
        for i, row in enumerate(rows):
            server.stdin.write(json.dumps(dict(input=row["input"], context=row["context"], width=16)) + "\n")
            server.stdin.flush()
            answer = json.loads(server.stdout.readline())
            n = answer["syllables"]
            candidates = [c for c in answer["candidates"] if c["end"] == n]
            predictions = {}
            for anchor, counts in stats.items():
                if not candidates:
                    continue
                scores, _ = evaluator.evaluate(answer["context"], answer["input"],
                                                [c["text"] for c in candidates], context_anchor=anchor)
                best = max(range(len(scores)), key=lambda j:scores[j] + .1 * candidates[j]["score"] / n)
                prediction = candidates[best]["text"]
                predictions[anchor] = prediction
                correct = prediction in row["gold"]
                counts.update(total=1, correct=correct, single_total=n == 1,
                              single_correct=n == 1 and correct)
                counts[row["category"] + "/correct"] += correct
                counts[row["category"] + "/total"] += 1
            results.append(dict(row, predictions=predictions))
            if (i + 1) % 100 == 0:
                print(json.dumps(dict(cases=i+1, stats=stats)), flush=True)
    finally:
        server.stdin.close()
        server.wait(timeout=15)
    selected = max(stats, key=lambda k:(stats[k]["correct"], -k))
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(selected=selected, stats=stats, scope="dev-only same-pool N-best; 60/40 new/replay strata",
                  model_sha256=hashlib.file_digest(open(args.model, "rb"), "sha256").hexdigest(),
                  source_hashes={str(p):hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                                 for p in (args.data / "dev.jsonl", args.replay / "dev.jsonl")})
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
