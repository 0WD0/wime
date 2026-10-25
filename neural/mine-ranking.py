#!/usr/bin/env python3
"""Mine same-key hard negatives from the real Wime lattice on TRAIN only."""
import argparse
from collections import Counter, defaultdict
import heapq
import json
import os
from pathlib import Path
import random
import subprocess
import time

from conditional import ROOT
from inference import ConditionalEvaluator
from replay import Protection, fingerprint, rows
from ranking import ranking_composition, spelling_equivalent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "replay", "diverse", "database", "grammar-dir", "grammar-module", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protect", type=Path, action="append", required=True)
    parser.add_argument("--per-domain", type=int, default=6000)
    parser.add_argument("--width", type=int, default=16)
    args = parser.parse_args()
    protection = Protection(args.protect)
    heaps = defaultdict(list)
    counts = Counter()
    # A bounded, reproducible mixture; neither held-out failures nor their
    # target strings influence which training examples get mined.
    import hashlib
    for name, path in (("old", args.replay), ("new", args.diverse)):
        for row in rows(path):
            if row["split"] != "train":
                raise ValueError("cannot mine a held-out row")
            if protection.contains(row):
                counts["protected_source"] += 1
                continue
            rng = random.Random(row["id"])
            row = ranking_composition(row, rng)
            # A bare one-character homophone has no reliable intended reading.
            if len(row["gold"][0]) == 1 and not row["context"]:
                continue
            if protection.contains(row):
                counts["protected_span"] += 1
                continue
            category = row.get("category", "chat")
            row = dict(row, domain=name)
            key = -int(hashlib.sha256(row["id"].encode()).hexdigest(), 16)
            item = (key, row["id"], row)
            heap = heaps[name, category]
            limit = args.per_domain // (2 if name == "old" else 5)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif key > heap[0][0]:
                heapq.heapreplace(heap, item)
    samples = sorted((item[2] for heap in heaps.values() for item in heap), key=lambda row:row["id"])
    env = dict(os.environ, WIME_DATABASE=str(args.database), WIME_GRAMMAR=str(args.grammar_dir),
               WIME_GRAMMAR_MODULE=str(args.grammar_module))
    server = subprocess.Popen(["emacs", "-Q", "--batch", "-L", str(ROOT / "emacs-packages/wime"),
                               "--eval", "(setq load-prefer-newer t)", "-l",
                               str(ROOT / "emacs-packages/wime/test/search-conditional.el")],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    evaluator = ConditionalEvaluator(str(args.model), context_anchor=2)
    args.out.mkdir(parents=True, exist_ok=True)

    def search(row, prefix=""):
        server.stdin.write(json.dumps(dict(input=row["input"], context=row["context"],
                                          prefix=prefix, width=args.width)) + "\n")
        server.stdin.flush()
        line = server.stdout.readline()
        if not line:
            raise RuntimeError("dictionary worker exited")
        return json.loads(line)

    start = time.monotonic()
    try:
        with (args.out / "train.jsonl").open("w") as output:
            for i, row in enumerate(samples):
                answer = search(row)
                n = answer["syllables"]
                candidates = {c["text"]:c["score"]/n for c in answer["candidates"]
                              if c["end"] == n and len(c["text"]) == n}
                gold = row["gold"][0]
                if gold not in candidates:
                    oracle = search(row, gold)
                    matching = [c for c in oracle["candidates"] if c["text"] == gold and c["end"] == n]
                    if not matching:
                        counts["unreachable"] += 1
                        continue
                    candidates[gold] = matching[0]["score"]/n
                negatives = [c for c in candidates if not any(spelling_equivalent(c, g) for g in row["gold"])]
                if not negatives:
                    counts["no_negative"] += 1
                    continue
                texts = [gold] + negatives
                # Use exactly the context accepted by the editable-buffer policy.
                row = dict(row, context=answer["context"], input=answer["input"])
                scores, _ = evaluator.evaluate(row["context"], row["input"], texts)
                joint = [s + .1*candidates[t] for s, t in zip(scores, texts)]
                hard = max(range(1, len(texts)), key=lambda j:joint[j])
                gap = joint[0] - joint[hard]
                if gap > .5:
                    counts["easy"] += 1
                    continue
                # Both strings have been returned for these EXACT typed keys.
                pair = dict(row, negative=texts[hard], priors=[candidates[gold], candidates[texts[hard]]],
                            initial_margin=gap)
                output.write(json.dumps(pair, ensure_ascii=False) + "\n")
                counts["pairs"] += 1
                counts["domain/"+row["domain"]] += 1
                counts["category/"+row.get("category", "chat")] += 1
                counts["length/"+str(n)] += 1
                counts["already_wrong"] += gap <= 0
                if (i+1) % 500 == 0:
                    print(json.dumps(dict(visited=i+1, counts=counts, seconds=time.monotonic()-start)), flush=True)
    finally:
        server.stdin.close()
        server.wait(timeout=15)
    manifest = dict(counts=counts, sampled=len(samples), seconds=time.monotonic()-start,
                    source_files={str(p):fingerprint(p) for p in (args.model, args.replay, args.diverse, args.database)},
                    protected_files=protection.files, width=args.width, context_anchor=2,
                    compositions="whole or committed-prefix suffix; no hidden continuation",
                    prior_weight=.1, max_initial_margin=.5,
                    sha256=fingerprint(args.out / "train.jsonl"))
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
