#!/usr/bin/env python3
"""Score a checkpoint against previously frozen plaintext development pools."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from composition import ENCODING
from inference import ConditionalEvaluator


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--pools", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = json.loads(args.pools.read_text())
    model = ConditionalEvaluator(str(args.model))
    priors = (0.0, .025, .05, .1, .2)
    counts = {w:Counter() for w in priors}
    for i, row in enumerate(rows):
        if row["split"] != "dev" or row["domain"] not in ("old", "new"):
            raise ValueError("this calibration accepts development examples only")
        if not re.fullmatch(r"(?:[a-z;]{2})+", row["input"]):
            raise ValueError("expected complete plain phonetic inputs")
        pool = row["pool"]
        # Pools were generated in editable temporary buffers with the default
        # Wime context policy: no properties, fields, narrowing or prompts.
        scores, _ = model.evaluate(row["context"][-64:], row["input"], [c["text"] for c in pool])
        for c, s in zip(pool, scores):
            c["candidate_score"] = s
        for weight in priors:
            best = max(pool, key=lambda c:c["candidate_score"] + weight * c["prior"])
            good = best["text"] in row["gold"]
            counts[weight].update(total=1, correct=good)
            for group in (row["domain"], row["domain"] + "/" + row["category"]):
                counts[weight][group + "/total"] += 1
                counts[weight][group + "/correct"] += good
        if (i + 1) % 200 == 0:
            print(json.dumps(dict(cases=i + 1)), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(model=dict(path=str(args.model), sha256=digest(args.model)),
                  pools=dict(path=str(args.pools), sha256=digest(args.pools)), encoding=ENCODING,
                  grid=[dict(prior_weight=w, counts=c,
                             objective=sum(c[d + "/correct"] / c[d + "/total"] for d in ("old", "new")) / 2)
                        for w, c in counts.items()])
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.out / "results.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
