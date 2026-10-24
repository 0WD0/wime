#!/usr/bin/env python3
"""Post-prediction attribution on paired frozen results; never a search policy.

Score the union of observed pools/drafts and dictionary-reachable labels under
both checkpoints. This separates missed search opportunities from preferences
that wider search cannot fix. Gold is used ONLY by this diagnostic program.
"""
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
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "middle", "after", "old-model", "new-model", "database",
                 "grammar-dir", "grammar-module", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    files = [args.before, args.middle, args.after]
    stages = [{x["id"]: x for x in json.loads(f.read_text())} for f in files]
    old, middle, after = stages
    if not all(set(s) == set(old) for s in stages):
        raise ValueError("different case sets")
    for key, row in after.items():
        for previous in (old[key], middle[key]):
            if any(previous[k] != row[k] for k in ("input", "context", "gold", "pool")):
                # Neural scores may differ, but the dictionary pool must not.
                if (any(previous[k] != row[k] for k in ("input", "context", "gold")) or
                        [(c["text"], c["prior"]) for c in previous["pool"]] !=
                        [(c["text"], c["prior"]) for c in row["pool"]]):
                    raise ValueError("changed test or dictionary pool")
    selected = [x for x in after.values() if old[x["id"]]["constrained"] in x["gold"]
                and x["constrained"] not in x["gold"]]
    env = dict(os.environ, WIME_DATABASE=str(args.database), WIME_GRAMMAR=str(args.grammar_dir),
               WIME_GRAMMAR_MODULE=str(args.grammar_module))
    server = subprocess.Popen(["emacs", "-Q", "--batch", "-L", str(ROOT / "emacs-packages/wime"),
                               "--eval", "(setq load-prefer-newer t)", "-l",
                               str(ROOT / "emacs-packages/wime/test/search-conditional.el")],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    evaluators = {"old": ConditionalEvaluator(str(args.old_model)),
                  "new": ConditionalEvaluator(str(args.new_model))}
    counts, rows = Counter(), []
    try:
        for row in selected:
            key = row["id"]
            priors = {}
            for stage in stages:
                for c in stage[key]["pool"] + stage[key]["trace"]:
                    priors[c["text"]] = max(priors.get(c["text"], -float("inf")), c["prior"])
            for target in row["gold"]:
                server.stdin.write(json.dumps(dict(input=row["input"], context=row["context"],
                                                   prefix=target, width=16)) + "\n")
                server.stdin.flush()
                answer = json.loads(server.stdout.readline())
                for c in answer["candidates"]:
                    if c["text"] == target and c["end"] == answer["syllables"]:
                        priors[target] = max(priors.get(target, -float("inf")),
                                             c["score"] / answer["syllables"])
            texts = sorted(priors)
            scores = {}
            for name, model, anchor in (("old", "old", 0), ("trained", "new", 0), ("anchored", "new", 2)):
                values, _ = evaluators[model].evaluate(answer["context"], answer["input"], texts,
                                                       context_anchor=anchor)
                scores[name] = {t: dict(neural=s, joint=s + .1 * priors[t]) for t, s in zip(texts, values)}
            reachable = [t for t in row["gold"] if t in priors]
            if not reachable:
                raise ValueError("old-correct label became unreachable")
            target = max(reachable, key=lambda t: scores["anchored"][t]["joint"])
            prediction = row["constrained"]
            gold_score, pred_score = [scores["anchored"][t] for t in (target, prediction)]
            kind = ("search_missed_higher_scoring_gold" if gold_score["joint"] > pred_score["joint"] + 1e-5
                    else "prior_overrode_neural_preference" if gold_score["neural"] > pred_score["neural"]
                    else "neural_preferred_wrong_answer")
            anchor_only = middle[key]["constrained"] in row["gold"]
            visited = any(x["text"] in row["gold"] for x in row["trace"])
            item = {k: row[k] for k in ("id", "input", "context", "gold", "category", "syllables")}
            item.update(before=old[key]["constrained"], trained=middle[key]["constrained"],
                        after=prediction, mechanism=kind, anchor_only=anchor_only,
                        gold_in_pool=row["gold_in_beam"], gold_visited=visited,
                        joint_margin={s: scores[s][target]["joint"] - scores[s][prediction]["joint"] for s in scores},
                        union_best={s: max(texts, key=lambda t:scores[s][t]["joint"]) for s in scores},
                        candidates=[dict(text=t, prior=priors[t], **{s:scores[s][t] for s in scores}) for t in texts])
            rows.append(item)
            counts.update(total=1, **{kind: 1, "category/" + row["category"]: 1},
                          anchor_only=anchor_only, gold_in_pool=row["gold_in_beam"], gold_visited=visited)
            if len(rows) % 25 == 0:
                print(json.dumps(counts), flush=True)
    finally:
        server.stdin.close()
        server.wait(timeout=15)
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = dict(counts=counts, scope="diagnostic only; union contains labels, never a deployable accuracy",
                    hashes={str(f): hashlib.file_digest(f.open("rb"), "sha256").hexdigest()
                            for f in files + [args.old_model, args.new_model]})
    (args.out / "cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
