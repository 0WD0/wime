#!/usr/bin/env python3
"""Compare octagram, conditioned N-best, and draft/verify/constrained search.

The Emacs process owns the real dictionary lattice. Gold labels are read only
for metrics, never sent to search or model. The optional CUDA worker shares this inference implementation; actual Emacs
CPU/GPU latency is measured separately.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from conditional import ROOT
from inference import ConditionalEvaluator




def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--database", required=True)
    p.add_argument("--grammar-dir", required=True)
    p.add_argument("--grammar-module", required=True)
    p.add_argument("--limit", type=int, default=1024)
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--seed-draft", choices=["traditional", "neural"], default="traditional",
                   help="use joint neural/prior scoring to choose the initial complete draft")
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--prior-weight", type=float, default=0.0)
    p.add_argument("--reference-baseline", type=Path,
                   help="abort if any octagram output differs from this prior results.json")
    p.add_argument("--audit", action="store_true",
                   help="record candidate scores and gold reachability after prediction, for failure analysis")
    args = p.parse_args()
    if args.width < 1 or args.budget < 1:
        p.error("width and budget must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.cases.read_text()
    cases = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(l) for l in raw.splitlines()]
    cases = cases[:args.limit]
    reference = ({r["id"]: r["octagram"] for r in json.loads(args.reference_baseline.read_text())}
                 if args.reference_baseline else None)
    env = dict(os.environ, WIME_DATABASE=str(Path(args.database).resolve()),
               WIME_GRAMMAR=args.grammar_dir, WIME_GRAMMAR_MODULE=str(Path(args.grammar_module).resolve()))
    server = subprocess.Popen(["emacs", "-Q", "--batch", "-L", str(ROOT / "emacs-packages/wime"),
                               "--eval", "(setq load-prefer-newer t)", "-l",
                               str(ROOT / "emacs-packages/wime/test/search-conditional.el")],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)

    def search(code, context, prefix="", width=None):
        width = args.width if width is None else width
        server.stdin.write(json.dumps(dict(input=code, context=context, prefix=prefix, width=width)) + "\n")
        server.stdin.flush()
        line = server.stdout.readline()
        if not line:
            raise RuntimeError("search process exited")
        return json.loads(line)

    evaluator = ConditionalEvaluator(args.model)
    stats, results, latencies = Counter(), [], []
    started = time.monotonic()
    try:
        for i, case in enumerate(cases):
            code, context, gold = case["input"], case.get("context", ""), case["gold"]
            base = search(code, context, width=1)
            beam = search(code, context)
            # The real input parser removes auxiliary codes only for the model;
            # every dictionary search still receives the complete typed input.
            model_code, syllables = beam["input"], beam["syllables"]
            model_context = beam["context"]
            records = [c for c in beam["candidates"] if c["end"] == syllables]
            candidates = [c["text"] for c in records]
            original = base["candidates"][0]["text"] if base["candidates"] else ""
            if reference is not None and original != reference[case["id"]]:
                raise RuntimeError(f"octagram reference mismatch for {case['id']}; check immutable artifacts")
            draft = candidates[0] if candidates else original
            best, nbest, best_score = draft, draft, -float("inf")
            calls, visited, recoveries, trace = 0, set(), [], []
            pool_scores = None
            if candidates and syllables > 1:
                path_score = records[0]["score"]
                start = time.monotonic()
                scores, distributions = evaluator.evaluate(model_context, model_code, candidates)
                pool_scores = scores.copy()
                nbest = candidates[int(np.argmax(scores))]
                seed_index = (max(range(len(records)), key=lambda j: scores[j] +
                                  args.prior_weight * records[j]["score"] / syllables)
                              if args.seed_draft == "neural" else 0)
                draft, path_score = candidates[seed_index], records[seed_index]["score"]
                # All seeds are real dictionary candidates; no gold enters selection.
                while calls < args.budget:
                    if calls:
                        scores, distributions = evaluator.evaluate(model_context, model_code, [draft])
                        index = 0
                    else:
                        index = seed_index
                    calls += 1
                    visited.add(draft)
                    trace.append(dict(text=draft, neural=scores[index], prior=path_score / syllables))
                    joint = scores[index] + args.prior_weight * path_score / syllables
                    if joint > best_score:
                        best, best_score = draft, joint
                    if calls == args.budget:
                        break
                    proposed = None
                    for prefix in evaluator.proposals(draft, distributions[index], args.margin):
                        answer = search(code, context, prefix)
                        full = [c for c in answer["candidates"] if c["end"] == syllables]
                        if full and full[0]["text"] not in visited:
                            proposed, path_score = full[0]["text"], full[0]["score"]
                            if proposed not in candidates:
                                recoveries.append(proposed)
                            break
                    if proposed is None:
                        break
                    draft = proposed
                latencies.append(1000 * (time.monotonic() - start))
            # Gold-constrained lookup is diagnostic only, after prediction is fixed.
            oracle = search(code, context, gold[0])
            oracle_reachable = any(c["end"] == syllables for c in oracle["candidates"])
            stats.update(total=1, octagram=original in gold, beam=(candidates or [""])[0] in gold,
                         nbest=nbest in gold, constrained=best in gold,
                         beam_oracle=any(t in gold for t in candidates),
                         rescue=best in gold and original not in gold,
                         regression=best not in gold and original in gold,
                         outside_beam_rescue=best in gold and best not in candidates,
                         calls=calls,
                         oracle_reachable=oracle_reachable)
            results.append(dict(id=case["id"], input=code, context=context, gold=gold,
                                octagram=original, beam=(candidates or [""])[0], nbest=nbest,
                                constrained=best, calls=calls, outside_beam=recoveries,
                                document=case.get("document"),
                                category=case.get("category"), family=case.get("family"),
                                seed_draft=args.seed_draft,
                                trace=trace))
            if args.audit:
                results[-1].update(
                    variant=case.get("variant"), parent=case.get("parent"),
                    syllables=syllables, oracle_reachable=oracle_reachable,
                    gold_in_beam=any(t in gold for t in candidates),
                    pool=[dict(text=c["text"], prior=c["score"] / syllables,
                               neural=pool_scores[j] if pool_scores is not None else None)
                          for j, c in enumerate(records)])
            if (i + 1) % 100 == 0:
                print(json.dumps(dict(stats, seconds=time.monotonic() - started)), flush=True)
    finally:
        server.stdin.close()
        server.wait(timeout=15)
    summary = dict(stats, margin=args.margin, budget=args.budget, prior_weight=args.prior_weight,
                   width=args.width, seed_draft=args.seed_draft,
                   device=evaluator.device,
                   precision="float32, TF32 disabled",
                   model=args.model, cases=str(args.cases), seconds=time.monotonic() - started,
                   reference_baseline=str(args.reference_baseline) if args.reference_baseline else None,
                   combined_ms_p50=float(np.percentile(latencies, 50)) if latencies else None,
                   combined_ms_p95=float(np.percentile(latencies, 95)) if latencies else None)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
