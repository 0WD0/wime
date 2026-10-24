#!/usr/bin/env python3
"""Compare octagram, conditioned N-best, and draft/verify/constrained search.

The Emacs process owns the real dictionary lattice. Gold labels are read only
for metrics, never sent to search or model. CUDA is for experiment throughput;
production uses the Rust CPU worker, measured separately.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from conditional import load_model, prompt, ROOT


class Evaluator:
    def __init__(self, path, device="cuda"):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model, self.vocab, self.metadata = load_model(path)
        self.model.to(device).eval()
        self.device = device
        self.index = {c: i for i, c in enumerate(self.vocab)}
        self.chinese = torch.tensor([i for i, c in enumerate(self.vocab)
                                     if len(c) == 1 and "\u3400" <= c <= "\u9fff"], device=device)

    @torch.inference_mode()
    def evaluate(self, context, code, texts):
        pre = [2] + [self.index.get(c, 1) for c in prompt(context, code, self.model.cfg.context)]
        encoded = [[self.index.get(c, 1) for c in text] for text in texts]
        if any(len(pre) + len(t) > self.model.cfg.context for t in encoded):
            raise ValueError("candidate exceeds model window")
        size = max(map(len, encoded)) + len(pre) - 1
        inputs = [pre + tail[:-1] + [0] * (size - len(pre) - len(tail) + 1) for tail in encoded]
        # Match the deployed float32 CPU worker. BF16 can flip close candidate
        # comparisons and prefix thresholds even with the same saved weights.
        logits, _ = self.model(torch.tensor(inputs, device=self.device))
        distributions = logits[:, len(pre) - 1:].float().log_softmax(-1)
        scores = [float(distributions[i, torch.arange(len(t)), t].mean()) for i, t in enumerate(encoded)]
        return scores, distributions

    def proposals(self, draft, distribution, margin):
        edits = []
        values, indices = distribution[:len(draft), self.chinese].topk(4, dim=-1)
        for position, char in enumerate(draft):
            current = float(distribution[position, self.index.get(char, 1)])
            for score, token in zip(values[position].tolist(), indices[position].tolist()):
                replacement = self.vocab[int(self.chinese[token])]
                if score > current + margin and replacement != char:
                    edits.append(draft[:position] + replacement)
        return edits[:16]


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
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--prior-weight", type=float, default=0.0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.cases.read_text()
    cases = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(l) for l in raw.splitlines()]
    cases = cases[:args.limit]
    env = dict(os.environ, WIME_DATABASE=str(Path(args.database).resolve()),
               WIME_GRAMMAR=args.grammar_dir, WIME_GRAMMAR_MODULE=str(Path(args.grammar_module).resolve()))
    server = subprocess.Popen(["emacs", "-Q", "--batch", "-L", str(ROOT / "emacs-packages/wime"),
                               "--eval", "(setq load-prefer-newer t)", "-l",
                               str(ROOT / "emacs-packages/wime/test/search-conditional.el")],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)

    def search(code, prefix="", width=8):
        server.stdin.write(json.dumps(dict(input=code, prefix=prefix, width=width)) + "\n")
        server.stdin.flush()
        line = server.stdout.readline()
        if not line:
            raise RuntimeError("search process exited")
        return json.loads(line)

    evaluator = Evaluator(args.model)
    stats, results, latencies = Counter(), [], []
    started = time.monotonic()
    try:
        for i, case in enumerate(cases):
            code, context, gold = case["input"], case.get("context", ""), case["gold"]
            base = search(code, width=1)
            beam = search(code)
            # The real input parser removes auxiliary codes only for the model;
            # every dictionary search still receives the complete typed input.
            model_code, syllables = beam["input"], beam["syllables"]
            records = [c for c in beam["candidates"] if c["end"] == syllables]
            candidates = [c["text"] for c in records]
            original = base["candidates"][0]["text"] if base["candidates"] else ""
            draft = candidates[0] if candidates else original
            best, nbest, best_score = draft, draft, -float("inf")
            calls, visited, recoveries, trace = 0, set(), [], []
            if candidates:
                path_score = records[0]["score"]
                start = time.monotonic()
                scores, distributions = evaluator.evaluate(context, model_code, candidates)
                nbest = candidates[int(np.argmax(scores))]
                # Verification starts from traditional draft, not a gold-ranked candidate.
                while calls < args.budget:
                    if calls:
                        scores, distributions = evaluator.evaluate(context, model_code, [draft])
                    calls += 1
                    visited.add(draft)
                    trace.append(dict(text=draft, neural=scores[0], prior=path_score / syllables))
                    joint = scores[0] + args.prior_weight * path_score / syllables
                    if joint > best_score:
                        best, best_score = draft, joint
                    if calls == args.budget:
                        break
                    proposed = None
                    for prefix in evaluator.proposals(draft, distributions[0], args.margin):
                        answer = search(code, prefix)
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
            stats.update(total=1, octagram=original in gold, beam=(candidates or [""])[0] in gold,
                         nbest=nbest in gold, constrained=best in gold,
                         beam_oracle=any(t in gold for t in candidates),
                         rescue=best in gold and original not in gold,
                         regression=best not in gold and original in gold,
                         outside_beam_rescue=best in gold and best not in candidates,
                         calls=calls,
                         oracle_reachable=any(c["end"] == syllables
                                              for c in search(code, gold[0])["candidates"]))
            results.append(dict(id=case["id"], input=code, context=context, gold=gold,
                                octagram=original, beam=(candidates or [""])[0], nbest=nbest,
                                constrained=best, calls=calls, outside_beam=recoveries,
                                category=case.get("category"), family=case.get("family"),
                                trace=trace))
            if (i + 1) % 100 == 0:
                print(json.dumps(dict(stats, seconds=time.monotonic() - started)), flush=True)
    finally:
        server.stdin.close()
        server.wait(timeout=15)
    summary = dict(stats, margin=args.margin, budget=args.budget, prior_weight=args.prior_weight,
                   device=evaluator.device,
                   precision="float32, TF32 disabled",
                   model=args.model, cases=str(args.cases), seconds=time.monotonic() - started,
                   combined_ms_p50=float(np.percentile(latencies, 50)) if latencies else None,
                   combined_ms_p95=float(np.percentile(latencies, 95)) if latencies else None)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
