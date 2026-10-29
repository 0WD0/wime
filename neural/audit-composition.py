#!/usr/bin/env python3
"""Check fixed-prefix conditioning against whole-input teacher forcing."""
import argparse
import json
from pathlib import Path

import torch

from composition import ENCODING, composition_prompt, prompt
from dataset_registry import Registry
from inference import ConditionalEvaluator
from replay import fingerprint
from trajectories import states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    path, snapshot = Registry().evaluation(args.dataset)
    evaluator = ConditionalEvaluator(args.model, device=args.device)
    window, results = evaluator.model.cfg.context, []
    for row in json.loads(path.read_text()):
        _, pairs = states(row)
        continuations = [pair["continuation"] for pair in pairs]
        if not continuations:
            continue
        _, whole = evaluator.evaluate(row["context"], row["input"], row["gold"][:1])
        for state in continuations:
            fixed = state["committed"]
            end = len(fixed["text"])
            expected = prompt(row["context"], row["input"], window) + fixed["text"]
            actual = composition_prompt(state["context"], state["input"], window, fixed)
            if actual != expected:
                raise AssertionError(f"conditioning changed: {row['id']} at {end}")
            _, suffix = evaluator.evaluate(state["context"], state["input"],
                                             [row["gold"][0][end:]], fixed)
            torch.testing.assert_close(whole[:, end:], suffix, rtol=0, atol=1e-5)
            results.append(dict(id=row["id"], boundary=end,
                                max_log_probability_difference=float((whole[:, end:] - suffix).abs().max())))
    report = dict(encoding=ENCODING, dataset=snapshot, model_sha256=fingerprint(args.model),
                  checked_boundaries=len(results), device=args.device,
                  max_log_probability_difference=max((r["max_log_probability_difference"] for r in results), default=0),
                  scope="Conditional teacher-forced distributions, all vocabulary entries. "
                        "This does not guarantee identical finite-beam or fused-LM rankings.",
                  results=results)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "dataset")}))


if __name__ == "__main__":
    main()
